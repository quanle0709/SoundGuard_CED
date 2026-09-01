"""Near-real-time VAD and transcript lifecycle for SoundGuard."""

from __future__ import annotations

import math
import queue
import re
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SpeechEvent:
    kind: str
    utterance_id: int
    audio: np.ndarray | None = None


class LiveSpeechStateMachine:
    """Pure frame-driven speech state machine, independent of Silero and I/O."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        pre_roll_ms: int = 250,
        end_silence_ms: int = 700,
        post_roll_ms: int = 500,
        partial_interval: float = 1.5,
        max_utterance_seconds: float = 12.0,
    ) -> None:
        if sample_rate <= 0 or frame_samples <= 0:
            raise ValueError("Sample rate and frame size must be positive.")
        if pre_roll_ms < 0 or post_roll_ms < 0 or end_silence_ms <= 0:
            raise ValueError("Roll durations cannot be negative and end silence must be positive.")
        if partial_interval <= 0 or max_utterance_seconds <= partial_interval:
            raise ValueError("Maximum utterance must exceed the partial interval.")
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        frame_seconds = frame_samples / sample_rate
        self.pre_roll_frames = max(1, math.ceil((pre_roll_ms / 1000) / frame_seconds))
        self.end_silence_frames = max(1, math.ceil((end_silence_ms / 1000) / frame_seconds))
        self.post_roll_frames = math.ceil((post_roll_ms / 1000) / frame_seconds)
        self.partial_frames = max(1, math.ceil(partial_interval / frame_seconds))
        self.max_frames = max(1, math.ceil(max_utterance_seconds / frame_seconds))
        self.pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self.utterance: list[np.ndarray] = []
        self.speaking = False
        self.silence_frames = 0
        self.collecting_post_roll = False
        self.post_roll_collected = 0
        self.next_partial_at = self.partial_frames
        self.utterance_id = 0

    def _snapshot(self) -> np.ndarray:
        return np.concatenate(self.utterance).astype(np.float32, copy=True)

    def _finish(self) -> SpeechEvent:
        event = SpeechEvent("FINAL_DUE", self.utterance_id, self._snapshot())
        recent = self.utterance[-self.pre_roll_frames:]
        self.pre_roll.clear()
        self.pre_roll.extend(frame.copy() for frame in recent)
        self.utterance = []
        self.speaking = False
        self.silence_frames = 0
        self.collecting_post_roll = False
        self.post_roll_collected = 0
        self.next_partial_at = self.partial_frames
        return event

    def force_finalize(self) -> SpeechEvent | None:
        """Return the complete latest snapshot when capture must stop."""
        return self._finish() if self.speaking and self.utterance else None

    def process_frame(self, samples: np.ndarray, has_speech: bool) -> list[SpeechEvent]:
        frame = np.asarray(samples, dtype=np.float32).reshape(-1)
        if frame.size != self.frame_samples:
            raise ValueError(f"Expected {self.frame_samples} samples, got {frame.size}.")
        events: list[SpeechEvent] = []
        if not self.speaking:
            self.pre_roll.append(frame.copy())
            if not has_speech:
                return events
            self.speaking = True
            self.utterance_id += 1
            self.utterance = [item.copy() for item in self.pre_roll]
            self.silence_frames = 0
            self.next_partial_at = self.partial_frames
            events.append(SpeechEvent("SPEECH_STARTED", self.utterance_id))
        else:
            self.utterance.append(frame.copy())

        # The current frame is already in ``utterance`` before any transition
        # is evaluated. FINAL snapshots are always freshly built by _finish().
        if has_speech:
            self.silence_frames = 0
            self.collecting_post_roll = False
            self.post_roll_collected = 0
        else:
            self.silence_frames += 1
        frame_count = len(self.utterance)
        if frame_count >= self.max_frames:
            events.append(self._finish())
        elif not has_speech and self.silence_frames >= self.end_silence_frames:
            if not self.collecting_post_roll:
                self.collecting_post_roll = True
                self.post_roll_collected = 0
            else:
                self.post_roll_collected += 1
            if self.post_roll_collected >= self.post_roll_frames:
                events.append(self._finish())
        elif not self.collecting_post_roll and frame_count >= self.next_partial_at:
            events.append(SpeechEvent("PARTIAL_DUE", self.utterance_id, self._snapshot()))
            self.next_partial_at += self.partial_frames
        return events


class TranscriptTracker:
    def __init__(self) -> None:
        self.partials: dict[int, str] = {}
        self.finalized_ids: set[int] = set()
        self.history: list[str] = []
        self.current_utterance_id: int | None = None
        self.last_stable_partial = ""

    @staticmethod
    def normalize_transcript(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    @staticmethod
    def _is_preferred_partial(previous: str, candidate: str) -> bool:
        if not previous:
            return True
        previous_words = previous.casefold().split()
        candidate_words = candidate.casefold().split()
        if candidate_words[:len(previous_words)] == previous_words:
            return True
        if previous_words[:len(candidate_words)] == candidate_words:
            return len(candidate_words) >= len(previous_words)
        if len(candidate_words) < len(previous_words):
            return False

        # For a non-prefix revision, require both genuinely new information
        # and preserved word order. This rejects equal-length word shuffles.
        new_tokens = set(candidate_words) - set(previous_words)
        if not new_tokens:
            return False
        lengths = [0] * (len(candidate_words) + 1)
        for previous_word in previous_words:
            diagonal = 0
            for index, candidate_word in enumerate(candidate_words, start=1):
                saved = lengths[index]
                if previous_word == candidate_word:
                    lengths[index] = diagonal + 1
                else:
                    lengths[index] = max(lengths[index], lengths[index - 1])
                diagonal = saved
        common_in_order = lengths[-1]
        overlap = common_in_order / max(len(previous_words), len(candidate_words))
        return overlap >= 0.75

    def accept_partial(self, utterance_id: int, text: str) -> str | None:
        value = self.normalize_transcript(text)
        if not value or utterance_id in self.finalized_ids:
            return None
        if self.current_utterance_id != utterance_id:
            self.current_utterance_id = utterance_id
            self.last_stable_partial = ""
        previous = self.last_stable_partial
        if value == previous:
            return None
        if not self._is_preferred_partial(previous, value):
            # Google can temporarily return a shorter fragment, reordered
            # words, or a weakly related hypothesis for accumulated audio.
            return None
        self.partials[utterance_id] = value
        self.last_stable_partial = value
        return value

    def accept_final(self, utterance_id: int, text: str) -> str | None:
        if utterance_id in self.finalized_ids:
            return None
        value = self.normalize_transcript(text)
        if not value and self.current_utterance_id == utterance_id:
            value = self.last_stable_partial
        if not value:
            return None
        self.finalized_ids.add(utterance_id)
        self.partials.pop(utterance_id, None)
        self.history.append(value)
        return value

    def mark_final_displayed(self, utterance_id: int) -> None:
        """Clear temporary state only after its FINAL line is permanent."""
        if self.current_utterance_id == utterance_id:
            self.current_utterance_id = None
            self.last_stable_partial = ""

    def reset(self) -> None:
        self.partials.clear()
        self.finalized_ids.clear()
        self.history.clear()
        self.current_utterance_id = None
        self.last_stable_partial = ""


class TranscriptDisplay:
    """Own one temporary terminal line; permanent lines are never rewritten."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.partial_visible = False
        self._lock = threading.RLock()

    def show_partial(self, text: str) -> None:
        with self._lock:
            self.stream.write(f"\r\x1b[2KPARTIAL: {text}")
            self.stream.flush()
            self.partial_visible = True

    def clear_partial(self) -> None:
        with self._lock:
            if self.partial_visible:
                self.stream.write("\r\x1b[2K")
                self.stream.flush()
                self.partial_visible = False

    def print_permanent(self, text: str) -> None:
        with self._lock:
            had_partial = self.partial_visible
            self.clear_partial()
            if had_partial:
                self.stream.write("\n")
            self.stream.write(f"{text}\n")
            self.stream.flush()


@dataclass(frozen=True)
class RecognitionJob:
    kind: str
    utterance_id: int
    audio: np.ndarray
    capture_started_at: float = 0.0
    capture_ended_at: float = 0.0
    event_sequence: int = 0


@dataclass(frozen=True)
class RecognitionResult:
    kind: str
    utterance_id: int
    transcript: str = ""
    error: str = ""
    capture_started_at: float = 0.0
    capture_ended_at: float = 0.0
    event_sequence: int = 0
    source: str = "transcript"
    logs: tuple[str, ...] = ()


class RecognitionWorker:
    """A bounded daemon worker; slow recognition never blocks audio capture."""

    def __init__(self, use_dtln: bool, recognizer: Callable[[np.ndarray, bool], str] | None = None,
                 save_live_utterances: bool = False) -> None:
        self.use_dtln = use_dtln
        self.recognizer = recognizer
        self.save_live_utterances = save_live_utterances
        self.jobs: queue.Queue[RecognitionJob | None] = queue.Queue(maxsize=8)
        self.results: queue.Queue[RecognitionResult] = queue.Queue(maxsize=16)
        self.thread = threading.Thread(target=self._run, name="live-stt-worker", daemon=True)
        self.jobs_dropped = 0
        self.results_dropped = 0

    def start(self) -> None:
        self.thread.start()

    def submit(self, job: RecognitionJob) -> bool:
        try:
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            self.jobs_dropped += 1
            return False

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                logs: list[str] = []
                if self.recognizer is not None:
                    transcript = self.recognizer(job.audio, self.use_dtln)
                else:
                    transcript = recognize_snapshot(
                        job.audio,
                        self.use_dtln,
                        utterance_id=job.utterance_id,
                        is_final=job.kind == "FINAL",
                        save_live_utterances=self.save_live_utterances,
                        log_messages=logs,
                    )
                result = RecognitionResult(
                    job.kind, job.utterance_id, transcript=transcript,
                    capture_started_at=job.capture_started_at,
                    capture_ended_at=job.capture_ended_at,
                    event_sequence=job.event_sequence,
                    logs=tuple(logs),
                )
            except Exception as exc:
                result = RecognitionResult(
                    job.kind, job.utterance_id, error=f"{type(exc).__name__}: {exc}",
                    capture_started_at=job.capture_started_at,
                    capture_ended_at=job.capture_ended_at,
                    event_sequence=job.event_sequence,
                )
            try:
                self.results.put_nowait(result)
            except queue.Full:
                self.results_dropped += 1

    def stop(self) -> None:
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            # Preserve shutdown and FINAL delivery by evicting stale work.
            try:
                self.jobs.get_nowait()
                self.jobs.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass

    def join(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout)


def recognize_snapshot(audio: np.ndarray, use_dtln: bool, *, utterance_id: int = 0,
                       is_final: bool = False, save_live_utterances: bool = False,
                       log_messages: list[str] | None = None) -> str:
    """Recognize one accumulated utterance; heavy dependencies load on demand."""
    import soundfile as sf
    from soundguard.audio.audio_capture import create_speech_optimized_wav
    from soundguard.speech.speech_enhancer import enhance_audio_file
    from soundguard.speech.speech_recognizer import transcribe_audio_file

    debug_save = save_live_utterances and is_final
    live_dir = (Path("test_outputs") / "live_utterances" if debug_save
                else Path(tempfile.gettempdir()) / "soundguard_live_stt")
    live_dir.mkdir(parents=True, exist_ok=True)
    token = f"utterance_{utterance_id:04d}" if debug_save else uuid.uuid4().hex
    raw_path = live_dir / f"{token}_raw.wav"
    denoised_path = live_dir / f"{token}_dtln.wav"
    optimized_path = live_dir / f"{token}_speech_optimized.wav"
    sf.write(raw_path, np.asarray(audio, dtype=np.float32), 16000, subtype="PCM_16")
    try:
        source = raw_path
        if use_dtln:
            try:
                source = Path(enhance_audio_file(raw_path, denoised_path, verbose=False))
            except Exception as exc:
                messages = log_messages if log_messages is not None else []
                messages.append(f"DTLN warning: {type(exc).__name__}: {exc}")
                messages.append("DTLN fallback: using the raw utterance for Speech-to-Text.")
        speech_path = create_speech_optimized_wav(
            source,
            output_path=optimized_path,
            preserve_tail_ms=500 if is_final else 0,
        )
        return transcribe_audio_file(str(speech_path))
    finally:
        for path in (raw_path, denoised_path, optimized_path):
            if debug_save:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def process_recognition_result(
    result: RecognitionResult,
    tracker: TranscriptTracker,
    display: TranscriptDisplay,
    emergency_system,
    partial_pending: set[int] | None = None,
) -> dict | None:
    """Apply one worker result and permanently print accepted FINAL decisions."""
    if result.kind == "PARTIAL" and partial_pending is not None:
        partial_pending.discard(result.utterance_id)
    for message in result.logs:
        display.print_permanent(message)
    if result.error:
        display.print_permanent(f"STT {result.kind.lower()} error: {result.error}")
        return None
    if result.kind == "PARTIAL":
        value = tracker.accept_partial(result.utterance_id, result.transcript)
        if value:
            display.show_partial(value)
        return None

    value = tracker.accept_final(result.utterance_id, result.transcript)
    decision = None
    if value:
        display.print_permanent(f"FINAL: {value}")
        decision = emergency_system.evaluate(
            transcript=value,
            sound_label="unknown",
            sound_confidence=0.0,
        )
        active = decision["event_state"] in {"EVENT_STARTED", "EVENT_CONTINUING"}
        cooldown_active = bool(active and not decision["emitted_this_cycle"])
        decision = {**decision, "cooldown_active": cooldown_active}
        display.print_permanent(
            f"Help request: {decision['help_request_detected']}"
        )
        display.print_permanent(f"Alert level: {decision['alert_level']}")
        display.print_permanent(
            f"Alert message: {decision['alert_text'] or '[none]'}"
        )
        display.print_permanent(f"Event state: {decision['event_state']}")
        display.print_permanent(f"Context: {decision['context']}")
        display.print_permanent(f"Cooldown active: {cooldown_active}")
        display.print_permanent(
            f"Emitted this cycle: {decision['emitted_this_cycle']}"
        )
    else:
        display.print_permanent("FINAL: [empty]")
    tracker.mark_final_displayed(result.utterance_id)
    display.print_permanent("[WAITING FOR SPEECH]")
    return decision


def run_live_stt(
    *, device_index: int | None, vad_threshold: float, use_dtln: bool,
    partial_interval: float, end_silence_ms: int,
    max_utterance_seconds: float, pre_roll_ms: int, post_roll_ms: int,
    queue_seconds: float, save_live_utterances: bool, emergency_system,
) -> int:
    from soundguard.audio.streaming_audio import StreamingAudioInput
    from soundguard.speech.voice_activity_detector import (
        detect_speech_frame,
        reset_streaming_vad,
    )

    machine = LiveSpeechStateMachine(
        pre_roll_ms=pre_roll_ms,
        end_silence_ms=end_silence_ms,
        post_roll_ms=post_roll_ms,
        partial_interval=partial_interval,
        max_utterance_seconds=max_utterance_seconds,
    )
    tracker = TranscriptTracker()
    display = TranscriptDisplay()
    worker = RecognitionWorker(
        use_dtln=use_dtln,
        save_live_utterances=save_live_utterances,
    )
    worker.start()
    partial_pending: set[int] = set()

    def consume_results() -> None:
        while True:
            try:
                result = worker.results.get_nowait()
            except queue.Empty:
                return
            process_recognition_result(
                result,
                tracker,
                display,
                emergency_system,
                partial_pending,
            )

    print("[WAITING FOR SPEECH]")
    try:
        with StreamingAudioInput(queue_seconds=queue_seconds, device_index=device_index) as audio_input:
            while True:
                consume_results()
                try:
                    frame = audio_input.get(timeout=0.1)
                except queue.Empty:
                    continue
                vad = detect_speech_frame(frame.samples, threshold=vad_threshold)
                for event in machine.process_frame(frame.samples, vad["has_speech"]):
                    if event.kind == "SPEECH_STARTED":
                        print("[SPEECH STARTED]")
                    elif event.kind == "PARTIAL_DUE" and event.audio is not None:
                        if event.utterance_id not in partial_pending:
                            accepted = worker.submit(RecognitionJob("PARTIAL", event.utterance_id, event.audio))
                            if accepted:
                                partial_pending.add(event.utterance_id)
                            else:
                                print("STT partial skipped: recognition queue is full")
                    elif event.kind == "FINAL_DUE" and event.audio is not None:
                        partial_pending.discard(event.utterance_id)
                        if not worker.submit(RecognitionJob("FINAL", event.utterance_id, event.audio)):
                            print("STT final skipped: recognition queue is full")
                            print("[WAITING FOR SPEECH]")
                        reset_streaming_vad()
    except KeyboardInterrupt:
        print("\nLive STT stopped. Microphone closed.")
        return 0
    except Exception as exc:
        print(f"Live STT error: {type(exc).__name__}: {exc}")
        return 1
    finally:
        worker.stop()
