"""Shared single-stream microphone pipeline for CED and utterance STT."""

from __future__ import annotations

import datetime
import json
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from display_transport import (
    HUDTransport,
    get_alert_display_state,
)
from emergency_system import evaluate_sound
from emergency_v3 import select_emergency_evidence
from hud_awareness import evaluate_ced_for_hud

from live_speech_to_text import (
    LiveSpeechStateMachine,
    RecognitionJob,
    RecognitionResult,
    RecognitionWorker,
    TranscriptDisplay,
    TranscriptTracker,
)
from streaming_audio import AudioFrame, AudioStreamHub


@dataclass(frozen=True)
class PipelineEvent:
    kind: str
    source: str
    capture_started_at: float
    capture_ended_at: float
    event_sequence: int
    payload: object = None


class CEDChunker:
    """Build fixed raw-audio windows; overlap defaults to zero."""

    def __init__(self, sample_rate: int = 16000, chunk_seconds: float = 5.0,
                 overlap_seconds: float = 0.0) -> None:
        if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
            raise ValueError("CED overlap must be non-negative and shorter than its chunk.")
        self.sample_rate = sample_rate
        self.chunk_samples = max(1, int(chunk_seconds * sample_rate))
        self.hop_samples = max(1, int((chunk_seconds - overlap_seconds) * sample_rate))
        self.samples = np.empty(0, dtype=np.float32)
        self.capture_started_at: float | None = None
        self.latest_frame: AudioFrame | None = None

    def add(self, frame: AudioFrame) -> list[PipelineEvent]:
        if self.capture_started_at is None:
            self.capture_started_at = frame.capture_started_at
        self.latest_frame = frame
        self.samples = np.concatenate((self.samples, frame.samples.astype(np.float32, copy=False)))
        events: list[PipelineEvent] = []
        while self.samples.size >= self.chunk_samples:
            audio = self.samples[:self.chunk_samples].copy()
            duration = self.chunk_samples / self.sample_rate
            started = float(self.capture_started_at)
            ended = started + duration
            events.append(PipelineEvent(
                "CED_DUE", "ced", started, ended,
                frame.event_sequence * 10, audio,
            ))
            self.samples = self.samples[self.hop_samples:]
            self.capture_started_at = started + self.hop_samples / self.sample_rate
        return events

    def flush(self) -> PipelineEvent | None:
        if not self.samples.size or self.latest_frame is None or self.capture_started_at is None:
            return None
        return PipelineEvent(
            "CED_DUE", "ced", self.capture_started_at,
            self.latest_frame.capture_ended_at, self.latest_frame.event_sequence * 10,
            self.samples.copy(),
        )


def classify_raw_audio(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """Classify a raw snapshot without ever routing DTLN audio to CED."""
    import soundfile as sf
    from sound_classifier import classify_audio_file

    path = Path(tempfile.gettempdir()) / f"soundguard_ced_{uuid.uuid4().hex}.wav"
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
    try:
        return classify_audio_file(str(path))
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


class CEDBranch:
    def __init__(self, frames: queue.Queue, events: queue.Queue,
                 chunk_seconds: float, overlap_seconds: float,
                 classifier: Callable[[np.ndarray, int], dict] = classify_raw_audio,
                 one_shot: bool = False, emergency_v3_specialist=None) -> None:
        self.frames, self.events = frames, events
        self.chunker = CEDChunker(chunk_seconds=chunk_seconds,
                                  overlap_seconds=overlap_seconds)
        self.classifier = classifier
        self.one_shot = one_shot
        self.emergency_v3_specialist = emergency_v3_specialist
        self.stop_requested = threading.Event()
        self.inference_stop_requested = threading.Event()
        self.inference_queue: queue.Queue[PipelineEvent] = queue.Queue(maxsize=3)
        self.thread = threading.Thread(
            target=self._run, name="ced-capture-worker", daemon=True
        )
        self.inference_thread = threading.Thread(
            target=self._run_inference, name="ced-inference-worker", daemon=True
        )
        self.frames_consumed = 0
        self.windows_processed = 0
        self.windows_dropped = 0
        self.events_dropped = 0
        self.inference_durations: list[float] = []
        self.max_window_queue_depth = 0

    def start(self) -> None:
        self.inference_thread.start()
        self.thread.start()

    def _emit(self, event: PipelineEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.events_dropped += 1

    def _classify(self, due: PipelineEvent) -> None:
        inference_started_at = time.time()
        inference_started_monotonic = time.monotonic()
        try:
            result = self.classifier(due.payload, 16000)
        except Exception as exc:
            result = {"label": "unknown", "confidence": 0.0,
                      "top_predictions": [], "error": f"{type(exc).__name__}: {exc}"}
        if self.emergency_v3_specialist is not None:
            result = dict(result)
            result["emergency_v3"] = self.emergency_v3_specialist.analyze_samples(
                due.payload, 16000
            )
        inference_ended_at = time.time()
        inference_duration = time.monotonic() - inference_started_monotonic
        result = dict(result)
        result["_ced_timing"] = {
            "inference_started_at": inference_started_at,
            "inference_ended_at": inference_ended_at,
            "inference_seconds": inference_duration,
        }
        self.inference_durations.append(inference_duration)
        self.windows_processed += 1
        self._emit(PipelineEvent(
            "CED", "ced", due.capture_started_at, due.capture_ended_at,
            due.event_sequence, result,
        ))

    def _enqueue_window(self, due: PipelineEvent) -> None:
        try:
            self.inference_queue.put_nowait(due)
            self.max_window_queue_depth = max(
                self.max_window_queue_depth, self.inference_queue.qsize()
            )
            return
        except queue.Full:
            self.windows_dropped += 1
        try:
            self.inference_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.inference_queue.put_nowait(due)
            self.max_window_queue_depth = max(
                self.max_window_queue_depth, self.inference_queue.qsize()
            )
        except queue.Full:
            self.windows_dropped += 1

    def _run_inference(self) -> None:
        while (not self.inference_stop_requested.is_set() or
               not self.inference_queue.empty()):
            try:
                due = self.inference_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._classify(due)

    def _run(self) -> None:
        one_shot_frames: list[AudioFrame] = []
        while not self.stop_requested.is_set() or not self.frames.empty():
            try:
                frame = self.frames.get(timeout=0.05)
            except queue.Empty:
                continue
            self.frames_consumed += 1
            if self.one_shot:
                one_shot_frames.append(frame)
                continue
            dues = self.chunker.add(frame)
            for due in dues:
                self._enqueue_window(due)
        if self.one_shot and one_shot_frames:
            audio = np.concatenate([frame.samples for frame in one_shot_frames])
            tail = PipelineEvent(
                "CED_DUE", "ced", one_shot_frames[0].capture_started_at,
                one_shot_frames[-1].capture_ended_at,
                one_shot_frames[-1].event_sequence * 10, audio,
            )
        else:
            tail = self.chunker.flush()
        if tail is not None:
            self._enqueue_window(tail)

    def stop(self) -> None:
        self.stop_requested.set()

    def join(self, timeout: float = 10.0) -> None:
        self.thread.join(timeout)
        if self.thread.is_alive():
            return
        self.inference_stop_requested.set()
        self.inference_thread.join(timeout)


class UtteranceTranscriber:
    """Shared frame/VAD/state-machine/recognition implementation for all modes."""

    def __init__(self, frames: queue.Queue, events: queue.Queue, *,
                 vad_threshold: float, use_dtln: bool, partial_interval: float,
                 end_silence_ms: int, max_utterance_seconds: float,
                 pre_roll_ms: int, post_roll_ms: int,
                 save_live_utterances: bool = False,
                 vad: Callable[[np.ndarray, float], dict] | None = None,
                 recognizer=None) -> None:
        self.frames, self.events = frames, events
        self.vad_threshold = vad_threshold
        self.machine = LiveSpeechStateMachine(
            partial_interval=partial_interval, end_silence_ms=end_silence_ms,
            max_utterance_seconds=max_utterance_seconds,
            pre_roll_ms=pre_roll_ms, post_roll_ms=post_roll_ms,
        )
        self.worker = RecognitionWorker(use_dtln, recognizer, save_live_utterances)
        self.vad = vad
        self.stop_requested = threading.Event()
        self.force_final = False
        self.thread = threading.Thread(target=self._run, name="utterance-worker", daemon=True)
        self.utterance_started_at = 0.0
        self.latest_frame: AudioFrame | None = None
        self.partial_pending: set[int] = set()
        self.speech_frames_produced = 0
        self.events_dropped = 0

    def start(self) -> None:
        self.worker.start()
        self.thread.start()

    def _has_speech(self, samples: np.ndarray) -> bool:
        if self.vad is not None:
            return bool(self.vad(samples, self.vad_threshold)["has_speech"])
        from voice_activity_detector import detect_speech_frame
        return bool(detect_speech_frame(samples, threshold=self.vad_threshold)["has_speech"])

    def _emit(self, event: PipelineEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.events_dropped += 1

    def _submit(self, kind: str, speech_event, frame: AudioFrame) -> None:
        sequence = frame.event_sequence * 10 + (2 if kind == "PARTIAL" else 3)
        job = RecognitionJob(
            kind, speech_event.utterance_id, speech_event.audio,
            self.utterance_started_at, frame.capture_ended_at, sequence,
        )
        if kind == "PARTIAL" and speech_event.utterance_id in self.partial_pending:
            return
        if self.worker.submit(job):
            if kind == "PARTIAL":
                self.partial_pending.add(speech_event.utterance_id)
        else:
            self._emit(PipelineEvent(
                "STT_ERROR", "transcript", job.capture_started_at,
                job.capture_ended_at, job.event_sequence,
                f"{kind} recognition queue is full",
            ))

    def _handle_state_event(self, event, frame: AudioFrame) -> None:
        if event.kind == "SPEECH_STARTED":
            self.utterance_started_at = frame.capture_started_at
            self._emit(PipelineEvent(
                "SPEECH_STARTED", "transcript", frame.capture_started_at,
                frame.capture_ended_at, frame.event_sequence * 10 + 1,
                event.utterance_id,
            ))
        elif event.kind == "PARTIAL_DUE" and event.audio is not None:
            self._submit("PARTIAL", event, frame)
        elif event.kind == "FINAL_DUE" and event.audio is not None:
            self.partial_pending.discard(event.utterance_id)
            self._submit("FINAL", event, frame)

    def _run(self) -> None:
        while not self.stop_requested.is_set() or not self.frames.empty():
            try:
                frame = self.frames.get(timeout=0.05)
            except queue.Empty:
                continue
            self.latest_frame = frame
            has_speech = self._has_speech(frame.samples)
            if has_speech:
                self.speech_frames_produced += 1
            for event in self.machine.process_frame(frame.samples, has_speech):
                self._handle_state_event(event, frame)
        if self.force_final and self.latest_frame is not None:
            final = self.machine.force_finalize()
            if final is not None:
                self._handle_state_event(final, self.latest_frame)

    def drain_results(self) -> list[RecognitionResult]:
        results = []
        while True:
            try:
                result = self.worker.results.get_nowait()
            except queue.Empty:
                return results
            self.partial_pending.discard(result.utterance_id)
            results.append(result)

    def stop(self, force_final: bool = False) -> None:
        self.force_final = force_final
        self.stop_requested.set()

    def join_capture(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout)

    def stop_recognition(self, timeout: float = 10.0) -> None:
        self.worker.stop()
        self.worker.join(timeout)


class MicrophonePipeline:
    def __init__(self, *, mode: str, emergency_system, device_index=None,
                 duration: float = 5.0, ced_overlap_seconds: float = 0.0,
                 vad_threshold: float = 0.5, use_dtln: bool = True,
                 partial_interval: float = 1.5, end_silence_ms: int = 700,
                 max_utterance_seconds: float = 12.0, pre_roll_ms: int = 250,
                 post_roll_ms: int = 500, queue_seconds: float = 4.0,
                 save_live_utterances: bool = False, stream_factory=None,
                 classifier=classify_raw_audio, vad=None, recognizer=None,
                 display: TranscriptDisplay | None = None,
                 hud: HUDTransport | None = None,
                 show_all_detected_sounds_on_hud: bool = False,
                 ced_display_ttl_seconds: float = 5.5,
                 counter_log_interval_seconds: float = 5.0,
                 emergency_v3_specialist=None,
                 clock=time.monotonic) -> None:
        if mode not in {"mic", "continuous", "live-stt"}:
            raise ValueError(f"Unsupported microphone mode: {mode}")
        self.mode, self.emergency_system = mode, emergency_system
        self.duration, self.clock = duration, clock
        self.display = display or TranscriptDisplay()
        self.hud = hud or HUDTransport()
        self.show_all_detected_sounds_on_hud = show_all_detected_sounds_on_hud
        if ced_display_ttl_seconds <= 0:
            raise ValueError("CED display TTL must be greater than zero.")
        self.ced_display_ttl_seconds = float(ced_display_ttl_seconds)
        if counter_log_interval_seconds <= 0:
            raise ValueError("Counter log interval must be greater than zero.")
        self.counter_log_interval_seconds = float(counter_log_interval_seconds)
        self.next_counter_log_at: float | None = None
        self.active_ced_display_label = ""
        self.ced_display_expires_at: float | None = None
        self.ced_last_confirmed_at: float | None = None
        self.ced_top_k_recoveries = 0
        self.ced_held_label_refreshes = 0
        self.ced_ttl_expirations = 0
        self.ced_clear_requests = 0
        self.active_hud_alert = ""
        self.tracker = TranscriptTracker()
        self.events: queue.Queue[PipelineEvent] = queue.Queue(maxsize=64)
        self.hub = AudioStreamHub(
            queue_seconds=queue_seconds,
            ced_queue_seconds=max(queue_seconds, duration * 2.0),
            device_index=device_index,
            stream_factory=stream_factory,
        )
        self.speech = UtteranceTranscriber(
            self.hub.speech_frames, self.events, vad_threshold=vad_threshold,
            use_dtln=use_dtln, partial_interval=partial_interval,
            end_silence_ms=end_silence_ms,
            max_utterance_seconds=max_utterance_seconds,
            pre_roll_ms=pre_roll_ms, post_roll_ms=post_roll_ms,
            save_live_utterances=save_live_utterances,
            vad=vad, recognizer=recognizer,
        )
        self.ced = CEDBranch(
            self.hub.ced_frames, self.events, duration, ced_overlap_seconds,
            classifier, one_shot=mode == "mic",
            emergency_v3_specialist=emergency_v3_specialist,
        )
        self.final_seen = False
        self.ced_events_dispatched = 0
        self.stt_partial_count = 0
        self.stt_final_count = 0
        self.last_sequence_displayed = {"ced": -1, "transcript": -1}

    @staticmethod
    def _timestamp(value: float) -> str:
        return datetime.datetime.fromtimestamp(value).isoformat(timespec="milliseconds")

    def _dispatch_recognition(self, result: RecognitionResult) -> None:
        for message in result.logs:
            self.display.print_permanent(message)
        if result.error:
            self.display.print_permanent(f"STT {result.kind.lower()} error: {result.error}")
            return
        if result.kind == "PARTIAL":
            self.stt_partial_count += 1
            value = self.tracker.accept_partial(result.utterance_id, result.transcript)
            if value and result.event_sequence >= self.last_sequence_displayed["transcript"]:
                self.last_sequence_displayed["transcript"] = result.event_sequence
                self.display.show_partial(value)
                self.hud.set_partial_subtitle(value)
                self.display.print_permanent(
                    "STT PARTIAL timing: " + json.dumps({
                        "capture_ended_at": result.capture_ended_at,
                        "hud_sent_at": time.time(),
                    }, sort_keys=True)
                )
            return
        self.stt_final_count += 1
        value = self.tracker.accept_final(result.utterance_id, result.transcript)
        self.final_seen = True
        if value:
            self.display.print_permanent(f"FINAL: {value}")
            self.hud.set_subtitle(value)
            self.display.print_permanent(
                "STT FINAL timing: " + json.dumps({
                    "capture_ended_at": result.capture_ended_at,
                    "hud_sent_at": time.time(),
                }, sort_keys=True)
            )
            decision = self.emergency_system.process_transcript_event(
                value, timestamp=self._timestamp(result.capture_ended_at))
            self._print_decision(decision)
        else:
            self.display.print_permanent("FINAL: [empty]")
        self.tracker.mark_final_displayed(result.utterance_id)
        if self.mode != "mic":
            self.display.print_permanent("[WAITING FOR SPEECH]")

    def _print_decision(self, decision: dict) -> None:
        self.display.print_permanent(f"Help request: {decision['help_request_detected']}")
        self.display.print_permanent(f"Alert level: {decision['alert_level']}")
        self.display.print_permanent(f"Alert message: {decision['alert_text'] or '[none]'}")
        self.display.print_permanent(f"Event state: {decision['event_state']}")
        self.display.print_permanent(f"Context: {decision['context']}")
        self.display.print_permanent(f"Emitted this cycle: {decision['emitted_this_cycle']}")
        event_label, help_active = get_alert_display_state(
            self.emergency_system.active_sound,
            self.emergency_system.help_active,
        )
        if event_label or help_active:
            self.active_hud_alert = event_label or "SOS"
            self.hud.set_alert_state(event_label, help_active)
        else:
            self.active_hud_alert = ""
            self.hud.set_alert_state()

    def _clear_ced_display(self) -> None:
        if not self.active_ced_display_label:
            self.ced_display_expires_at = None
            return
        self.active_ced_display_label = ""
        self.ced_display_expires_at = None
        self.ced_last_confirmed_at = None
        self.ced_clear_requests += 1
        self.hud.set_environmental_sound("")

    def _expire_ced_display_if_needed(self) -> None:
        if (self.active_ced_display_label and
                self.ced_display_expires_at is not None and
                self.clock() >= self.ced_display_expires_at):
            self.ced_ttl_expirations += 1
            self._clear_ced_display()

    def _select_ced_display_label(self, result: dict,
                                  decision: dict) -> tuple[str, list[dict], bool]:
        del decision
        def prediction_score(item: dict) -> float:
            try:
                return float(item.get("score", 0.0))
            except (TypeError, ValueError):
                return 0.0
        candidates = [{
            "label": str(result.get("label", "unknown")),
            "score": float(result.get("confidence", 0.0)),
            "source": "top1",
        }]
        candidates.extend({**prediction, "source": "top_k"}
                          for prediction in result.get("top_predictions") or [])
        ranked = sorted(
            candidates,
            key=prediction_score,
            reverse=True,
        )
        evaluations = []
        seen = set()
        for prediction in ranked:
            label = str(prediction.get("label", "unknown"))
            try:
                score = float(prediction.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            key = (label, score)
            if key in seen:
                continue
            seen.add(key)
            evaluation = evaluate_ced_for_hud(
                label, score, self.emergency_system.context
            )
            emergency_evaluation = evaluate_sound(
                label, score, self.emergency_system.context
            )
            evaluation["emergency_threshold"] = emergency_evaluation.get(
                "threshold"
            )
            evaluation["passes_emergency_threshold"] = bool(
                emergency_evaluation.get("matched")
            )
            evaluation["source"] = prediction.get("source", "top_k")
            evaluations.append(evaluation)
        accepted = [item for item in evaluations if item["accepted"]]
        if not accepted:
            return "", evaluations, False
        selected = max(accepted, key=lambda item: item["score"])
        recovered = selected["source"] != "top1"
        return str(selected["display_label"]), evaluations, recovered

    def _update_ced_display(self, result: dict, decision: dict) -> None:
        if not self.show_all_detected_sounds_on_hud:
            self._clear_ced_display()
            return
        previous_label = self.active_ced_display_label
        previous_confirmed = self.ced_last_confirmed_at
        frames_before = int(getattr(self.hud, "counters", {}).get(
            "C_frames_sent", 0
        ))
        display_label, evaluations, recovered = self._select_ced_display_label(
            result, decision
        )
        if display_label:
            if recovered:
                self.ced_top_k_recoveries += 1
            if display_label == previous_label:
                self.ced_held_label_refreshes += 1
            self.active_ced_display_label = display_label
            self.ced_last_confirmed_at = self.clock()
            self.ced_display_expires_at = self.ced_last_confirmed_at + self.ced_display_ttl_seconds
            self.hud.set_environmental_sound(display_label)
        elif decision.get("event_state") == "EVENT_ENDED":
            self._clear_ced_display()
        else:
            # Speech/background top-1 results are absence of fresh display
            # evidence, not reliable proof that the prior event has ended.
            self._expire_ced_display_if_needed()
        frames_after = int(getattr(self.hud, "counters", {}).get(
            "C_frames_sent", 0
        ))
        now = self.clock()
        timing = result.get("_ced_timing") or {}
        self.display.print_permanent(
            "CED window audit: " + json.dumps({
                "window_start": result.get("_window_start"),
                "window_end": result.get("_window_end"),
                "inference_start": timing.get("inference_started_at"),
                "inference_end": timing.get("inference_ended_at"),
                "inference_seconds": timing.get("inference_seconds"),
                "top1": result.get("label"),
                "top1_score": result.get("confidence"),
                "top5": result.get("top_predictions") or [],
                "hud_evaluations": evaluations,
                "selected_hud_label": display_label,
                "active_hud_label": self.active_ced_display_label,
                "previous_hud_label": previous_label,
                "previous_confirmed_at": previous_confirmed,
                "last_confirmed_at": self.ced_last_confirmed_at,
                "ttl_remaining": max(
                    0.0, (self.ced_display_expires_at or now) - now
                ),
                "c_frame_sent": frames_after > frames_before,
                "clear_requested": bool(previous_label and not self.active_ced_display_label),
            }, sort_keys=True)
        )

    def _dispatch_event(self, event: PipelineEvent) -> None:
        if event.kind == "SPEECH_STARTED":
            self.display.print_permanent("[SPEECH STARTED]")
            self.display.print_permanent(
                f"VAD speech start: {event.capture_started_at:.6f}"
            )
            self.hud.set_partial_subtitle("")
        elif event.kind == "STT_ERROR":
            self.display.print_permanent(str(event.payload))
        elif event.kind == "CED":
            self.ced_events_dispatched += 1
            result = event.payload
            result = dict(result)
            result["_window_start"] = event.capture_started_at
            result["_window_end"] = event.capture_ended_at
            label = result.get("label", "unknown")
            confidence = float(result.get("confidence", 0.0))
            emergency_label, emergency_confidence, emergency_source = select_emergency_evidence(
                str(label), confidence, result.get("emergency_v3"),
                self.emergency_system.context,
            )
            self.display.print_permanent(f"Detected sound: {label}")
            self.display.print_permanent(f"Confidence: {confidence}")
            decision = self.emergency_system.process_sound_event(
                emergency_label, emergency_confidence,
                timestamp=self._timestamp(event.capture_ended_at))
            if result.get("emergency_v3") is not None:
                specialist = result["emergency_v3"]
                self.display.print_permanent(
                    "Emergency V3: "
                    f"source={emergency_source}, "
                    f"detected={specialist.get('detected', False)}, "
                    f"category={specialist.get('category', 'none')}"
                )
            self._print_decision(decision)
            self._update_ced_display(result, decision)

    def _pump(self) -> None:
        self._expire_ced_display_if_needed()
        for result in self.speech.drain_results():
            self._dispatch_recognition(result)
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            self._dispatch_event(event)

    def _report_runtime_counters_if_due(self) -> None:
        now = self.clock()
        if self.next_counter_log_at is None:
            self.next_counter_log_at = now + self.counter_log_interval_seconds
            return
        if now < self.next_counter_log_at:
            return
        counters = self.counters
        names = (
            "raw_frames_captured", "speech_frames_produced",
            "ced_frames_produced", "ced_frames_consumed",
            "ced_frames_dropped", "ced_windows_processed",
            "ced_events_dispatched", "C_frames_sent",
        )
        self.display.print_permanent(
            "Live counters: " + ", ".join(
                f"{name}={counters[name]}" for name in names
            )
        )
        while self.next_counter_log_at <= now:
            self.next_counter_log_at += self.counter_log_interval_seconds

    def run(self) -> int:
        self.display.print_permanent("[WAITING FOR SPEECH]")
        self.hud.set_status("LIVE")
        started = self.clock()
        interrupted = False
        try:
            self.speech.start()
            self.ced.start()
            with self.hub:
                while True:
                    self.hud.update()
                    self._pump()
                    self._report_runtime_counters_if_due()
                    if self.mode == "mic" and self.final_seen:
                        break
                    if self.mode == "mic" and self.clock() - started >= self.duration:
                        break
                    time.sleep(0.01)
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:
            self.display.print_permanent(f"Audio pipeline error: {type(exc).__name__}: {exc}")
            return 1
        finally:
            self.speech.stop(force_final=self.mode == "mic")
            self.ced.stop()
            self.speech.join_capture()
            self.ced.join()
            # Let already-submitted FINAL recognition complete before stopping.
            time.sleep(0.02)
            self.speech.stop_recognition()
            self._pump()
            counters = self.counters
            self.display.print_permanent(
                "Queue counters: " + ", ".join(f"{key}={value}" for key, value in counters.items())
            )
        if interrupted:
            self.display.print_permanent("Shutdown requested. Microphone and workers closed.")
        return 0

    @property
    def counters(self) -> dict[str, int]:
        values = dict(self.hub.counters)
        values["speech_frames_produced"] = self.speech.speech_frames_produced
        values["speech_events_dropped"] = self.speech.events_dropped
        values["recognition_jobs_dropped"] = self.speech.worker.jobs_dropped
        values["recognition_results_dropped"] = self.speech.worker.results_dropped
        values["ced_frames_consumed"] = self.ced.frames_consumed
        values["ced_events_dropped"] = self.ced.events_dropped
        values["ced_windows_processed"] = self.ced.windows_processed
        values["ced_windows_dropped"] = self.ced.windows_dropped
        values["ced_window_queue_max_depth"] = self.ced.max_window_queue_depth
        values["ced_inference_seconds_total"] = sum(self.ced.inference_durations)
        values["ced_inference_seconds_max"] = max(
            self.ced.inference_durations, default=0.0
        )
        values["ced_events_dispatched"] = self.ced_events_dispatched
        values["ced_top_k_recoveries"] = self.ced_top_k_recoveries
        values["ced_held_label_refreshes"] = self.ced_held_label_refreshes
        values["ced_ttl_expirations"] = self.ced_ttl_expirations
        values["ced_clear_requests"] = self.ced_clear_requests
        values["stt_partial_count"] = self.stt_partial_count
        values["stt_final_count"] = self.stt_final_count
        hud_counters = getattr(self.hud, "counters", {})
        values["C_frames_sent"] = int(hud_counters.get("C_frames_sent", 0))
        return values
