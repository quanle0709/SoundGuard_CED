"""Bounded, continuous microphone capture for live speech recognition."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    overflowed: bool = False
    capture_started_at: float = 0.0
    capture_ended_at: float = 0.0
    event_sequence: int = 0


class AudioStreamHub:
    """One physical stream fanned out to independent bounded raw queues."""

    def __init__(self, sample_rate: int = 16000, frame_samples: int = 512,
                 queue_seconds: float = 4.0, device_index: int | None = None,
                 stream_factory=None, clock=time.time,
                 ced_queue_seconds: float | None = None,
                 familiar_sound_enabled=None,
                 familiar_queue_seconds: float = 6.0) -> None:
        if (sample_rate <= 0 or frame_samples <= 0 or queue_seconds <= 0 or
                (ced_queue_seconds is not None and ced_queue_seconds <= 0)):
            raise ValueError("Audio stream dimensions must be positive.")
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device_index = device_index
        frame_seconds = frame_samples / sample_rate
        speech_capacity = max(1, int(queue_seconds / frame_seconds))
        ced_capacity = max(
            1, int((ced_queue_seconds or queue_seconds) / frame_seconds)
        )
        self.speech_frames: queue.Queue[AudioFrame] = queue.Queue(
            maxsize=speech_capacity
        )
        self.ced_frames: queue.Queue[AudioFrame] = queue.Queue(
            maxsize=ced_capacity
        )
        self.familiar_sound_frames: queue.Queue[AudioFrame] = queue.Queue(
            maxsize=max(1, int(familiar_queue_seconds / frame_seconds))
        )
        self._familiar_sound_enabled = familiar_sound_enabled or (lambda: False)
        self.raw_frames_captured = 0
        self.ced_frames_produced = 0
        self.dropped_speech_frames = 0
        self.dropped_ced_frames = 0
        self.familiar_frames_produced = 0
        self.dropped_familiar_frames = 0
        self.stream_statuses: deque[str] = deque(maxlen=32)
        self._stream_factory = stream_factory
        self._clock = clock
        self._stream = None
        self._sequence = 0
        self._lock = threading.Lock()
        self.capture_started_at: float | None = None
        self.capture_ended_at: float | None = None

    @staticmethod
    def _offer(target: queue.Queue, frame: AudioFrame) -> bool:
        try:
            target.put_nowait(frame)
            return False
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(frame)
            except queue.Full:
                pass
            return True

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        ended = self._clock()
        started = ended - self.frame_samples / self.sample_rate
        raw = np.asarray(indata[:, 0], dtype=np.float32).reshape(-1).copy()
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            self.raw_frames_captured += 1
            if self.capture_started_at is None:
                self.capture_started_at = started
            self.capture_ended_at = ended
        if status:
            self.stream_statuses.append(str(status))
        speech_frame = AudioFrame(raw.copy(), bool(status), started, ended, sequence)
        ced_frame = AudioFrame(raw.copy(), bool(status), started, ended, sequence)
        if self._offer(self.speech_frames, speech_frame):
            self.dropped_speech_frames += 1
        if self._offer(self.ced_frames, ced_frame):
            self.dropped_ced_frames += 1
        self.ced_frames_produced += 1
        try:
            familiar_enabled = bool(self._familiar_sound_enabled())
        except Exception:
            familiar_enabled = False
        if familiar_enabled:
            familiar_frame = AudioFrame(raw.copy(), bool(status), started, ended, sequence)
            if self._offer(self.familiar_sound_frames, familiar_frame):
                self.dropped_familiar_frames += 1
            self.familiar_frames_produced += 1

    def __enter__(self) -> "AudioStreamHub":
        if self._stream_factory is None:
            import sounddevice as sd
            factory = sd.InputStream
        else:
            factory = self._stream_factory
        self._stream = factory(
            samplerate=self.sample_rate, blocksize=self.frame_samples,
            channels=1, dtype="float32", device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    @property
    def counters(self) -> dict[str, int]:
        return {
            "raw_frames_captured": self.raw_frames_captured,
            "ced_frames_produced": self.ced_frames_produced,
            "speech_frames_dropped": self.dropped_speech_frames,
            "ced_frames_dropped": self.dropped_ced_frames,
            "familiar_frames_produced": self.familiar_frames_produced,
            "familiar_frames_dropped": self.dropped_familiar_frames,
        }


class StreamingAudioInput:
    """Keep a sounddevice stream open and expose copied frames via a queue."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        queue_seconds: float = 4.0,
        device_index: int | None = None,
    ) -> None:
        if sample_rate <= 0 or frame_samples <= 0 or queue_seconds <= 0:
            raise ValueError("Audio stream dimensions must be positive.")
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device_index = device_index
        frame_seconds = frame_samples / sample_rate
        self.frames: queue.Queue[AudioFrame] = queue.Queue(
            maxsize=max(1, int(queue_seconds / frame_seconds))
        )
        self.dropped_frames = 0
        self.stream_statuses: deque[str] = deque(maxlen=32)
        self._stream = None

    def _enqueue(self, samples: np.ndarray, overflowed: bool = False) -> None:
        frame = AudioFrame(np.asarray(samples, dtype=np.float32).reshape(-1).copy(), overflowed)
        try:
            self.frames.put_nowait(frame)
            return
        except queue.Full:
            self.dropped_frames += 1
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            self.stream_statuses.append(str(status))
        self._enqueue(indata[:, 0], overflowed=bool(status))

    def __enter__(self) -> "StreamingAudioInput":
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=1,
            dtype="float32",
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def get(self, timeout: float = 0.1) -> AudioFrame:
        return self.frames.get(timeout=timeout)

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
