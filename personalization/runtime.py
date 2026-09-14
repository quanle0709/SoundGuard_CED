"""Fail-open live services for familiar sounds and familiar voices."""

from __future__ import annotations

import queue
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .recognition import (
    BoundedRecognitionWorker,
    EmbeddingClient,
    RecognitionStore,
    TemporalDecisionGate,
    open_set_match,
)


class PersonalizedRecognitionRuntime:
    """Owns optional workers; callers never need to trust model availability."""

    def __init__(
        self,
        *,
        familiar_sounds: bool = False,
        familiar_voices: bool = False,
        store: RecognitionStore | None = None,
        client: EmbeddingClient | None = None,
        sample_rate: int = 16_000,
        sound_cooldown_seconds: float = 5.0,
    ) -> None:
        self.familiar_sounds = bool(familiar_sounds)
        self.familiar_voices = bool(familiar_voices)
        self.store = store or RecognitionStore()
        self.client = client or EmbeddingClient()
        self.sample_rate = int(sample_rate)
        # Snapshot once at startup so the real-time callback never touches disk.
        self.sound_profiles = (
            self.store.matcher_profiles("sound") if self.familiar_sounds else []
        )
        self.voice_profiles = (
            self.store.matcher_profiles("voice") if self.familiar_voices else []
        )
        self.sound_gate = TemporalDecisionGate(
            required=2, window=3, cooldown_seconds=sound_cooldown_seconds
        )
        self.sound_worker = BoundedRecognitionWorker(
            "sound", self._recognize_sound, maxsize=2
        ) if self.familiar_sounds else None
        self.voice_worker = BoundedRecognitionWorker(
            "voice", self._recognize_voice, maxsize=2
        ) if self.familiar_voices else None
        self._voice_cache: dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        for worker in (self.sound_worker, self.voice_worker):
            if worker is not None:
                worker.start()
        self.started = True
        preload = getattr(self.client, "preload_async", None)
        if callable(preload):
            # Only enabled features with an already-built profile may allocate a
            # model. Each request returns immediately and is duplicate-safe.
            if self.sound_enabled:
                preload("sound")
            if self.voice_enabled:
                preload("voice")

    def _embed_samples(self, kind: str, samples: np.ndarray) -> tuple[np.ndarray, dict]:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stream:
            path = Path(stream.name)
        try:
            sf.write(path, audio, self.sample_rate, subtype="PCM_16")
            return self.client.embed(kind, path)
        finally:
            path.unlink(missing_ok=True)

    def _recognize_sound(self, samples: np.ndarray) -> dict:
        vector, metadata = self._embed_samples("sound", samples)
        result = open_set_match(vector, self.sound_profiles)
        return {**self.sound_gate.apply(result), "model": metadata}

    def _recognize_voice(self, samples: np.ndarray) -> dict:
        vector, metadata = self._embed_samples("voice", samples)
        result = open_set_match(vector, self.voice_profiles)
        return {**result, "model": metadata}

    @property
    def sound_enabled(self) -> bool:
        return self.sound_worker is not None and bool(self.sound_profiles)

    @property
    def voice_enabled(self) -> bool:
        return self.voice_worker is not None and bool(self.voice_profiles)

    def submit_sound(self, samples: np.ndarray, key: str | None = None) -> bool:
        if not self.sound_enabled:
            return False
        self.start()
        return self.sound_worker.submit(key or str(time.time_ns()), np.asarray(samples).copy())

    def submit_voice(self, utterance_id: int | str, samples: np.ndarray) -> bool:
        if not self.voice_enabled:
            return False
        self.start()
        return self.voice_worker.submit(str(utterance_id), np.asarray(samples).copy())

    def drain_sound_results(self) -> list[dict]:
        values = []
        if self.sound_worker is None:
            return values
        while True:
            try:
                item = self.sound_worker.results.get_nowait()
            except queue.Empty:
                return values
            values.append({"key": item.key, **item.result})

    def take_voice_result(self, utterance_id: int | str, timeout: float = 0.0) -> dict | None:
        if self.voice_worker is None:
            return None
        key = str(utterance_id)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._cache_lock:
                if key in self._voice_cache:
                    return self._voice_cache.pop(key)
            try:
                wait = max(0.0, deadline - time.monotonic())
                item = self.voice_worker.results.get(timeout=wait) if wait else self.voice_worker.results.get_nowait()
            except queue.Empty:
                return None
            if item.key == key:
                return item.result
            with self._cache_lock:
                self._voice_cache[item.key] = item.result

    @property
    def counters(self) -> dict:
        values = {
            "sound_enabled": self.sound_enabled,
            "voice_enabled": self.voice_enabled,
        }
        status = getattr(self.client, "status", None)
        if callable(status):
            for kind, state in status().items():
                values[f"familiar_{kind}_model_status"] = state["status"]
        for name, worker in (("sound", self.sound_worker), ("voice", self.voice_worker)):
            for key, value in (worker.counters if worker else {}).items():
                values[f"familiar_{name}_{key}"] = value
        return values

    def close(self) -> None:
        for worker in (self.sound_worker, self.voice_worker):
            if worker is not None:
                worker.stop()
        self.client.close()
        self.started = False
