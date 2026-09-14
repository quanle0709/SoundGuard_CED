"""Local enrollment, open-set matching, and isolated embedding clients.

Raw recordings and embeddings are deliberately stored outside user_profile.json.
The recognizers are optional: model failure raises ModelUnavailable at the web
boundary and degrades to the unmodified SoundGuard core at the live boundary.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from soundguard.paths import REPOSITORY_ROOT


SCHEMA_VERSION = "1.0"
PROFILE_KINDS = frozenset({"sound", "voice"})
TYPE_DIRS = {"sound": "familiar_sounds", "voice": "familiar_voices"}
DEFAULT_THRESHOLDS = {"sound": 0.72, "voice": 0.72}
DEFAULT_MARGINS = {"sound": 0.08, "voice": 0.06}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
DEFAULT_ROOT = Path(__file__).resolve().parent / "enrollments"
DEFAULT_WORKER_PYTHON = (
    REPOSITORY_ROOT / "benchmark_data" / "external"
    / "personalized_recognition_venv" / "Scripts" / "python.exe"
)
WORKER_MODULE = "personalization.recognition_model_worker"


class RecognitionError(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def l2_normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not vector.size or not math.isfinite(norm) or norm <= 1e-8:
        raise RecognitionError("embedding is empty or non-finite")
    vector = vector / norm
    if not np.isfinite(vector).all():
        raise RecognitionError("embedding contains non-finite values")
    return vector


def build_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise RecognitionError("no valid embeddings")
    return l2_normalize(np.mean(np.stack([l2_normalize(x) for x in embeddings]), axis=0))


def open_set_match(
    query: np.ndarray,
    profiles: list[dict],
    *,
    threshold: float | None = None,
    margin: float | None = None,
) -> dict:
    query = l2_normalize(query)
    ranked = sorted(
        (
            {
                "id": item["id"],
                "display_name": item["display_name"],
                "score": float(np.dot(query, l2_normalize(item["embedding"]))),
                "threshold": float(item.get("threshold", 0.72)),
                "margin_required": float(item.get("margin", 0.06)),
                "priority": int(item.get("priority", 3)),
            }
            for item in profiles
            if item.get("enabled", False)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    if not ranked:
        return {
            "label": "UNKNOWN", "accepted": False, "reason": "no_enabled_profiles",
            "score": None, "runner_up_score": None, "margin": None,
        }
    best = ranked[0]
    runner = ranked[1]["score"] if len(ranked) > 1 else None
    actual_threshold = float(threshold if threshold is not None else best["threshold"])
    required_margin = float(margin if margin is not None else best["margin_required"])
    actual_margin = best["score"] - runner if runner is not None else 1.0
    accepted = best["score"] >= actual_threshold and actual_margin >= required_margin
    reason = "accepted" if accepted else (
        "below_threshold" if best["score"] < actual_threshold else "ambiguous_margin"
    )
    return {
        **best,
        "label": best["display_name"] if accepted else "UNKNOWN",
        "accepted": accepted,
        "reason": reason,
        "runner_up_score": runner,
        "margin": actual_margin,
        "threshold": actual_threshold,
        "margin_required": required_margin,
        "ranked": ranked,
    }


def validate_audio_bytes(raw: bytes, kind: str) -> tuple[np.ndarray, int, dict]:
    if kind not in PROFILE_KINDS:
        raise RecognitionError("invalid recognition profile type")
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise RecognitionError("audio upload is empty or exceeds 12 MiB")
    try:
        samples, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RecognitionError("audio could not be decoded") from exc
    if sample_rate < 8_000 or sample_rate > 192_000 or samples.size == 0:
        raise RecognitionError("unsupported or empty audio")
    mono = np.mean(samples, axis=1, dtype=np.float32)
    duration = mono.size / sample_rate
    minimum, maximum = ((0.4, 15.0) if kind == "sound" else (0.8, 20.0))
    if duration < minimum or duration > maximum:
        raise RecognitionError(
            f"{kind} sample duration must be between {minimum:g} and {maximum:g} seconds"
        )
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    peak = float(np.max(np.abs(mono)))
    clipping_ratio = float(np.mean(np.abs(mono) >= 0.995))
    if rms < 0.003:
        raise RecognitionError("audio is silent or too quiet")
    if clipping_ratio > 0.05:
        raise RecognitionError("audio is severely clipped")
    frame = max(1, int(sample_rate * 0.025))
    usable = mono[: mono.size - (mono.size % frame)]
    frame_rms = (
        np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1))
        if usable.size else np.asarray([rms])
    )
    speech_ratio = float(np.mean(frame_rms >= max(0.006, rms * 0.35)))
    if kind == "voice" and speech_ratio < 0.25:
        raise RecognitionError("voice sample contains too little active speech")
    return mono, int(sample_rate), {
        "duration_seconds": round(duration, 4),
        "sample_rate_hz": int(sample_rate),
        "channels": int(samples.shape[1]),
        "rms": round(rms, 7),
        "peak": round(peak, 7),
        "clipping_ratio": round(clipping_ratio, 7),
        "speech_ratio": round(speech_ratio, 4) if kind == "voice" else None,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "quality": "valid",
    }


class EmbeddingClient:
    """Persistent JSON-lines client for the isolated optional model runtime."""

    def __init__(self, python_path: str | Path | None = None,
                 timeout_seconds: float = 180.0) -> None:
        configured = os.environ.get("SOUNDGUARD_PERSONALIZED_PYTHON", "").strip()
        self.python_path = Path(python_path or configured or DEFAULT_WORKER_PYTHON)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("model timeout must be positive")
        self._processes: dict[str, subprocess.Popen] = {}
        self._process_lock = threading.RLock()
        self._kind_locks = {kind: threading.Lock() for kind in PROFILE_KINDS}
        self._state_lock = threading.Lock()
        self._states = {
            kind: {
                "status": "idle", "started_at": None, "ready_at": None,
                "failed_at": None, "error": None, "metadata": None,
            }
            for kind in PROFILE_KINDS
        }
        self._preload_threads: dict[str, threading.Thread] = {}
        self._closing = False

    @property
    def loaded(self) -> bool:
        return any(value["status"] == "ready" for value in self.status().values())

    def status(self, kind: str | None = None) -> dict:
        """Return a JSON-safe snapshot of per-model lifecycle state."""
        if kind is not None and kind not in PROFILE_KINDS:
            raise ValueError("kind must be sound or voice")
        with self._state_lock:
            values = {
                name: {
                    key: (dict(value) if isinstance(value, dict) else value)
                    for key, value in state.items()
                }
                for name, state in self._states.items()
                if kind is None or name == kind
            }
        return values[kind] if kind is not None else values

    def _set_state(self, kind: str, status: str, **fields) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._state_lock:
            state = self._states[kind]
            state["status"] = status
            if status == "loading":
                state.update({
                    "started_at": now, "ready_at": None, "failed_at": None,
                    "error": None,
                })
            elif status == "ready":
                state.update({"ready_at": now, "failed_at": None, "error": None})
            elif status == "failed":
                state.update({"failed_at": now, "ready_at": None})
            state.update(fields)

    def _start(self, kind: str) -> subprocess.Popen:
        with self._process_lock:
            existing = self._processes.get(kind)
            if existing is not None and existing.poll() is None:
                return existing
            if not self.python_path.is_file():
                raise ModelUnavailable(
                    f"optional recognition Python is missing: {self.python_path}"
                )
            process = subprocess.Popen(
                [str(self.python_path), "-m", WORKER_MODULE],
                cwd=str(REPOSITORY_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            )
            self._processes[kind] = process
            return process

    def _request(self, kind: str, request: dict) -> dict:
        process = self._start(kind)
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("model worker pipes are unavailable")
        process.stdin.write(json.dumps({"kind": kind, **request}) + "\n")
        process.stdin.flush()
        response_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        reader = threading.Thread(
            target=lambda: response_queue.put(process.stdout.readline()),
            name=f"{kind}-model-response", daemon=True,
        )
        reader.start()
        try:
            line = response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                f"{kind} model request exceeded {self.timeout_seconds:g} seconds"
            ) from exc
        if not line:
            raise RuntimeError("model worker exited without a response")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "model worker failed"))
        return response

    def _perform_preload(self, kind: str) -> dict:
        try:
            with self._kind_locks[kind]:
                response = self._request(kind, {"action": "preload"})
            metadata = dict(response["metadata"])
            self._set_state(kind, "ready", metadata=metadata)
            return metadata
        except Exception as exc:
            self._close_kind(kind)
            self._set_state(kind, "failed", error=f"{type(exc).__name__}: {exc}")
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc

    def preload_async(self, kind: str) -> bool:
        """Start one lower-priority model load and return without waiting.

        Repeated calls while loading or ready are no-ops, preventing duplicate
        worker processes and duplicate model instances.
        """
        if kind not in PROFILE_KINDS:
            raise ValueError("kind must be sound or voice")
        with self._state_lock:
            if self._closing or self._states[kind]["status"] in {"loading", "ready"}:
                return False
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._states[kind].update({
                "status": "loading", "started_at": now, "ready_at": None,
                "failed_at": None, "error": None, "metadata": None,
            })

        def load() -> None:
            try:
                self._perform_preload(kind)
            except ModelUnavailable:
                # Failure is exposed through status; optional callers stay alive.
                pass

        thread = threading.Thread(
            target=load, name=f"familiar-{kind}-model-preload", daemon=True
        )
        self._preload_threads[kind] = thread
        thread.start()
        return True

    def embed(self, kind: str, audio_path: str | Path) -> tuple[np.ndarray, dict]:
        if kind not in PROFILE_KINDS:
            raise ValueError("kind must be sound or voice")
        current_status = self.status(kind)["status"]
        if current_status == "loading":
            raise ModelUnavailable(f"{kind} recognition model is loading")
        if current_status != "ready":
            self._set_state(kind, "loading")
        try:
            with self._kind_locks[kind]:
                response = self._request(kind, {
                    "action": "embed", "audio_path": str(Path(audio_path).resolve())
                })
            metadata = dict(response["metadata"])
            self._set_state(kind, "ready", metadata=metadata)
            return (
                l2_normalize(np.asarray(response["embedding"], dtype=np.float32)),
                metadata,
            )
        except Exception as exc:
            self._close_kind(kind)
            self._set_state(kind, "failed", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, ModelUnavailable):
                raise
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc

    def _close_kind(self, kind: str) -> None:
        with self._process_lock:
            process = self._processes.pop(kind, None)
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def close(self) -> None:
        with self._state_lock:
            self._closing = True
        for kind in PROFILE_KINDS:
            with self._kind_locks[kind]:
                self._close_kind(kind)
            self._set_state(kind, "idle", metadata=None, error=None)


class RecognitionStore:
    def __init__(self, root: str | Path | None = None) -> None:
        configured = os.environ.get("SOUNDGUARD_RECOGNITION_ROOT", "").strip()
        self.root = Path(root or configured or DEFAULT_ROOT)
        self._lock = threading.RLock()

    def _type_root(self, kind: str) -> Path:
        if kind not in PROFILE_KINDS:
            raise RecognitionError("invalid recognition profile type")
        return self.root / TYPE_DIRS[kind]

    def _profile_dir(self, kind: str, profile_id: str) -> Path:
        try:
            parsed = str(uuid.UUID(profile_id))
        except ValueError as exc:
            raise RecognitionError("invalid profile id") from exc
        return self._type_root(kind) / "profiles" / parsed

    def _profile_path(self, kind: str, profile_id: str) -> Path:
        return self._profile_dir(kind, profile_id) / "profile.json"

    def _load(self, kind: str, profile_id: str) -> dict:
        try:
            value = json.loads(self._profile_path(kind, profile_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RecognitionError("profile was not found or is malformed") from exc
        if value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != kind:
            raise RecognitionError("profile schema is unsupported")
        return value

    @staticmethod
    def _public(profile: dict) -> dict:
        value = dict(profile)
        value.pop("prototype_file", None)
        return value

    def list(self, kind: str) -> list[dict]:
        root = self._type_root(kind) / "profiles"
        if not root.exists():
            return []
        values = []
        for path in sorted(root.glob("*/profile.json")):
            try:
                values.append(self._public(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
        return values

    def create(self, kind: str, display_name: str, **fields) -> dict:
        name = " ".join(str(display_name).split())
        if not name or len(name) > 80 or any(ord(ch) < 32 for ch in name):
            raise RecognitionError("display name must contain 1-80 printable characters")
        priority = int(fields.get("priority", 3))
        if priority < 1 or priority > 5:
            raise RecognitionError("priority must be between 1 and 5")
        if kind == "voice" and not bool(fields.get("consent_acknowledged")):
            raise RecognitionError("voice-profile consent acknowledgment is required")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        profile_id = str(uuid.uuid4())
        profile = {
            "schema_version": SCHEMA_VERSION, "id": profile_id, "kind": kind,
            "display_name": name, "relationship": str(fields.get("relationship", ""))[:80],
            "contexts": list(fields.get("contexts") or []), "priority": priority,
            "enabled": False, "built": False, "sample_count": 0,
            "threshold": float(fields.get("threshold", DEFAULT_THRESHOLDS[kind])),
            "margin": float(fields.get("margin", DEFAULT_MARGINS[kind])),
            "consent_acknowledged": bool(fields.get("consent_acknowledged", False)),
            "samples": [], "model": None, "preprocessing_version": None,
            "embedding_dimension": None, "quality_summary": None,
            "created_at": now, "updated_at": now,
        }
        with self._lock:
            _atomic_json(self._profile_path(kind, profile_id), profile)
        return self._public(profile)

    def update(self, kind: str, profile_id: str, fields: dict) -> dict:
        allowed = {"display_name", "relationship", "contexts", "priority", "enabled", "threshold", "margin"}
        if set(fields) - allowed:
            raise RecognitionError("unsupported profile update field")
        with self._lock:
            profile = self._load(kind, profile_id)
            if fields.get("enabled") and not profile.get("built"):
                raise RecognitionError("build the profile before enabling it")
            if "display_name" in fields:
                name = " ".join(str(fields["display_name"]).split())
                if not name or len(name) > 80:
                    raise RecognitionError("display name must contain 1-80 characters")
                fields["display_name"] = name
            if "priority" in fields and not 1 <= int(fields["priority"]) <= 5:
                raise RecognitionError("priority must be between 1 and 5")
            for key in ("threshold", "margin"):
                if key in fields and not 0 <= float(fields[key]) <= 1:
                    raise RecognitionError(f"{key} must be between 0 and 1")
            profile.update(fields)
            profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _atomic_json(self._profile_path(kind, profile_id), profile)
            return self._public(profile)

    def add_sample(self, kind: str, profile_id: str, raw: bytes) -> dict:
        audio, sample_rate, quality = validate_audio_bytes(raw, kind)
        with self._lock:
            profile = self._load(kind, profile_id)
            maximum = 10
            if len(profile["samples"]) >= maximum:
                raise RecognitionError(f"{kind} profile accepts at most {maximum} samples")
            if any(item.get("sha256") == quality["sha256"] for item in profile["samples"]):
                raise RecognitionError("this recording is already enrolled")
            sample_id = str(uuid.uuid4())
            sample_dir = self._profile_dir(kind, profile_id) / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            destination = sample_dir / f"{sample_id}.wav"
            temporary = destination.with_suffix(".wav.tmp")
            sf.write(temporary, audio, sample_rate, format="WAV", subtype="PCM_16")
            os.replace(temporary, destination)
            item = {"id": sample_id, "file": destination.name, **quality}
            profile["samples"].append(item)
            profile["sample_count"] = len(profile["samples"])
            profile["built"] = False
            profile["enabled"] = False
            profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _atomic_json(self._profile_path(kind, profile_id), profile)
            return item

    def delete_sample(self, kind: str, profile_id: str, sample_id: str) -> dict:
        with self._lock:
            profile = self._load(kind, profile_id)
            matching = [item for item in profile["samples"] if item["id"] == sample_id]
            if not matching:
                raise RecognitionError("sample was not found")
            (self._profile_dir(kind, profile_id) / "samples" / matching[0]["file"]).unlink(missing_ok=True)
            profile["samples"] = [item for item in profile["samples"] if item["id"] != sample_id]
            profile["sample_count"] = len(profile["samples"])
            profile["built"] = False
            profile["enabled"] = False
            _atomic_json(self._profile_path(kind, profile_id), profile)
            return self._public(profile)

    def build(self, kind: str, profile_id: str, embed: Callable) -> dict:
        with self._lock:
            profile = self._load(kind, profile_id)
            minimum = 2 if kind == "sound" else 3
            if len(profile["samples"]) < minimum:
                raise RecognitionError(f"at least {minimum} valid samples are required")
            embeddings, metadata = [], None
            for item in profile["samples"]:
                vector, metadata = embed(kind, self._profile_dir(kind, profile_id) / "samples" / item["file"])
                embeddings.append(vector)
            centroid = build_centroid(embeddings)
            prototype = self._profile_dir(kind, profile_id) / "prototype.npy"
            temporary = prototype.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, centroid, allow_pickle=False)
            os.replace(temporary, prototype)
            profile.update({
                "built": True, "enabled": True, "prototype_file": prototype.name,
                "model": metadata.get("model") if metadata else None,
                "model_revision": metadata.get("revision") if metadata else None,
                "model_sha256": metadata.get("sha256") if metadata else None,
                "preprocessing_version": metadata.get("preprocessing") if metadata else None,
                "embedding_dimension": int(centroid.size),
                "quality_summary": {
                    "valid_samples": len(embeddings),
                    "total_duration_seconds": round(sum(x["duration_seconds"] for x in profile["samples"]), 3),
                    "mean_pairwise_similarity": round(float(np.mean([
                        np.dot(l2_normalize(x), centroid) for x in embeddings
                    ])), 6),
                },
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            _atomic_json(self._profile_path(kind, profile_id), profile)
            return self._public(profile)

    def matcher_profiles(self, kind: str) -> list[dict]:
        result = []
        for profile in self.list(kind):
            if not profile.get("built") or not profile.get("enabled"):
                continue
            try:
                stored = self._load(kind, profile["id"])
                vector = np.load(
                    self._profile_dir(kind, profile["id"]) / stored["prototype_file"],
                    allow_pickle=False,
                )
                result.append({**profile, "embedding": vector})
            except (OSError, ValueError, KeyError):
                continue
        return result

    def test(self, kind: str, raw: bytes, embed: Callable) -> dict:
        audio, sample_rate, quality = validate_audio_bytes(raw, kind)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stream:
            path = Path(stream.name)
        try:
            sf.write(path, audio, sample_rate, subtype="PCM_16")
            vector, metadata = embed(kind, path)
            return {**open_set_match(vector, self.matcher_profiles(kind)), "quality": quality, "model": metadata}
        finally:
            path.unlink(missing_ok=True)

    def delete(self, kind: str, profile_id: str, *, raw_only: bool = False) -> dict:
        with self._lock:
            profile = self._load(kind, profile_id)
            directory = self._profile_dir(kind, profile_id)
            if raw_only:
                shutil.rmtree(directory / "samples", ignore_errors=True)
                profile["samples"] = []
                profile["sample_count"] = 0
                _atomic_json(self._profile_path(kind, profile_id), profile)
                return self._public(profile)
            shutil.rmtree(directory)
            return {"deleted": True, "id": profile_id}


class TemporalDecisionGate:
    def __init__(self, required: int = 2, window: int = 3, cooldown_seconds: float = 5.0, clock=time.monotonic):
        self.required, self.window = required, window
        self.cooldown_seconds, self.clock = cooldown_seconds, clock
        self.history: deque[str] = deque(maxlen=window)
        self.last_emitted: dict[str, float] = {}

    def apply(self, result: dict) -> dict:
        label = result["label"] if result.get("accepted") else "UNKNOWN"
        self.history.append(label)
        if label == "UNKNOWN" or sum(item == label for item in self.history) < self.required:
            return {**result, "accepted": False, "label": "UNKNOWN", "reason": "temporal_evidence"}
        now = self.clock()
        if now - self.last_emitted.get(label, -1e30) < self.cooldown_seconds:
            return {**result, "accepted": False, "label": "UNKNOWN", "reason": "cooldown"}
        self.last_emitted[label] = now
        return result


@dataclass(frozen=True)
class AsyncRecognitionResult:
    kind: str
    key: str
    result: dict


class BoundedRecognitionWorker:
    """Low-priority, drop-oldest worker with exception isolation and counters."""

    def __init__(self, kind: str, recognize: Callable[[object], dict], maxsize: int = 2):
        self.kind, self.recognize = kind, recognize
        self.tasks: queue.Queue = queue.Queue(maxsize=maxsize)
        self.results: queue.Queue[AsyncRecognitionResult] = queue.Queue(maxsize=maxsize * 2)
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"familiar-{kind}-worker", daemon=True)
        self._started = False
        self.produced = self.consumed = self.dropped = self.failures = 0
        self.latencies: list[float] = []

    def start(self) -> None:
        if not self._started:
            self.thread.start()
            self._started = True

    def submit(self, key: str, payload: object) -> bool:
        self.produced += 1
        try:
            self.tasks.put_nowait((key, payload))
            return True
        except queue.Full:
            self.dropped += 1
        try:
            self.tasks.get_nowait()
        except queue.Empty:
            pass
        try:
            self.tasks.put_nowait((key, payload))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        while not self.stop_requested.is_set() or not self.tasks.empty():
            try:
                key, payload = self.tasks.get(timeout=0.05)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                result = self.recognize(payload)
            except Exception as exc:
                self.failures += 1
                result = {"accepted": False, "label": "UNKNOWN", "reason": "worker_error", "error": f"{type(exc).__name__}: {exc}"}
            self.consumed += 1
            self.latencies.append(time.perf_counter() - started)
            try:
                self.results.put_nowait(AsyncRecognitionResult(self.kind, key, result))
            except queue.Full:
                pass

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_requested.set()
        if self._started:
            self.thread.join(timeout)

    @property
    def counters(self) -> dict:
        return {"produced": self.produced, "consumed": self.consumed, "dropped": self.dropped, "failures": self.failures}


def decode_audio_base64(value: str) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise RecognitionError("audio_base64 is invalid") from exc
    return raw
