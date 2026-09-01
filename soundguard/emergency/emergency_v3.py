"""Opt-in, failure-isolated client for the EfficientSED emergency specialist."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from soundguard.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
ENABLE_ENV = "SOUNDGUARD_ENABLE_EMERGENCY_V3"
PYTHON_ENV = "SOUNDGUARD_EFFICIENTSED_PYTHON"
WORKER = ROOT / "benchmark" / "experiments" / "emergency_v3" / "efficientsed_worker.py"
DEFAULT_PYTHON = (
    ROOT / "benchmark_data" / "external" / "efficientsed_venv" / "Scripts" / "python.exe"
)
CATEGORY_LABELS = {
    "glass_breaking": "glass breaking",
    "vehicle_horn": "vehicle horn",
    "fire": "fire",
    "siren": "siren",
    "baby_crying": "baby cry",
}


def emergency_v3_enabled(explicit: bool = False) -> bool:
    configured = os.environ.get(ENABLE_ENV, "").strip().lower()
    return explicit or configured in {"1", "true", "yes", "on"}


class EmergencyV3Specialist:
    """Start the isolated model lazily and reuse one worker process."""

    def __init__(
        self,
        python_path: str | Path | None = None,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        logger: Callable[[str], None] = print,
    ) -> None:
        self.python_path = Path(
            python_path or os.environ.get(PYTHON_ENV, "") or DEFAULT_PYTHON
        )
        self.popen_factory = popen_factory
        self.logger = logger
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._failure_logged = False

    @property
    def loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _log_failure_once(self, message: str) -> None:
        if not self._failure_logged:
            self.logger(f"Emergency V3 unavailable; using V2: {message}")
            self._failure_logged = True

    def _start(self) -> subprocess.Popen:
        if self.loaded:
            return self._process  # type: ignore[return-value]
        if not self.python_path.is_file():
            raise FileNotFoundError(f"isolated Python not found: {self.python_path}")
        if not WORKER.is_file():
            raise FileNotFoundError(f"EfficientSED worker not found: {WORKER}")
        self._process = self.popen_factory(
            [str(self.python_path), str(WORKER)],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self._process

    def analyze_file(self, audio_path: str | Path) -> dict:
        with self._lock:
            try:
                process = self._start()
                if process.stdin is None or process.stdout is None:
                    raise RuntimeError("worker pipes are unavailable")
                request = json.dumps({"audio_path": str(Path(audio_path).resolve())})
                process.stdin.write(request + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("worker exited without a response")
                result = json.loads(line)
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "unknown worker error"))
                result["source"] = "efficientsed_fmn10_strong"
                result["sound_label"] = CATEGORY_LABELS.get(result.get("category"), "unknown")
                return result
            except Exception as exc:
                self._log_failure_once(f"{type(exc).__name__}: {exc}")
                self.close()
                return {
                    "ok": False,
                    "detected": False,
                    "source": "v2_fallback",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    def analyze_samples(self, audio: np.ndarray, sample_rate: int = 16_000) -> dict:
        import soundfile as sf

        path = Path(tempfile.gettempdir()) / f"soundguard_emergency_v3_{uuid.uuid4().hex}.wav"
        sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
        try:
            return self.analyze_file(path)
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def create_emergency_v3_specialist(explicit: bool = False, **kwargs):
    return EmergencyV3Specialist(**kwargs) if emergency_v3_enabled(explicit) else None


def select_emergency_evidence(
    ced_label: str,
    ced_confidence: float,
    specialist_result: dict | None,
    context: str = "neutral",
) -> tuple[str, float, str]:
    """Apply the frozen V2 OR specialist policy without changing CED output."""
    from soundguard.emergency.emergency_system import evaluate_sound

    if evaluate_sound(ced_label, ced_confidence, context).get("is_dangerous"):
        return ced_label, float(ced_confidence), "ced_tiny_v2"
    if specialist_result and specialist_result.get("detected"):
        label = specialist_result.get("sound_label", "unknown")
        if label in CATEGORY_LABELS.values():
            return str(label), 1.0, "efficientsed"
    return ced_label, float(ced_confidence), "ced_tiny_v2"
