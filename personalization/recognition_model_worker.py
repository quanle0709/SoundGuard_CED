"""Isolated CPU embedding worker for optional personalized recognition.

The process speaks one-request/one-response JSON lines on standard I/O.  Model
imports and weights stay out of the latency-critical SoundGuard process.
"""

from __future__ import annotations

import hashlib
import builtins
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = REPOSITORY_ROOT / "benchmark_data" / "external" / "personalized_recognition_models"
EFFICIENTAT_ROOT = MODEL_CACHE / "EfficientAT"
EFFICIENTAT_REVISION = "a425fdce92572e602a1d5634799bd9f1f2efa806"
EFFICIENTAT_WEIGHT_URL = (
    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt"
)
WESPEAKER_REPOSITORY = "Wespeaker/wespeaker-ecapa-tdnn512-LM"
WESPEAKER_REVISION = "a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b"
WESPEAKER_FILENAME = "voxceleb_ECAPA512_LM.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not value.size or not np.isfinite(value).all() or norm <= 1e-8:
        raise RuntimeError("model produced an invalid embedding")
    return value / norm


def _rss_bytes() -> int | None:
    """Best-effort resident memory without adding a runtime dependency."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            get_current = ctypes.windll.kernel32.GetCurrentProcess
            get_current.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            get_memory.restype = wintypes.BOOL
            handle = get_current()
            if get_memory(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.WorkingSetSize)
        else:
            import resource
            maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(maximum * (1024 if sys.platform != "darwin" else 1))
    except Exception:
        pass
    return None


def _lower_process_priority() -> None:
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
            )
        else:
            os.nice(5)
    except Exception:
        pass


class EfficientATEmbedder:
    def __init__(self) -> None:
        started = time.perf_counter()
        root = Path(os.environ.get("SOUNDGUARD_EFFICIENTAT_ROOT", EFFICIENTAT_ROOT))
        if not (root / "models" / "mn" / "model.py").is_file():
            raise RuntimeError(f"pinned EfficientAT source is missing: {root}")
        previous_directory = Path.cwd()
        try:
            # Upstream resolves both metadata and its weight cache relative to cwd.
            os.chdir(root)
            sys.path.insert(0, str(root))
            import torch
            from helpers.utils import NAME_TO_WIDTH
            from models.mn.model import get_model
            from models.preprocess import AugmentMelSTFT

            torch.set_num_threads(max(
                1, int(os.environ.get("SOUNDGUARD_MODEL_THREADS", "4"))
            ))
            self.torch = torch
            # Upstream prints the complete network; stdout is reserved for IPC.
            original_print = builtins.print
            builtins.print = lambda *args, **kwargs: None
            try:
                self.model = get_model(
                    width_mult=NAME_TO_WIDTH("mn10_as"), pretrained_name="mn10_as",
                    strides=(2, 2, 2, 2), head_type="mlp",
                ).cpu().eval()
            finally:
                builtins.print = original_print
            self.mel = AugmentMelSTFT(
                n_mels=128, sr=32_000, win_length=800, hopsize=320
            ).cpu().eval()
        finally:
            os.chdir(previous_directory)
        weights = root / "resources" / "mn10_as_mAP_471.pt"
        self.metadata = {
            "model": "EfficientAT mn10_as",
            "revision": EFFICIENTAT_REVISION,
            "sha256": _sha256(weights) if weights.is_file() else None,
            "source": EFFICIENTAT_WEIGHT_URL,
            "preprocessing": "efficientat-32k-mel128-win800-hop320-v1",
            "sample_rate_hz": 32_000,
        }
        self.load_seconds = time.perf_counter() - started

    def embed(self, path: Path) -> tuple[np.ndarray, dict]:
        import librosa

        started = time.perf_counter()
        cpu_started = time.process_time()
        waveform, _ = librosa.load(path, sr=32_000, mono=True)
        if waveform.size < 320:
            raise RuntimeError("sound input is too short")
        tensor = self.torch.from_numpy(np.asarray(waveform, dtype=np.float32)[None, :])
        with self.torch.inference_mode():
            spectrum = self.mel(tensor)
            _, features = self.model(spectrum.unsqueeze(0))
        metadata = dict(self.metadata)
        metadata.update({
            "load_seconds": self.load_seconds,
            "inference_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "worker_rss_bytes": _rss_bytes(),
        })
        return _normalise(features.detach().cpu().numpy()), metadata


class WeSpeakerEmbedder:
    def __init__(self) -> None:
        started = time.perf_counter()
        os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "huggingface"))
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort

        model_path = Path(hf_hub_download(
            repo_id=WESPEAKER_REPOSITORY,
            filename=WESPEAKER_FILENAME,
            revision=WESPEAKER_REVISION,
        ))
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(
            1, int(os.environ.get("SOUNDGUARD_MODEL_THREADS", "2"))
        )
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.metadata = {
            "model": "WeSpeaker ECAPA-TDNN512-LM",
            "revision": WESPEAKER_REVISION,
            "sha256": _sha256(model_path),
            "source": f"hf://{WESPEAKER_REPOSITORY}/{WESPEAKER_FILENAME}",
            "preprocessing": "wespeaker-kaldi-fbank80-cmn-16k-v1",
            "sample_rate_hz": 16_000,
        }
        self.load_seconds = time.perf_counter() - started

    def embed(self, path: Path) -> tuple[np.ndarray, dict]:
        import librosa
        import torch
        import torchaudio

        started = time.perf_counter()
        cpu_started = time.process_time()
        waveform, _ = librosa.load(path, sr=16_000, mono=True)
        if waveform.size < 12_800:
            raise RuntimeError("voice input must contain at least 0.8 seconds")
        tensor = torch.from_numpy(np.asarray(waveform * 32768.0, dtype=np.float32)).unsqueeze(0)
        features = torchaudio.compliance.kaldi.fbank(
            tensor,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=16_000,
            window_type="hamming",
            use_energy=False,
        )
        features = features - features.mean(dim=0, keepdim=True)
        embs = self.session.run(
            ["embs"], {"feats": features.unsqueeze(0).numpy()}
        )[0]
        metadata = dict(self.metadata)
        metadata.update({
            "load_seconds": self.load_seconds,
            "inference_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "worker_rss_bytes": _rss_bytes(),
        })
        return _normalise(embs), metadata


class Worker:
    def __init__(self) -> None:
        self.models: dict[str, object] = {}

    def embed(self, kind: str, path: Path) -> tuple[np.ndarray, dict]:
        if kind not in {"sound", "voice"}:
            raise ValueError("kind must be sound or voice")
        if not path.is_file():
            raise FileNotFoundError(path)
        if kind not in self.models:
            self.models[kind] = EfficientATEmbedder() if kind == "sound" else WeSpeakerEmbedder()
        return self.models[kind].embed(path)  # type: ignore[attr-defined]


def main() -> None:
    worker_temp = MODEL_CACHE.parent / "personalized_recognition_tmp"
    numba_cache = MODEL_CACHE.parent / "personalized_recognition_numba_cache"
    worker_temp.mkdir(parents=True, exist_ok=True)
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(worker_temp)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
    import tempfile
    tempfile.tempdir = str(worker_temp)
    _lower_process_priority()
    if os.environ.get("SOUNDGUARD_MODEL_DEBUG"):
        import faulthandler
        faulthandler.dump_traceback_later(10, repeat=True)
    worker = Worker()
    protocol = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                embedding, metadata = worker.embed(
                    str(request.get("kind", "")),
                    Path(str(request.get("audio_path", ""))),
                )
            response = {
                "ok": True,
                "embedding": embedding.tolist(),
                "metadata": metadata,
            }
        except Exception as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
            }
        print(json.dumps(response, separators=(",", ":")), file=protocol, flush=True)


if __name__ == "__main__":
    main()
