"""Isolated adapter for the upstream EfficientSED fmn10 strong checkpoint.

Run this module with the dedicated EfficientSED virtual environment. It imports
the model architecture from the separately cloned, ignored upstream repository;
no third-party source is copied into SoundGuard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "benchmark_data" / "external" / "efficientsed_repo"
CHECKPOINT = UPSTREAM / "resources" / "fmn10_strong.pt"
CHECKPOINT_URL = (
    "https://github.com/theMoro/EfficientSED/releases/download/v0.0.1/"
    "fmn10_strong.pt"
)
SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 10.0
FRAME_SECONDS = 0.04


def _upstream_imports():
    if not UPSTREAM.is_dir():
        raise RuntimeError(f"EfficientSED checkout is missing: {UPSTREAM}")
    sys.path.insert(0, str(UPSTREAM))
    from data_util.audioset_classes import as_strong_train_classes
    from models.efficient_cnns.fmn.fmn_wrapper import FrameMNWrapper

    return as_strong_train_classes, FrameMNWrapper


def download_checkpoint() -> None:
    """Download only the selected asset from the authorized upstream repo."""
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        return
    temporary = CHECKPOINT.with_suffix(".pt.partial")
    urllib.request.urlretrieve(CHECKPOINT_URL, temporary)
    os.replace(temporary, CHECKPOINT)


def checkpoint_sha256() -> str:
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remap_checkpoint(state_dict: dict[str, object]) -> dict[str, object]:
    state_dict = {
        ("model." + key[len("net.") :] if key.startswith("net.") else key): value
        for key, value in state_dict.items()
    }
    state_dict = {
        (
            "model.fmn." + key[len("model.model.") :]
            if key.startswith("model.model.")
            else key
        ): value
        for key, value in state_dict.items()
    }
    return {
        (key[len("model.") :] if not key.startswith("model.fmn.") else key): value
        for key, value in state_dict.items()
    }


def load_model(device: str = "cpu"):
    import torch

    labels, frame_mn_wrapper = _upstream_imports()
    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"Checkpoint missing: {CHECKPOINT}. Run this module with --download first."
        )

    class StrongFrameModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = frame_mn_wrapper(1.0)
            embedding_dim = self.model.state_dict()["fmn.features.16.1.bias"].shape[0]
            self.strong_head = torch.nn.Linear(embedding_dim, len(labels))
            self.weak_head = torch.nn.Linear(embedding_dim, len(labels))

        def mel_forward(self, waveform):
            return self.model.mel_forward(waveform)

        def forward(self, mel):
            sequence = self.model(mel)
            if sequence.size(-2) != 250:
                sequence = torch.nn.functional.interpolate(
                    sequence.transpose(1, 2), size=250, mode="linear"
                ).transpose(1, 2)
            return self.strong_head(sequence).transpose(1, 2)

    model = StrongFrameModel()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(_remap_checkpoint(state), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    return model, list(labels)


def load_waveform(path: Path):
    import librosa
    import torch

    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    duration = len(waveform) / SAMPLE_RATE
    target = int(SEGMENT_SECONDS * SAMPLE_RATE)
    if len(waveform) > target:
        waveform = waveform[:target]
        duration = SEGMENT_SECONDS
    elif len(waveform) < target:
        waveform = __import__("numpy").pad(waveform, (0, target - len(waveform)))
    return torch.from_numpy(waveform).float(), duration


def infer_one(model, labels: list[str], path: Path, device: str = "cpu") -> dict:
    import torch

    waveform, duration = load_waveform(path)
    started = time.perf_counter()
    with torch.inference_mode():
        mel = model.mel_forward(waveform[None, :].to(device))
        probabilities = torch.sigmoid(model(mel))[0].cpu()
    inference_ms = (time.perf_counter() - started) * 1000.0
    valid_frames = min(probabilities.shape[1], max(1, int(duration / FRAME_SECONDS + 0.999)))
    probabilities = probabilities[:, :valid_frames]
    top_values, top_indices = torch.topk(probabilities.transpose(0, 1), k=5, dim=1)
    top_frames = [
        [
            {"label": labels[int(index)], "probability": round(float(value), 7)}
            for value, index in zip(values, indices)
        ]
        for values, indices in zip(top_values, top_indices)
    ]
    return {
        "path": str(path),
        "duration_seconds": duration,
        "frame_seconds": FRAME_SECONDS,
        "frame_count": valid_frames,
        "inference_ms": inference_ms,
        "probabilities": probabilities.numpy(),
        "top_frames": top_frames,
    }


def model_metadata(model, labels: list[str]) -> dict:
    import torch

    return {
        "model": "fmn10_strong",
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_sha256": checkpoint_sha256(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "label_count": len(labels),
        "sample_rate": SAMPLE_RATE,
        "frame_seconds": FRAME_SECONDS,
        "device": next(model.parameters()).device.type,
        "torch": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke-audio", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.download:
        download_checkpoint()
        print(json.dumps({"checkpoint": str(CHECKPOINT), "sha256": checkpoint_sha256()}))
        if not args.smoke_audio:
            return
    model, labels = load_model(args.device)
    result = model_metadata(model, labels)
    if args.smoke_audio:
        prediction = infer_one(model, labels, args.smoke_audio, args.device)
        probabilities = prediction.pop("probabilities")
        maxima = probabilities.max(axis=1)
        order = maxima.argsort()[-10:][::-1]
        result["smoke"] = {
            **prediction,
            "top_clip_labels": [
                {"label": labels[int(index)], "max_probability": float(maxima[index])}
                for index in order
            ],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
