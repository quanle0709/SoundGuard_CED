"""Record dependency versions without importing optional packages."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path


PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "librosa",
    "numpy",
    "scipy",
    "pandas",
    "timm",
    "nnAudio",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    target = Path(sys.argv[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {name: package_version(name) for name in PACKAGES},
    }
    try:
        import torch

        payload["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:  # pragma: no cover - environment diagnostic
        payload["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
