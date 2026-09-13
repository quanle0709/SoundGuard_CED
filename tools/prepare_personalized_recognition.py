"""Prepare and smoke-test the optional pinned personalized model runtime."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "benchmark_data" / "external"
VENV = EXTERNAL / "personalized_recognition_venv"
MODELS = EXTERNAL / "personalized_recognition_models"
EFFICIENTAT = MODELS / "EfficientAT"
EFFICIENTAT_REPOSITORY = "https://github.com/fschmid56/EfficientAT.git"
EFFICIENTAT_REVISION = "a425fdce92572e602a1d5634799bd9f1f2efa806"


def run(command, **kwargs):
    env = kwargs.pop("env", None)
    cwd = str(kwargs.pop("cwd", ROOT))
    subprocess.run([str(item) for item in command], cwd=cwd, env=env,
                   check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-sound", help="Optional WAV for EfficientAT smoke")
    parser.add_argument("--smoke-voice", help="Optional WAV for WeSpeaker smoke")
    args = parser.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    task_temp = EXTERNAL / "personalized_recognition_tmp"
    pip_cache = EXTERNAL / "personalized_recognition_pip_cache"
    task_temp.mkdir(parents=True, exist_ok=True)
    pip_cache.mkdir(parents=True, exist_ok=True)
    if not (EFFICIENTAT / ".git").exists():
        run(["git", "clone", EFFICIENTAT_REPOSITORY, EFFICIENTAT])
    run(["git", "checkout", "--detach", EFFICIENTAT_REVISION], cwd=EFFICIENTAT)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=EFFICIENTAT, text=True
    ).strip()
    if actual != EFFICIENTAT_REVISION:
        raise RuntimeError(f"EfficientAT revision mismatch: {actual}")
    python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        run([sys.executable, "-m", "venv", "--system-site-packages", VENV])
    env = dict(os.environ)
    env.update({
        "TEMP": str(task_temp), "TMP": str(task_temp),
        "PIP_CACHE_DIR": str(pip_cache), "HF_HOME": str(MODELS / "huggingface"),
    })
    run([python, "-m", "pip", "install", "-r",
         ROOT / "requirements-personalized-recognition.txt"], env=env)
    if args.smoke_sound or args.smoke_voice:
        sys.path.insert(0, str(ROOT))
        from personalization.recognition import EmbeddingClient
        client = EmbeddingClient(python)
        try:
            for kind, path in (("sound", args.smoke_sound), ("voice", args.smoke_voice)):
                if path:
                    vector, metadata = client.embed(kind, path)
                    print(kind, vector.shape, metadata)
        finally:
            client.close()
    print(f"Optional model runtime ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
