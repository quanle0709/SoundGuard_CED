import tempfile
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


_TEMP_DIR = Path(tempfile.gettempdir())
_RAW_WAV_PATH = _TEMP_DIR / "soundguard_raw_recording.wav"
_PROCESSED_WAV_PATH = _TEMP_DIR / "soundguard_speech_recording.wav"


def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    if not isinstance(devices, list):
        return []

    return [
        {
            "index": index,
            "name": device["name"],
            "max_input_channels": device["max_input_channels"],
        }
        for index, device in enumerate(devices)
        if device.get("max_input_channels", 0) > 0
    ]


def _remove_old_temp_files() -> None:
    for path in (_RAW_WAV_PATH, _PROCESSED_WAV_PATH):
        if path.exists():
            path.unlink()


def cleanup_temp_recordings() -> None:
    _remove_old_temp_files()


def _countdown() -> None:
    for remaining in range(3, 0, -1):
        print(f"Đếm ngược: {remaining}...")
        time.sleep(1)
    print("Bắt đầu nói")


def record_audio(duration_seconds: int, device_index: int | None = None) -> Path:
    """Record mono audio from the selected microphone and save it as a WAV file."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0")

    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Unable to query audio devices: {exc}") from exc

    if not devices:
        raise RuntimeError("No microphone is available.")

    input_devices = [device for device in devices if device.get("max_input_channels", 0) > 0]
    if not input_devices:
        raise RuntimeError("No microphone is available.")

    if device_index is not None:
        if device_index < 0 or device_index >= len(devices):
            raise RuntimeError(f"Device index {device_index} is out of range.")
        selected_device = devices[device_index]
        if selected_device.get("max_input_channels", 0) <= 0:
            raise RuntimeError(f"Device {device_index} does not support input.")

    _remove_old_temp_files()
    _countdown()

    sample_rate = 16000
    channels = 1
    dtype = "int16"

    try:
        audio_frames = sd.rec(
            int(duration_seconds * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            device=device_index,
        )
        sd.wait()
    except Exception as exc:
        raise RuntimeError(f"Microphone capture failed: {exc}") from exc

    audio_data = np.squeeze(audio_frames).astype(np.int16)
    sf.write(_RAW_WAV_PATH, audio_data, sample_rate, subtype="PCM_16")
    print("Đã thu xong")
    return _RAW_WAV_PATH


def create_speech_optimized_wav(
    raw_wav_path: Path | str,
    output_path: Path | str | None = None,
    preserve_tail_ms: int = 0,
) -> Path:
    """Create a speech-optimized mono WAV from the raw recording."""
    raw_path = Path(raw_wav_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw recording not found: {raw_path}")
    if preserve_tail_ms < 0:
        raise ValueError("preserve_tail_ms must not be negative")

    audio, sample_rate = sf.read(raw_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    if audio.size == 0:
        raise ValueError("Recording is empty.")

    audio = audio - np.mean(audio)

    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        audio = np.zeros_like(audio)
        peak = 0.0

    threshold = max(0.001, peak * 0.003)
    start = 0
    end = len(audio)

    for idx in range(0, len(audio) - 1):
        if np.abs(audio[idx]) > threshold:
            start = idx
            break

    for idx in range(len(audio) - 1, -1, -1):
        if np.abs(audio[idx]) > threshold:
            end = idx + 1
            break

    # Live FINAL recognition keeps a protected tail so a quiet final word is
    # never removed solely because it falls below the amplitude threshold.
    preserve_tail_samples = int(sample_rate * preserve_tail_ms / 1000)
    if preserve_tail_samples:
        end = len(audio)

    audio = audio[start:end]

    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        target_peak = 0.8
        gain = min(3.0, target_peak / peak)
        audio = np.clip(audio * gain, -0.95, 0.95)

    destination = Path(output_path) if output_path is not None else _PROCESSED_WAV_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate, subtype="PCM_16")
    return destination


def get_peak_amplitude(wav_path: Path | str) -> float:
    audio, _ = sf.read(Path(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return float(np.max(np.abs(audio))) if audio.size else 0.0
