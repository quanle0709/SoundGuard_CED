from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000

_VAD_MODEL = None
_GET_SPEECH_TIMESTAMPS = None


def _load_vad():
    global _VAD_MODEL, _GET_SPEECH_TIMESTAMPS

    if (
        _VAD_MODEL is not None
        and _GET_SPEECH_TIMESTAMPS is not None
    ):
        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS

    import torch
    from silero_vad import (
        get_speech_timestamps,
        load_silero_vad,
    )

    torch.set_num_threads(1)

    _VAD_MODEL = load_silero_vad(onnx=False)
    _GET_SPEECH_TIMESTAMPS = get_speech_timestamps

    return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS


def detect_speech_frame(
    audio_frame: np.ndarray,
    threshold: float = 0.5,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict:
    """Classify one streaming frame without opening an audio device.

    Silero supports 512-sample frames at 16 kHz. The model and torch remain
    lazily imported through the existing loader.
    """
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError("Streaming Silero VAD requires 16000 Hz audio.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("VAD threshold must be between 0.0 and 1.0.")
    frame = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
    if frame.size != 512:
        raise ValueError("Streaming Silero VAD requires exactly 512 samples.")

    model, _ = _load_vad()
    import torch

    probability = float(model(torch.from_numpy(frame), sample_rate).item())
    return {
        "has_speech": probability >= threshold,
        "probability": probability,
        "threshold": threshold,
    }


def reset_streaming_vad() -> None:
    """Reset Silero recurrent state if the model has already been loaded."""
    if _VAD_MODEL is not None and hasattr(_VAD_MODEL, "reset_states"):
        _VAD_MODEL.reset_states()


def detect_speech(
    wav_path: str | Path,
    threshold: float = 0.5,
) -> dict:
    audio_path = Path(wav_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file audio: {audio_path}"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "VAD threshold phải nằm trong khoảng 0.0 đến 1.0."
        )

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError("File audio đang rỗng.")

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            "Silero VAD yêu cầu audio 16000 Hz, "
            f"nhưng file đang là {sample_rate} Hz."
        )

    model, get_speech_timestamps = _load_vad()

    import torch

    audio_tensor = torch.from_numpy(audio)

    speech_segments = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=threshold,
        sampling_rate=TARGET_SAMPLE_RATE,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        return_seconds=True,
    )

    audio_duration = (
        len(audio) / TARGET_SAMPLE_RATE
    )

    speech_duration = sum(
        float(segment["end"])
        - float(segment["start"])
        for segment in speech_segments
    )

    speech_ratio = (
        speech_duration / audio_duration
        if audio_duration > 0
        else 0.0
    )

    return {
        "has_speech": len(speech_segments) > 0,
        "segments": speech_segments,
        "audio_duration": audio_duration,
        "speech_duration": speech_duration,
        "speech_ratio": speech_ratio,
        "threshold": threshold,
    }
