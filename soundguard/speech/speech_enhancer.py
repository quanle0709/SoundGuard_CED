from pathlib import Path
import time

import numpy as np
import soundfile as sf

from soundguard.paths import REPOSITORY_ROOT


TARGET_SAMPLE_RATE = 16000
BLOCK_LENGTH = 512
BLOCK_SHIFT = 128

PROJECT_DIR = REPOSITORY_ROOT
MODEL_DIR = PROJECT_DIR / "external" / "DTLN-master" / "pretrained_model"
MODEL_1_PATH = MODEL_DIR / "model_1.tflite"
MODEL_2_PATH = MODEL_DIR / "model_2.tflite"

_INTERPRETER_1 = None
_INTERPRETER_2 = None


def _load_models():
    global _INTERPRETER_1, _INTERPRETER_2

    if _INTERPRETER_1 is not None and _INTERPRETER_2 is not None:
        return _INTERPRETER_1, _INTERPRETER_2

    import tensorflow as tf

    if not MODEL_1_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy DTLN model 1: {MODEL_1_PATH}"
        )

    if not MODEL_2_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy DTLN model 2: {MODEL_2_PATH}"
        )

    _INTERPRETER_1 = tf.lite.Interpreter(
        model_path=str(MODEL_1_PATH)
    )

    _INTERPRETER_2 = tf.lite.Interpreter(
        model_path=str(MODEL_2_PATH)
    )

    _INTERPRETER_1.allocate_tensors()
    _INTERPRETER_2.allocate_tensors()

    return _INTERPRETER_1, _INTERPRETER_2


def enhance_audio_file(
    input_wav: str | Path,
    output_wav: str | Path,
    verbose: bool = True,
) -> str:
    input_path = Path(input_wav)
    output_path = Path(output_wav)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy audio đầu vào: {input_path}"
        )

    audio, sample_rate = sf.read(
        input_path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError(
            "File audio đầu vào đang rỗng."
        )

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            "DTLN yêu cầu audio 16000 Hz, "
            f"nhưng file đang là {sample_rate} Hz."
        )

    interpreter_1, interpreter_2 = _load_models()

    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()

    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    states_1 = np.zeros(
        input_details_1[1]["shape"],
        dtype=np.float32,
    )

    states_2 = np.zeros(
        input_details_2[1]["shape"],
        dtype=np.float32,
    )

    output_audio = np.zeros_like(
        audio,
        dtype=np.float32,
    )

    input_buffer = np.zeros(
        BLOCK_LENGTH,
        dtype=np.float32,
    )

    output_buffer = np.zeros(
        BLOCK_LENGTH,
        dtype=np.float32,
    )

    number_of_blocks = (
        audio.shape[0] - (BLOCK_LENGTH - BLOCK_SHIFT)
    ) // BLOCK_SHIFT

    if number_of_blocks <= 0:
        raise ValueError(
            "File audio quá ngắn để DTLN xử lý."
        )

    start_time = time.perf_counter()

    for block_index in range(number_of_blocks):
        input_buffer[:-BLOCK_SHIFT] = (
            input_buffer[BLOCK_SHIFT:]
        )

        start = block_index * BLOCK_SHIFT
        end = start + BLOCK_SHIFT

        input_buffer[-BLOCK_SHIFT:] = audio[start:end]

        input_fft = np.fft.rfft(input_buffer)
        input_magnitude = np.abs(input_fft)
        input_phase = np.angle(input_fft)

        input_magnitude = np.reshape(
            input_magnitude,
            (1, 1, -1),
        ).astype(np.float32)

        interpreter_1.set_tensor(
            input_details_1[1]["index"],
            states_1,
        )

        interpreter_1.set_tensor(
            input_details_1[0]["index"],
            input_magnitude,
        )

        interpreter_1.invoke()

        output_mask = interpreter_1.get_tensor(
            output_details_1[0]["index"]
        )

        states_1 = interpreter_1.get_tensor(
            output_details_1[1]["index"]
        )

        estimated_complex = (
            input_magnitude
            * output_mask
            * np.exp(1j * input_phase)
        )

        estimated_block = np.fft.irfft(
            estimated_complex
        )

        estimated_block = np.reshape(
            estimated_block,
            (1, 1, -1),
        ).astype(np.float32)

        interpreter_2.set_tensor(
            input_details_2[1]["index"],
            states_2,
        )

        interpreter_2.set_tensor(
            input_details_2[0]["index"],
            estimated_block,
        )

        interpreter_2.invoke()

        output_block = interpreter_2.get_tensor(
            output_details_2[0]["index"]
        )

        states_2 = interpreter_2.get_tensor(
            output_details_2[1]["index"]
        )

        output_buffer[:-BLOCK_SHIFT] = (
            output_buffer[BLOCK_SHIFT:]
        )

        output_buffer[-BLOCK_SHIFT:] = 0.0
        output_buffer += np.squeeze(output_block)

        output_audio[start:end] = (
            output_buffer[:BLOCK_SHIFT]
        )

    processing_seconds = (
        time.perf_counter() - start_time
    )

    output_audio = np.nan_to_num(
        output_audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    output_audio = np.clip(
        output_audio,
        -1.0,
        1.0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        output_path,
        output_audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )

    audio_duration = (
        len(audio) / TARGET_SAMPLE_RATE
    )

    real_time_factor = (
        processing_seconds / audio_duration
        if audio_duration > 0
        else 0.0
    )

    if verbose:
        print(f"DTLN input: {input_path}")
        print(f"DTLN output: {output_path}")
        print(
            f"Audio duration: "
            f"{audio_duration:.3f} seconds"
        )
        print(
            f"Processing time: "
            f"{processing_seconds:.3f} seconds"
        )
        print(
            f"Real-time factor: "
            f"{real_time_factor:.3f}"
        )

    return str(output_path)
