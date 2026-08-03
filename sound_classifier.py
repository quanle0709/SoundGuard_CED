import numpy as np
import librosa
import torch
from transformers import pipeline


_MODEL = None


def _get_classifier():
    global _MODEL
    if _MODEL is None:
        device = 0 if torch.cuda.is_available() else -1
        _MODEL = pipeline(
            task="audio-classification",
            model="mispeech/ced-tiny",
            trust_remote_code=True,
            device=device,
        )
    return _MODEL


def load_audio_for_ced(file_path):
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    audio = audio.astype(np.float32)

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sr


def classify_audio_file(file_path):
    audio, sr = load_audio_for_ced(file_path)
    classifier = _get_classifier()

    predictions = classifier(
        {"array": audio, "sampling_rate": sr},
        top_k=None,
        function_to_apply="sigmoid",
    )

    if not predictions:
        return {"label": "unknown", "confidence": 0.0, "top_predictions": []}

    ranked_predictions = sorted(
        (
            {"label": item["label"], "score": float(item["score"])}
            for item in predictions
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:5]

    top_prediction = ranked_predictions[0]
    return {
        "label": top_prediction["label"],
        "confidence": float(top_prediction["score"]),
        "top_predictions": ranked_predictions,
    }
