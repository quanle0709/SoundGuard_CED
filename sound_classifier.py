import numpy as np
import librosa
import torch
from transformers import pipeline

from sound_taxonomy import map_ced_label, semantic_ced_enabled


_MODEL = None
MODEL_ID = "mispeech/ced-tiny"
MODEL_REVISION = "ace276d29dd0bb3f3517b0fa8cf300738c409019"


def _resolve_model_source():
    """Prefer the frozen local snapshot and avoid optional Hub probes offline."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        return MODEL_ID, {
            "revision": MODEL_REVISION,
            "local_files_only": True,
        }
    except (OSError, ValueError):
        # A fresh installation may not have the frozen snapshot yet. In that
        # case the normal Hub path remains available, but is pinned to the same
        # immutable revision used by the benchmark baseline.
        return MODEL_ID, {"revision": MODEL_REVISION}


def _get_classifier():
    global _MODEL
    if _MODEL is None:
        device = 0 if torch.cuda.is_available() else -1
        model_source, model_kwargs = _resolve_model_source()
        _MODEL = pipeline(
            task="audio-classification",
            model=model_source,
            trust_remote_code=True,
            device=device,
            **model_kwargs,
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
    result = {
        "label": top_prediction["label"],
        "confidence": float(top_prediction["score"]),
        "top_predictions": ranked_predictions,
    }
    if semantic_ced_enabled():
        mapped = map_ced_label(top_prediction["label"])
        if mapped:
            result["raw_label"] = top_prediction["label"]
            result["label"] = mapped
            result["semantic_mapping"] = "soundguard_taxonomy_v1"
    return result
