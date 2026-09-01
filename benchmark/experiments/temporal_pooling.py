"""Controlled clip-window pooling experiment for the frozen ESC-50 set."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "benchmark_results" / "improvement" / "temporal_pooling_predictions.csv"


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_predictions() -> list[dict]:
    manifest = _read(ROOT / "benchmark_data" / "external" / "esc50" / "manifest.csv")
    if CACHE.exists():
        rows = _read(CACHE)
        if len(rows) == len(manifest) * 3:
            return rows
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from soundguard.detection.sound_classifier import _get_classifier, load_audio_for_ced

    classifier = _get_classifier()
    rows = []
    for index, case in enumerate(manifest, 1):
        audio, rate = load_audio_for_ced(ROOT / case["path"])
        window_samples = rate
        windows = [audio[start:min(start + window_samples, len(audio))]
                   for start in range(0, len(audio), window_samples)
                   if len(audio[start:min(start + window_samples, len(audio))]) >= rate // 2]
        outputs = classifier([{"array": window, "sampling_rate": rate} for window in windows],
                             top_k=None, function_to_apply="sigmoid", batch_size=len(windows))
        scores: dict[str, list[float]] = {}
        for output in outputs:
            for item in output:
                scores.setdefault(item["label"], []).append(float(item["score"]))
        aggregations = {
            "max_1s": {label: max(values) for label, values in scores.items()},
            "top2_mean_1s": {label: float(np.mean(sorted(values, reverse=True)[:2])) for label, values in scores.items()},
            "p90_1s": {label: float(np.percentile(values, 90)) for label, values in scores.items()},
        }
        for method, values in aggregations.items():
            raw_label, confidence = max(values.items(), key=lambda item: item[1])
            rows.append({"sample_id": case["sample_id"], "true_label": case["true_label"],
                         "method": method, "raw_label": raw_label, "confidence": confidence,
                         "window_count": len(windows), "window_seconds": 1.0})
        if index % 20 == 0:
            print(f"temporal pooling {index}/{len(manifest)}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows
