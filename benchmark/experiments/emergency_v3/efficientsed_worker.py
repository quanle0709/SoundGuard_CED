"""Persistent JSON-lines worker for the isolated EfficientSED runtime."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

try:
    from .efficientsed_adapter import infer_one, load_model
except ImportError:  # Direct script execution in the isolated worker.
    from efficientsed_adapter import infer_one, load_model

THRESHOLD = 0.05
CATEGORIES = ("glass_breaking", "vehicle_horn", "fire", "siren", "baby_crying")


def load_mapping(labels: list[str]) -> dict[str, list[int]]:
    taxonomy = Path(__file__).with_name("efficientsed_safety_taxonomy.json")
    payload = json.loads(taxonomy.read_text(encoding="utf-8"))
    mapping = {category: [] for category in CATEGORIES}
    for entry in payload["entries"]:
        if entry["classification"] in {"EXACT", "STRONG_SEMANTIC_ALIAS"}:
            mapping[entry["soundguard_category"]].append(labels.index(entry["raw_label"]))
    return mapping


def main() -> None:
    with redirect_stdout(sys.stderr):
        model, labels = load_model("cpu")
    mapping = load_mapping(labels)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                result = infer_one(model, labels, Path(request["audio_path"]), "cpu")
            probabilities = result["probabilities"]
            scores = {
                category: float(probabilities[indices].max())
                for category, indices in mapping.items()
            }
            category = max(CATEGORIES, key=scores.get)
            response = {
                "ok": True,
                "detected": scores[category] >= THRESHOLD,
                "category": category,
                "score": scores[category],
                "threshold": THRESHOLD,
                "scores": scores,
                "inference_ms": result["inference_ms"],
            }
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
