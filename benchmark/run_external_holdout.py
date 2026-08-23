"""Run exactly the frozen SoundGuard V3 baseline on the frozen external set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.external_holdout_common import (  # noqa: E402
    DECONTAM_DIR,
    MANIFEST_FIELDS,
    RESULTS_DIR,
    ROOT,
    TARGET_CLASSES,
    read_csv,
    relative,
    sha256_file,
    write_csv,
    write_json,
)


FREEZE = DECONTAM_DIR / "freeze_manifest" / "final_external_freeze_manifest.json"
INCLUDED = DECONTAM_DIR / "included_manifests" / "external_included_manifest.csv"
RAW_JSONL = RESULTS_DIR / "raw_predictions.jsonl"
START_MARKER = RESULTS_DIR / "inference_started.json"
COMPLETE_MARKER = RESULTS_DIR / "inference_completed.json"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def verify_freeze() -> dict:
    if not FREEZE.is_file():
        raise RuntimeError("Final pre-prediction freeze does not exist")
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    for rel_path, expected in {
        **payload["artifact_sha256"],
        **payload["evaluator_code_sha256"],
    }.items():
        path = ROOT / rel_path
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen artifact changed before inference: {rel_path}")
    return payload


def existing_predictions() -> dict[str, dict]:
    if not RAW_JSONL.is_file():
        return {}
    rows = {}
    for line in RAW_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[f"{row['source_dataset']}:{row['sample_id']}"] = row
    return rows


def create_or_verify_start_marker(freeze: dict, resume: bool) -> None:
    freeze_hash = sha256_file(FREEZE)
    if START_MARKER.exists():
        marker = json.loads(START_MARKER.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError("Inference has already started; use --resume only for an interrupted identical run")
        if marker["freeze_manifest_sha256"] != freeze_hash:
            raise RuntimeError("Freeze manifest changed since inference started")
        if COMPLETE_MARKER.exists():
            raise RuntimeError("External inference is already complete; rerun is forbidden")
        return
    if resume:
        raise RuntimeError("Cannot resume because no inference-start marker exists")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    marker = {
        "started_at": now(),
        "freeze_manifest": relative(FREEZE),
        "freeze_manifest_sha256": freeze_hash,
        "baseline": freeze["frozen_baseline"],
        "rule": "This marker is written before the first external model call; completed rows may only be resumed, never retuned or selectively rerun.",
    }
    with START_MARKER.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(marker, indent=2) + "\n")


def flatten_prediction(row: dict) -> dict:
    scores = row.pop("efficientsed_scores")
    decisions = row.pop("class_decisions")
    return {
        **row,
        **{f"efficientsed_{key}_score": value for key, value in scores.items()},
        **{f"predict_{key}": value for key, value in decisions.items()},
    }


def write_split_outputs(rows: list[dict]) -> None:
    flattened = [flatten_prediction(dict(row)) for row in rows]
    write_csv(RESULTS_DIR / "raw_predictions.csv", flattened)
    for dataset, folder in (
        ("FSD50K_eval", "fsd50k"),
        ("DCASE2017_source_events", "dcase"),
    ):
        selected = [row for row in flattened if row["source_dataset"] == dataset]
        write_csv(RESULTS_DIR / folder / "raw_predictions.csv", selected)
        output = RESULTS_DIR / folder / "raw_predictions.json"
        write_json(output, selected)


def run(resume: bool) -> None:
    freeze = verify_freeze()
    manifest = sorted(
        read_csv(INCLUDED), key=lambda row: (row["source_dataset"], row["sample_id"])
    )
    create_or_verify_start_marker(freeze, resume)
    completed = existing_predictions()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING"] = "1"
    from emergency_system import evaluate_sound
    from emergency_v3 import EmergencyV3Specialist
    from sound_classifier import classify_audio_file

    specialist = EmergencyV3Specialist(logger=lambda message: print(message, file=sys.stderr))
    try:
        with RAW_JSONL.open("a", encoding="utf-8") as stream:
            for position, case in enumerate(manifest, 1):
                key = f"{case['source_dataset']}:{case['sample_id']}"
                if key in completed:
                    continue
                audio_path = ROOT / case["audio_path"]
                ced = classify_audio_file(audio_path)
                alert = evaluate_sound(ced["label"], float(ced["confidence"]), "neutral")
                v2_dangerous = bool(alert.get("is_dangerous"))
                efficient = specialist.analyze_file(audio_path)
                if not efficient.get("ok"):
                    raise RuntimeError(
                        f"Frozen EfficientSED inference failed for {key}: {efficient.get('error')}"
                    )
                scores = {name: float(efficient["scores"][name]) for name in TARGET_CLASSES}
                class_decisions = {
                    name: bool(
                        (v2_dangerous and ced["label"] == name)
                        or scores[name] >= float(efficient["threshold"])
                    )
                    for name in TARGET_CLASSES
                }
                if v2_dangerous and ced["label"] in TARGET_CLASSES:
                    predicted_category = ced["label"]
                    category_source = "ced_tiny_v2"
                elif efficient["detected"]:
                    predicted_category = efficient["category"]
                    category_source = "efficientsed"
                elif v2_dangerous:
                    predicted_category = "other_emergency"
                    category_source = "ced_tiny_v2"
                else:
                    predicted_category = "non_emergency"
                    category_source = "none"
                prediction = {
                    "sample_id": case["sample_id"],
                    "source_dataset": case["source_dataset"],
                    "audio_path": case["audio_path"],
                    "true_label": case["mapped_soundguard_label"],
                    "is_target": case["is_target"].lower() == "true",
                    "is_hard_negative": case["is_hard_negative"].lower() == "true",
                    "duration": float(case["duration"]),
                    "ced_raw_label": ced.get("raw_label", ced["label"]),
                    "ced_mapped_label": ced["label"],
                    "ced_confidence": float(ced["confidence"]),
                    "ced_top_predictions": ced["top_predictions"],
                    "v2_dangerous": v2_dangerous,
                    "v2_alert_level": alert.get("alert_level", alert.get("level", "")),
                    "efficientsed_detected": bool(efficient["detected"]),
                    "efficientsed_top_category": efficient["category"],
                    "efficientsed_top_score": float(efficient["score"]),
                    "efficientsed_threshold": float(efficient["threshold"]),
                    "efficientsed_scores": scores,
                    "efficientsed_inference_ms": float(efficient["inference_ms"]),
                    "class_decisions": class_decisions,
                    "combined_emergency_prediction": bool(v2_dangerous or efficient["detected"]),
                    "predicted_category": predicted_category,
                    "category_source": category_source,
                }
                stream.write(json.dumps(prediction, separators=(",", ":"), ensure_ascii=False) + "\n")
                stream.flush()
                completed[key] = prediction
                if position % 25 == 0 or position == len(manifest):
                    print(f"Frozen external inference {position}/{len(manifest)}", flush=True)
    finally:
        specialist.close()

    rows = [completed[f"{case['source_dataset']}:{case['sample_id']}"] for case in manifest]
    write_split_outputs(rows)
    write_json(
        COMPLETE_MARKER,
        {
            "completed_at": now(),
            "sample_count": len(rows),
            "freeze_manifest_sha256": sha256_file(FREEZE),
            "post_hoc_threshold_tuning": False,
            "selective_rerun": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.resume)


if __name__ == "__main__":
    main()

