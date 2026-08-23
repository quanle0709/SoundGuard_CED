"""Frozen ESC-50 evaluation for the EfficientSED Emergency V3 specialist."""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from efficientsed_adapter import (
    FRAME_SECONDS,
    ROOT,
    infer_one,
    load_model,
    model_metadata,
)

EXPERIMENT = ROOT / "benchmark" / "experiments" / "emergency_v3"
OUT = ROOT / "benchmark_results" / "emergency_v3"
ONLY = OUT / "efficientsed_only"
MANIFEST = ROOT / "benchmark_data" / "external" / "esc50" / "manifest.csv"
V2_PREDICTIONS = ROOT / "benchmark_results" / "improved" / "emergency" / "predictions.csv"
CATEGORIES = ("glass_breaking", "vehicle_horn", "fire", "siren", "baby_crying")
SAFETY_TRUTH = set(CATEGORIES)
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 0.96, 0.05))
MAX_ACCEPTABLE_FPR = 0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer fields for empty CSV: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def binary_metrics(truth: list[bool], predicted: list[bool]) -> dict[str, float | int]:
    tp = sum(actual and guess for actual, guess in zip(truth, predicted))
    fp = sum(not actual and guess for actual, guess in zip(truth, predicted))
    tn = sum(not actual and not guess for actual, guess in zip(truth, predicted))
    fn = sum(actual and not guess for actual, guess in zip(truth, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "n": len(truth),
    }


def load_taxonomy(labels: list[str]) -> tuple[dict[str, str], dict[str, list[int]]]:
    payload = json.loads((EXPERIMENT / "efficientsed_safety_taxonomy.json").read_text(encoding="utf-8"))
    included = {
        entry["raw_label"]: entry["soundguard_category"]
        for entry in payload["entries"]
        if entry["classification"] in {"EXACT", "STRONG_SEMANTIC_ALIAS"}
    }
    unknown = sorted(set(included) - set(labels))
    if unknown:
        raise RuntimeError(f"Taxonomy labels absent from checkpoint ontology: {unknown}")
    indices = {
        category: [labels.index(label) for label, mapped in included.items() if mapped == category]
        for category in CATEGORIES
    }
    return included, indices


def longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def select_threshold(rows: list[dict], dev_indices: list[int]) -> tuple[float, list[dict]]:
    sweep = []
    truth = [rows[index]["truth_emergency"] for index in dev_indices]
    for threshold in THRESHOLDS:
        predicted = [rows[index]["max_safety_score"] >= threshold for index in dev_indices]
        metrics = binary_metrics(truth, predicted)
        sweep.append({"threshold": threshold, "split": "development_folds_1_4", **metrics})
    eligible = [row for row in sweep if row["false_positive_rate"] <= MAX_ACCEPTABLE_FPR]
    if not eligible:
        eligible = sweep
    selected = max(
        eligible,
        key=lambda row: (
            row["recall"],
            -row["false_negative_rate"],
            -row["fp"],
            row["precision"],
            row["f1"],
            row["threshold"],
        ),
    )
    return float(selected["threshold"]), sweep


def evaluate_category(rows: list[dict], indices: list[int], threshold: float) -> list[dict]:
    output = []
    for category in CATEGORIES:
        truth = [rows[index]["true_label"] == category for index in indices]
        pred = [rows[index]["category_scores"][category] >= threshold for index in indices]
        output.append({"category": category, **binary_metrics(truth, pred)})
    return output


def confusion(rows: list[dict], indices: list[int], threshold: float) -> list[dict]:
    labels = list(CATEGORIES) + ["non_emergency"]
    counts = Counter()
    for index in indices:
        row = rows[index]
        actual = row["true_label"] if row["truth_emergency"] else "non_emergency"
        predicted = (
            row["top_category"] if row["max_safety_score"] >= threshold else "non_emergency"
        )
        counts[(actual, predicted)] += 1
    return [
        {"actual": actual, **{predicted: counts[(actual, predicted)] for predicted in labels}}
        for actual in labels
    ]


def policy_metrics(rows: list[dict], indices: list[int], threshold: float) -> list[dict]:
    truth = [rows[index]["truth_emergency"] for index in indices]
    policies = {
        "A_v2_only": [rows[index]["v2_prediction"] for index in indices],
        "B_v2_or_efficientsed": [
            rows[index]["v2_prediction"] or rows[index]["max_safety_score"] >= threshold
            for index in indices
        ],
        "C_v2_or_persistent_efficientsed": [
            rows[index]["v2_prediction"]
            or (
                rows[index]["max_safety_score"] >= threshold
                and rows[index]["top_category_longest_run"] >= 3
            )
            for index in indices
        ],
        "D_efficientsed_only": [
            rows[index]["max_safety_score"] >= threshold for index in indices
        ],
    }
    output = []
    for name, prediction in policies.items():
        metrics = binary_metrics(truth, prediction)
        recalls = {}
        for category in CATEGORIES:
            selected = [position for position, index in enumerate(indices) if rows[index]["true_label"] == category]
            recalls[f"{category}_recall"] = (
                sum(prediction[position] for position in selected) / len(selected) if selected else 0.0
            )
        output.append({"policy": name, **metrics, **recalls})
    return output


def working_set_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.WorkingSetSize)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ONLY.mkdir(parents=True, exist_ok=True)
    started_load = time.perf_counter()
    with (OUT / "upstream_model_stdout.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        model, labels = load_model("cpu")
    load_seconds = time.perf_counter() - started_load
    metadata = model_metadata(model, labels)
    metadata["load_seconds"] = load_seconds
    metadata["working_set_after_load_bytes"] = working_set_bytes()
    included, category_indices = load_taxonomy(labels)

    manifest = read_csv(MANIFEST)
    v2 = {row["sample_id"]: row for row in read_csv(V2_PREDICTIONS)}
    rows = []
    framewise_payloads = []
    latency = []
    scores_cache = []
    raw_path = ONLY / "raw_frame_predictions.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as raw_stream:
        for position, case in enumerate(manifest, 1):
            result = infer_one(model, labels, ROOT / case["path"], "cpu")
            probabilities = result.pop("probabilities")
            latency.append(result["inference_ms"])
            category_frames = {
                category: probabilities[indices].max(axis=0)
                for category, indices in category_indices.items()
            }
            scores_cache.append(np.stack([category_frames[name] for name in CATEGORIES]))
            category_scores = {name: float(values.max()) for name, values in category_frames.items()}
            top_category = max(CATEGORIES, key=category_scores.get)
            top_values = category_frames[top_category]
            top_index = int(top_values.argmax())
            truth_emergency = case["true_label"] in SAFETY_TRUTH
            row = {
                **case,
                "truth_emergency": truth_emergency,
                "v2_prediction": v2[case["sample_id"]]["emergency_prediction"].lower() == "true",
                "category_scores": category_scores,
                "top_category": top_category,
                "max_safety_score": category_scores[top_category],
                "top_category_longest_run": 0,
                "inference_ms": result["inference_ms"],
            }
            rows.append(row)
            raw_safety = {
                raw_label: probabilities[labels.index(raw_label)].round(7).tolist()
                for raw_label in included
            }
            raw_stream.write(
                json.dumps(
                    {
                        "sample_id": case["sample_id"],
                        "frame_seconds": FRAME_SECONDS,
                        "frame_count": result["frame_count"],
                        "raw_safety_probabilities": raw_safety,
                        "top_k_per_frame": result["top_frames"],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            if position % 20 == 0 or position == len(manifest):
                print(f"EfficientSED inference {position}/{len(manifest)}", flush=True)

    dev_indices = [index for index, row in enumerate(rows) if int(row["fold"]) != 5]
    holdout_indices = [index for index, row in enumerate(rows) if int(row["fold"]) == 5]
    all_indices = list(range(len(rows)))
    threshold, sweep = select_threshold(rows, dev_indices)
    write_csv(OUT / "threshold_sweep.csv", sweep)

    for row, category_score_array in zip(rows, scores_cache):
        top_values = category_score_array[CATEGORIES.index(row["top_category"])]
        row["top_category_longest_run"] = longest_run(top_values >= threshold)

    prediction_rows = []
    for row in rows:
        predicted = row["max_safety_score"] >= threshold
        prediction_rows.append(
            {
                **{key: value for key, value in row.items() if key not in {"category_scores"}},
                **{f"{key}_max": value for key, value in row["category_scores"].items()},
                "selected_threshold": threshold,
                "efficientsed_prediction": predicted,
                "predicted_category": row["top_category"] if predicted else "non_emergency",
            }
        )
    write_csv(ONLY / "predictions.csv", prediction_rows)

    split_metrics = {}
    for split, indices in (
        ("development_folds_1_4", dev_indices),
        ("holdout_fold_5", holdout_indices),
        ("all_development_audio", all_indices),
    ):
        truth = [rows[index]["truth_emergency"] for index in indices]
        pred = [rows[index]["max_safety_score"] >= threshold for index in indices]
        split_metrics[split] = binary_metrics(truth, pred)
        write_csv(ONLY / f"per_class_metrics_{split}.csv", evaluate_category(rows, indices, threshold))
    write_json(
        ONLY / "metrics.json",
        {
            "selected_threshold": threshold,
            "selection_split": "ESC-50 folds 1-4",
            "holdout_split": "ESC-50 fold 5; inspected only after threshold freeze",
            "maximum_acceptable_development_fpr": MAX_ACCEPTABLE_FPR,
            "splits": split_metrics,
        },
    )
    write_csv(ONLY / "per_class_metrics.csv", evaluate_category(rows, all_indices, threshold))
    write_csv(ONLY / "confusion_matrix.csv", confusion(rows, all_indices, threshold))

    framewise_rows = []
    for row, category_score_array in zip(rows, scores_cache):
        if not row["truth_emergency"]:
            continue
        truth_values = category_score_array[CATEGORIES.index(row["true_label"])]
        above = truth_values >= threshold
        max_index = int(truth_values.argmax())
        framewise_rows.append(
            {
                "sample_id": row["sample_id"],
                "category": row["true_label"],
                "maximum_probability": float(truth_values[max_index]),
                "time_of_max_seconds": max_index * FRAME_SECONDS,
                "temporal_mean_probability": float(truth_values.mean()),
                "frames_above_threshold": int(above.sum()),
                "longest_consecutive_frames": longest_run(above),
                "longest_duration_seconds": longest_run(above) * FRAME_SECONDS,
                "correct_event_detected": bool(above.any()),
                "brief_only_not_mean_detected": bool(above.any() and truth_values.mean() < threshold),
                "selected_threshold": threshold,
            }
        )
    write_csv(OUT / "framewise_analysis.csv", framewise_rows)

    development_policies = policy_metrics(rows, dev_indices, threshold)
    holdout_policies = policy_metrics(rows, holdout_indices, threshold)
    all_policies = policy_metrics(rows, all_indices, threshold)
    policy_rows = [
        {"split": split, **row}
        for split, values in (
            ("development_folds_1_4", development_policies),
            ("holdout_fold_5", holdout_policies),
            ("all_development_audio", all_policies),
        )
        for row in values
    ]
    write_csv(OUT / "fusion_policy_comparison.csv", policy_rows)
    eligible_policies = [
        row
        for row in development_policies
        if row["policy"] in {"B_v2_or_efficientsed", "C_v2_or_persistent_efficientsed"}
        and row["false_positive_rate"] <= MAX_ACCEPTABLE_FPR
    ]
    selected_policy = max(
        eligible_policies,
        key=lambda row: (row["recall"], -row["fp"], row["precision"], row["f1"]),
        default=None,
    )
    gate_passed = bool(selected_policy and selected_policy["recall"] >= 0.70)
    write_json(
        OUT / "development_gate.json",
        {
            "passed": gate_passed,
            "required_recall": 0.70,
            "maximum_acceptable_fpr": MAX_ACCEPTABLE_FPR,
            "selected_policy": selected_policy,
            "threshold": threshold,
        },
    )
    latency_sorted = sorted(latency)
    latency_payload = {
        "device": "cpu",
        "sample_count": len(latency),
        "scope": "mel preprocessing plus model inference; excludes file decode and model load",
        "mean_ms": statistics.mean(latency),
        "median_ms": statistics.median(latency),
        "p95_ms": latency_sorted[math.ceil(0.95 * len(latency_sorted)) - 1],
        "min_ms": min(latency),
        "max_ms": max(latency),
        "model_load_seconds": load_seconds,
        "gpu_available": False,
    }
    write_json(OUT / "latency" / "cpu.json", latency_payload)
    metadata["working_set_after_benchmark_bytes"] = working_set_bytes()
    metadata["checkpoint_file_size_bytes"] = (
        ROOT / "benchmark_data" / "external" / "efficientsed_repo" / "resources" / "fmn10_strong.pt"
    ).stat().st_size
    write_json(OUT / "model_metadata.json", metadata)
    print(json.dumps({"threshold": threshold, "gate_passed": gate_passed, "selected_policy": selected_policy}, indent=2))


if __name__ == "__main__":
    main()
