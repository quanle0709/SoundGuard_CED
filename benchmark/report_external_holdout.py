"""Calculate preregistered external metrics and KHKT-ready evidence outputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.external_holdout_common import (  # noqa: E402
    DECONTAM_DIR,
    RESULTS_DIR,
    ROOT,
    TARGET_CLASSES,
    read_csv,
    write_csv,
    write_json,
)


def binary_metrics(truth: list[bool], prediction: list[bool]) -> dict:
    tp = sum(a and p for a, p in zip(truth, prediction))
    fp = sum(not a and p for a, p in zip(truth, prediction))
    tn = sum(not a and not p for a, p in zip(truth, prediction))
    fn = sum(a and not p for a, p in zip(truth, prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(truth),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def load_external() -> list[dict]:
    complete = RESULTS_DIR / "inference_completed.json"
    if not complete.is_file():
        raise RuntimeError("Frozen external inference is not complete")
    rows = []
    for line in (RESULTS_DIR / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    expected = len(
        read_csv(DECONTAM_DIR / "included_manifests" / "external_included_manifest.csv")
    )
    if len(rows) != expected:
        raise RuntimeError(f"Prediction count {len(rows)} does not match frozen manifest {expected}")
    return rows


def class_metrics(rows: list[dict]) -> list[dict]:
    output = []
    for category in TARGET_CLASSES:
        truth = [row["true_label"] == category for row in rows]
        prediction = [bool(row["class_decisions"][category]) for row in rows]
        output.append({"class": category, "support": sum(truth), **binary_metrics(truth, prediction)})
    return output


def macro(per_class: list[dict]) -> dict:
    return {
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "classes": len(per_class),
    }


def confusion(rows: list[dict]) -> tuple[list[str], np.ndarray, list[dict]]:
    labels = [*TARGET_CLASSES, "non_emergency", "other_emergency"]
    counts = Counter((row["true_label"], row["predicted_category"]) for row in rows)
    matrix = np.array([[counts[(actual, predicted)] for predicted in labels] for actual in labels])
    csv_rows = [
        {"actual": actual, **{predicted: int(matrix[i, j]) for j, predicted in enumerate(labels)}}
        for i, actual in enumerate(labels)
    ]
    return labels, matrix, csv_rows


def internal_comparator() -> tuple[list[dict], dict]:
    efficient = read_csv(
        ROOT / "benchmark_results" / "emergency_v3" / "efficientsed_only" / "predictions.csv"
    )
    v2 = {
        row["sample_id"]: row
        for row in read_csv(
            ROOT / "benchmark_results" / "improved" / "emergency" / "predictions.csv"
        )
    }
    rows = []
    for row in efficient:
        base = v2[row["sample_id"]]
        v2_emergency = base["emergency_prediction"].lower() == "true"
        decisions = {
            category: bool(
                (v2_emergency and base["predicted_label"] == category)
                or float(row[f"{category}_max"]) >= 0.05
            )
            for category in TARGET_CLASSES
        }
        rows.append(
            {
                "true_label": row["true_label"],
                "is_target": row["true_label"] in TARGET_CLASSES,
                "is_hard_negative": row["true_label"] not in TARGET_CLASSES,
                "class_decisions": decisions,
                "combined_emergency_prediction": bool(
                    v2_emergency or row["efficientsed_prediction"].lower() == "true"
                ),
            }
        )
    per_class = class_metrics(rows)
    binary = binary_metrics(
        [row["is_target"] for row in rows],
        [row["combined_emergency_prediction"] for row in rows],
    )
    return per_class, binary


def save_figures(per_class: list[dict], internal_per_class: list[dict], matrix: np.ndarray, labels: list[str], hard_fpr: float) -> None:
    figure_dir = RESULTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(per_class))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric in zip((-width, 0, width), ("precision", "recall", "f1")):
        ax.bar(x + offset, [row[metric] for row in per_class], width, label=metric.title())
    ax.set_xticks(x, [row["class"].replace("_", "\n") for row in per_class])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("External decontaminated holdout: per-class results")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "external_per_class_precision_recall_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), [label.replace("_", "\n") for label in labels], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), [label.replace("_", "\n") for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("External category confusion (single top category)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(figure_dir / "external_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, [row["f1"] for row in internal_per_class], 0.36, label="Internal / development")
    ax.bar(x + 0.18, [row["f1"] for row in per_class], 0.36, label="External decontaminated")
    ax.set_xticks(x, [row["class"].replace("_", "\n") for row in per_class])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Internal vs external class F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "internal_vs_external_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Correct rejection", "False alert"], [1 - hard_fpr, hard_fpr], color=["#4c78a8", "#e45756"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of hard negatives")
    ax.set_title("External hard-negative alert behavior")
    fig.tight_layout()
    fig.savefig(figure_dir / "hard_negative_false_positive_rate.png", dpi=180)
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    rows = load_external()
    per_class = class_metrics(rows)
    macro_result = macro(per_class)
    binary = binary_metrics(
        [bool(row["is_target"]) for row in rows],
        [bool(row["combined_emergency_prediction"]) for row in rows],
    )
    hard = [row for row in rows if row["is_hard_negative"]]
    hard_binary = binary_metrics(
        [False] * len(hard), [bool(row["combined_emergency_prediction"]) for row in hard]
    )
    fsd = [row for row in rows if row["source_dataset"] == "FSD50K_eval"]
    dcase = [row for row in rows if row["source_dataset"] == "DCASE2017_source_events"]
    datasets = {}
    for name, selected in (("fsd50k", fsd), ("dcase", dcase)):
        class_rows = class_metrics(selected)
        datasets[name] = {
            "n": len(selected),
            "target_n": sum(bool(row["is_target"]) for row in selected),
            "hard_negative_n": sum(bool(row["is_hard_negative"]) for row in selected),
            "binary": binary_metrics(
                [bool(row["is_target"]) for row in selected],
                [bool(row["combined_emergency_prediction"]) for row in selected],
            ),
            "per_class": class_rows,
        }

    labels, matrix, matrix_rows = confusion(rows)
    internal_per_class, internal_binary = internal_comparator()
    internal_macro = macro(internal_per_class)
    comparison = [
        {
            "metric": metric,
            "internal_development": internal,
            "external_decontaminated": external,
            "delta_external_minus_internal": external - internal,
        }
        for metric, internal, external in (
            ("target_macro_precision", internal_macro["macro_precision"], macro_result["macro_precision"]),
            ("target_macro_recall", internal_macro["macro_recall"], macro_result["macro_recall"]),
            ("target_macro_f1", internal_macro["macro_f1"], macro_result["macro_f1"]),
            ("emergency_recall", internal_binary["recall"], binary["recall"]),
            ("false_negative_rate", internal_binary["false_negative_rate"], binary["false_negative_rate"]),
            ("hard_negative_fpr", internal_binary["false_positive_rate"], hard_binary["false_positive_rate"]),
        )
    ]
    metrics = {
        "methodology": "Decontaminated external holdout derived from official FSD50K/DCASE evaluation data",
        "overall": {
            "total_n": len(rows),
            "target_n": sum(bool(row["is_target"]) for row in rows),
            "hard_negative_n": len(hard),
            **macro_result,
            "emergency": binary,
            "hard_negative": hard_binary,
        },
        "per_class": per_class,
        "datasets": datasets,
        "post_hoc_threshold_tuning": False,
    }
    write_json(RESULTS_DIR / "metrics.json", metrics)
    write_csv(RESULTS_DIR / "per_class_metrics.csv", per_class)
    write_csv(RESULTS_DIR / "confusion_matrix.csv", matrix_rows)
    write_csv(RESULTS_DIR / "internal_vs_external.csv", comparison)
    write_json(
        RESULTS_DIR / "internal_vs_external.json",
        {
            "internal_scope": "ESC-50 development audio; same frozen combined class-decision rule reconstructed from saved predictions",
            "external_scope": metrics["methodology"],
            "comparison": comparison,
        },
    )
    for name in ("fsd50k", "dcase"):
        write_json(RESULTS_DIR / name / "metrics.json", datasets[name])
        write_csv(RESULTS_DIR / name / "per_class_metrics.csv", datasets[name]["per_class"])
    save_figures(per_class, internal_per_class, matrix, labels, hard_binary["false_positive_rate"])

    strongest = sorted(per_class, key=lambda row: (row["f1"], row["support"]), reverse=True)
    report = f"""# SoundGuard External Validation — {DATE_TITLE}

## 1. Decontamination Summary

See `../external_decontamination_{DATE_TAG}/decontamination_summary.json`. Exact-ID, raw/decoded hash, and conservative acoustic-landmark checks were completed before prediction. The dataset is a **decontaminated external holdout derived from official FSD50K/DCASE evaluation data**, not an untouched official evaluation split.

## 2. Frozen External Dataset

| Dataset | Final N | Target N | Hard-negative N |
|---|---:|---:|---:|
| FSD50K-derived | {len(fsd)} | {sum(bool(r['is_target']) for r in fsd)} | {sum(bool(r['is_hard_negative']) for r in fsd)} |
| DCASE-derived | {len(dcase)} | {sum(bool(r['is_target']) for r in dcase)} | 0 |
| Total | {len(rows)} | {sum(bool(r['is_target']) for r in rows)} | {len(hard)} |

## 3. External Results

| Metric | Result |
|---|---:|
| Target Macro Precision | {fmt(macro_result['macro_precision'])} |
| Target Macro Recall | {fmt(macro_result['macro_recall'])} |
| Target Macro F1 | {fmt(macro_result['macro_f1'])} |
| Emergency Recall | {fmt(binary['recall'])} |
| False Negative Rate | {fmt(binary['false_negative_rate'])} |
| Hard-negative FPR | {fmt(hard_binary['false_positive_rate'])} |

## 4. KHKT Main Results

1. **External target Macro-F1:** {fmt(macro_result['macro_f1'])}. KHKT claim: category performance generalizes beyond the development corpus. Report use: primary external-results table.
2. **Emergency Recall:** {fmt(binary['recall'])} ({binary['tp']}/{binary['tp'] + binary['fn']}). KHKT claim: proportion of relevant safety events detected. Report use: safety-results table.
3. **False Negative Rate:** {fmt(binary['false_negative_rate'])} ({binary['fn']} missed events). KHKT claim: empirical missed-event risk. Report use: limitations and safety table.
4. **Hard-negative FPR:** {fmt(hard_binary['false_positive_rate'])} ({hard_binary['fp']}/{len(hard)}). KHKT claim: the system does not alert indiscriminately on preregistered unrelated sound families. Report use: false-alert figure/table.
5. **Strongest class F1:** {strongest[0]['class']} = {fmt(strongest[0]['f1'])}, N={strongest[0]['support']}. KHKT claim: strongest independently evaluated capability. Report use: per-class chart.

## 5. Internal vs External

The comparison is in `internal_vs_external.csv`; deltas are external minus internal and are reported as a generalization gap, without threshold changes.

## 6. Leakage / Validity Statement

Contamination auditing occurred before prediction. Exact ESC-50 Freesound-ID overlaps were excluded using provenance, then source-file/decoded-PCM hashes and conservative time-offset-consistent acoustic landmarks were checked against available SoundGuard development audio. Model, mappings, thresholds, sample population, and hard-negative rules were frozen before the first external inference. No sample was selected from model output and no post-hoc tuning was performed.

## 7. Reproducibility

The model/checkpoint hashes, CED revision, preprocessing, threshold 0.05, mappings, manifests, code hashes, dependency snapshot, and inference start/completion markers are preserved. Raw predictions are retained for every included sample.
"""
    (RESULTS_DIR / "FINAL_EXTERNAL_REPORT.md").write_text(report, encoding="utf-8")


DATE_TAG = "20260820"
DATE_TITLE = "2026-08-20"


if __name__ == "__main__":
    main()

