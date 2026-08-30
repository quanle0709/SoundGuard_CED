"""Create plots and derived latency summaries from frozen Emergency V3 results."""

from __future__ import annotations

import csv
import json
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmark_results" / "emergency_v3"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[ceil(quantile * len(values)) - 1]


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "median_ms": percentile(ordered, 0.50),
        "p90_ms": percentile(ordered, 0.90),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    sweep = read_csv(OUT / "threshold_sweep.csv")
    thresholds = [float(row["threshold"]) for row in sweep]
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, [float(row["recall"]) for row in sweep], marker="o", label="Recall")
    plt.plot(thresholds, [float(row["precision"]) for row in sweep], marker="o", label="Precision")
    plt.plot(thresholds, [float(row["false_positive_rate"]) for row in sweep], marker="o", label="FPR")
    plt.axvline(0.05, color="black", linestyle="--", label="Selected 0.05")
    plt.xlabel("EfficientSED safety threshold")
    plt.ylabel("Ratio")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "threshold_tradeoff.png", dpi=160)
    plt.close()

    matrix_rows = read_csv(OUT / "efficientsed_only" / "confusion_matrix.csv")
    labels = [row["actual"] for row in matrix_rows]
    values = [[int(row[label]) for label in labels] for row in matrix_rows]
    plt.figure(figsize=(8, 6))
    plt.imshow(values, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            plt.text(column_index, row_index, str(value), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUT / "efficientsed_only" / "confusion_matrix.png", dpi=160)
    plt.close()

    specialist = read_csv(OUT / "efficientsed_only" / "predictions.csv")
    v2 = {
        row["sample_id"]: row
        for row in read_csv(ROOT / "benchmark_results/improved/emergency/predictions.csv")
    }
    specialist_latency = [float(row["inference_ms"]) for row in specialist]
    v2_latency = [float(v2[row["sample_id"]]["latency_ms"]) for row in specialist]
    combined_latency = [left + right for left, right in zip(v2_latency, specialist_latency)]
    payload = {
        "measurement_note": (
            "Sequential local computation estimate on the identical 280 clips; "
            "file capture, STT, HUD, model load, and subprocess startup excluded."
        ),
        "v2_ced_plus_emergency": distribution(v2_latency),
        "efficientsed_mel_plus_inference": distribution(specialist_latency),
        "combined_v3_sequential": distribution(combined_latency),
    }
    (OUT / "latency" / "combined.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
