from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter


def classification_metrics(truths: list[str], predictions: list[str]) -> dict:
    labels = sorted(set(truths) | set(predictions))
    rows = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truths, predictions))
        fp = sum(t != label and p == label for t, p in zip(truths, predictions))
        fn = sum(t == label and p != label for t, p in zip(truths, predictions))
        support = sum(t == label for t in truths)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "precision": precision, "recall": recall,
                     "f1": f1, "support": support, "tp": tp, "fp": fp, "fn": fn})
    count = len(truths)
    accuracy = sum(t == p for t, p in zip(truths, predictions)) / count if count else 0.0
    macro = lambda key: statistics.fmean(row[key] for row in rows) if rows else 0.0
    weighted_f1 = sum(row["f1"] * row["support"] for row in rows) / count if count else 0.0
    return {"count": count, "accuracy": accuracy, "macro_precision": macro("precision"),
            "macro_recall": macro("recall"), "macro_f1": macro("f1"),
            "weighted_f1": weighted_f1, "per_class": rows, "labels": labels}


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p90_ms": percentile(values, 0.90), "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99), "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def normalize_vi(text: str) -> str:
    value = unicodedata.normalize("NFC", text or "").lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    matrix = [[(0, 0, 0, 0)] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        matrix[i][0] = (i, 0, i, 0)
    for j in range(1, len(hypothesis) + 1):
        matrix[0][j] = (j, 0, 0, j)
    for i, ref in enumerate(reference, 1):
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                matrix[i][j] = matrix[i - 1][j - 1]
                continue
            candidates = []
            distance, sub, delete, insert = matrix[i - 1][j - 1]
            candidates.append((distance + 1, sub + 1, delete, insert))
            distance, sub, delete, insert = matrix[i - 1][j]
            candidates.append((distance + 1, sub, delete + 1, insert))
            distance, sub, delete, insert = matrix[i][j - 1]
            candidates.append((distance + 1, sub, delete, insert + 1))
            matrix[i][j] = min(candidates)
    _, substitutions, deletions, insertions = matrix[-1][-1]
    return substitutions, deletions, insertions


def error_rates(reference: str, hypothesis: str) -> dict:
    reference, hypothesis = normalize_vi(reference), normalize_vi(hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    substitutions, deletions, insertions = edit_counts(ref_words, hyp_words)
    char_sub, char_del, char_ins = edit_counts(list(reference), list(hypothesis))
    return {
        "reference": reference, "hypothesis": hypothesis,
        "wer": (substitutions + deletions + insertions) / len(ref_words) if ref_words else 0.0,
        "cer": (char_sub + char_del + char_ins) / len(reference) if reference else 0.0,
        "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        "reference_words": len(ref_words), "reference_characters": len(reference),
    }


def binary_metrics(truths: list[bool], predictions: list[bool]) -> dict:
    counts = Counter((truth, prediction) for truth, prediction in zip(truths, predictions))
    tp, fp = counts[(True, True)], counts[(False, True)]
    tn, fn = counts[(False, False)], counts[(True, False)]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"sample_count": len(truths), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(truths) if truths else 0.0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0}
