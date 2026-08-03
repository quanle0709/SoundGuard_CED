"""Pure, dependency-free metrics for SoundGuard fixed-file benchmarks."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1, previous[column] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def _truth(value) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _number(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def calculate_metrics(rows: list[dict], run_id: str | None = None,
                      include_breakdowns: bool = True) -> dict:
    if run_id is not None:
        rows = [row for row in rows if row.get("run_id") == run_id]
    included_run_ids = sorted({str(row.get("run_id", "")).strip() for row in rows
                               if str(row.get("run_id", "")).strip()})
    valid = [row for row in rows if row.get("status") == "completed"]
    def stage_succeeded(row, stage, legacy_field):
        status = str(row.get(f"{stage}_status", "")).strip().lower()
        return status == "success" or (not status and str(row.get(legacy_field, "")).strip() != "")

    expected_speech_rows = [r for r in rows if _truth(r.get("expected_speech")) is True]
    non_speech_rows = [r for r in rows if _truth(r.get("expected_speech")) is False]
    speech_with_truth = [r for r in expected_speech_rows
                         if normalize_text(r.get("ground_truth_transcript", ""))]
    speech_missing_truth = [r for r in expected_speech_rows
                            if not normalize_text(r.get("ground_truth_transcript", ""))]
    transcript_rows = [r for r in speech_with_truth
                       if stage_succeeded(r, "stt", "transcript")]
    stt_failed_rows = [r for r in speech_with_truth
                       if not stage_succeeded(r, "stt", "transcript")]
    word_edits = word_total = char_edits = char_total = exact = empty = 0
    for row in transcript_rows:
        reference = normalize_text(row["ground_truth_transcript"])
        hypothesis = normalize_text(row.get("transcript", ""))
        ref_words, hyp_words = reference.split(), hypothesis.split()
        word_edits += edit_distance(ref_words, hyp_words)
        word_total += len(ref_words)
        char_edits += edit_distance(list(reference), list(hypothesis))
        char_total += len(reference)
        exact += reference == hypothesis
        empty += not hypothesis

    ced_valid = [r for r in rows if stage_succeeded(r, "ced", "predicted_sound")]
    sound_rows = [r for r in ced_valid if str(r.get("ground_truth_sound", "")).strip()]
    category_rows = [r for r in ced_valid if str(r.get("ground_truth_category", "")).strip()]
    labels = sorted({str(r.get("ground_truth_category", "")) for r in category_rows} |
                    {str(r.get("predicted_category", "")) for r in category_rows})
    confusion = Counter((r["ground_truth_category"], r.get("predicted_category", ""))
                        for r in category_rows)
    per_class = {}
    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(count for (truth, pred), count in confusion.items()
                 if pred == label and truth != label)
        fn = sum(count for (truth, pred), count in confusion.items()
                 if truth == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0, "support": tp + fn}

    alert_rows = [r for r in ced_valid if str(r.get("expected_alert_level", "")).strip()]
    help_rows = [(r, _truth(r.get("expected_help_request"))) for r in transcript_rows]
    help_rows = [(r, truth) for r, truth in help_rows if truth is not None]
    vad_valid = [r for r in rows if stage_succeeded(r, "vad", "vad_detected")]
    vad_rows = [(r, _truth(r.get("expected_speech"))) for r in vad_valid]
    vad_rows = [(r, truth) for r, truth in vad_rows if truth is not None]
    emergency_rows = [(r, _truth(r.get("expected_emergency"))) for r in rows
                      if _truth(r.get("emergency_detected")) is not None]
    emergency_rows = [(r, truth) for r, truth in emergency_rows if truth is not None]
    false_emergencies = sum(not truth and _truth(r.get("emergency_detected")) is True
                            for r, truth in emergency_rows)
    missed_emergencies = sum(truth and _truth(r.get("emergency_detected")) is not True
                             for r, truth in emergency_rows)

    latencies = defaultdict(list)
    for row in rows:
        for name in ("vad_latency_seconds", "dtln_latency_seconds", "ced_latency_seconds",
                     "stt_latency_seconds", "total_latency_seconds", "dtln_realtime_factor",
                     "vad_model_load_or_cold_start_latency_seconds", "vad_inference_latency_seconds",
                     "ced_model_load_or_cold_start_latency_seconds", "ced_inference_latency_seconds",
                     "dtln_model_load_or_cold_start_latency_seconds", "dtln_inference_latency_seconds",
                     "dtln_inference_realtime_factor", "stt_model_load_or_cold_start_latency_seconds",
                     "stt_inference_latency_seconds"):
            value = _number(row.get(name))
            if value is not None:
                latencies[name].append(value)
    latency_summary = {name: {"count": len(values), "mean": sum(values) / len(values),
                              "min": min(values), "max": max(values)}
                       for name, values in latencies.items() if values}

    pairs = defaultdict(dict)
    for row in transcript_rows:
        if row.get("condition") in {"dtln_enabled", "dtln_disabled"}:
            reference = normalize_text(row["ground_truth_transcript"]).split()
            hypothesis = normalize_text(row.get("transcript", "")).split()
            pairs[(row.get("sample_id"), row.get("run_id"))][row["condition"]] = (
                edit_distance(reference, hypothesis) / len(reference) if reference else 0.0)
    paired = Counter()
    paired_count = 0
    for pair in pairs.values():
        if set(pair) != {"dtln_enabled", "dtln_disabled"}:
            continue
        paired_count += 1
        delta = pair["dtln_enabled"] - pair["dtln_disabled"]
        paired["improved" if delta < 0 else "worsened" if delta > 0 else "unchanged"] += 1

    result = {
        "selected_run_id": run_id, "included_run_ids": included_run_ids,
        "unique_wav_count": len({r.get("source_sha256") or r.get("relative_path")
                                 for r in rows if r.get("source_sha256") or r.get("relative_path")}),
        "unique_sample_count": len({r.get("sample_id") for r in rows if r.get("sample_id")}),
        "sample_condition_row_count": len(rows), "paired_dtln_sample_count": paired_count,
        "dataset_rows": len(rows), "completed_rows": len(valid),
        "failed_rows": len(rows) - len(valid),
        "transcript_ground_truth_rows": len(transcript_rows),
        "speech_rows_eligible_for_transcript_metrics": len(speech_with_truth),
        "non_speech_rows_excluded_from_transcript_metrics": len(non_speech_rows),
        "speech_rows_missing_transcript_ground_truth": len(speech_missing_truth),
        "stt_success_rows_evaluated": len(transcript_rows),
        "stt_failed_rows_excluded": len(stt_failed_rows),
        "wer": word_edits / word_total if word_total else None,
        "cer": char_edits / char_total if char_total else None,
        "exact_transcript_match": exact / len(transcript_rows) if transcript_rows else None,
        "empty_transcript_rate": empty / len(transcript_rows) if transcript_rows else None,
        "sound_ground_truth_rows": len(sound_rows),
        "ced_valid_rows": len(ced_valid),
        "missing_sound_ground_truth": len(ced_valid) - len(sound_rows),
        "top1_sound_accuracy": (sum(normalize_text(r["ground_truth_sound"]) ==
                                    normalize_text(r.get("predicted_sound", "")) for r in sound_rows)
                                / len(sound_rows) if sound_rows else None),
        "canonical_category_accuracy": (sum(r["ground_truth_category"] == r.get("predicted_category", "")
                                            for r in category_rows) / len(category_rows)
                                        if category_rows else None),
        "per_class": per_class,
        "confusion_matrix": {f"{truth} -> {pred}": count
                             for (truth, pred), count in sorted(confusion.items())},
        "expected_alert_level_accuracy": (sum(r["expected_alert_level"] == r.get("sound_alert_level", "")
                                               for r in alert_rows) / len(alert_rows)
                                          if alert_rows else None),
        "false_emergencies": false_emergencies, "missed_emergencies": missed_emergencies,
        "positive_emergency_rows": sum(truth for _, truth in emergency_rows),
        "negative_emergency_rows": sum(not truth for _, truth in emergency_rows),
        "help_request_accuracy": (sum(truth == (_truth(r.get("help_request_detected")) is True)
                                      for r, truth in help_rows) / len(help_rows) if help_rows else None),
        "vad_correctness": (sum(truth == (_truth(r.get("vad_detected")) is True)
                                for r, truth in vad_rows) / len(vad_rows) if vad_rows else None),
        "stage_status_counts": {
            stage: dict(Counter(str(r.get(f"{stage}_status", "legacy_or_unknown") or "legacy_or_unknown")
                                for r in rows))
            for stage in ("vad", "ced", "dtln", "stt")
        },
        "latencies": latency_summary, "paired_dtln": dict(paired),
    }
    if include_breakdowns:
        groups = {
            "clean_speech_dtln_disabled": [r for r in rows if r.get("scenario") == "clean_speech"
                                            and r.get("condition") == "dtln_disabled"],
            "noisy_speech_dtln_enabled": [r for r in rows if r.get("scenario") == "noisy_speech"
                                           and r.get("condition") == "dtln_enabled"],
            "noisy_speech_dtln_disabled": [r for r in rows if r.get("scenario") == "noisy_speech"
                                            and r.get("condition") == "dtln_disabled"],
        }
        result["condition_metrics"] = {
            name: calculate_metrics(group, include_breakdowns=False)
            for name, group in groups.items()
        }
    return result
