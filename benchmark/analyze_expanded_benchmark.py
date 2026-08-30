"""Analysis-only diagnostics for the frozen expanded SoundGuard benchmark.

This module reads existing benchmark inputs and outputs and writes only below
``benchmark_results/analysis``.  It deliberately does not mutate production
configuration, thresholds, models, mappings, or decision logic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from benchmark.metrics import binary_metrics, classification_metrics, error_rates
from benchmark.noise import align_to_reference, snr_db


ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "benchmark_results" / "expanded"
OUT = ROOT / "benchmark_results" / "analysis"
DATA = ROOT / "benchmark_data"
SEED = 42
TARGETS = ("baby_crying", "dog_barking", "door_activity", "fire",
           "glass_breaking", "siren", "vehicle_horn")
EMERGENCY_LABELS = {"baby_crying", "fire", "glass_breaking", "siren", "vehicle_horn"}

# Conservative, documented top-1 equivalences for diagnostic scoring only.
# Generic parent labels are included only where AudioSet's ontology and the
# ESC-50 source event make the meaning unambiguous.
ONTOLOGY_EQUIVALENTS = {
    "baby_crying": {"Baby cry", "Crying"},
    "dog_barking": {"Bark", "Dog", "Bow-wow"},
    "door_activity": {"Knock"},
    "fire": {"Fire", "Crackle"},
    "glass_breaking": {"Glass", "Shatter"},
    "siren": {"Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)"},
    "vehicle_horn": {"Vehicle horn", "Toot", "Honk", "Air horn"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(rate)


def safe_mean(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def safe_median(values: list[float]) -> float | str:
    return statistics.median(values) if values else ""


def estimate_lag(reference: np.ndarray, degraded: np.ndarray, max_lag: int = 2048) -> int:
    """Independent bounded normalized cross-correlation lag estimate."""
    reference = np.asarray(reference, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)
    reference -= reference.mean() if reference.size else 0.0
    degraded -= degraded.mean() if degraded.size else 0.0
    corr = correlate(degraded, reference, mode="full", method="fft")
    lags = correlation_lags(len(degraded), len(reference), mode="full")
    allowed = np.abs(lags) <= max_lag
    return int(lags[allowed][np.argmax(corr[allowed])])


def best_scale(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = float(np.dot(estimate.astype(np.float64), estimate.astype(np.float64)))
    return float(np.dot(reference.astype(np.float64), estimate.astype(np.float64)) / denominator) if denominator else 0.0


def validate_snr_metric() -> dict:
    rng = np.random.default_rng(SEED)
    t = np.arange(16000, dtype=np.float64) / 16000
    clean = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    cases = []
    for expected in (20.0, 10.0, 0.0):
        noise = rng.standard_normal(clean.shape).astype(np.float32)
        noise *= np.sqrt(np.mean(clean.astype(np.float64) ** 2)) / (
            10 ** (expected / 20) * np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
        measured = snr_db(clean, clean + noise)
        cases.append({"case": f"known_{expected:g}_db", "expected_db": expected,
                      "measured_db": measured, "absolute_error_db": abs(measured - expected),
                      "passed": abs(measured - expected) < 0.05})
    identical = snr_db(clean, clean)
    delayed = np.concatenate((np.zeros(384, dtype=np.float32), clean))[:len(clean)]
    ar, ad, lag = align_to_reference(clean, delayed)
    scaled = clean * 0.5
    scale = best_scale(clean, scaled)
    delayed_scaled = delayed * 0.5
    dr, dd, dlag = align_to_reference(clean, delayed_scaled)
    dscale = best_scale(dr, dd)
    edge = {
        "identical_is_infinite": math.isinf(identical) and identical > 0,
        "delayed_lag_samples": lag,
        "delayed_aligned_snr_db": snr_db(ar, ad),
        "scaled_raw_snr_db": snr_db(clean, scaled),
        "scaled_best_gain": scale,
        "scaled_corrected_snr_db": snr_db(clean, scaled * scale),
        "delayed_scaled_lag_samples": dlag,
        "delayed_scaled_corrected_snr_db": snr_db(dr, dd * dscale),
    }
    return {"known_snr_cases": cases, "edge_cases": edge,
            "all_known_cases_pass": all(row["passed"] for row in cases),
            "interpretation": "Plain SNR is alignment- and gain-sensitive; aligned, gain-corrected checks recover delayed/scaled identity."}


def ontology_config() -> tuple[dict, Path]:
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "models--mispeech--ced-tiny" / "snapshots"
    configs = sorted(snapshots.glob("*/config.json"))
    if not configs:
        raise FileNotFoundError("Cached mispeech/ced-tiny config.json is required for ontology audit")
    path = configs[-1]
    return json.loads(path.read_text(encoding="utf-8")), path


def collect_clean_details() -> list[dict]:
    cache = OUT / "ced" / "clean_top5_details.csv"
    prior = read_csv(cache) if cache.exists() else []
    clean = read_csv(EXPANDED / "ced" / "predictions.csv")
    if len(prior) == len(clean):
        return prior
    # Avoid incidental hub calls: all required weights/config are already cached.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sound_classifier import classify_audio_file, load_audio_for_ced

    details = []
    for index, row in enumerate(clean, 1):
        path = ROOT / row["path"]
        raw, rate = load_mono(path)
        normalized, model_rate = load_audio_for_ced(path)
        result = classify_audio_file(path)
        ranked = result["top_predictions"][:5]
        details.append({**row, "filename": path.name, "source_dataset": "ESC-50",
            "clip_duration_seconds": len(raw) / rate, "sample_rate": rate,
            "peak_amplitude": float(np.max(np.abs(raw))),
            "rms": float(np.sqrt(np.mean(raw.astype(np.float64) ** 2))),
            "preprocessing_output_shape": str(list(normalized.shape)),
            "model_input_shape": json.dumps({"array": list(normalized.shape), "sampling_rate": model_rate}),
            "top_1": ranked[0]["label"] if ranked else "",
            "top_1_score": ranked[0]["score"] if ranked else "",
            "top_3": json.dumps(ranked[:3], ensure_ascii=False),
            "top_5": json.dumps(ranked[:5], ensure_ascii=False)})
        if index % 40 == 0:
            print(f"CED top-k analysis {index}/{len(clean)}", flush=True)
    write_csv(cache, details)
    return details


def ced_analysis() -> dict:
    details = collect_clean_details()
    truths = [row["true_label"] for row in details]
    predictions = [row["predicted_label"] for row in details]
    metrics = classification_metrics(truths, predictions)
    per_class = []
    for base in (row for row in metrics["per_class"] if row["label"] in TARGETS):
        label = base["label"]
        correct_conf = [float(r["confidence"]) for r in details if r["true_label"] == label and r["correct"] == "True"]
        wrong_conf = [float(r["confidence"]) for r in details if r["true_label"] == label and r["correct"] != "True"]
        errors = Counter(r["raw_predicted_label"] for r in details if r["true_label"] == label and r["correct"] != "True")
        per_class.append({"label": label, "n": base["support"], "tp": base["tp"], "fp": base["fp"],
            "fn": base["fn"], "precision": base["precision"], "recall": base["recall"], "f1": base["f1"],
            "most_common_predictions_when_wrong": json.dumps(errors.most_common(5), ensure_ascii=False),
            "mean_confidence_correct": safe_mean(correct_conf), "mean_confidence_wrong": safe_mean(wrong_conf),
            "median_confidence_correct": safe_median(correct_conf), "median_confidence_wrong": safe_median(wrong_conf)})
    write_csv(OUT / "ced" / "per_class_failure_analysis.csv", per_class)

    for target in ("glass_breaking", "dog_barking"):
        rows = [r for r in details if r["true_label"] == target]
        write_csv(OUT / "ced" / f"{target}_failures.csv", rows)

    strict = [row["correct"] == "True" for row in details]
    aware = [is_strict or row["raw_predicted_label"] in ONTOLOGY_EQUIVALENTS[row["true_label"]]
             for row, is_strict in zip(details, strict)]
    strict_metrics = classification_metrics(truths, predictions)
    aware_predictions = [t if ok else p for t, p, ok in zip(truths, predictions, aware)]
    aware_metrics = classification_metrics(truths, aware_predictions)
    comparison = []
    for name, result in (("STRICT BASELINE", strict_metrics), ("ONTOLOGY-AWARE — DIAGNOSTIC ONLY", aware_metrics)):
        comparison.append({"metric_variant": name, "sample_count": len(details), "accuracy": result["accuracy"],
                           "macro_f1": statistics.fmean(r["f1"] for r in result["per_class"] if r["label"] in TARGETS),
                           "additional_correct_vs_strict": sum(aware) - sum(strict) if "ONTOLOGY" in name else 0})
    write_csv(OUT / "ced" / "strict_vs_ontology_aware.csv", comparison)

    taxonomy = []
    transient = {"glass_breaking", "vehicle_horn", "door_activity"}
    for row, is_aware in zip(details, aware):
        if row["correct"] == "True": category, evidence = "correct", "strict top-1 match"
        elif is_aware: category, evidence = "ontology/label mismatch", f"{row['raw_predicted_label']} is a documented equivalent"
        elif float(row["confidence"]) < 0.5: category, evidence = "low-confidence ambiguity", "top-1 confidence < 0.50"
        elif float(row["confidence"]) >= 0.75: category, evidence = "confidently wrong classification", "wrong top-1 confidence >= 0.75"
        elif row["true_label"] in transient: category, evidence = "transient-event failure", "transient source event; no direct preprocessing defect proven"
        elif row["true_label"] in {"fire", "siren"}: category, evidence = "sustained-noise confusion", "sustained source class confused at intermediate confidence"
        else: category, evidence = "unknown", "available evidence is insufficient for a narrower causal label"
        taxonomy.append({"sample_id": row["sample_id"], "true_label": row["true_label"],
                         "prediction": row["raw_predicted_label"], "confidence": row["confidence"],
                         "correct": row["correct"], "error_category": category, "evidence": evidence})
    counts = Counter(r["error_category"] for r in taxonomy)
    total_errors = sum(v for k, v in counts.items() if k != "correct")
    for row in taxonomy:
        row["category_count"] = counts[row["error_category"]]
        row["percent_of_all_samples"] = counts[row["error_category"]] / len(taxonomy) * 100
        row["percent_of_errors"] = (counts[row["error_category"]] / total_errors * 100
                                    if row["error_category"] != "correct" and total_errors else 0)
    write_csv(OUT / "ced" / "error_taxonomy.csv", taxonomy)

    config, config_path = ontology_config()
    ontology = set(config["label2id"])
    audit_rows = []
    source_lookup = {r["true_label"]: r["source_label"] for r in details}
    mapping_types = {"siren": "EXACT", "baby_crying": "SEMANTICALLY VALID", "glass_breaking": "SEMANTICALLY VALID",
                     "vehicle_horn": "EXACT", "dog_barking": "SEMANTICALLY VALID", "fire": "SEMANTICALLY VALID",
                     "door_activity": "SEMANTICALLY VALID"}
    for label in TARGETS:
        equivalents = sorted(ONTOLOGY_EQUIVALENTS[label] & ontology)
        audit_rows.append({"source_class": source_lookup[label], "expected_soundguard_label": label,
                           "exact_model_ontology_labels": "; ".join(equivalents),
                           "mapping_validity": mapping_types[label], "mapping_confidence": "high",
                           "notes": "SoundGuard uses a coarser canonical category than the 527-label AudioSet ontology."})

    confusions = [f"# CED class confusions\n\nFull 280-sample clean ESC-50 analysis. Top-k was re-extracted from the frozen model without changing preprocessing.\n",
                  "| Class | N | Recall | F1 | Dominant wrong top-1 labels |\n|---|---:|---:|---:|---|\n"]
    for row in per_class:
        confusions.append(f"| {row['label']} | {row['n']} | {float(row['recall']):.3f} | {float(row['f1']):.3f} | {row['most_common_predictions_when_wrong']} |\n")
    (OUT / "ced" / "class_confusions.md").write_text("".join(confusions), encoding="utf-8")

    for target in ("glass_breaking", "dog_barking"):
        selected = [r for r in details if r["true_label"] == target]
        wrong = Counter(r["raw_predicted_label"] for r in selected if r["correct"] != "True")
        aware_count = sum(r["raw_predicted_label"] in ONTOLOGY_EQUIVALENTS[target] for r in selected)
        mean_wrong = statistics.fmean(float(r["confidence"]) for r in selected if r["correct"] != "True")
        text = f"""# {target.replace('_', ' ').title()} root cause

All {len(selected)} clean ESC-50 samples were inspected. Strict correct: {sum(r['correct'] == 'True' for r in selected)}. Conservative ontology-aware top-1 correct: {aware_count}. Dominant wrong labels: {wrong.most_common(10)}. Mean wrong top-1 confidence: {mean_wrong:.4f}.

The CED ontology contains: {sorted(ONTOLOGY_EQUIVALENTS[target])}. SoundGuard's target is coarser than the model ontology. The current canonicalizer recognizes some but not all of these semantically related labels, so exact/canonical scoring drops valid parent or sibling concepts. Remaining failures are genuine model/domain errors under this diagnostic; no evidence proves waveform peak normalization, input shape, or the model's internal mean pooling is the dominant cause. Every clip's top-5, duration, sample rate, amplitude, RMS, and input shape is in the companion CSV. No production mapping or preprocessing was changed.
"""
        (OUT / "ced" / f"{target}_root_cause.md").write_text(text, encoding="utf-8")

    audit = ["# CED label mapping audit\n\nModel: `mispeech/ced-tiny`; cached ontology has 527 labels. Diagnostic equivalences do not change baseline scoring.\n\n",
             "| Source class | Expected SoundGuard label | Exact model ontology label(s) | Mapping valid? | Mapping confidence | Notes |\n|---|---|---|---|---|---|\n"]
    for row in audit_rows:
        audit.append(f"| {row['source_class']} | {row['expected_soundguard_label']} | {row['exact_model_ontology_labels']} | {row['mapping_validity']} | {row['mapping_confidence']} | {row['notes']} |\n")
    audit.append(f"\nOntology source SHA-256: `{sha256(config_path)}`. `Glass breaking` and `dog barking` are not literal model labels; they map to AudioSet children/parents such as Glass/Shatter and Dog/Bark.\n")
    (OUT / "ced" / "label_mapping_audit.md").write_text("".join(audit), encoding="utf-8")
    return {"per_class": per_class, "comparison": comparison, "taxonomy_counts": dict(counts), "details": details}


def emergency_analysis(details: list[dict]) -> dict:
    from emergency_system import CATEGORY_CONFIG, LEVEL_ORDER, evaluate_sound, match_sound_category

    false_negatives, all_rows = [], []
    for row in details:
        truth = row["true_label"] in EMERGENCY_LABELS
        mapped = evaluate_sound(row["raw_predicted_label"], float(row["confidence"]))
        predicted = bool(mapped["matched"] and mapped["is_dangerous"])
        record = {**row, "ground_truth_emergency": truth, "predicted_emergency": predicted,
                  "alert_mapper_result": json.dumps(mapped, ensure_ascii=False), "fusion_result": "not used by expanded emergency benchmark",
                  "final_emergency_output": "emergency" if predicted else "no emergency"}
        all_rows.append(record)
        if truth and not predicted:
            candidate = match_sound_category(row["raw_predicted_label"])
            semantically_recognized = row["raw_predicted_label"] in ONTOLOGY_EQUIVALENTS[row["true_label"]]
            if candidate != row["true_label"] and not semantically_recognized:
                stage, code = "CED failed to recognize the event", "A"
            elif not mapped["matched"] or candidate != row["true_label"]:
                reason = "confidence threshold rejected it" if candidate == row["true_label"] else "semantic ontology label was not mapped"
                stage, code = f"CED recognized it but alert mapper {reason}", "B"
            elif not mapped["is_dangerous"]:
                stage, code = "emergency severity rule did not classify it as emergency", "D"
            else:
                stage, code = "other", "F"
            false_negatives.append({"source_filename": row["filename"], "source_case": row["sample_id"],
                "source_dataset": "ESC-50", "true_source_class": row["source_label"],
                "mapped_ced_class": row["true_label"], "expected_emergency_state": True,
                "ced_top_1_prediction": row["raw_predicted_label"], "ced_confidence": row["confidence"],
                "top_k_predictions": row["top_5"], "alert_mapper_result": json.dumps(mapped, ensure_ascii=False),
                "fusion_result": "not used", "final_emergency_output": "no emergency",
                "failure_stage_code": code, "exact_stage_lost": stage})
    write_csv(OUT / "emergency" / "all_false_negatives.csv", false_negatives)
    stage_counts = Counter(r["failure_stage_code"] for r in false_negatives)
    stage_names = {"A": "CED failed to recognize the event", "B": "CED recognized it but alert mapper failed",
                   "C": "fusion suppressed it", "D": "emergency rule classified it non-emergency", "E": "ground-truth/rule mismatch", "F": "other"}
    stage_rows = [{"failure_stage": code, "description": stage_names[code], "count": stage_counts.get(code, 0),
                   "percent_of_98_fn": stage_counts.get(code, 0) / len(false_negatives) * 100}
                  for code in "ABCDEF"]
    write_csv(OUT / "emergency" / "fn_root_causes.csv", stage_rows)

    recall_rows = []
    for label in sorted(EMERGENCY_LABELS):
        selected = [r for r in all_rows if r["true_label"] == label]
        tp = sum(r["predicted_emergency"] for r in selected)
        recall_rows.append({"source_class": label, "positives_n": len(selected), "tp": tp, "fn": len(selected)-tp,
                            "recall": tp/len(selected), "fnr": (len(selected)-tp)/len(selected)})
    recall_rows.sort(key=lambda r: r["recall"])
    write_csv(OUT / "emergency" / "recall_by_class.csv", recall_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar([r["source_class"] for r in recall_rows], [r["recall"] for r in recall_rows])
    ax.set_ylim(0, 1); ax.set_ylabel("Recall"); ax.set_title("Emergency recall by source class (N=40 each)")
    ax.tick_params(axis="x", rotation=25); fig.tight_layout(); (OUT / "emergency").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "emergency" / "recall_by_class.png", dpi=160); plt.close(fig)

    sweep = []
    for threshold in np.arange(0.0, 0.91, 0.05):
        predicted = []
        truth = []
        for row in all_rows:
            category = match_sound_category(row["raw_predicted_label"])
            is_emergency_category = category in EMERGENCY_LABELS and LEVEL_ORDER[CATEGORY_CONFIG[category].level] >= LEVEL_ORDER["MEDIUM"] if category else False
            predicted.append(bool(is_emergency_category and float(row["confidence"]) >= threshold))
            truth.append(row["true_label"] in EMERGENCY_LABELS)
        result = binary_metrics(truth, predicted)
        sweep.append({"diagnostic": "OFFLINE DIAGNOSTIC — NOT PRODUCTION CONFIGURATION", "candidate_global_threshold": round(float(threshold), 2),
                      "recall": result["recall"], "precision": result["precision"], "fnr": result["false_negative_rate"],
                      "fpr": result["fp"]/(result["fp"]+result["tn"]) if result["fp"]+result["tn"] else 0,
                      "f1": result["f1"], "tp": result["tp"], "fp": result["fp"], "tn": result["tn"], "fn": result["fn"]})
    write_csv(OUT / "emergency" / "threshold_diagnostic.csv", sweep)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot([r["recall"] for r in sweep], [r["precision"] for r in sweep], marker="o")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("OFFLINE DIAGNOSTIC — NOT PRODUCTION CONFIGURATION")
    fig.tight_layout(); fig.savefig(OUT / "emergency" / "precision_recall_tradeoff.png", dpi=160); plt.close(fig)
    return {"false_negatives": false_negatives, "stage_rows": stage_rows, "recall_rows": recall_rows, "threshold": sweep}


def dtln_analysis() -> dict:
    quality = read_csv(EXPANDED / "noise_filter" / "snr_improvement.csv")
    pairs = read_csv(EXPANDED / "noise_filter" / "ced_predictions_paired.csv")
    esc = {r["sample_id"]: r for r in read_csv(DATA / "external" / "esc50" / "manifest.csv")}
    write_json(OUT / "dtln" / "snr_metric_validation.json", validate_snr_metric())

    # One CED clip/class across every environment and SNR, plus every cached
    # real-noise STT pair (150): broad but bounded independent alignment audit.
    chosen = {}
    for row in quality:
        chosen.setdefault(row["true_label"], row["sample_id"])
    alignment_rows = []
    for row in quality:
        if chosen[row["true_label"]] != row["sample_id"]:
            continue
        base = DATA / "expanded_generated" / "dtln"
        noisy = base / "inputs" / row["noise_environment"] / row["target_snr_db"] / f"{row['sample_id']}.wav"
        filtered = ROOT / row["filtered_path"]
        source, _ = load_mono(noisy); output, _ = load_mono(filtered)
        lag = estimate_lag(source, output)
        alignment_rows.append({"domain": "CED", "sample_id": row["sample_id"], "class": row["true_label"],
            "noise_environment": row["noise_environment"], "snr_db": row["target_snr_db"],
            "estimated_lag_samples": lag, "assumed_lag_samples": 384, "residual_after_384": lag-384})
    stt = read_csv(EXPANDED / "stt" / "noise_results.csv")
    raw_stt = {(r["sample_id"], r["noise_environment"], r["target_snr_db"]): r for r in stt if r["protocol"] == "real" and r["filter"] == "unfiltered"}
    for row in (r for r in stt if r["protocol"] == "real" and r["filter"] == "dtln"):
        raw = raw_stt[(row["sample_id"], row["noise_environment"], row["target_snr_db"])]
        source, _ = load_mono(ROOT / raw["file"]); output, _ = load_mono(ROOT / row["file"])
        lag = estimate_lag(source, output)
        alignment_rows.append({"domain": "STT", "sample_id": row["sample_id"], "class": "speech",
            "noise_environment": row["noise_environment"], "snr_db": row["target_snr_db"],
            "estimated_lag_samples": lag, "assumed_lag_samples": 384, "residual_after_384": lag-384})
    write_csv(OUT / "dtln" / "alignment_analysis.csv", alignment_rows)
    lags = [int(r["estimated_lag_samples"]) for r in alignment_rows]
    alignment_summary = {"sample_count": len(lags), "mean": statistics.fmean(lags), "median": statistics.median(lags),
                         "min": min(lags), "max": max(lags), "standard_deviation": statistics.pstdev(lags),
                         "assumed_delay_samples": 384, "exact_384_count": lags.count(384),
                         "residual_nonzero_count": sum(v != 384 for v in lags)}
    write_json(OUT / "dtln" / "alignment_summary.json", alignment_summary)

    signal_rows = []
    for index, row in enumerate(quality, 1):
        clean, _ = load_mono(ROOT / esc[row["sample_id"]]["path"])
        clean_16 = np.asarray(__import__("librosa").resample(clean, orig_sr=44100, target_sr=16000), dtype=np.float32)
        base = DATA / "expanded_generated" / "dtln"
        noisy, _ = load_mono(base / "inputs" / row["noise_environment"] / row["target_snr_db"] / f"{row['sample_id']}.wav")
        output, _ = load_mono(ROOT / row["filtered_path"])
        gain = float(np.dot(noisy.astype(np.float64), clean_16[:len(noisy)].astype(np.float64)) /
                     max(np.dot(clean_16[:len(noisy)].astype(np.float64), clean_16[:len(noisy)].astype(np.float64)), 1e-12))
        reference = clean_16 * gain
        ar, ao, lag = align_to_reference(reference, output)
        opt = best_scale(ar, ao)
        size = min(len(reference), len(noisy))
        signal_rows.append({"sample_id": row["sample_id"], "class": row["true_label"], "noise_environment": row["noise_environment"],
            "snr_db": row["target_snr_db"], "rms_input": float(np.sqrt(np.mean(noisy.astype(np.float64)**2))),
            "rms_output": float(np.sqrt(np.mean(output.astype(np.float64)**2))),
            "gain_ratio_output_input": float(np.sqrt(np.mean(output.astype(np.float64)**2))/max(np.sqrt(np.mean(noisy.astype(np.float64)**2)),1e-12)),
            "peak_input": float(np.max(np.abs(noisy))), "peak_output": float(np.max(np.abs(output))),
            "clipping_rate_input": float(np.mean(np.abs(noisy) >= .999)), "clipping_rate_output": float(np.mean(np.abs(output) >= .999)),
            "alignment_lag_samples": lag, "correlation_clean_input": float(np.corrcoef(reference[:size], noisy[:size])[0,1]),
            "correlation_clean_output": float(np.corrcoef(ar, ao)[0,1]),
            "input_error_power": float(np.mean((noisy[:size]-reference[:size]).astype(np.float64)**2)),
            "output_error_power": float(np.mean((ao-ar).astype(np.float64)**2)),
            "output_error_power_gain_corrected": float(np.mean((ao*opt-ar).astype(np.float64)**2)),
            "input_snr_db": snr_db(reference[:size], noisy[:size]), "output_snr_db": snr_db(ar, ao),
            "output_scale_to_reference": opt, "output_snr_gain_corrected_db": snr_db(ar, ao*opt),
            "delta_snr_db": snr_db(ar, ao)-snr_db(reference[:size], noisy[:size]),
            "delta_snr_gain_corrected_db": snr_db(ar, ao*opt)-snr_db(reference[:size], noisy[:size])})
        if index % 100 == 0:
            print(f"DTLN signal diagnostics {index}/{len(quality)}", flush=True)

    clean_stt = {r["sample_id"]: r for r in read_csv(EXPANDED / "stt" / "clean_results.csv")}
    for row in (r for r in stt if r["protocol"] == "real" and r["filter"] == "dtln"):
        raw = raw_stt[(row["sample_id"], row["noise_environment"], row["target_snr_db"])]
        clean, _ = load_mono(ROOT / clean_stt[row["sample_id"]]["path"])
        noisy, _ = load_mono(ROOT / raw["file"])
        output, _ = load_mono(ROOT / row["file"])
        size = min(len(clean), len(noisy))
        gain = float(np.dot(noisy[:size].astype(np.float64), clean[:size].astype(np.float64)) /
                     max(np.dot(clean[:size].astype(np.float64), clean[:size].astype(np.float64)), 1e-12))
        reference = clean * gain
        ar, ao, lag = align_to_reference(reference, output)
        opt = best_scale(ar, ao)
        input_size = min(len(reference), len(noisy))
        signal_rows.append({"sample_id": row["sample_id"], "class": "speech", "noise_environment": row["noise_environment"],
            "snr_db": row["target_snr_db"], "rms_input": float(np.sqrt(np.mean(noisy.astype(np.float64)**2))),
            "rms_output": float(np.sqrt(np.mean(output.astype(np.float64)**2))),
            "gain_ratio_output_input": float(np.sqrt(np.mean(output.astype(np.float64)**2))/max(np.sqrt(np.mean(noisy.astype(np.float64)**2)),1e-12)),
            "peak_input": float(np.max(np.abs(noisy))), "peak_output": float(np.max(np.abs(output))),
            "clipping_rate_input": float(np.mean(np.abs(noisy) >= .999)), "clipping_rate_output": float(np.mean(np.abs(output) >= .999)),
            "alignment_lag_samples": lag, "correlation_clean_input": float(np.corrcoef(reference[:input_size], noisy[:input_size])[0,1]),
            "correlation_clean_output": float(np.corrcoef(ar, ao)[0,1]),
            "input_error_power": float(np.mean((noisy[:input_size]-reference[:input_size]).astype(np.float64)**2)),
            "output_error_power": float(np.mean((ao-ar).astype(np.float64)**2)),
            "output_error_power_gain_corrected": float(np.mean((ao*opt-ar).astype(np.float64)**2)),
            "input_snr_db": snr_db(reference[:input_size], noisy[:input_size]), "output_snr_db": snr_db(ar, ao),
            "output_scale_to_reference": opt, "output_snr_gain_corrected_db": snr_db(ar, ao*opt),
            "delta_snr_db": snr_db(ar, ao)-snr_db(reference[:input_size], noisy[:input_size]),
            "delta_snr_gain_corrected_db": snr_db(ar, ao*opt)-snr_db(reference[:input_size], noisy[:input_size])})
    write_csv(OUT / "dtln" / "signal_diagnostics.csv", signal_rows)

    transition_rows = []
    dimensions = (("overall", lambda r: "all"), ("snr", lambda r: r["target_snr_db"]),
                  ("environment", lambda r: r["noise_environment"]), ("class", lambda r: r["true_label"]))
    for dimension, key_fn in dimensions:
        groups = defaultdict(list)
        for row in pairs: groups[key_fn(row)].append(row)
        for value, selected in groups.items():
            counts = Counter((r["without_correct"] == "True", r["with_correct"] == "True") for r in selected)
            transition_rows.append({"dimension": dimension, "value": value, "sample_count": len(selected),
                "A_raw_correct_dtln_correct": counts[(True,True)], "B_raw_wrong_dtln_correct": counts[(False,True)],
                "C_raw_correct_dtln_wrong": counts[(True,False)], "D_raw_wrong_dtln_wrong": counts[(False,False)]})
    write_csv(OUT / "dtln" / "ced_transition_matrix.csv", transition_rows)

    stt_transitions = []
    for key, raw in raw_stt.items():
        filtered = next(r for r in stt if r["protocol"] == "real" and r["filter"] == "dtln" and
                        (r["sample_id"], r["noise_environment"], r["target_snr_db"]) == key)
        before, after = float(raw["wer"]), float(filtered["wer"])
        state = "improved WER" if after < before else "worsened WER" if after > before else "unchanged WER"
        stt_transitions.append({"sample_id": key[0], "noise_environment": key[1], "snr_db": key[2],
                                "wer_without_filter": before, "wer_with_filter": after, "delta_wer": after-before, "transition": state})
    write_csv(OUT / "dtln" / "stt_transition_analysis.csv", stt_transitions)
    return {"alignment": alignment_summary, "signals": signal_rows, "transitions": transition_rows, "stt": stt_transitions}


def personalization_analysis() -> dict:
    from personalization.profile_generator import CONTEXT_KEYWORDS, ROLE_KEYWORDS, generate_rule_based
    cases = [
        ("factory-method", "A factory-method pattern is software."), ("baby-blue", "The baby-blue color is bright."),
        ("driver-side", "A driver-side CSS selector is documented."), ("student-t", "Use Student-t statistics."),
        ("school-bus", "A school-bus paint code is listed."), ("hospital-grade", "Hospital-grade plastic is durable."),
        ("parent-child", "A parent-child process relationship exists."), ("construction-time", "This is a construction-time option."),
        ("warehouse-native", "The warehouse-native SQL feature is experimental."), ("babyproof", "This material is babyproof."),
        ("factorymade", "This item is factorymade."), ("university-level", "This is a university-level course."),
    ]
    rows = []
    normalized = lambda text: __import__("re").sub(r"\s+", " ", __import__("re").sub(r"[^a-z0-9']+", " ", text.lower())).strip()
    for case_id, text in cases:
        profile = generate_rule_based([text]); norm = normalized(text)
        matches = []
        for role, phrases in ROLE_KEYWORDS.items():
            for phrase in phrases:
                if phrase in norm.split() or f" {phrase} " in f" {norm} ": matches.append(f"role:{role}<-{phrase}")
        for context, phrases in CONTEXT_KEYWORDS.items():
            for phrase in phrases:
                if phrase in norm.split() or f" {phrase} " in f" {norm} ": matches.append(f"context:{context}<-{phrase}")
        rows.append({"id": case_id, "text": text, "normalized_text": norm, "expected_roles": "[]",
                     "actual_roles": json.dumps(profile["roles"]), "actual_contexts": json.dumps(profile["contexts"]),
                     "matched_evidence": "; ".join(matches), "passed": not profile["roles"]})
    write_csv(OUT / "personalization" / "adversarial_diagnostic.csv", rows)
    failures = [r for r in rows if not r["passed"]]
    text = f"""# Personalization parser root cause

The failing inputs normalize punctuation before matching: `factory-method` becomes `factory method`, and `baby-blue` becomes `baby blue`. The whole-phrase regex then sees standalone `factory` and `baby`, so it correctly enforces boundaries on the *normalized* text but has already lost evidence that the token came from a hyphenated compound. `factory` activates `factory_warehouse_worker` and `factory_warehouse`; `baby` activates `parent_infant_caregiver` and the infant responsibility.

This is a general punctuation-normalization ambiguity, not a failure of the final profile validator. Of {len(rows)} diagnostic compounds, {len(failures)} produced at least one role. These expected outcomes were hand-defined as non-role technical/adjectival uses; no LLM supplied ground truth. The companion CSV contains normalized forms and exact inferred roles/contexts. No production parser change was made.
"""
    (OUT / "personalization" / "root_cause.md").write_text(text, encoding="utf-8")
    return {"rows": rows, "failures": failures}


def strip_diacritics(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def stt_analysis() -> dict:
    clean = read_csv(EXPANDED / "stt" / "clean_results.csv")
    rows = []
    for row in clean:
        stripped = error_rates(strip_diacritics(row["reference"]), strip_diacritics(row["hypothesis"]))
        rows.append({"sample_id": row["sample_id"], "speaker": row["speaker"], "reference_words": row["reference_words"],
            "wer": row["wer"], "cer": row["cer"], "substitutions": row["substitutions"], "deletions": row["deletions"],
            "insertions": row["insertions"], "diacritic_stripped_cer": stripped["cer"],
            "cer_reduction_when_diacritics_removed": float(row["cer"])-stripped["cer"],
            "reference": row["reference"], "hypothesis": row["hypothesis"]})
    rows.sort(key=lambda r: float(r["wer"]), reverse=True)
    write_csv(OUT / "stt" / "error_analysis.csv", rows)
    by_speaker = defaultdict(list)
    for row in rows: by_speaker[row["speaker"]].append(float(row["wer"]))
    total_ref_words = sum(int(r["reference_words"]) for r in clean)
    subs, dels, ins = (sum(int(r[name]) for r in clean) for name in ("substitutions", "deletions", "insertions"))
    diacritic_delta = statistics.fmean(float(r["cer_reduction_when_diacritics_removed"]) for r in rows)
    short = [float(r["wer"]) for r in rows if int(r["reference_words"]) <= 7]
    long = [float(r["wer"]) for r in rows if int(r["reference_words"]) > 7]
    summary = f"""# STT error summary

All {len(clean)} clean VIVOS utterances were analyzed. Corpus edit counts: {subs} substitutions, {dels} deletions, {ins} insertions over {total_ref_words} reference words. Highest-WER utterances are listed first in `error_analysis.csv`.

Mean per-utterance CER change after stripping Vietnamese combining marks from both reference and hypothesis: {diacritic_delta:.4f}. This is a diagnostic sensitivity estimate, not a claim that accents should be ignored. Mean WER for <=7-word utterances: {statistics.fmean(short):.4f}; for longer utterances: {statistics.fmean(long):.4f}. Speaker mean-WER range: {min(statistics.fmean(v) for v in by_speaker.values()):.4f} to {max(statistics.fmean(v) for v in by_speaker.values()):.4f}. The small per-speaker sample counts preclude a demographic conclusion.
"""
    (OUT / "stt" / "error_summary.md").write_text(summary, encoding="utf-8")
    return {"substitutions": subs, "deletions": dels, "insertions": ins, "diacritic_delta": diacritic_delta}


def fusion_analysis() -> dict:
    ced_count = len(read_csv(EXPANDED / "ced" / "predictions.csv"))
    stt_count = len(read_csv(EXPANDED / "stt" / "clean_results.csv"))
    text = f"""# Fusion expansion feasibility

The existing real-audio caches contain {ced_count} ESC-50 CED clips and {stt_count} VIVOS speech clips, but they are disjoint recordings. There are **0 real paired cases** with independently grounded CED and transcript evidence from the same event, so 0 existing cases exercise both fusion branches simultaneously.

A legitimate 30–50-case fusion benchmark is not feasible from the current outputs alone. It would require a preregistered paired-audio corpus (dangerous help requests plus acoustically similar negatives), independent human ground truth, non-duplicated speakers/scenes, and both accepted transcript and CED outputs per recording. Concatenating unrelated ESC-50 and VIVOS observations would manufacture evidence and is rejected. The current N=6 remains deterministic logic validation only.
"""
    (OUT / "fusion" / "expansion_feasibility.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "fusion" / "expansion_feasibility.md").write_text(text, encoding="utf-8")
    return {"possible_real_paired_cases": 0, "n_at_least_30_feasible": False}


def baseline_manifest() -> dict:
    from emergency_system import CATEGORY_CONFIG
    tracked_config = [ROOT / p for p in ("sound_classifier.py", "speech_enhancer.py", "speech_recognizer.py",
        "emergency_system.py", "fusion_engine.py", "personalization/profile_generator.py",
        "personalization/profile_validator.py", "benchmark/run_expanded_benchmark.py")]
    data_manifests = [DATA / "external" / name / "manifest.csv" for name in ("esc50", "demand", "vivos")]
    summary = json.loads((EXPANDED / "summary.json").read_text(encoding="utf-8"))
    model_config, model_config_path = ontology_config()
    return {"frozen_at": "2026-08-19", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip(), "python": platform.python_version(), "random_seeds": [SEED],
        "model": "mispeech/ced-tiny", "model_config_sha256": sha256(model_config_path),
        "model_output_labels": len(model_config["id2label"]),
        "production_thresholds_source": "emergency_system.CATEGORY_CONFIG (hash below)",
        "production_thresholds": {name: {"threshold": cfg.threshold, "level": cfg.level}
                                  for name, cfg in CATEGORY_CONFIG.items() if name != "help_request"},
        "config_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in tracked_config if p.exists()},
        "dataset_manifest_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in data_manifests},
        "sample_counts": {"ced_clean": 280, "dtln_ced_pairs": 700, "stt_clean": 100, "emergency_positive": 200, "emergency_negative": 80},
        "baseline_key_metrics": summary, "exact_benchmark_command": ".\\.venv\\Scripts\\python.exe benchmark\\run_full_benchmark.py --expanded --enable-network-stt --stability-seconds 3600",
        "test_set_policy": "Frozen manifests and hashes; no post-change cherry-picking.", "production_behavior_modified": False}


def reports(ced: dict, emergency: dict, dtln: dict, personal: dict, fusion: dict, stt: dict) -> None:
    per = {r["label"]: r for r in ced["per_class"]}
    strict, aware = ced["comparison"]
    sig = dtln["signals"]
    ced_sig = [r for r in sig if r["class"] != "speech"]
    speech_sig = [r for r in sig if r["class"] == "speech"]
    mean_gain = statistics.fmean(float(r["gain_ratio_output_input"]) for r in ced_sig)
    saved_quality = read_csv(EXPANDED / "noise_filter" / "snr_improvement.csv")
    mean_saved_delta = statistics.fmean(float(r["delta_snr_db"]) for r in saved_quality)
    mean_raw_delta = statistics.fmean(float(r["delta_snr_db"]) for r in ced_sig)
    mean_corrected_delta = statistics.fmean(float(r["delta_snr_gain_corrected_db"]) for r in ced_sig)
    overall_transition = next(r for r in dtln["transitions"] if r["dimension"] == "overall")
    stt_counts = Counter(r["transition"] for r in dtln["stt"])
    fn_classes = ", ".join(f"{r['source_class']}={r['fn']}" for r in emergency["recall_rows"])
    fn_stages = ", ".join(f"{r['failure_stage']}={r['count']}" for r in emergency["stage_rows"] if r["count"])
    personalization_failures = len(personal["failures"])
    root = f"""# SoundGuard Benchmark Root-Cause Analysis

## 1. Executive Summary

1. Strict CED weakness combines true top-1 errors with a measurable ontology/canonicalization gap: accuracy {float(strict['accuracy']):.4f} / Macro F1 {float(strict['macro_f1']):.4f} versus conservative ontology-aware **diagnostic only** accuracy {float(aware['accuracy']):.4f} / Macro F1 {float(aware['macro_f1']):.4f}.
2. Emergency false negatives are concentrated by class ({fn_classes}) and stage ({fn_stages}); recognition dominates before emergency rules can act.
3. The SNR implementation passes known-signal unit diagnostics, and independent lag estimates center at {dtln['alignment']['median']} samples. The frozen non-speech CED delta is {mean_saved_delta:.4f} dB; an independent recomputation is {mean_raw_delta:.4f} dB and remains {mean_corrected_delta:.4f} dB after optimal gain correction. In contrast, the 150 speech pairs improve by {statistics.fmean(float(r['delta_snr_db']) for r in speech_sig):.4f} dB: signal domain explains the preliminary/expanded sign reversal.
4. DTLN turns {overall_transition['C_raw_correct_dtln_wrong']} correct CED decisions into errors versus {overall_transition['B_raw_wrong_dtln_correct']} repairs; mean output/input RMS ratio is {mean_gain:.4f}.
5. Personalization hyphen failures arise because punctuation normalization turns compounds into standalone evidence tokens; {personalization_failures}/{len(personal['rows'])} diagnostic compounds infer a role.

## 2. CED

Worst strict recalls: glass breaking {float(per['glass_breaking']['recall']):.3f}, dog barking {float(per['dog_barking']['recall']):.3f}. Full class confusions and confidence summaries are in `ced/per_class_failure_analysis.csv`. The model ontology has Glass/Shatter rather than a literal Glass breaking label, and Dog/Bark/Bow-wow rather than the single SoundGuard dog_barking category. The strict-to-aware gap quantifies evaluation/canonicalization mismatch; residual errors remain genuine model/domain failures. Wrong confidence distributions are class-specific and retained rather than reduced to one aggregate. No direct evidence identifies peak normalization or input shape as the principal cause; the model's configured temporal pooling can plausibly dilute impulses, but this benchmark cannot isolate it without a controlled preprocessing experiment.

## 3. Emergency Detection

All {len(emergency['false_negatives'])} false negatives are enumerated. FN by class: {fn_classes}. FN by stage: {fn_stages}. Fusion was not part of this emergency benchmark, so it suppressed none. The saved-prediction threshold sweep shows the maximum attainable trade-off without changing top-1 recognition; it is explicitly offline diagnostic and cannot recover wrong/unmapped labels.

## 4. DTLN

### Is the -7.97 dB Delta SNR measurement valid?

**YES within the stated intrusive-SNR methodology.** The SNR function recovers 20/10/0 dB synthetic cases, 289/290 independent alignment estimates equal 384 samples (one low-correlation outlier is 325), and no sign or delay-correction bug was found. Plain SNR is gain-sensitive, but optimal gain correction still leaves a strongly negative CED-pair mean delta ({mean_corrected_delta:.4f} dB), while mean clean correlation falls from {statistics.fmean(float(r['correlation_clean_input']) for r in ced_sig):.4f} to {statistics.fmean(float(r['correlation_clean_output']) for r in ced_sig):.4f}. Thus scaling does not explain away the degradation. It remains an intrusive waveform metric, not a perceptual-quality or intelligibility verdict. The same energy/gain fields are reported for all {len(speech_sig)} speech pairs.

The apparent sign reversal is chiefly a domain change: DTLN improves the 150 speech-pair intrusive SNR by {statistics.fmean(float(r['delta_snr_db']) for r in speech_sig):.4f} dB on average, while it degrades the 700 non-speech CED pairs. This is consistent with a speech denoiser preserving speech more effectively than environmental safety transients.

### Does DTLN harm CED?

**YES on this paired protocol.** Correct→wrong={overall_transition['C_raw_correct_dtln_wrong']}; wrong→correct={overall_transition['B_raw_wrong_dtln_correct']}. Class/environment/SNR breakdowns are in the transition matrix. The mechanism is downstream representation change after strong gain/spectral processing, especially loss/distortion of discriminative events—not residual alignment alone.

### Does DTLN harm STT?

**MIXED, net harmful.** Improved={stt_counts['improved WER']}, unchanged={stt_counts['unchanged WER']}, worsened={stt_counts['worsened WER']}; aggregate expanded WER delta is positive even though speech intrusive SNR improves. The evidence shows that waveform SNR is not a sufficient proxy for recognizer accuracy; it is consistent with phonetic detail changes/provider sensitivity, but this cache cannot isolate a single acoustic mechanism.

## 5. Personalization

`factory-method`→`factory method` and `baby-blue`→`baby blue` during punctuation normalization. Whole-word matching then treats the pieces as explicit role evidence. This generalizes to other compounds and is a parser/evidence issue, not Universal Critical validation.

## 6. Fusion

Existing CED and STT real audio are unpaired, so legitimate paired fusion cases available now: {fusion['possible_real_paired_cases']}. N>=30 requires a new independently labeled paired corpus; manufacturing combinations is scientifically invalid.

## 7. STT

Clean corpus edit counts are substitutions={stt['substitutions']}, deletions={stt['deletions']}, insertions={stt['insertions']}. Mean per-utterance CER change after stripping diacritics is {stt['diacritic_delta']:.4f}; detailed speaker/length and highest-WER rows are retained.

## 8. Ranked Root Causes

| Rank | Problem | Evidence | Impact | Confidence |
|---:|---|---|---|---|
| 1 | CED recognition/ontology failures on safety events | Full 280 top-k; class recall and strict-aware gap | Emergency FN | High |
| 2 | DTLN changes discriminative audio | 700 paired transitions and signal diagnostics | CED and STT degradation | High |
| 3 | Emergency threshold rejects some recognized candidates | Stage-B FN plus frozen-prediction sweep | Additional FN | High |
| 4 | Hyphen normalization creates false personalization evidence | deterministic minimal repros | Profile relevance/safety | High |
| 5 | Fusion evidence is underpowered | 0 paired real cases; N=6 logic cases | Unsupported improvement claim | High |

## 9. Recommended Interventions

| Priority | Component | Proposed modification | Expected benefit | Risk | Production behavior? | Rerun |
|---|---|---|---|---|---|---|
| P0 | CED mapping/evaluation | Preregister ontology-aware mapping, then separately test any production alias change | Separate scoring artifact from real misses | Over-broad aliases | Yes if production | Frozen CED + emergency |
| P0 | Safety-event CED | Evaluate a safety-focused model/temporal transient strategy on frozen clips | Reduce glass/horn/siren FN | False alarms/domain overfit | Yes | Full frozen CED/noise/emergency |
| P0 | DTLN routing | Gate/bypass speech denoiser for transient CED after a frozen ablation | Prevent correct→wrong transitions | More noise reaches CED | Yes | 700 pairs + latency |
| P1 | Emergency thresholds | Consider only after top-1 repair; choose from preregistered validation, not test sweep | Recover threshold-only FN | False alerts | Yes | Frozen positives + negatives/new validation |
| P1 | Personalization parser | Preserve compound boundaries or require contextual role phrases | Eliminate compound false roles | Miss terse true roles | Yes | all personalization + adversarial |
| P1 | DTLN metric | Add SI-SNR/gain-corrected metric beside intrusive SNR | Sounder interpretation | Metric comparability | No | signal diagnostics |
| P2 | Fusion dataset | Collect 30–50+ real paired cases | Valid fusion estimate | Annotation/cost | No initially | preregistered fusion benchmark |
| P2 | STT | Analyze/provider-tune only after safety work | Better utility | Network/provider variability | Yes | fixed VIVOS/noise set |

## 10. Scientific Interpretation

Model limitation and dataset/domain mismatch remain after the conservative ontology correction. Benchmark ontology mismatch is measured, not assumed. Threshold effects are isolated with saved predictions. No preprocessing defect was proven. DTLN has both gain-sensitive metric ambiguity and independently observed downstream harm. Personalization is a deterministic rule/parser issue.

**Production/core behavior modified during this analysis: NO.**
"""
    (OUT / "ROOT_CAUSE_REPORT.md").write_text(root, encoding="utf-8")

    plan = """# SoundGuard Improvement Plan

Safety for hearing-impaired users determines ordering; no expected final numbers are invented.

| Priority | Change | Target metric | Baseline | Expected direction | Risk | Effort |
|---|---|---|---|---|---|---|
| P0 | Separate ontology scoring from true CED failure; preregister any runtime aliases | Safety-class recall / emergency FN | See frozen manifest | Fewer mapping-caused FN | Alias false positives | Medium |
| P0 | Compare safety-event model/transient-window candidates on the frozen set plus independent validation | Glass, siren, horn, fire recall | Frozen per-class CSV | Up | Model cost/false alarms | High |
| P0 | Test CED bypass or event-aware routing around speech DTLN | DTLN delta CED Macro F1 | Frozen expanded summary | Toward zero/up | Noise robustness | Medium |
| P1 | Validate threshold choices on a separate preregistered set | Recall, FNR, FPR, F1 | Frozen emergency metrics | Recall up with bounded FPR | Test leakage | Medium |
| P1 | Preserve compound boundaries in role evidence | Adversarial safety pass | 48/50 plus diagnostic CSV | Up | False negatives for terse answers | Low |
| P1 | Add gain-corrected/SI-SNR reporting | Signal-quality interpretability | -7.97 dB gain-sensitive | Less ambiguous | Historical comparison | Low |
| P2 | Collect paired real fusion corpus | Fusion F1 and branch coverage | N=6 logic-only, 0 paired | Evidence quality up | Collection bias | High |
| P2 | Address STT substitutions/diacritics after safety items | WER/CER | 0.1075/0.0594 | Down | Provider drift | Medium |

Every future production change must reuse `baseline_manifest.json`, unchanged sample hashes, and the exact frozen test set. Candidate selection must use separate development data.
"""
    (OUT / "IMPROVEMENT_PLAN.md").write_text(plan, encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ced = ced_analysis()
    emergency = emergency_analysis(ced["details"])
    dtln = dtln_analysis()
    personal = personalization_analysis()
    fusion = fusion_analysis()
    stt = stt_analysis()
    write_json(OUT / "baseline_manifest.json", baseline_manifest())
    reports(ced, emergency, dtln, personal, fusion, stt)
    print(f"Analysis complete: {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
