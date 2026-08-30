"""Evaluate frozen SoundGuard improvement candidates without production changes."""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark.metrics import binary_metrics
from benchmark.run_expanded_benchmark import aggregate_errors, target_classification_metrics
from benchmark.experiments.semantic_mapping import map_candidate
from benchmark.experiments.temporal_pooling import generate_predictions


ROOT = Path(__file__).resolve().parents[2]
EXPANDED = ROOT / "benchmark_results" / "expanded"
ANALYSIS = ROOT / "benchmark_results" / "analysis"
OUT = ROOT / "benchmark_results" / "improvement"
TARGETS = {"baby_crying", "dog_barking", "door_activity", "fire", "glass_breaking", "siren", "vehicle_horn"}
SAFETY = {"baby_crying", "fire", "glass_breaking", "siren", "vehicle_horn"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if rows:
            writer.writeheader(); writer.writerows(rows)


def canonical(raw: str) -> str:
    from personalization.sound_labels import canonicalize_label
    return canonicalize_label(raw) or "other"


def candidate_category(raw: str) -> tuple[str, dict | None]:
    baseline = canonical(raw)
    value, rule = map_candidate(raw, None if baseline == "other" else baseline)
    return value or "other", rule


def score(name: str, rows: list[dict], semantic: bool) -> dict:
    from emergency_system import EmergencySystem
    truths, predictions, emergency_truth, emergency_predictions = [], [], [], []
    normalized_rows = []
    for row in rows:
        raw = row["raw_label"]
        predicted, rule = candidate_category(raw) if semantic else (canonical(raw), None)
        truths.append(row["true_label"]); predictions.append(predicted)
        emergency_truth.append(row["true_label"] in SAFETY)
        label_for_alert = predicted if semantic and rule else raw
        result = EmergencySystem(decision_mode="single_shot").process_sound_event(label_for_alert, float(row["confidence"]))
        emergency_predictions.append(result["alert_level"] in {"MEDIUM", "HIGH", "CRITICAL"} and
                                     result["event_state"] in {"EVENT_STARTED", "EVENT_CONTINUING"})
        normalized_rows.append({**row, "predicted": predicted, "mapping_rule": rule["raw_label"] if rule else ""})
    metrics = target_classification_metrics(truths, predictions)
    per_class = {row["label"]: row for row in metrics["per_class"]}
    emergency = binary_metrics(emergency_truth, emergency_predictions)
    return {"candidate": name, "sample_count": len(rows), "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "safety_recall": statistics.fmean(per_class[label]["recall"] for label in SAFETY),
            "glass_recall": per_class["glass_breaking"]["recall"],
            "vehicle_horn_recall": per_class["vehicle_horn"]["recall"],
            "fire_recall": per_class["fire"]["recall"], "siren_recall": per_class["siren"]["recall"],
            "baby_crying_recall": per_class["baby_crying"]["recall"],
            "emergency_recall": emergency["recall"], "emergency_fnr": emergency["false_negative_rate"],
            "emergency_precision": emergency["precision"], "emergency_fp": emergency["fp"],
            "per_class": list(per_class.values()), "rows": normalized_rows,
            "emergency_predictions": emergency_predictions}


def semantic_experiment(clean: list[dict]) -> tuple[dict, dict]:
    baseline = score("Baseline", clean, False)
    candidate = score("Semantic mapping", clean, True)
    result_rows = []
    for result in (baseline, candidate):
        result_rows.append({"row_type": "summary", **{k: v for k, v in result.items()
                           if k not in {"per_class", "rows", "emergency_predictions"}}})
        result_rows.extend({"row_type": "per_class", "candidate": result["candidate"], **row}
                           for row in result["per_class"])
    write_csv(OUT / "semantic_mapper_results.csv", result_rows)
    transitions = []
    for before, after, before_em, after_em in zip(baseline["rows"], candidate["rows"],
                                                  baseline["emergency_predictions"], candidate["emergency_predictions"]):
        before_correct = before["predicted"] == before["true_label"]
        after_correct = after["predicted"] == after["true_label"]
        if before_correct != after_correct or before_em != after_em:
            transitions.append({"sample_id": before["sample_id"], "true_label": before["true_label"],
                "raw_label": before["raw_label"], "confidence": before["confidence"],
                "baseline_category": before["predicted"], "candidate_category": after["predicted"],
                "alias": after["mapping_rule"], "newly_recovered_tp": not before_correct and after_correct,
                "newly_created_classification_fp": before_correct and not after_correct,
                "baseline_emergency": before_em, "candidate_emergency": after_em,
                "new_emergency_tp": (before["true_label"] in SAFETY) and not before_em and after_em,
                "new_emergency_fp": (before["true_label"] not in SAFETY) and not before_em and after_em})
    write_csv(OUT / "semantic_mapper_transitions.csv", transitions)
    recovered = Counter(r["alias"] for r in transitions if r["newly_recovered_tp"])
    created = Counter(r["alias"] for r in transitions if r["newly_created_classification_fp"])
    report = f"""# Semantic mapper experiment

Frozen ESC-50 is now a development/improvement set, not an untouched holdout. The taxonomy was defined from ontology semantics before scoring transitions. Broad labels (`Animal`, `Vehicle`, generic `Breaking`, `Crying, sobbing`) are explicitly rejected.

| Metric | Baseline | Candidate |
|---|---:|---:|
| CED Macro F1 | {baseline['macro_f1']:.4f} | {candidate['macro_f1']:.4f} |
| Emergency recall | {baseline['emergency_recall']:.4f} | {candidate['emergency_recall']:.4f} |
| Emergency FNR | {baseline['emergency_fnr']:.4f} | {candidate['emergency_fnr']:.4f} |
| Emergency precision | {baseline['emergency_precision']:.4f} | {candidate['emergency_precision']:.4f} |
| Emergency false positives | {baseline['emergency_fp']} | {candidate['emergency_fp']} |

Recovered strict-category TPs by alias: `{dict(recovered)}`. Newly created classification errors by alias: `{dict(created)}`. Every transition is retained in `semantic_mapper_transitions.csv`; no observed transition is hidden. The experiment cannot estimate false-positive behavior outside these seven source classes, so semantic validity remains a required gate even when observed FP is zero.
"""
    (OUT / "semantic_mapper_report.md").write_text(report, encoding="utf-8")
    return baseline, candidate


def pooling_experiment(clean: list[dict], baseline: dict, semantic: dict) -> tuple[dict, dict, list[dict]]:
    temporal = generate_predictions()
    candidates = [baseline, semantic]
    for method in sorted({row["method"] for row in temporal}):
        rows = [row for row in temporal if row["method"] == method]
        candidates.append(score(f"Pooling: {method}", rows, False))
        candidates.append(score(f"Combined: {method}", rows, True))
    pooling_only = [r for r in candidates if r["candidate"].startswith("Pooling:")]
    combined = [r for r in candidates if r["candidate"].startswith("Combined:")]
    rank = lambda r: (r["emergency_recall"], r["safety_recall"], -r["emergency_fp"], r["macro_f1"])
    best_pooling, best_combined = max(pooling_only, key=rank), max(combined, key=rank)
    rows = [{k: v for k, v in result.items() if k not in {"per_class", "rows", "emergency_predictions"}}
            for result in candidates]
    write_csv(OUT / "temporal_pooling_comparison.csv", rows)
    report = f"""# Temporal pooling experiment

Controlled alternatives use fixed non-overlapping 1-second windows and only three preregistered aggregations: max, top-2 mean, and 90th percentile. The current whole-clip output remains the baseline. No production pooling was changed.

Best pooling-only candidate by the safety-first gate: **{best_pooling['candidate']}**, Macro F1 {best_pooling['macro_f1']:.4f}, safety recall {best_pooling['safety_recall']:.4f}, emergency recall {best_pooling['emergency_recall']:.4f}, FP {best_pooling['emergency_fp']}.

Best combined candidate: **{best_combined['candidate']}**, Macro F1 {best_combined['macro_f1']:.4f}, safety recall {best_combined['safety_recall']:.4f}, emergency recall {best_combined['emergency_recall']:.4f}, FP {best_combined['emergency_fp']}.

This development set has already been inspected. Results support candidate selection only; they are not final holdout evidence.
"""
    (OUT / "temporal_pooling_report.md").write_text(report, encoding="utf-8")
    return best_pooling, best_combined, candidates


def dtln_routing_experiment() -> list[dict]:
    pairs = read_csv(EXPANDED / "noise_filter" / "ced_predictions_paired.csv")
    stt = read_csv(EXPANDED / "stt" / "noise_results.csv")
    real_raw = [r for r in stt if r["protocol"] == "real" and r["filter"] == "unfiltered"]
    real_dtln = [r for r in stt if r["protocol"] == "real" and r["filter"] == "dtln"]
    def ced(column: str) -> tuple[float, float]:
        metric = target_classification_metrics([r["true_label"] for r in pairs], [r[column] for r in pairs])
        per = {r["label"]: r for r in metric["per_class"]}
        return metric["macro_f1"], statistics.fmean(per[label]["recall"] for label in SAFETY)
    raw_ced, raw_safety = ced("without_filter"); filtered_ced, filtered_safety = ced("with_filter")
    raw_stt, filtered_stt = aggregate_errors(real_raw), aggregate_errors(real_dtln)
    signal = json.loads((EXPANDED / "noise_filter" / "signal_summary.json").read_text(encoding="utf-8"))
    rows = [
        {"architecture": "A: DTLN -> CED; DTLN -> STT", "ced_route": "DTLN", "stt_route": "DTLN",
         "ced_macro_f1": filtered_ced, "ced_safety_recall": filtered_safety, "stt_wer": filtered_stt["wer"], "stt_cer": filtered_stt["cer"],
         "standalone_dtln_mean_delta_snr_db": signal["mean_delta_snr_db"]},
        {"architecture": "B: raw -> CED; DTLN -> STT", "ced_route": "raw", "stt_route": "DTLN",
         "ced_macro_f1": raw_ced, "ced_safety_recall": raw_safety, "stt_wer": filtered_stt["wer"], "stt_cer": filtered_stt["cer"],
         "standalone_dtln_mean_delta_snr_db": signal["mean_delta_snr_db"]},
        {"architecture": "C: raw -> CED; raw -> STT; DTLN standalone", "ced_route": "raw", "stt_route": "raw",
         "ced_macro_f1": raw_ced, "ced_safety_recall": raw_safety, "stt_wer": raw_stt["wer"], "stt_cer": raw_stt["cer"],
         "standalone_dtln_mean_delta_snr_db": signal["mean_delta_snr_db"]},
    ]
    write_csv(OUT / "dtln_routing_comparison.csv", rows)
    return rows


def _safe_normalize(text: str) -> str:
    value = str(text).lower().replace("â€™", "'")
    value = re.sub(r"(?<=[a-z0-9])[-‐‑–—](?=[a-z0-9])", "compoundjoiner", value)
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@contextmanager
def safe_compounds():
    with patch("personalization.profile_generator._normalize_evidence_text", _safe_normalize):
        yield


def personalization_experiment() -> dict:
    from personalization.profile_generator import generate_profile, generate_rule_based
    from personalization.profile_validator import UNIVERSAL_MIN_PRIORITY, validate_profile
    from personalization.role_knowledge import DEFAULT_PRIORITIES
    from personalization.schemas import ProfileValidationError
    cases = json.loads((ROOT / "benchmark_data" / "expanded_personalization_cases.json").read_text(encoding="utf-8"))
    adversarial = read_csv(ANALYSIS / "personalization" / "adversarial_diagnostic.csv")
    with safe_compounds():
        assertion_results = []
        profiles = []
        for case in cases:
            profile = generate_rule_based([case["text"]]); profiles.append(profile)
            assertion_results.extend(profile["priority_profile"][label] == expected
                                     for label, expected in case["priority_assertions"].items())
        safety = [all(profile["priority_profile"][label] >= minimum for label, minimum in UNIVERSAL_MIN_PRIORITY.items())
                  for profile in profiles]
        negations = ["I do not drive.", "I never use a bus.", "I have no construction work.", "I live without a baby.",
            "I am not a student.", "I do not work at a factory.", "I never walk to work.", "I am not pregnant.",
            "I do not work in healthcare.", "I live without an older adult."]
        safety.extend(not generate_rule_based([text])["roles"] for text in negations)
        contrasts = [("I do not drive, but I walk to work on foot.", "pedestrian_commuter"),
            ("I never use a bus; however I ride a motorbike.", "motorcyclist_cyclist"),
            ("I am not a student, but I work in a hospital as a nurse.", "healthcare_care_worker"),
            ("I do not work in construction; however I work in a factory.", "factory_warehouse_worker"),
            ("I am not a driver, but I am a cyclist in road traffic.", "motorcyclist_cyclist")]
        safety.extend(expected in generate_rule_based([text])["roles"] for text, expected in contrasts)
        substrings = ["I paint driveway artwork.", "The studentized statistic is useful.", "This is a homeopathic note.",
                      "A factory-method pattern is software.", "The baby-blue color is bright."]
        safety.extend(not generate_rule_based([text])["roles"] for text in substrings)
        malformed = [
            {**generate_rule_based(["I drive."]), "priority_profile": {**DEFAULT_PRIORITIES, "speech": 9}},
            {k:v for k,v in generate_rule_based(["I drive."]).items() if k != "contexts"},
            {**generate_rule_based(["I drive."]), "priority_profile": {**DEFAULT_PRIORITIES, "made_up": 3}},
            {**generate_rule_based(["I drive."]), "profile_version": "999"},
            {**generate_rule_based(["I drive."]), "roles": ["unknown_role"]},
        ]
        for payload in malformed:
            try: validate_profile(payload); rejected = False
            except ProfileValidationError: rejected = True
            safety.append(rejected)
        saved = {name: os.environ.get(name) for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "SOUNDGUARD_AI_PROVIDER")}
        os.environ.pop("OPENAI_API_KEY", None); os.environ.pop("OPENROUTER_API_KEY", None); os.environ["SOUNDGUARD_AI_PROVIDER"] = "openai"
        try:
            safety.extend(generate_profile([text])[1].startswith("deterministic_fallback:") for text in
                          ("I drive.", "I am a student.", "I care for a baby.", "I work in construction.", "I walk on foot."))
        finally:
            for name, value in saved.items():
                if value is None: os.environ.pop(name, None)
                else: os.environ[name] = value
        adversarial_pass = [not generate_rule_based([row["text"]])["roles"] for row in adversarial]
    result = {"baseline_priority_accuracy": 1.0, "candidate_priority_accuracy": statistics.fmean(assertion_results),
              "baseline_universal_critical": 1.0, "candidate_universal_critical": statistics.fmean(safety[:20]),
              "baseline_fixed_safety": .96, "candidate_fixed_safety": statistics.fmean(safety),
              "fixed_safety_passed": sum(safety), "fixed_safety_count": len(safety),
              "adversarial_passed": sum(adversarial_pass), "adversarial_count": len(adversarial_pass),
              "compound_failures_remaining": len(adversarial_pass)-sum(adversarial_pass)}
    (OUT / "personalization_compound_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean_details = read_csv(ANALYSIS / "ced" / "clean_top5_details.csv")
    clean = [{"sample_id": row["sample_id"], "true_label": row["true_label"],
              "raw_label": row["raw_predicted_label"], "confidence": row["confidence"]} for row in clean_details]
    baseline, semantic = semantic_experiment(clean)
    pooling, combined, candidates = pooling_experiment(clean, baseline, semantic)
    routing = dtln_routing_experiment()
    personal = personalization_experiment()
    summary = {"baseline": {k:v for k,v in baseline.items() if k not in {"per_class","rows","emergency_predictions"}},
               "semantic": {k:v for k,v in semantic.items() if k not in {"per_class","rows","emergency_predictions"}},
               "best_pooling": {k:v for k,v in pooling.items() if k not in {"per_class","rows","emergency_predictions"}},
               "best_combined": {k:v for k,v in combined.items() if k not in {"per_class","rows","emergency_predictions"}},
               "routing": routing, "personalization": personal}
    (OUT / "candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
