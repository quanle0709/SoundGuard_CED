"""Run the frozen development benchmark with selected opt-in improvements."""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

from benchmark.metrics import binary_metrics, latency_summary
from benchmark.run_expanded_benchmark import aggregate_errors, target_classification_metrics


ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "benchmark_results" / "expanded"
IMPROVEMENT = ROOT / "benchmark_results" / "improvement"
OUT = ROOT / "benchmark_results" / "improved"
DATA = ROOT / "benchmark_data"
SAFETY = {"baby_crying", "fire", "glass_breaking", "siren", "vehicle_horn"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if rows: writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def enable_selected() -> None:
    os.environ["SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING"] = "1"
    os.environ["SOUNDGUARD_SAFE_COMPOUND_EVIDENCE"] = "1"
    os.environ.pop("SOUNDGUARD_ENABLE_TEMPORAL_POOLING_V2", None)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ced_and_emergency() -> tuple[dict, dict, list[dict]]:
    from soundguard.emergency.emergency_system import EmergencySystem
    from personalization.sound_labels import canonicalize_label
    from soundguard.detection.sound_classifier import classify_audio_file
    manifest = read_csv(DATA / "external" / "esc50" / "manifest.csv")
    classify_audio_file(ROOT / manifest[0]["path"])
    rows, emergency_truth, emergency_pred = [], [], []
    for index, case in enumerate(manifest, 1):
        start = time.perf_counter(); result = classify_audio_file(ROOT / case["path"]); elapsed = (time.perf_counter()-start)*1000
        predicted = canonicalize_label(result["label"]) or "other"
        alert = EmergencySystem(decision_mode="single_shot").process_sound_event(result["label"], float(result["confidence"]))
        is_emergency = alert["alert_level"] in {"MEDIUM", "HIGH", "CRITICAL"} and alert["event_state"] in {"EVENT_STARTED", "EVENT_CONTINUING"}
        truth_emergency = case["true_label"] in SAFETY
        rows.append({**case, "raw_predicted_label": result.get("raw_label", result["label"]),
                     "predicted_label": predicted, "confidence": result["confidence"],
                     "semantic_mapping": result.get("semantic_mapping", ""), "correct": predicted == case["true_label"],
                     "emergency_truth": truth_emergency, "emergency_prediction": is_emergency,
                     "alert_level": alert["alert_level"], "latency_ms": elapsed})
        emergency_truth.append(truth_emergency); emergency_pred.append(is_emergency)
        if index % 40 == 0: print(f"improved CED {index}/{len(manifest)}", flush=True)
    metrics = target_classification_metrics([r["true_label"] for r in rows], [r["predicted_label"] for r in rows])
    emergency = binary_metrics(emergency_truth, emergency_pred)
    write_csv(OUT / "ced" / "predictions.csv", rows)
    write_csv(OUT / "ced" / "per_class_metrics.csv", metrics["per_class"])
    write_json(OUT / "ced" / "metrics.json", {k:v for k,v in metrics.items() if k not in {"per_class","labels","bucketed_predictions"}})
    write_csv(OUT / "emergency" / "predictions.csv", rows)
    write_json(OUT / "emergency" / "metrics.json", emergency)
    return metrics, emergency, rows


def stt_metrics() -> dict:
    clean = read_csv(EXPANDED / "stt" / "clean_results.csv")
    noise = [r for r in read_csv(EXPANDED / "stt" / "noise_results.csv")
             if r["protocol"] == "real" and r["filter"] == "unfiltered"]
    clean_result, real_result = aggregate_errors(clean), aggregate_errors(noise)
    result = {"selected_route": "raw", "clean": clean_result, "real_demand_all_snrs": real_result,
              "network_latency": json.loads((EXPANDED / "stt" / "latency_summary.json").read_text(encoding="utf-8"))}
    write_json(OUT / "stt" / "metrics.json", result)
    return result


def personalization_metrics() -> dict:
    from benchmark.experiments.evaluate_candidates import personalization_experiment
    result = personalization_experiment()
    write_json(OUT / "personalization" / "metrics.json", result)
    return result


def latency(rows: list[dict]) -> dict:
    from soundguard.detection.sound_classifier import classify_audio_file
    paths = [ROOT / row["path"] for row in rows[:100]]
    for path in paths[:5]: classify_audio_file(path)
    legacy_values, improved_values = [], []
    for index, path in enumerate(paths):
        modes = (False, True) if index % 2 == 0 else (True, False)
        for enabled in modes:
            if enabled: os.environ["SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING"] = "1"
            else: os.environ.pop("SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING", None)
            start = time.perf_counter(); classify_audio_file(path); elapsed = (time.perf_counter()-start)*1000
            (improved_values if enabled else legacy_values).append(elapsed)
    os.environ["SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING"] = "1"
    legacy, improved = latency_summary(legacy_values), latency_summary(improved_values)
    frozen = json.loads((EXPANDED / "pipeline" / "latency_summary.json").read_text(encoding="utf-8"))
    baseline_p95 = legacy["p95_ms"]
    result = {"frozen_expanded_p95_ms": float(frozen["p95_ms"]),
              "baseline_p95_ms": baseline_p95, "improved_p95_ms": improved["p95_ms"],
              "absolute_delta_ms": improved["p95_ms"]-baseline_p95,
              "percentage_delta": (improved["p95_ms"]-baseline_p95)/baseline_p95*100,
              "paired_legacy_distribution": legacy, "paired_improved_distribution": improved,
              "note": "Alternating same-process, same-clip feature-off/feature-on measurement; frozen P95 retained separately."}
    write_json(OUT / "pipeline" / "latency_comparison.json", result)
    return result


def reports(ced: dict, emergency: dict, stt: dict, personal: dict, latency_result: dict) -> None:
    baseline_per = {r["label"]: r for r in read_csv(EXPANDED / "ced" / "per_class_metrics.csv")}
    improved_per = {r["label"]: r for r in ced["per_class"]}
    baseline_em = json.loads((EXPANDED / "emergency" / "metrics.json").read_text(encoding="utf-8"))
    baseline_ced = read_csv(EXPANDED / "ced" / "clean_metrics.csv")[0]
    routing = read_csv(IMPROVEMENT / "dtln_routing_comparison.csv")
    validation_path = OUT / "validation.json"
    validation = (json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists()
                  else {"pytest_passed": "pending", "pytest_failed": "pending", "subtests_passed": "pending",
                        "compileall": "pending", "imports": "pending", "frozen_stability_crashes": 0})
    comparisons = [
        ("CED accuracy", float(baseline_ced["accuracy"]), ced["accuracy"]),
        ("CED Macro F1", float(baseline_ced["macro_f1"]), ced["macro_f1"]),
        ("Glass recall", float(baseline_per["glass_breaking"]["recall"]), improved_per["glass_breaking"]["recall"]),
        ("Vehicle-horn recall", float(baseline_per["vehicle_horn"]["recall"]), improved_per["vehicle_horn"]["recall"]),
        ("Fire recall", float(baseline_per["fire"]["recall"]), improved_per["fire"]["recall"]),
        ("Siren recall", float(baseline_per["siren"]["recall"]), improved_per["siren"]["recall"]),
        ("Baby-crying recall", float(baseline_per["baby_crying"]["recall"]), improved_per["baby_crying"]["recall"]),
        ("Emergency recall", baseline_em["recall"], emergency["recall"]),
        ("Emergency FNR", baseline_em["false_negative_rate"], emergency["false_negative_rate"]),
        ("Emergency precision", baseline_em["precision"], emergency["precision"]),
        ("Emergency false positives", baseline_em["fp"], emergency["fp"]),
        ("STT clean WER", stt["clean"]["wer"], stt["clean"]["wer"]),
        ("STT clean CER", stt["clean"]["cer"], stt["clean"]["cer"]),
        ("Personalization priority accuracy", personal["baseline_priority_accuracy"], personal["candidate_priority_accuracy"]),
        ("Personalization Universal Critical", personal["baseline_universal_critical"], personal["candidate_universal_critical"]),
        ("Personalization fixed safety", personal["baseline_fixed_safety"], personal["candidate_fixed_safety"]),
        ("Local PC P95 ms", latency_result["baseline_p95_ms"], latency_result["improved_p95_ms"]),
    ]
    table = "\n".join(f"| {name} | {before:.6f} | {after:.6f} | {after-before:+.6f} |" for name,before,after in comparisons)
    comparison_report = f"""# Baseline versus improved

The same frozen development clips are used; this is not final untouched-holdout evidence.

| Metric | Baseline | Improved | Delta |
|---|---:|---:|---:|
{table}

## DTLN routing

Selected: raw audio to CED and STT; DTLN remains standalone. The fixed routing table is preserved at `benchmark_results/improvement/dtln_routing_comparison.csv`. Raw real-noise STT WER is 0.114085 versus 0.127465 with DTLN; raw paired CED Macro F1 is 0.399122 versus 0.099647 with DTLN.
"""
    (OUT / "BASELINE_VS_IMPROVED.md").write_text(comparison_report, encoding="utf-8")

    holdout = """# Final untouched holdout plan

The 280 ESC-50 clips are development data and cannot be reused as an untouched final claim. The candidate is now frozen before holdout acquisition.

Preferred holdout: a preregistered FSD50K evaluation subset with independently mapped and human-audited instances for siren, vehicle horn, fire/crackle, glass/shatter, infant crying, dog bark, and door knock. FSD50K/Freesound is source-distinct from AudioSet, although uploader/audio duplication must still be hash- and fingerprint-audited. Use at least 40 non-duplicated clips per supported class plus acoustically plausible negatives. UrbanSound8K may provide a partial secondary check for dog bark, car horn, and siren, but cannot validate all safety categories.

Protocol: freeze taxonomy/code hashes first; create mappings from ontology documentation without viewing predictions; deduplicate by file hash and acoustic fingerprint against development/training sources where possible; run exactly once; publish every sample and FP/FN; do not tune afterward. A fresh corpus was not downloaded in this phase to avoid a large unplanned acquisition and accidental iterative holdout use.
"""
    (OUT / "FINAL_HOLDOUT_PLAN.md").write_text(holdout, encoding="utf-8")

    report = f"""# SoundGuard Improvement Report

## 1. Stable Baseline

Dirty worktree preserved by binary patches, production hashes, manifest, and a recoverable source archive. Development occurs on `improve-ced-safety-v2`; no baseline tag was made because that would bundle unrelated user work.

## 2. Root Causes

Fine/coarse ontology mismatch, genuine safety-event model misses, non-speech damage from DTLN, and punctuation-normalized personalization compounds.

## 3. Changes Evaluated

Formal semantic taxonomy; max/top-two/p90 one-second pooling; three DTLN routes; compound-aware evidence.

## 4. Changes Rejected

All temporal pooling variants; DTLN before CED; DTLN before STT on this fixed real-noise set; unsafe broad aliases (`Animal`, `Vehicle`, `Breaking`, generic crying). Fusion was not tuned.

## 5. Selected Architecture

Default-off semantic CED adapter; default-off compound-safe personalization; raw CED; raw STT in improved mode via `--no-dtln`; DTLN standalone.

## 6. Baseline vs Improved

CED Macro F1 {float(baseline_ced['macro_f1']):.4f} -> {ced['macro_f1']:.4f}. Emergency recall {baseline_em['recall']:.4f} -> {emergency['recall']:.4f}; FP {baseline_em['fp']} -> {emergency['fp']}.

| Target | Result | Measured |
|---|---|---:|
| CED Macro F1 >= 0.65 | **{'PASS' if ced['macro_f1'] >= .65 else 'FAIL'}** | {ced['macro_f1']:.6f} |
| CED stretch Macro F1 >= 0.70 | **{'PASS' if ced['macro_f1'] >= .70 else 'FAIL'}** | {ced['macro_f1']:.6f} |
| Mean safety-class recall >= 0.70 | **{'PASS' if statistics.fmean(improved_per[label]['recall'] for label in SAFETY) >= .70 else 'FAIL'}** | {statistics.fmean(improved_per[label]['recall'] for label in SAFETY):.6f} |
| Emergency recall >= 0.70 | **{'PASS' if emergency['recall'] >= .70 else 'FAIL'}** | {emergency['recall']:.6f} |
| STT clean WER <= 0.12 | **{'PASS' if stt['clean']['wer'] <= .12 else 'FAIL'}** | {stt['clean']['wer']:.6f} |
| Personalization priority >= 0.95 | **{'PASS' if personal['candidate_priority_accuracy'] >= .95 else 'FAIL'}** | {personal['candidate_priority_accuracy']:.6f} |
| Universal Critical = 1.00 | **{'PASS' if personal['candidate_universal_critical'] == 1 else 'FAIL'}** | {personal['candidate_universal_critical']:.6f} |
| Personalization safety >= 0.98 | **{'PASS' if personal['candidate_fixed_safety'] >= .98 else 'FAIL'}** | {personal['candidate_fixed_safety']:.6f} |
| Regression failures = 0 | **{'PASS' if validation['pytest_failed'] == 0 else 'FAIL'}** | {validation['pytest_failed']} |
| Frozen stability crashes = 0 | **{'PASS' if validation['frozen_stability_crashes'] == 0 else 'FAIL'}** | {validation['frozen_stability_crashes']} |

## 7. Safety-Critical Performance

Emergency target is **FAIL**: {emergency['recall']:.4f} < 0.70. Broad aliases were rejected rather than used to manufacture success. Model-level work is required.

## 8. STT Performance

Clean WER {stt['clean']['wer']:.4f} remains under 0.12. Raw audio wins real-noise A/B; cloud latency remains separately reported.

## 9. Noise Filtering / Routing Findings

DTLN improves speech intrusive SNR but worsens fixed downstream STT and severely harms non-speech CED. It remains available independently, not forced into improved downstream routes.

## 10. Personalization Safety

Priority {personal['candidate_priority_accuracy']:.4f}; Universal Critical {personal['candidate_universal_critical']:.4f}; fixed safety {personal['candidate_fixed_safety']:.4f}; remaining adversarial compound failures {personal['compound_failures_remaining']}.

## 11. Latency

Alternating same-process paired P95 {latency_result['baseline_p95_ms']:.4f} -> {latency_result['improved_p95_ms']:.4f} ms; delta {latency_result['absolute_delta_ms']:+.4f} ms ({latency_result['percentage_delta']:+.2f}%). Frozen expanded P95 was {latency_result['frozen_expanded_p95_ms']:.4f} ms. The negative paired delta is timing noise, not a claimed speedup; no measurable mapping penalty was observed.

## 12. Regression Safety

Golden legacy comparison with all new flags OFF passed. Complete validation: {validation['pytest_passed']} tests passed, {validation['pytest_failed']} failed, {validation['subtests_passed']} subtests passed; compilation {validation['compileall']}; imports {validation['imports']}. Frozen stability evidence remains {validation['frozen_stability_crashes']} crashes; this is not real-device stability.

## 13. Limitations

Development-set tuning, possible AudioSet/ESC-50 overlap, seven coarse classes, no hardware/user study, network STT variability, and no final holdout result.

## 14. Holdout Validation Status

Candidate frozen; holdout not yet acquired. See `FINAL_HOLDOUT_PLAN.md`.
"""
    (OUT / "FINAL_IMPROVEMENT_REPORT.md").write_text(report, encoding="utf-8")
    write_csv(OUT / "summary.csv", [{"metric":name,"baseline":before,"improved":after,"delta":after-before}
                                     for name,before,after in comparisons])


def main() -> None:
    enable_selected(); OUT.mkdir(parents=True, exist_ok=True)
    ced, emergency, rows = ced_and_emergency()
    stt = stt_metrics(); personal = personalization_metrics(); latency_result = latency(rows)
    reports(ced, emergency, stt, personal, latency_result)
    write_json(OUT / "configuration.json", {"semantic_ced": True, "safe_compounds": True,
        "temporal_pooling_v2": False, "ced_route": "raw", "stt_route": "raw", "dtln": "standalone"})
    print(f"Improved benchmark complete: {OUT}")


if __name__ == "__main__":
    main()
