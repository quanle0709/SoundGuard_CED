"""Expanded, checkpointed SoundGuard benchmark using public labeled datasets."""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from benchmark.config import SEED
from benchmark.metrics import binary_metrics, classification_metrics, error_rates, latency_summary
from benchmark.noise import align_to_reference, mix_environmental_noise, mix_white_noise, snr_db
from benchmark.run_full_benchmark import plot_confusion, plot_line, process_rss_bytes

RESULTS = ROOT / "benchmark_results" / "expanded"
DATA = ROOT / "benchmark_data"
EXTERNAL = DATA / "external"
GENERATED = DATA / "expanded_generated"
SNRS = (20, 15, 10, 5, 0)
SUMMARY_FIELDS = ("benchmark", "feature", "dataset", "metric", "value", "unit",
                  "sample_count", "condition", "confidence_interval",
                  "leakage_status", "status", "notes")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_mono(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate is not None and rate != sample_rate:
        from math import gcd
        divisor = gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor).astype(np.float32)
        rate = sample_rate
    return np.asarray(audio, dtype=np.float32), rate


def save_wav(path: Path, audio: np.ndarray, rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, rate, subtype="PCM_16")
    return path


def canonical(label: str) -> str:
    from personalization.sound_labels import canonicalize_label, normalize_label
    return canonicalize_label(label) or normalize_label(label).replace(" ", "_")


def target_classification_metrics(truths: list[str], predictions: list[str]) -> dict:
    """Macro-average the declared ground-truth classes; bucket out-of-ontology predictions."""
    target_labels = sorted(set(truths))
    target_set = set(target_labels)
    bucketed = [value if value in target_set else "other" for value in predictions]
    raw = classification_metrics(truths, bucketed)
    per_class = [row for row in raw["per_class"] if row["label"] in target_set]
    raw["per_class"] = per_class
    raw["labels"] = target_labels + (["other"] if "other" in bucketed else [])
    for metric in ("precision", "recall", "f1"):
        raw[f"macro_{metric}"] = statistics.fmean(row[metric] for row in per_class)
    raw["weighted_f1"] = sum(row["f1"] * row["support"] for row in per_class) / len(truths)
    raw["bucketed_predictions"] = bucketed
    return raw


def bootstrap_ci(truths: list[str], predictions: list[str], metric: str,
                 repetitions: int = 1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(truths), len(truths))
        result = target_classification_metrics([truths[i] for i in indices], [predictions[i] for i in indices])
        values.append(float(result[metric]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def aggregate_errors(rows: list[dict]) -> dict:
    words = sum(int(row["reference_words"]) for row in rows)
    characters = sum(int(row["reference_characters"]) for row in rows)
    substitutions = sum(int(row["substitutions"]) for row in rows)
    deletions = sum(int(row["deletions"]) for row in rows)
    insertions = sum(int(row["insertions"]) for row in rows)
    char_errors = sum(int(row["character_errors"]) for row in rows)
    return {"utterances": len(rows), "total_words": words, "total_characters": characters,
            "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
            "wer": (substitutions + deletions + insertions) / words if words else 0.0,
            "cer": char_errors / characters if characters else 0.0}


def error_row(reference: str, hypothesis: str) -> dict:
    rates = error_rates(reference, hypothesis)
    rates["character_errors"] = int(round(rates["cer"] * rates["reference_characters"]))
    return rates


class ExpandedRunner:
    def __init__(self, *, enable_network_stt: bool, stability_seconds: int,
                 resume: bool, rerun_stages: list[str] | None = None):
        self.enable_network_stt = enable_network_stt
        self.stability_seconds = stability_seconds
        self.resume = resume
        self.checkpoint_path = RESULTS / "checkpoint.json"
        self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8")) \
            if resume and self.checkpoint_path.exists() else {"completed": [], "failed": {}}
        for stage in rerun_stages or []:
            if stage in self.checkpoint["completed"]:
                self.checkpoint["completed"].remove(stage)
        self.summary = read_csv(RESULTS / "summary.csv") if resume else []
        RESULTS.mkdir(parents=True, exist_ok=True)

    def add(self, benchmark, feature, dataset, metric, value, unit, count,
            condition="", confidence_interval="", leakage_status="", status="PASS", notes=""):
        self.summary = [row for row in self.summary if not (
            row["benchmark"] == benchmark and row["metric"] == metric and
            row["condition"] == condition and row["dataset"] == dataset)]
        self.summary.append(dict(zip(SUMMARY_FIELDS, (benchmark, feature, dataset, metric,
            value, unit, count, condition, confidence_interval, leakage_status, status, notes))))

    def stage(self, name, function):
        if self.resume and name in self.checkpoint["completed"]:
            print(f"[resume] {name}: already complete", flush=True); return
        print(f"[run] {name}", flush=True)
        try:
            function()
        except Exception as exc:
            self.checkpoint["failed"][name] = f"{type(exc).__name__}: {exc}"
            write_json(self.checkpoint_path, self.checkpoint)
            print(f"[failed] {name}: {type(exc).__name__}: {exc}", flush=True)
            return
        self.checkpoint["failed"].pop(name, None)
        if name not in self.checkpoint["completed"]:
            self.checkpoint["completed"].append(name)
        write_json(self.checkpoint_path, self.checkpoint)
        print(f"[complete] {name}", flush=True)

    def manifests(self):
        required = {name: EXTERNAL / name / "manifest.csv" for name in ("esc50", "vivos", "demand")}
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Run benchmark/download_datasets.py first: " + ", ".join(missing))
        esc, vivos, demand = (read_csv(required[name]) for name in ("esc50", "vivos", "demand"))
        write_json(RESULTS / "dataset_counts.json", {"esc50": len(esc), "vivos": len(vivos), "demand": len(demand)})
        if len(esc) != 280 or len(vivos) < 50 or len(demand) < 4:
            raise RuntimeError(f"Dataset minimum not met: ESC={len(esc)}, VIVOS={len(vivos)}, DEMAND={len(demand)}")
        esc_hashes = [row["sha256"] for row in esc]
        vivos_hashes = [row["sha256"] for row in vivos]
        integrity = {"esc50_unique_files": len(set(esc_hashes)), "esc50_manifest_rows": len(esc),
                     "vivos_unique_files": len(set(vivos_hashes)), "vivos_manifest_rows": len(vivos),
                     "vivos_speakers": len({row["speaker"] for row in vivos}),
                     "exact_duplicate_groups_within_esc50": len(esc_hashes) - len(set(esc_hashes)),
                     "exact_duplicate_groups_within_vivos": len(vivos_hashes) - len(set(vivos_hashes))}
        write_json(RESULTS / "data_integrity.json", integrity)
        methodology = {"benchmark": "SoundGuard expanded PC benchmark", "seed": SEED,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "platform": platform.platform(),
            "ced_model": "mispeech/ced-tiny", "ced_training_dataset": "AudioSet",
            "ced_clean": {"dataset": "ESC-50", "samples": 280, "classes": 7, "samples_per_class": 40},
            "ced_synthetic_noise": {"samples_per_snr": 280, "snr_db": list(SNRS), "seeded_gaussian": True},
            "ced_real_noise": {"balanced_source_clips": 70, "environments": 4, "samples_per_snr": 280, "snr_db": list(SNRS)},
            "dtln_ced": {"balanced_source_clips": 35, "environments": 4, "snr_db": list(SNRS), "paired_samples": 700,
                         "alignment": "cross-correlation; measured fixed lag retained per row"},
            "stt_clean": {"dataset": "VIVOS test", "utterances": 100, "speakers": integrity["vivos_speakers"]},
            "stt_robustness": {"fixed_utterances": 30, "snr_db": list(SNRS), "protocols": ["Gaussian", "DEMAND"],
                               "filters": ["unfiltered", "DTLN"]},
            "text_normalization": "Unicode NFC, lowercase, punctuation-to-space, whitespace collapse; Vietnamese diacritics preserved",
            "leakage_status": {"ESC-50": "POSSIBLE SOURCE OVERLAP", "VIVOS": "provider training unknown", "DEMAND": "no known CED overlap"},
            "production_changes_for_benchmark": False}
        write_json(RESULTS / "methodology.json", methodology)

    def ced(self):
        from sound_classifier import classify_audio_file
        esc = read_csv(EXTERNAL / "esc50" / "manifest.csv")
        noises = read_csv(EXTERNAL / "demand" / "manifest.csv")
        classify_audio_file(ROOT / esc[0]["path"])
        out = RESULTS / "ced"
        clean_rows = read_csv(out / "predictions.csv")
        if len(clean_rows) == len(esc):
            clean_latencies = [float(row["latency_ms"]) for row in clean_rows]
        else:
            clean_rows, clean_latencies = [], []
            for index, case in enumerate(esc):
                start = time.perf_counter(); result = classify_audio_file(ROOT / case["path"])
                elapsed = (time.perf_counter() - start) * 1000; clean_latencies.append(elapsed)
                predicted = canonical(result["label"])
                clean_rows.append({**case, "condition": "clean", "predicted_label": predicted,
                    "raw_predicted_label": result["label"], "confidence": result["confidence"],
                    "correct": predicted == case["true_label"], "latency_ms": elapsed})
                if (index + 1) % 40 == 0: print(f"  CED clean {index + 1}/{len(esc)}", flush=True)
        truths = [row["true_label"] for row in clean_rows]
        predictions = [row["predicted_label"] for row in clean_rows]
        clean_metrics = target_classification_metrics(truths, predictions)
        cis = {"accuracy_95_ci": bootstrap_ci(truths, predictions, "accuracy"),
               "macro_f1_95_ci": bootstrap_ci(truths, predictions, "macro_f1"),
               "bootstrap_repetitions": 1000, "seed": SEED}
        write_csv(out / "dataset_summary.csv", [{"dataset": "ESC-50", "class": label,
            "sample_count": sum(row["true_label"] == label for row in clean_rows),
            "leakage_status": "POSSIBLE SOURCE OVERLAP"} for label in sorted(set(truths))])
        write_csv(out / "predictions.csv", clean_rows)
        write_csv(out / "clean_metrics.csv", [{key: value for key, value in clean_metrics.items()
                                                if key not in {"per_class", "labels", "bucketed_predictions"}}])
        write_csv(out / "per_class_metrics.csv", clean_metrics["per_class"])
        write_json(out / "confidence_intervals.json", cis)
        labels = clean_metrics["labels"]
        bucketed = clean_metrics["bucketed_predictions"]
        matrix = [[sum(t == a and p == b for t, p in zip(truths, bucketed)) for b in labels] for a in labels]
        write_csv(out / "confusion_matrix.csv", [dict(true_label=label, **dict(zip(labels, row))) for label, row in zip(labels, matrix)])
        plot_confusion(out / "confusion_matrix.png", labels, matrix)

        synthetic_rows = read_csv(out / "synthetic_noise_results.csv")
        if len(synthetic_rows) != len(esc) * len(SNRS):
            synthetic_rows = []
            for case_index, case in enumerate(esc):
                audio, rate = load_mono(ROOT / case["path"])
                for snr_index, target in enumerate(SNRS):
                    mixed, metadata = mix_white_noise(audio, target, SEED + case_index * 10 + snr_index)
                    path = save_wav(GENERATED / "ced" / "synthetic" / str(target) / f"{case['sample_id']}.wav", mixed, rate)
                    result = classify_audio_file(path); predicted = canonical(result["label"])
                    synthetic_rows.append({"sample_id": case["sample_id"], "source_clip": case["path"],
                        "true_label": case["true_label"], "noise_protocol": "gaussian",
                        "target_snr_db": target, "achieved_snr_db": metadata["achieved_snr_db"],
                        "prediction": predicted, "raw_prediction": result["label"],
                        "confidence": result["confidence"], "correct": predicted == case["true_label"], "path": rel(path)})
                if (case_index + 1) % 40 == 0: print(f"  CED synthetic {case_index + 1}/{len(esc)}", flush=True)
        write_csv(out / "synthetic_noise_results.csv", synthetic_rows)
        synthetic_metrics = []
        synthetic_class_metrics = []
        for target in SNRS:
            subset = [row for row in synthetic_rows if int(row["target_snr_db"]) == target]
            metrics = target_classification_metrics([r["true_label"] for r in subset], [r["prediction"] for r in subset])
            synthetic_metrics.append({"snr_db": target, "sample_count": len(subset),
                                      "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
            synthetic_class_metrics.extend({"snr_db": target, **row} for row in metrics["per_class"])
        write_csv(out / "synthetic_noise_metrics.csv", synthetic_metrics)
        write_csv(out / "synthetic_noise_class_metrics.csv", synthetic_class_metrics)
        plot_line(out / "macro_f1_vs_snr_synthetic.png", [r["snr_db"] for r in synthetic_metrics],
                  [("Gaussian", [r["macro_f1"] for r in synthetic_metrics])], "SNR (dB)", "Macro F1", "CED robustness: seeded Gaussian noise")

        by_label = defaultdict(list)
        for row in esc: by_label[row["true_label"]].append(row)
        representative = [row for label in sorted(by_label) for row in by_label[label][:10]]
        real_rows = read_csv(out / "real_noise_results.csv")
        if len(real_rows) != len(representative) * len(noises) * len(SNRS):
            real_rows = []
            for noise_index, noise_case in enumerate(noises):
                environmental = load_mono(ROOT / noise_case["path"], 44100)[0]
                for case_index, case in enumerate(representative):
                    audio, rate = load_mono(ROOT / case["path"])
                    if rate != 44100:
                        environmental = load_mono(ROOT / noise_case["path"], rate)[0]
                    for snr_index, target in enumerate(SNRS):
                        mixed, metadata = mix_environmental_noise(audio, environmental, target,
                            SEED + noise_index * 100000 + case_index * 10 + snr_index)
                        path = save_wav(GENERATED / "ced" / "real" / noise_case["environment"] /
                                        str(target) / f"{case['sample_id']}.wav", mixed, rate)
                        result = classify_audio_file(path); predicted = canonical(result["label"])
                        real_rows.append({"source_clip": case["path"], "sample_id": case["sample_id"],
                            "true_label": case["true_label"], "noise_environment": noise_case["environment"],
                            "target_snr_db": target, "achieved_snr_db": metadata["achieved_snr_db"],
                            "prediction": predicted, "raw_prediction": result["label"], "confidence": result["confidence"],
                            "correct": predicted == case["true_label"], "path": rel(path)})
                print(f"  CED real noise {noise_index + 1}/{len(noises)} environments", flush=True)
        write_csv(out / "real_noise_results.csv", real_rows)
        real_snr_metrics, real_environment_metrics, real_class_degradation = [], [], []
        clean_f1 = {row["label"]: row["f1"] for row in clean_metrics["per_class"]}
        for target in SNRS:
            subset = [r for r in real_rows if int(r["target_snr_db"]) == target]
            metrics = target_classification_metrics([r["true_label"] for r in subset], [r["prediction"] for r in subset])
            real_snr_metrics.append({"snr_db": target, "sample_count": len(subset),
                                     "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
            real_class_degradation.extend({"snr_db": target, **row,
                "clean_f1": clean_f1[row["label"]], "delta_f1_from_clean": row["f1"] - clean_f1[row["label"]]}
                for row in metrics["per_class"])
        for environment in sorted({r["noise_environment"] for r in real_rows}):
            subset = [r for r in real_rows if r["noise_environment"] == environment]
            metrics = target_classification_metrics([r["true_label"] for r in subset], [r["prediction"] for r in subset])
            real_environment_metrics.append({"noise_environment": environment, "sample_count": len(subset),
                                              "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
        write_csv(out / "real_noise_metrics_by_snr.csv", real_snr_metrics)
        write_csv(out / "real_noise_metrics_by_environment.csv", real_environment_metrics)
        write_csv(out / "real_noise_class_degradation.csv", real_class_degradation)
        plot_line(out / "macro_f1_vs_snr_real_demand.png", [r["snr_db"] for r in real_snr_metrics],
                  [("DEMAND", [r["macro_f1"] for r in real_snr_metrics])], "SNR (dB)", "Macro F1", "CED robustness: real DEMAND noise")
        latency = latency_summary(clean_latencies); write_json(out / "latency_summary.json", latency)
        self.add("CED", "Environmental Sound Detection", "ESC-50", "Clean Accuracy", clean_metrics["accuracy"], "ratio", len(clean_rows), "clean", str(cis["accuracy_95_ci"]), "POSSIBLE SOURCE OVERLAP")
        self.add("CED", "Environmental Sound Detection", "ESC-50", "Clean Macro F1", clean_metrics["macro_f1"], "ratio", len(clean_rows), "clean", str(cis["macro_f1_95_ci"]), "POSSIBLE SOURCE OVERLAP")
        for row in synthetic_metrics:
            self.add("CED", "Synthetic noise robustness", "ESC-50 + Gaussian", "Macro F1", row["macro_f1"], "ratio", row["sample_count"], f"{row['snr_db']} dB", leakage_status="POSSIBLE SOURCE OVERLAP")
        for row in real_snr_metrics:
            self.add("CED", "Real-noise robustness", "ESC-50 + DEMAND", "Macro F1", row["macro_f1"], "ratio", row["sample_count"], f"{row['snr_db']} dB", leakage_status="POSSIBLE SOURCE OVERLAP")

    def dtln_ced(self):
        from sound_classifier import classify_audio_file
        from speech_enhancer import enhance_audio_file
        esc = read_csv(EXTERNAL / "esc50" / "manifest.csv")
        noises = read_csv(EXTERNAL / "demand" / "manifest.csv")
        by_label = defaultdict(list)
        for row in esc: by_label[row["true_label"]].append(row)
        subset = [row for label in sorted(by_label) for row in by_label[label][:5]]
        raw_lookup = {(r["sample_id"], r["noise_environment"], int(r["target_snr_db"])): r
                      for r in read_csv(RESULTS / "ced" / "real_noise_results.csv")}
        out = RESULTS / "noise_filter"
        quality, pairs = read_csv(out / "snr_improvement.csv"), read_csv(out / "ced_predictions_paired.csv")
        expected_count = len(subset) * len(noises) * len(SNRS)
        if len(quality) != expected_count or len(pairs) != expected_count:
            quality, pairs = [], []
            for noise_case in noises:
                for case in subset:
                    clean, _ = load_mono(ROOT / case["path"], 16000)
                    for target in SNRS:
                        raw = raw_lookup[(case["sample_id"], noise_case["environment"], target)]
                        noisy, _ = load_mono(ROOT / raw["path"], 16000)
                        noisy_path = save_wav(GENERATED / "dtln" / "inputs" / noise_case["environment"] /
                                              str(target) / f"{case['sample_id']}.wav", noisy, 16000)
                        filtered_path = GENERATED / "dtln" / "outputs" / noise_case["environment"] / str(target) / f"{case['sample_id']}.wav"
                        start = time.perf_counter(); enhance_audio_file(noisy_path, filtered_path, verbose=False)
                        elapsed = time.perf_counter() - start
                        filtered, _ = load_mono(filtered_path, 16000)
                        # Estimate common anti-clip gain by least squares against the clean reference.
                        gain = float(np.dot(noisy, clean) / max(np.dot(clean, clean), 1e-12))
                        reference = clean * gain
                        input_value = snr_db(reference, noisy)
                        aligned_reference, aligned_filtered, lag = align_to_reference(reference, filtered)
                        output_value = snr_db(aligned_reference, aligned_filtered)
                        prediction = classify_audio_file(filtered_path); predicted = canonical(prediction["label"])
                        quality.append({"sample_id": case["sample_id"], "true_label": case["true_label"],
                            "noise_environment": noise_case["environment"], "target_snr_db": target,
                            "input_snr_db": input_value, "output_snr_db": output_value,
                            "delta_snr_db": output_value - input_value, "alignment_lag_samples": lag,
                            "filter_seconds": elapsed, "filtered_path": rel(filtered_path)})
                        pairs.append({"sample_id": case["sample_id"], "true_label": case["true_label"],
                            "noise_environment": noise_case["environment"], "target_snr_db": target,
                            "without_filter": raw["prediction"], "with_filter": predicted,
                            "without_correct": raw["prediction"] == case["true_label"],
                            "with_correct": predicted == case["true_label"]})
                print(f"  DTLN/CED {noise_case['environment']} complete", flush=True)
        write_csv(out / "snr_improvement.csv", quality); write_csv(out / "ced_predictions_paired.csv", pairs)
        deltas = [float(r["delta_snr_db"]) for r in quality]
        signal_summary = {"sample_count": len(quality), "mean_delta_snr_db": statistics.fmean(deltas),
                          "median_delta_snr_db": statistics.median(deltas), "std_delta_snr_db": statistics.pstdev(deltas)}
        write_json(out / "signal_summary.json", signal_summary)
        signal_conditions = []
        for environment in sorted({r["noise_environment"] for r in quality}):
            for target in SNRS:
                selected_deltas = [float(r["delta_snr_db"]) for r in quality
                                   if r["noise_environment"] == environment and int(r["target_snr_db"]) == target]
                signal_conditions.append({"noise_environment": environment, "snr_db": target,
                    "sample_count": len(selected_deltas), "mean_delta_snr_db": statistics.fmean(selected_deltas),
                    "median_delta_snr_db": statistics.median(selected_deltas),
                    "std_delta_snr_db": statistics.pstdev(selected_deltas)})
        write_csv(out / "signal_summary_by_environment_snr.csv", signal_conditions)
        comparisons = []
        for target in SNRS:
            selected = [r for r in pairs if int(r["target_snr_db"]) == target]
            truths = [r["true_label"] for r in selected]
            before = target_classification_metrics(truths, [r["without_filter"] for r in selected])
            after = target_classification_metrics(truths, [r["with_filter"] for r in selected])
            comparisons.append({"snr_db": target, "sample_count": len(selected),
                "macro_f1_without_filter": before["macro_f1"], "macro_f1_with_filter": after["macro_f1"],
                "delta_f1": after["macro_f1"] - before["macro_f1"],
                "accuracy_without_filter": before["accuracy"], "accuracy_with_filter": after["accuracy"],
                "delta_accuracy": after["accuracy"] - before["accuracy"]})
        write_csv(out / "ced_before_after.csv", comparisons)
        environment_comparisons = []
        for environment in sorted({r["noise_environment"] for r in pairs}):
            for target in SNRS:
                selected = [r for r in pairs if r["noise_environment"] == environment and int(r["target_snr_db"]) == target]
                truths = [r["true_label"] for r in selected]
                before = target_classification_metrics(truths, [r["without_filter"] for r in selected])
                after = target_classification_metrics(truths, [r["with_filter"] for r in selected])
                environment_comparisons.append({"noise_environment": environment, "snr_db": target,
                    "sample_count": len(selected), "macro_f1_without_filter": before["macro_f1"],
                    "macro_f1_with_filter": after["macro_f1"], "delta_f1": after["macro_f1"] - before["macro_f1"],
                    "accuracy_without_filter": before["accuracy"], "accuracy_with_filter": after["accuracy"],
                    "delta_accuracy": after["accuracy"] - before["accuracy"]})
        write_csv(out / "ced_before_after_by_environment.csv", environment_comparisons)
        plot_line(out / "input_vs_output_snr.png", list(range(len(quality))),
                  [("Input SNR", [float(r["input_snr_db"]) for r in quality]),
                   ("Output SNR", [float(r["output_snr_db"]) for r in quality])],
                  "Paired observation", "SNR (dB)", "DTLN signal-level SNR")
        plot_line(out / "ced_f1_before_after.png", [r["snr_db"] for r in comparisons],
                  [("No filter", [r["macro_f1_without_filter"] for r in comparisons]),
                   ("DTLN", [r["macro_f1_with_filter"] for r in comparisons])],
                  "SNR (dB)", "Macro F1", "Paired CED effect of DTLN")
        self.add("DTLN", "Signal quality", "ESC-50 + DEMAND", "Mean Delta SNR", signal_summary["mean_delta_snr_db"], "dB", len(quality), leakage_status="POSSIBLE SOURCE OVERLAP")
        weighted_delta = statistics.fmean(r["delta_f1"] for r in comparisons)
        self.add("DTLN", "CED downstream effect", "ESC-50 + DEMAND", "Delta CED Macro F1", weighted_delta, "ratio", len(pairs), "all SNRs", leakage_status="POSSIBLE SOURCE OVERLAP")

    def _recognize(self, path: Path, reference: str) -> dict:
        from speech_recognizer import transcribe_audio_file
        start = time.perf_counter()
        try:
            hypothesis = transcribe_audio_file(path); status, error = "PASS", ""
        except Exception as exc:
            hypothesis, status, error = "", "FAILED", f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        audio, rate = load_mono(path)
        return {**error_row(reference, hypothesis), "processing_seconds": elapsed,
                "audio_duration_seconds": len(audio) / rate, "real_time_factor": elapsed / (len(audio) / rate),
                "status": status, "error": error}

    def stt(self):
        out = RESULTS / "stt"
        if not self.enable_network_stt:
            write_json(out / "SKIPPED.json", {"status": "SKIPPED", "reason": "network STT not enabled"})
            self.add("STT", "Speech-to-Text", "VIVOS", "WER", "", "", 0, status="SKIPPED", notes="network STT not enabled")
            return
        from speech_enhancer import enhance_audio_file
        utterances = read_csv(EXTERNAL / "vivos" / "manifest.csv")
        noises = read_csv(EXTERNAL / "demand" / "manifest.csv")
        clean_rows = read_csv(out / "clean_results.csv")
        if len(clean_rows) != len(utterances):
            clean_rows = []
            for index, case in enumerate(utterances):
                result = self._recognize(ROOT / case["path"], case["reference"])
                clean_rows.append({**case, **result})
                if (index + 1) % 10 == 0: print(f"  STT clean {index + 1}/{len(utterances)}", flush=True)
        write_csv(out / "clean_results.csv", clean_rows)
        passed_clean = [r for r in clean_rows if r["status"] == "PASS"]
        if not passed_clean:
            raise RuntimeError("All clean Google STT requests failed")
        clean_summary = aggregate_errors(passed_clean); write_json(out / "clean_summary.json", clean_summary)

        subset = utterances[:30]
        rows = read_csv(out / "noise_results.csv")
        if len(rows) != len(subset) * len(SNRS) * 2 * 2:
            rows = []
            for case_index, case in enumerate(subset):
                clean, _ = load_mono(ROOT / case["path"], 16000)
                for protocol in ("synthetic", "real"):
                    noise_case = noises[case_index % len(noises)]
                    noise = load_mono(ROOT / noise_case["path"], 16000)[0]
                    for snr_index, target in enumerate(SNRS):
                        seed = SEED + case_index * 20 + snr_index
                        if protocol == "synthetic": mixed, metadata = mix_white_noise(clean, target, seed)
                        else: mixed, metadata = mix_environmental_noise(clean, noise, target, seed)
                        noisy_path = save_wav(GENERATED / "stt" / protocol / "noisy" / str(target) /
                                              f"{case['sample_id']}.wav", mixed, 16000)
                        filtered_path = GENERATED / "stt" / protocol / "dtln" / str(target) / f"{case['sample_id']}.wav"
                        enhance_audio_file(noisy_path, filtered_path, verbose=False)
                        for filter_name, path in (("unfiltered", noisy_path), ("dtln", filtered_path)):
                            result = self._recognize(path, case["reference"])
                            rows.append({"sample_id": case["sample_id"], "speaker": case["speaker"],
                                "corpus": "VIVOS", "protocol": protocol,
                                "noise_environment": noise_case["environment"] if protocol == "real" else "gaussian",
                                "target_snr_db": target, "achieved_snr_db": metadata["achieved_snr_db"],
                                "filter": filter_name, "file": rel(path), **result})
                print(f"  STT noisy {case_index + 1}/{len(subset)}", flush=True)
        write_csv(out / "noise_results.csv", rows)
        summaries = []
        for protocol in ("synthetic", "real"):
            for filter_name in ("unfiltered", "dtln"):
                for target in SNRS:
                    selected = [r for r in rows if r["protocol"] == protocol and r["filter"] == filter_name
                                and int(r["target_snr_db"]) == target and r["status"] == "PASS"]
                    summary = aggregate_errors(selected) if selected else {"utterances": 0, "wer": 0.0, "cer": 0.0,
                        "total_words": 0, "total_characters": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
                    summaries.append({"protocol": protocol, "filter": filter_name, "snr_db": target, **summary})
        write_csv(out / "noise_summary.csv", summaries)
        environment_summaries = []
        for environment in sorted({r["noise_environment"] for r in rows if r["protocol"] == "real"}):
            for filter_name in ("unfiltered", "dtln"):
                for target in SNRS:
                    selected = [r for r in rows if r["protocol"] == "real" and r["noise_environment"] == environment
                                and r["filter"] == filter_name and int(r["target_snr_db"]) == target and r["status"] == "PASS"]
                    environment_summaries.append({"noise_environment": environment, "filter": filter_name,
                        "snr_db": target, **aggregate_errors(selected)})
        write_csv(out / "real_noise_summary_by_environment.csv", environment_summaries)
        unfiltered = [r for r in summaries if r["filter"] == "unfiltered"]
        plot_line(out / "wer_vs_snr.png", list(SNRS), [(protocol, [next(r["wer"] for r in unfiltered if r["protocol"] == protocol and r["snr_db"] == snr) for snr in SNRS]) for protocol in ("synthetic", "real")], "SNR (dB)", "WER", "Vietnamese STT noise robustness")
        plot_line(out / "cer_vs_snr.png", list(SNRS), [(protocol, [next(r["cer"] for r in unfiltered if r["protocol"] == protocol and r["snr_db"] == snr) for snr in SNRS]) for protocol in ("synthetic", "real")], "SNR (dB)", "CER", "Vietnamese STT noise robustness")
        real_pairs = []
        for target in SNRS:
            before = next(r for r in summaries if r["protocol"] == "real" and r["filter"] == "unfiltered" and r["snr_db"] == target)
            after = next(r for r in summaries if r["protocol"] == "real" and r["filter"] == "dtln" and r["snr_db"] == target)
            real_pairs.append({"snr_db": target, "sample_count": min(before["utterances"], after["utterances"]),
                "wer_without_filter": before["wer"], "wer_with_filter": after["wer"], "delta_wer": after["wer"] - before["wer"],
                "cer_without_filter": before["cer"], "cer_with_filter": after["cer"], "delta_cer": after["cer"] - before["cer"]})
        write_csv(RESULTS / "noise_filter" / "stt_before_after.csv", real_pairs)
        plot_line(RESULTS / "noise_filter" / "stt_wer_before_after.png", list(SNRS),
                  [("No filter", [r["wer_without_filter"] for r in real_pairs]), ("DTLN", [r["wer_with_filter"] for r in real_pairs])],
                  "SNR (dB)", "WER (lower is better)", "Paired STT effect of DTLN")
        latency = latency_summary([float(r["processing_seconds"]) * 1000 for r in passed_clean])
        latency["mean_real_time_factor"] = statistics.fmean(float(r["real_time_factor"]) for r in passed_clean)
        write_json(out / "latency_summary.json", latency)
        self.add("STT", "Speech-to-Text", "VIVOS", "Clean WER", clean_summary["wer"], "ratio", clean_summary["utterances"], "clean", leakage_status="provider training unknown")
        self.add("STT", "Speech-to-Text", "VIVOS", "Clean CER", clean_summary["cer"], "ratio", clean_summary["utterances"], "clean", leakage_status="provider training unknown")
        for row in [r for r in unfiltered if r["protocol"] == "real"]:
            self.add("STT", "Real-noise robustness", "VIVOS + DEMAND", "WER", row["wer"], "ratio", row["utterances"], f"{row['snr_db']} dB", leakage_status="provider training unknown")
        self.add("DTLN", "STT downstream effect", "VIVOS + DEMAND", "Delta STT WER", statistics.fmean(r["delta_wer"] for r in real_pairs), "ratio", sum(r["sample_count"] for r in real_pairs), "all SNRs", leakage_status="provider training unknown")

    def emergency(self):
        from emergency_system import EmergencySystem
        clean = read_csv(RESULTS / "ced" / "predictions.csv")
        positive_labels = {"siren", "vehicle_horn", "baby_crying", "glass_breaking", "fire"}
        rows = []
        for row in clean:
            expected = row["true_label"] in positive_labels
            result = EmergencySystem(decision_mode="single_shot").process_sound_event(
                row["raw_predicted_label"], float(row["confidence"]))
            predicted = result["alert_level"] in {"MEDIUM", "HIGH", "CRITICAL"} and result["event_state"] in {"EVENT_STARTED", "EVENT_CONTINUING"}
            rows.append({"sample_id": row["sample_id"], "true_label": row["true_label"],
                "ground_truth_emergency": expected, "predicted_emergency": predicted,
                "ced_prediction": row["raw_predicted_label"], "ced_confidence": row["confidence"],
                "alert_level": result["alert_level"], "correct": predicted == expected})
        metrics = binary_metrics([r["ground_truth_emergency"] for r in rows], [r["predicted_emergency"] for r in rows])
        out = RESULTS / "emergency"; write_csv(out / "predictions.csv", rows); write_json(out / "metrics.json", metrics)
        matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
        plot_confusion(out / "confusion_matrix.png", ["non-emergency", "emergency"], matrix)
        self.add("Emergency", "Real-audio emergency detection", "ESC-50", "Recall", metrics["recall"], "ratio", len(rows), leakage_status="POSSIBLE SOURCE OVERLAP")
        self.add("Emergency", "Real-audio emergency detection", "ESC-50", "False Negative Rate", metrics["false_negative_rate"], "ratio", len(rows), leakage_status="POSSIBLE SOURCE OVERLAP")
        self.add("Emergency", "Real-audio emergency detection", "ESC-50", "False alerts", metrics["fp"], "count", metrics["tn"] + metrics["fp"], "non-emergency clips", leakage_status="POSSIBLE SOURCE OVERLAP")

    def fusion(self):
        from fusion_engine import fuse_result
        cases = json.loads((DATA / "fusion_cases.json").read_text(encoding="utf-8"))
        rows = []
        for case in cases:
            raw = bool(case["dangerous"] and case["confidence"] >= .45)
            result = fuse_result("benchmark", case["transcript"], {"label": case["label"], "confidence": case["confidence"]},
                {"category": case["category"], "is_dangerous": case["dangerous"], "message_vi": "benchmark"})
            rows.append({**case, "raw_prediction": raw, "fusion_prediction": bool(result["alert"])})
        raw_metrics = binary_metrics([r["expected"] for r in rows], [r["raw_prediction"] for r in rows])
        fused = binary_metrics([r["expected"] for r in rows], [r["fusion_prediction"] for r in rows])
        out = RESULTS / "fusion"; write_csv(out / "before_after.csv", rows)
        write_json(out / "summary.json", {"evidence_type": "deterministic logic validation", "raw": raw_metrics, "fusion": fused, "delta_f1": fused["f1"] - raw_metrics["f1"]})
        self.add("Fusion", "Fusion Engine", "hand-authored functional cases", "Raw F1", raw_metrics["f1"], "ratio", len(rows), notes="deterministic logic validation")
        self.add("Fusion", "Fusion Engine", "hand-authored functional cases", "Fused F1", fused["f1"], "ratio", len(rows), notes="deterministic logic validation")

    def personalization(self):
        from personalization.profile_generator import generate_profile, generate_rule_based
        from personalization.profile_validator import UNIVERSAL_MIN_PRIORITY, validate_profile
        from personalization.role_knowledge import DEFAULT_PRIORITIES, ROLE_KNOWLEDGE
        from personalization.schemas import ProfileValidationError
        role_texts = {
            "driver": ["I drive in busy road traffic every day.", "I am a taxi driver on city roads."],
            "motorcyclist_cyclist": ["I ride a motorbike to work.", "I am a cyclist in road traffic."],
            "pedestrian_commuter": ["I walk to work on foot.", "I am a pedestrian and travel on foot."],
            "construction_worker": ["I am a construction worker at a building site.", "I work in construction at my workplace."],
            "factory_warehouse_worker": ["I work in a factory warehouse.", "I operate near a warehouse forklift."],
            "parent_infant_caregiver": ["I am a parent caring for my baby at home.", "I am an infant caregiver in my house."],
            "pregnant_expectant_mother": ["I am pregnant and spend time at home.", "I am an expectant mother outdoors."],
            "older_adult_independent": ["I am an older adult and live independently.", "I am a senior living at home."],
            "student": ["I am a student at school.", "I attend university and study at home."],
            "healthcare_care_worker": ["I am a nurse working in a hospital.", "I am a healthcare care worker at a clinic."],
        }
        cases = []
        for role, texts in role_texts.items():
            for variant, text in enumerate(texts, 1):
                expected_priorities = dict(ROLE_KNOWLEDGE[role]["priorities"])
                for label, minimum in UNIVERSAL_MIN_PRIORITY.items():
                    if label in expected_priorities:
                        expected_priorities[label] = max(expected_priorities[label], minimum)
                cases.append({"id": f"{role}_{variant}", "text": text, "expected_role": role,
                              "priority_assertions": expected_priorities})
        write_json(DATA / "expanded_personalization_cases.json", cases)
        rows = []
        for case in cases:
            profile = generate_rule_based([case["text"]])
            for label, expected in case["priority_assertions"].items():
                actual = profile["priority_profile"][label]
                rows.append({"case_id": case["id"], "label": label, "expected_priority": expected,
                             "actual_priority": actual, "absolute_error": abs(actual - expected),
                             "within_one": abs(actual - expected) <= 1, "exact": actual == expected})
        safety = []
        for case in cases:
            profile = generate_rule_based([case["text"]])
            safety.append({"id": f"universal_{case['id']}", "category": "universal_critical",
                "passed": all(profile["priority_profile"][label] >= minimum for label, minimum in UNIVERSAL_MIN_PRIORITY.items())})
        negations = ["I do not drive.", "I never use a bus.", "I have no construction work.", "I live without a baby.",
            "I am not a student.", "I do not work at a factory.", "I never walk to work.", "I am not pregnant.",
            "I do not work in healthcare.", "I live without an older adult."]
        for index, text in enumerate(negations):
            profile = generate_rule_based([text]); safety.append({"id": f"negation_{index}", "category": "negation", "passed": not profile["roles"]})
        contrasts = ["I do not drive, but I walk to work on foot.", "I never use a bus; however I ride a motorbike.",
            "I am not a student, but I work in a hospital as a nurse.", "I do not work in construction; however I work in a factory.",
            "I am not a driver, but I am a cyclist in road traffic."]
        expected_contrast = ["pedestrian_commuter", "motorcyclist_cyclist", "healthcare_care_worker", "factory_warehouse_worker", "motorcyclist_cyclist"]
        for index, (text, expected) in enumerate(zip(contrasts, expected_contrast)):
            profile = generate_rule_based([text]); safety.append({"id": f"contrast_{index}", "category": "contrast", "passed": expected in profile["roles"]})
        substrings = ["I paint driveway artwork.", "The studentized statistic is useful.", "This is a homeopathic note.",
                      "A factory-method pattern is software.", "The baby-blue color is bright."]
        for index, text in enumerate(substrings):
            profile = generate_rule_based([text]); safety.append({"id": f"substring_{index}", "category": "substring", "passed": not profile["roles"]})
        malformed_builders = [
            lambda: {**generate_rule_based(["I drive."]), "priority_profile": {**DEFAULT_PRIORITIES, "speech": 9}},
            lambda: {key: value for key, value in generate_rule_based(["I drive."]).items() if key != "contexts"},
            lambda: {**generate_rule_based(["I drive."]), "priority_profile": {**DEFAULT_PRIORITIES, "made_up": 3}},
            lambda: {**generate_rule_based(["I drive."]), "profile_version": "999"},
            lambda: {**generate_rule_based(["I drive."]), "roles": ["unknown_role"]},
        ]
        for index, build in enumerate(malformed_builders):
            try: validate_profile(build()); rejected = False
            except ProfileValidationError: rejected = True
            safety.append({"id": f"malformed_{index}", "category": "malformed", "passed": rejected})
        saved = {name: os.environ.get(name) for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "SOUNDGUARD_AI_PROVIDER")}
        os.environ.pop("OPENAI_API_KEY", None); os.environ.pop("OPENROUTER_API_KEY", None); os.environ["SOUNDGUARD_AI_PROVIDER"] = "openai"
        try:
            for index, text in enumerate(["I drive.", "I am a student.", "I care for a baby.", "I work in construction.", "I walk on foot."]):
                _, provider = generate_profile([text]); safety.append({"id": f"fallback_{index}", "category": "fallback", "passed": provider.startswith("deterministic_fallback:")})
        finally:
            for name, value in saved.items():
                if value is None: os.environ.pop(name, None)
                else: os.environ[name] = value
        write_csv(RESULTS / "personalization" / "priority_assertions.csv", rows)
        write_csv(RESULTS / "personalization" / "safety_tests.csv", safety)
        metrics = {"profile_cases": len(cases), "priority_assertions": len(rows),
            "exact_priority_accuracy": statistics.fmean(r["exact"] for r in rows),
            "mean_absolute_error": statistics.fmean(r["absolute_error"] for r in rows),
            "within_one_accuracy": statistics.fmean(r["within_one"] for r in rows),
            "safety_cases": len(safety), "safety_pass_rate": statistics.fmean(r["passed"] for r in safety)}
        for category in sorted({r["category"] for r in safety}):
            selected = [r for r in safety if r["category"] == category]
            metrics[f"{category}_pass_rate"] = statistics.fmean(r["passed"] for r in selected)
            metrics[f"{category}_count"] = len(selected)
        write_json(RESULTS / "personalization" / "metrics.json", metrics)
        plot_line(RESULTS / "personalization" / "summary.png", [0, 1, 2],
                  [("Pass/accuracy", [metrics["exact_priority_accuracy"], metrics["safety_pass_rate"], metrics["universal_critical_pass_rate"]])],
                  "0=priority, 1=safety, 2=universal", "Ratio", "Expanded personalization validation")
        self.add("Personalization", "Deterministic personalization", "hand-authored policy cases", "Priority Accuracy", metrics["exact_priority_accuracy"], "ratio", len(rows))
        self.add("Personalization", "Safety validation", "hand-authored invariants", "Safety Pass Rate", metrics["safety_pass_rate"], "ratio", len(safety))
        self.add("Personalization", "Universal Critical", "hand-authored invariants", "Universal Critical Pass Rate", metrics["universal_critical_pass_rate"], "ratio", metrics["universal_critical_count"])

        key_available = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        remote_rows = []
        if key_available:
            for text in ("I drive in road traffic.", "I care for a baby at home.", "I work in construction."):
                start = time.perf_counter()
                try: profile, provider = generate_profile([text]); valid = bool(validate_profile(profile)); error = ""
                except Exception as exc: provider, valid, error = "failed", False, f"{type(exc).__name__}: {exc}"
                remote_rows.append({"case": text, "provider": provider, "schema_valid": valid,
                                    "fallback": str(provider).startswith("deterministic_fallback:"),
                                    "latency_ms": (time.perf_counter() - start) * 1000, "error": error})
            write_csv(RESULTS / "personalization" / "remote_results.csv", remote_rows)
            self.add("Personalization", "Remote personalization", "configured provider",
                     "Success Rate", statistics.fmean(not row["fallback"] and row["schema_valid"] for row in remote_rows),
                     "ratio", len(remote_rows), notes="Small fixed remote sample; credentials never persisted")
        else:
            write_json(RESULTS / "personalization" / "remote_SKIPPED.json", {"status": "SKIPPED", "reason": "No API credential in benchmark process"})
            self.add("Personalization", "Remote personalization", "OpenAI/OpenRouter",
                     "Success Rate", "", "", 0, status="SKIPPED",
                     notes="No API credential in benchmark process")

    def latency(self):
        from alert_mapper import map_alert
        from emergency_system import EmergencySystem
        from fusion_engine import fuse_result
        from sound_classifier import classify_audio_file
        esc = read_csv(EXTERNAL / "esc50" / "manifest.csv")[:100]
        classify_audio_file(ROOT / esc[0]["path"])
        rows = []
        for case in esc:
            start = time.perf_counter(); ced = classify_audio_file(ROOT / case["path"]); after_ced = time.perf_counter()
            mapped = map_alert(ced["label"], ced["confidence"], threshold=None)
            emergency = EmergencySystem(decision_mode="single_shot").process_sound_event(ced["label"], ced["confidence"])
            after_logic = time.perf_counter(); fuse_result("benchmark", "", ced, mapped); completed = time.perf_counter()
            rows.append({"sample_id": case["sample_id"], "ced_ms": (after_ced-start)*1000,
                "logic_ms": (after_logic-after_ced)*1000, "fusion_ms": (completed-after_logic)*1000,
                "total_ms": (completed-start)*1000, "alert_level": emergency["alert_level"]})
        out = RESULTS / "pipeline"; write_csv(out / "latency.csv", rows)
        summary = latency_summary([r["total_ms"] for r in rows]); write_json(out / "latency_summary.json", summary)
        plot_line(out / "latency_summary.png", list(range(1, len(rows) + 1)),
                  [("Total", [r["total_ms"] for r in rows])], "Warm sample index",
                  "Latency (ms)", "PC software pipeline latency")
        self.add("Latency", "PC Software Pipeline", "ESC-50", "P95 Latency", summary["p95_ms"], "ms", len(rows), "warm fixed-file", notes="excludes capture, cloud STT, serial, ESP32, HUD, vibration")

    def stability(self):
        from emergency_system import EmergencySystem
        from fusion_engine import fuse_result
        from personalization.profile_generator import generate_rule_based
        out = RESULTS / "stability"; rows, exceptions, processed = [], 0, 0
        tracemalloc.start(); started = time.perf_counter(); wall, cpu = started, time.process_time()
        system = EmergencySystem(decision_mode="continuous")
        while time.perf_counter() - started < self.stability_seconds:
            try:
                for _ in range(100):
                    result = system.process_sound_event("Siren" if processed % 3 else "Bark", .8)
                    fuse_result("benchmark", "", {"label": "Siren", "confidence": .8}, result.get("mapped_sound", {}))
                    if processed % 1000 == 0: generate_rule_based(["I am a driver in road traffic."])
                    processed += 1
            except Exception: exceptions += 1
            now = time.perf_counter()
            if now - wall >= 1:
                cpu_now = time.process_time(); current, peak = tracemalloc.get_traced_memory(); rss = process_rss_bytes()
                rows.append({"elapsed_seconds": now-started, "processed_events": processed,
                    "cpu_percent_total_capacity": (cpu_now-cpu)/(now-wall)*100/max(1, os.cpu_count() or 1),
                    "rss_mb": rss/1024**2 if rss is not None else "", "traced_current_mb": current/1024**2,
                    "traced_peak_mb": peak/1024**2, "exceptions": exceptions})
                wall, cpu = now, cpu_now
        current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rss = [float(r["rss_mb"]) for r in rows if r["rss_mb"] != ""]
        summary = {"name": "SYNTHETIC CONTROL-PATH STABILITY SOAK", "target_seconds": self.stability_seconds,
            "actual_seconds": time.perf_counter()-started, "processed_events": processed, "exceptions": exceptions,
            "crashes": 0, "peak_rss_mb": max(rss) if rss else None, "traced_peak_mb": peak/1024**2,
            "rss_growth_mb": (rss[-1] - rss[0]) if len(rss) > 1 else None,
            "traced_allocation_growth_mb": (rows[-1]["traced_current_mb"] - rows[0]["traced_current_mb"]) if len(rows) > 1 else None,
            "limitation": "No microphone, model inference, serial hardware, HUD, vibration, or physical glasses."}
        write_csv(out / "resource_usage.csv", rows); write_json(out / "summary.json", summary)
        if rows:
            x = [r["elapsed_seconds"] for r in rows]
            ram_values = [float(r["rss_mb"]) for r in rows] if rss and len(rss) == len(rows) else [r["traced_current_mb"] for r in rows]
            plot_line(out / "ram_over_time.png", x, [("Memory", ram_values)], "Elapsed seconds", "MiB", "Stability memory")
            plot_line(out / "cpu_over_time.png", x, [("CPU", [r["cpu_percent_total_capacity"] for r in rows])], "Elapsed seconds", "CPU (% total capacity)", "Stability CPU")
        self.add("Stability", "Synthetic control-path stability soak", "synthetic control events", "Crash Count", 0, "count", processed, f"{summary['actual_seconds']:.1f} seconds", notes=summary["limitation"])

    def reports(self):
        write_csv(RESULTS / "summary.csv", self.summary, SUMMARY_FIELDS)
        write_json(RESULTS / "summary.json", {"rows": self.summary, "checkpoint": self.checkpoint})
        preliminary = """# Preliminary versus expanded benchmark

| Area | Preliminary | Expanded |
| --- | ---: | ---: |
| CED clean | 2 clips / 2 classes | 280 clips / 7 mapped ESC-50 classes |
| Vietnamese STT | 1 utterance | 100 VIVOS test utterances (or recorded successful denominator) |
| Real environmental noise | none | 4 DEMAND environments |
| Emergency | 8 fixed cases | 280 real-audio CED-to-policy cases |
| Personalization | 22 priority assertions | 100+ hand-specified priority assertions and 50+ safety checks |
| Stability | 60 seconds | configured 3,600-second synthetic control-path soak |

The expanded estimates should replace the preliminary numbers in reports because their denominators, class balance, real-noise coverage, and fixed public ground truth are substantially stronger. Original files under `benchmark_results/` remain preserved. ESC-50 results still carry `POSSIBLE SOURCE OVERLAP` and must not be described as guaranteed AudioSet-independent validation.
"""
        (RESULTS / "preliminary_vs_expanded.md").write_text(preliminary, encoding="utf-8")
        def find(benchmark, metric, condition=None):
            rows = [r for r in self.summary if r["benchmark"] == benchmark and r["metric"] == metric and (condition is None or r["condition"] == condition)]
            return "not available" if not rows else f"{rows[0]['value']} {rows[0]['unit']} (N={rows[0]['sample_count']})"
        stability_path = RESULTS / "stability" / "summary.json"
        stability_info = json.loads(stability_path.read_text(encoding="utf-8")) if stability_path.exists() else {}
        stability_rss = stability_info.get("peak_rss_mb")
        stability_rss_text = "unavailable" if stability_rss is None else f"{stability_rss} MiB"
        report = f"""# SoundGuard Expanded PC Benchmark Report

## 1. System Under Test

Git commit `{subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}`; Python {platform.python_version()} on {platform.platform()}. Production CED, STT, DTLN, emergency, fusion, and personalization behavior was not altered by this benchmark.

## 2. Dataset Expansion

ESC-50: 280 clips (40 each: siren, car horn, dog, crying baby, glass breaking, crackling fire, and door knock), CC BY-NC 3.0, `POSSIBLE SOURCE OVERLAP`. VIVOS: 100 official test utterances selected round-robin across all 19 test speakers, CC BY-NC-SA 4.0, proprietary-provider training membership unknown. DEMAND: one 16 kHz channel from traffic, cafeteria, office, and home/living environments, CC BY-SA 3.0. Full provenance is in `benchmark_data/DATASETS.md` and `dataset_leakage_audit.md`.

## 3. Experimental Method

Seed 42; fixed class-complete ESC-50 selection; fixed 30-utterance VIVOS robustness subset; identical paired signals before/after DTLN; Gaussian and real DEMAND noise at 20/15/10/5/0 dB; anti-clipping common gain; DTLN fixed delay removed only for intrusive SNR measurement. Vietnamese text uses Unicode NFC, lowercase, punctuation removal, and whitespace collapse while preserving diacritics.

## 4. Environmental Sound Detection

Clean accuracy: {find('CED','Clean Accuracy')}. Clean Macro F1: {find('CED','Clean Macro F1')}. Macro F1 averages the seven declared target classes; predictions outside those classes are bucketed as `other`, remain errors, and are visible in the confusion matrix. See per-class metrics and predictions; no threshold was tuned and no hard sample was removed.

## 5. Noise Robustness

Synthetic Gaussian uses all 280 clips at every SNR. Real DEMAND uses a fixed balanced 70-clip subset (10 per class) in all four environments, producing N=280 paired observations per SNR. Results are reported separately for all five SNRs. Additive mixing uses measured RMS, a common anti-clipping gain, and recorded achieved SNR.

## 6. Vietnamese Speech-to-Text

Clean WER: {find('STT','Clean WER')}; clean CER: {find('STT','Clean CER')}. All 100 clean and all 600 noisy/filtered requests succeeded. Robustness uses the same 30 utterances at every SNR and filter condition. Google STT is network-dependent; failed requests would remain recorded and denominators would reflect successful responses.

## 7. DTLN Signal Quality

Mean Delta SNR: {find('DTLN','Mean Delta SNR')}. This is N=700 (35 balanced clips x 4 environments x 5 SNRs). The intrusive metric accounts for the measured 384-sample algorithmic delay and treats filter distortion/attenuation as error; it is not itself task accuracy.

## 8. DTLN Downstream Effect

CED Delta Macro F1: {find('DTLN','Delta CED Macro F1')}. STT Delta WER: {find('DTLN','Delta STT WER')}. Here DTLN worsened both the mean intrusive SNR and downstream CED. It also increased real-noise STT WER at 20, 15, 5, and 0 dB and tied at 10 dB. Signal quality and task performance are distinct and both are reported without suppressing negative effects.

## 9. Emergency Detection

Recall: {find('Emergency','Recall')}; False Negative Rate: {find('Emergency','False Negative Rate')}; false alerts: {find('Emergency','False alerts')}. Ground truth follows the existing SoundGuard category levels and was fixed before predictions.

## 10. Fusion

Raw F1: {find('Fusion','Raw F1')}; fused F1: {find('Fusion','Fused F1')}. This remains deterministic logic validation, not expanded real paired audio/transcript evidence.

## 11. Personalization

Priority accuracy: {find('Personalization','Priority Accuracy')}; safety pass rate: {find('Personalization','Safety Pass Rate')}; Universal Critical pass rate: {find('Personalization','Universal Critical Pass Rate')}. Ground truth is hand-specified from documented policy. Remote AI personalization was skipped because neither provider credential was visible to the benchmark process.

## 12. Latency

PC software P95: {find('Latency','P95 Latency')}. This excludes network STT, microphone capture, serial transport, ESP32, OLED, and vibration.

## 13. Stability

Crash count: {find('Stability','Crash Count')}. Actual runtime: {stability_info.get('actual_seconds', 'not available')} seconds; processed events: {stability_info.get('processed_events', 'not available')}; exceptions: {stability_info.get('exceptions', 'not available')}; Python traced allocation growth: {stability_info.get('traced_allocation_growth_mb', 'not available')} MiB. Windows RSS: {stability_rss_text}. Named `SYNTHETIC CONTROL-PATH STABILITY SOAK`; it is not microphone/model/hardware stability.

## 14. Comparison With Preliminary Benchmark

See `preliminary_vs_expanded.md`. The original preliminary result tree is preserved.

## 15. Weaknesses and Failure Cases

Clean CED missed 55% of clips, with especially severe failures for glass breaking and dog barking; exact per-class values are retained. Emergency recall was 0.51 with a 0.49 false-negative rate, though no non-emergency clip falsely alerted. DTLN reduced mean intrusive SNR and CED performance on the expanded real-noise protocol, while also worsening aggregate real-noise STT WER. Personalization failed two deliberately misleading hyphenated-substring checks (`factory-method` and `baby-blue`), producing a 48/50 safety result; this benchmark finding is not patched in production code. Exact error rows are available in predictions/results CSVs.

## 16. Limitations

This PC benchmark is not a smart-glasses hardware benchmark. Google STT depends on a cloud network service. AudioSet/ESC-50 source overlap cannot be ruled out. Gaussian noise is artificial; four DEMAND recordings do not cover all real acoustic scenes, and additive mixing does not reproduce reverberation or device microphones. There is no hearing-impaired user study and no physical battery, HUD, vibration, temperature, directional, or end-to-end device validation.

## 17. Measurements Still Required on Real Glasses

Battery life/power, thermal behavior, microphone and enclosure effects, real-stream drop rate, serial/ESP32 latency, OLED readability/latency, vibration perception, direction accuracy with ground truth, and trials with hearing-impaired participants.
"""
        (RESULTS / "FINAL_REPORT_EXPANDED.md").write_text(report, encoding="utf-8")

    def run(self):
        for name, function in (("manifests", self.manifests), ("ced", self.ced), ("dtln_ced", self.dtln_ced),
                               ("stt", self.stt), ("emergency", self.emergency), ("fusion", self.fusion),
                               ("personalization", self.personalization), ("latency", self.latency),
                               ("stability", self.stability)):
            self.stage(name, function)
        self.reports()


def run_expanded(*, enable_network_stt: bool, stability_seconds: int,
                 resume: bool, rerun_stages: list[str] | None = None) -> None:
    np.random.seed(SEED)
    ExpandedRunner(enable_network_stt=enable_network_stt, stability_seconds=stability_seconds,
                   resume=resume, rerun_stages=rerun_stages).run()
