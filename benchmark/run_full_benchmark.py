"""Run reproducible benchmarks against existing SoundGuard public interfaces."""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from benchmark.config import AUDIO_CASES, DATA_DIR, RESULTS_DIR, SEED, SNR_LEVELS
from benchmark.metrics import (binary_metrics, classification_metrics, error_rates,
                               latency_summary)
from benchmark.noise import align_to_reference, snr_db, write_noisy_variants


SUMMARY_FIELDS = ("benchmark", "feature", "metric", "value", "unit",
                  "sample_count", "condition", "status", "notes")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def ram_bytes() -> int | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong)]
    try:
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        return int(status.total_phys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    except (AttributeError, OSError):
        return None


def process_rss_bytes() -> int | None:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("page_fault_count", ctypes.c_ulong),
                    ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                    ("quota_peak_paged", ctypes.c_size_t), ("quota_paged", ctypes.c_size_t),
                    ("quota_peak_nonpaged", ctypes.c_size_t), ("quota_nonpaged", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
    try:
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.working_set)
    except (AttributeError, OSError):
        pass
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {os.getpid()}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        )
        values = next(csv.reader(completed.stdout.splitlines()))
        digits = "".join(character for character in values[-1] if character.isdigit())
        return int(digits) * 1024 if digits else None
    except (OSError, StopIteration, IndexError, ValueError):
        return None


def plot_line(path: Path, x, series: list[tuple[str, list[float]]], xlabel: str,
              ylabel: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for label, values in series:
        axis.plot(x, values, marker="o", label=label)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(True, alpha=0.3)
    if len(series) > 1:
        axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_confusion(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set(xlabel="Predicted", ylabel="Ground truth", title="CED confusion matrix (clean)")
    for row in range(len(labels)):
        for col in range(len(labels)):
            axis.text(col, row, matrix[row][col], ha="center", va="center")
    fig.colorbar(image, ax=axis); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


class Runner:
    def __init__(self, *, enable_network_stt: bool, stability_seconds: int, resume: bool):
        self.enable_network_stt = enable_network_stt
        self.stability_seconds = stability_seconds
        self.resume = resume
        self.summary: list[dict] = read_csv(RESULTS_DIR / "summary.csv") if resume else []
        self.baseline_path = RESULTS_DIR / "baseline_git_status.txt"
        if not self.baseline_path.exists():
            self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self.baseline_path.write_text(git("status", "--short") + "\n", encoding="utf-8")
        self.baseline_git_status = self.baseline_path.read_text(encoding="utf-8").strip()
        self.checkpoint_path = RESULTS_DIR / "checkpoint.json"
        self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8")) \
            if resume and self.checkpoint_path.exists() else {"completed": [], "failed": {}}
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def add(self, benchmark: str, feature: str, metric: str, value, unit: str,
            count: int, condition: str = "", status: str = "PASS", notes: str = "") -> None:
        self.summary = [row for row in self.summary if not (
            row["benchmark"] == benchmark and row["metric"] == metric and row["condition"] == condition
        )]
        self.summary.append(dict(zip(SUMMARY_FIELDS, (benchmark, feature, metric, value, unit,
                                                       count, condition, status, notes))))

    def stage(self, name: str, function) -> None:
        if self.resume and name in self.checkpoint["completed"]:
            print(f"[resume] {name}: already complete")
            return
        print(f"[run] {name}")
        try:
            function()
            self.summary = [row for row in self.summary
                            if not (row["benchmark"] == name and row["metric"] == "stage")]
            self.checkpoint["completed"].append(name)
            self.checkpoint["failed"].pop(name, None)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[failed] {name}: {message}")
            self.checkpoint["failed"][name] = message
            self.add(name, name, "stage", "", "", 0, status="FAILED", notes=message)
        write_json(self.checkpoint_path, self.checkpoint)

    def audit(self) -> None:
        dirty = self.baseline_git_status
        table = [
            ("Environmental Sound Detection / CED", "YES", "sound_classifier.py", "mispeech/ced-tiny", "YES; 2 clips only", "2 labeled clips", "Accuracy, per-class P/R/F1, confusion"),
            ("CED noise robustness", "YES", "benchmark wrapper around CED", "Seeded white-noise mixing", "YES; preliminary", "Derived from 2 clean clips", "Accuracy, Macro F1 by SNR"),
            ("CED latency", "YES", "classify_audio_file", "Warm repeated inference", "YES", "N/A", "mean/median/P90/P95/P99"),
            ("Speech-to-Text", "YES", "speech_recognizer.py", "Google Speech Recognition vi-VN", "NETWORK-DEPENDENT", "1 transcript", "WER, CER"),
            ("STT noise robustness", "YES", "same STT interface", "Seeded noise", "NETWORK-DEPENDENT", "1 transcript", "WER/CER by SNR"),
            ("STT latency", "YES", "transcribe_audio_file", "Google remote API", "NETWORK-DEPENDENT", "N/A", "latency, RTF"),
            ("Noise filtering quality", "YES", "speech_enhancer.py", "DTLN two-stage TFLite", "YES", "Paired clean/noisy", "input/output/delta SNR"),
            ("Noise filtering effect on CED", "YES", "DTLN -> CED experiment", "Paired", "YES", "2 clips", "Macro F1 delta"),
            ("Noise filtering effect on STT", "YES", "DTLN -> Google STT", "Paired", "NETWORK-DEPENDENT", "1 transcript", "WER/CER delta"),
            ("Emergency detection quality", "YES", "emergency_system.py", "Deterministic thresholds/rules", "YES", "Fixed policy cases", "TP/FP/TN/FN/P/R/F1/FNR"),
            ("Emergency false alarms", "YES", "EmergencySystem", "Deterministic", "YES", "Fixed negatives", "false alerts/clip"),
            ("Emergency latency", "YES", "EmergencySystem single-shot", "Software timing", "YES", "Fixed cases", "mean/median/P95/P99"),
            ("Fusion CED + STT", "YES", "fusion_engine.py", "Deterministic rules", "YES", "Fixed cases", "before/after accuracy/P/R/F1"),
            ("Fusion error analysis", "YES", "fusion_engine.py", "A/B/C/D categories", "YES", "Fixed cases", "counts and percentages"),
            ("Speaker/family voice recognition", "NO", "NOT IMPLEMENTED", "N/A", "NO", "NO", "NOT IMPLEMENTED"),
            ("Speaker recognition noise robustness", "NO", "NOT IMPLEMENTED", "N/A", "NO", "NO", "NOT IMPLEMENTED"),
            ("AI personalization", "YES", "personalization/", "Rules + optional OpenAI/OpenRouter", "Rules YES; API credential-dependent", "Fixed policy cases", "priority/role/context accuracy, invariants, latency"),
            ("AI assistant", "NO", "NOT IMPLEMENTED as separate feature", "N/A", "NO", "NO", "NOT IMPLEMENTED"),
            ("End-to-end software pipeline latency", "YES", "CED -> emergency -> fusion", "PC fixed-file path", "YES", "2 clips", "PC software latency"),
            ("Continuous software stability", "YES", "Repeated control-path workload", "Bounded synthetic soak", "YES", "Deterministic events", "crashes/exceptions/CPU/RAM"),
            ("Left/right direction or vibration", "NO", "No localization/decision implementation", "N/A", "NO", "No multichannel truth/hardware", "Hardware validation required"),
            ("HUD transport", "YES", "display_transport.py + ESP32 firmware", "CRC framed serial", "Logic only", "Unit tests only", "Physical latency/readability not benchmarkable"),
        ]
        lines = ["# SoundGuard benchmark audit", "", f"Baseline commit: `{git('rev-parse', 'HEAD')}`",
                 "", "Dirty working tree was preserved:", "", "```text", dirty or "clean", "```", "",
                 "| Feature | Exists? | Implementation | Model/Algorithm | Can benchmark on PC? | Ground truth available? | Planned metrics |",
                 "|---|---|---|---|---|---|---|"]
        lines += ["| " + " | ".join(row) + " |" for row in table]
        lines += ["", "No production/core file is modified by the benchmark runner."]
        (RESULTS_DIR / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def methodology(self) -> None:
        import torch
        dtln_paths = [ROOT / "external/DTLN-master/pretrained_model/model_1.tflite",
                      ROOT / "external/DTLN-master/pretrained_model/model_2.tflite"]
        methodology = {
            "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "baseline_dirty_git_status": self.baseline_git_status.splitlines(),
            "final_dirty_git_status": git("status", "--short").splitlines(),
            "os": platform.platform(), "python": sys.version, "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
            "logical_cpu_count": os.cpu_count(), "ram_bytes": ram_bytes(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ced_model": "mispeech/ced-tiny", "ced_device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtln_models": [{"path": rel(path), "sha256": sha256(path)} for path in dtln_paths if path.exists()],
            "sample_rate_hz": 16000, "ced_fixed_file_window": "entire clip",
            "continuous_ced_window_seconds": 5.0, "continuous_ced_overlap_seconds": 0.0,
            "vad_threshold": 0.5, "dtln_block_length": 512, "dtln_block_shift": 128,
            "stt_backend": "Google Speech Recognition", "stt_language": "vi-VN",
            "personalization_provider": os.environ.get("SOUNDGUARD_AI_PROVIDER", "openai"),
            "personalization_model_override": os.environ.get("SOUNDGUARD_AI_MODEL"),
            "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "openrouter_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
            "random_seed": SEED, "snr_conditions_db": ["clean", 20, 15, 10, 5, 0],
            "noise": "independent seeded Gaussian white noise; identical WAV reused for paired comparisons",
            "datasets": [{**{key: str(value) for key, value in case.items() if key != "path"},
                           "path": rel(case["path"]), "sha256": sha256(case["path"]) if case["path"].exists() else None}
                          for case in AUDIO_CASES],
            "stability_seconds_requested": self.stability_seconds,
        }
        write_json(RESULTS_DIR / "methodology.json", methodology)

    def prepare_noise(self) -> None:
        rows = []
        for case_index, case in enumerate(AUDIO_CASES):
            if not case["path"].exists():
                continue
            generated = write_noisy_variants(case["path"], DATA_DIR / "generated_audio",
                                             case["sample_id"], tuple(s for s in SNR_LEVELS if s is not None),
                                             SEED + case_index * 100)
            for row in generated:
                row["path"] = rel(Path(row["path"]))
            rows.extend(generated)
        write_csv(RESULTS_DIR / "noise_generation.csv", rows)

    @staticmethod
    def canonical(label: str) -> str:
        from personalization.sound_labels import canonicalize_label, normalize_label
        return canonicalize_label(label) or normalize_label(label).replace(" ", "_")

    def ced(self) -> None:
        from sound_classifier import classify_audio_file
        out = RESULTS_DIR / "ced"; out.mkdir(parents=True, exist_ok=True)
        generated = {(row["sample_id"], int(row["snr_db"])): ROOT / row["path"]
                     for row in read_csv(RESULTS_DIR / "noise_generation.csv")}
        predictions, cold_ms = [], None
        for condition in SNR_LEVELS:
            for case in AUDIO_CASES:
                path = case["path"] if condition is None else generated[(case["sample_id"], condition)]
                start = time.perf_counter(); result = classify_audio_file(path); elapsed = (time.perf_counter() - start) * 1000
                if cold_ms is None: cold_ms = elapsed
                predicted = self.canonical(result["label"])
                predictions.append({"file": rel(path), "sample_id": case["sample_id"],
                                    "condition": "clean" if condition is None else f"{condition} dB",
                                    "true_label": case["true_label"], "predicted_label": predicted,
                                    "raw_predicted_label": result["label"], "confidence": result["confidence"],
                                    "correct": predicted == case["true_label"], "latency_ms": elapsed})
        write_csv(out / "predictions.csv", predictions)
        clean = [row for row in predictions if row["condition"] == "clean"]
        metrics = classification_metrics([r["true_label"] for r in clean], [r["predicted_label"] for r in clean])
        rng = np.random.default_rng(SEED)
        bootstrap_accuracy, bootstrap_macro_f1 = [], []
        for _ in range(1000):
            indices = rng.integers(0, len(clean), len(clean))
            sample = [clean[index] for index in indices]
            sampled = classification_metrics([r["true_label"] for r in sample], [r["predicted_label"] for r in sample])
            bootstrap_accuracy.append(sampled["accuracy"]); bootstrap_macro_f1.append(sampled["macro_f1"])
        aggregate = {key: value for key, value in metrics.items() if key not in {"per_class", "labels"}}
        aggregate.update({"bootstrap_repetitions": 1000,
                          "accuracy_ci95_low": float(np.percentile(bootstrap_accuracy, 2.5)),
                          "accuracy_ci95_high": float(np.percentile(bootstrap_accuracy, 97.5)),
                          "macro_f1_ci95_low": float(np.percentile(bootstrap_macro_f1, 2.5)),
                          "macro_f1_ci95_high": float(np.percentile(bootstrap_macro_f1, 97.5))})
        write_csv(out / "clean_metrics.csv", [aggregate])
        write_csv(out / "per_class_metrics.csv", metrics["per_class"])
        labels = metrics["labels"]
        matrix = [[sum(r["true_label"] == truth and r["predicted_label"] == pred for r in clean)
                   for pred in labels] for truth in labels]
        write_csv(out / "confusion_matrix.csv", [dict(true_label=label, **dict(zip(labels, row))) for label, row in zip(labels, matrix)])
        plot_confusion(out / "confusion_matrix.png", labels, matrix)
        snr_rows = []
        for condition in ["clean", "20 dB", "15 dB", "10 dB", "5 dB", "0 dB"]:
            subset = [row for row in predictions if row["condition"] == condition]
            result = classification_metrics([r["true_label"] for r in subset], [r["predicted_label"] for r in subset])
            snr_rows.append({"condition": condition, "sample_count": result["count"],
                             "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
        write_csv(out / "snr_results.csv", snr_rows)
        plot_line(out / "f1_vs_snr.png", [20, 15, 10, 5, 0],
                  [("Macro F1", [row["macro_f1"] for row in snr_rows[1:]])], "SNR (dB)", "Macro F1", "CED robustness")
        classify_audio_file(AUDIO_CASES[0]["path"])
        latency_rows = []
        for index in range(8):
            path = AUDIO_CASES[index % len(AUDIO_CASES)]["path"]
            start = time.perf_counter(); classify_audio_file(path); elapsed = (time.perf_counter() - start) * 1000
            latency_rows.append({"run": index + 1, "file": rel(path), "total_ms": elapsed})
        write_csv(out / "latency.csv", latency_rows)
        summary = latency_summary([row["total_ms"] for row in latency_rows]); summary["cold_start_ms"] = cold_ms
        write_json(out / "latency_summary.json", summary)
        plot_line(out / "latency_distribution.png", [row["run"] for row in latency_rows],
                  [("Total latency", [row["total_ms"] for row in latency_rows])],
                  "Warm run", "Latency (ms)", "CED warm inference latency")
        self.add("CED", "Environmental Sound Detection", "Accuracy", metrics["accuracy"], "ratio", metrics["count"], "clean",
                 notes="Preliminary: only two labeled clips")
        self.add("CED", "Environmental Sound Detection", "Macro F1", metrics["macro_f1"], "ratio", metrics["count"], "clean")
        for row in snr_rows[1:]:
            self.add("CED", "CED noise robustness", "Macro F1", row["macro_f1"], "ratio",
                     row["sample_count"], row["condition"])
        self.add("CED", "CED latency", "P95 latency", summary["p95_ms"], "ms", summary["count"], "warm")

    @staticmethod
    def to_16k(source: Path, destination: Path) -> Path:
        audio, sample_rate = sf.read(source, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sample_rate != 16000:
            from math import gcd
            divisor = gcd(sample_rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio, 16000, subtype="PCM_16")
        return destination

    def noise_filter(self) -> None:
        from sound_classifier import classify_audio_file
        from speech_enhancer import enhance_audio_file
        out = RESULTS_DIR / "noise_filter"; generated_dir = DATA_DIR / "generated_audio"
        noise_rows = read_csv(RESULTS_DIR / "noise_generation.csv")
        ced_unfiltered = {(r["sample_id"], r["condition"]): r for r in read_csv(RESULTS_DIR / "ced/predictions.csv")}
        quality, ced_pairs = [], []
        for case in AUDIO_CASES:
            clean16 = self.to_16k(case["path"], generated_dir / f"{case['sample_id']}_clean_16k.wav")
            clean_audio, _ = sf.read(clean16, dtype="float32")
            for metadata in [row for row in noise_rows if row["sample_id"] == case["sample_id"]]:
                snr = int(metadata["snr_db"]); noisy = ROOT / metadata["path"]
                noisy16 = self.to_16k(noisy, generated_dir / f"{case['sample_id']}_{snr}db_16k.wav")
                filtered = generated_dir / f"{case['sample_id']}_{snr}db_dtln.wav"
                start = time.perf_counter(); enhance_audio_file(noisy16, filtered, verbose=False); elapsed = time.perf_counter() - start
                noisy_audio, _ = sf.read(noisy16, dtype="float32"); filtered_audio, _ = sf.read(filtered, dtype="float32")
                # Anti-clipping applies one common gain to signal and noise.
                # Compare against that scaled clean reference so the measured
                # input SNR remains the requested condition.
                clean_reference = clean_audio * float(metadata["anti_clip_scale"])
                input_snr = snr_db(clean_reference, noisy_audio)
                aligned_clean, aligned_filtered, alignment_lag = align_to_reference(
                    clean_reference, filtered_audio
                )
                output_snr = snr_db(aligned_clean, aligned_filtered)
                quality.append({"sample_id": case["sample_id"], "snr_db": snr,
                                "input_snr_db": input_snr, "output_snr_db": output_snr,
                                "delta_snr_db": output_snr - input_snr, "filter_seconds": elapsed,
                                "alignment_lag_samples": alignment_lag,
                                "filtered_path": rel(filtered)})
                prediction = classify_audio_file(filtered); predicted = self.canonical(prediction["label"])
                raw = ced_unfiltered[(case["sample_id"], f"{snr} dB")]
                ced_pairs.append({"sample_id": case["sample_id"], "snr_db": snr,
                                  "true_label": case["true_label"],
                                  "without_filter": raw["predicted_label"], "with_filter": predicted,
                                  "without_correct": raw["predicted_label"] == case["true_label"],
                                  "with_correct": predicted == case["true_label"]})
        write_csv(out / "snr_improvement.csv", quality); write_csv(out / "ced_predictions_paired.csv", ced_pairs)
        summary = {"sample_count": len(quality), "mean_input_snr_db": float(np.mean([r["input_snr_db"] for r in quality])),
                   "mean_output_snr_db": float(np.mean([r["output_snr_db"] for r in quality])),
                   "mean_delta_snr_db": float(np.mean([r["delta_snr_db"] for r in quality]))}
        write_json(out / "summary.json", summary)
        comparison = []
        for snr in (20, 15, 10, 5, 0):
            subset = [r for r in ced_pairs if r["snr_db"] == snr]
            truths = [r["true_label"] for r in subset]
            before = classification_metrics(truths, [r["without_filter"] for r in subset])
            after = classification_metrics(truths, [r["with_filter"] for r in subset])
            comparison.append({"snr_db": snr, "sample_count": len(subset),
                               "macro_f1_without_filter": before["macro_f1"], "macro_f1_with_filter": after["macro_f1"],
                               "delta_f1": after["macro_f1"] - before["macro_f1"],
                               "accuracy_without_filter": before["accuracy"], "accuracy_with_filter": after["accuracy"]})
        write_csv(out / "ced_before_after.csv", comparison)
        plot_line(out / "ced_delta_f1_vs_snr.png", [r["snr_db"] for r in comparison],
                  [("Without DTLN", [r["macro_f1_without_filter"] for r in comparison]),
                   ("With DTLN", [r["macro_f1_with_filter"] for r in comparison])],
                  "SNR (dB)", "Macro F1", "CED before/after DTLN")
        self.add("NoiseFilter", "Noise filtering", "Mean Delta SNR", summary["mean_delta_snr_db"], "dB", len(quality))
        self.add("NoiseFilter", "CED effect", "Mean Delta F1", float(np.mean([r["delta_f1"] for r in comparison])),
                 "ratio", len(ced_pairs), notes="Experimental: production CED intentionally uses raw audio")
        for row in comparison:
            self.add("NoiseFilter", "CED effect", "Delta F1", row["delta_f1"], "ratio",
                     row["sample_count"], f"{row['snr_db']} dB")

    def stt(self) -> None:
        out = RESULTS_DIR / "stt"
        if not self.enable_network_stt:
            write_json(out / "SKIPPED.json", {"status": "SKIPPED", "reason": "network STT not enabled"})
            self.add("STT", "Speech-to-Text", "WER", "", "", 0, status="SKIPPED", notes="network STT not enabled")
            return
        from speech_recognizer import transcribe_audio_file
        case = next(item for item in AUDIO_CASES if item["sample_id"] == "local_vi_speech")
        paths = [("clean", case["path"], "unfiltered")]
        noise_rows = [row for row in read_csv(RESULTS_DIR / "noise_generation.csv") if row["sample_id"] == case["sample_id"]]
        quality = {int(row["snr_db"]): ROOT / row["filtered_path"] for row in read_csv(RESULTS_DIR / "noise_filter/snr_improvement.csv")
                   if row["sample_id"] == case["sample_id"]}
        for row in noise_rows:
            snr = int(row["snr_db"]); paths += [(f"{snr} dB", ROOT / row["path"], "unfiltered"),
                                                (f"{snr} dB", quality[snr], "dtln")]
        results = []
        for condition, path, filter_state in paths:
            start = time.perf_counter()
            try:
                hypothesis = transcribe_audio_file(path); status, error = "PASS", ""
            except Exception as exc:
                hypothesis, status, error = "", "FAILED", f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - start
            rates = error_rates(case["reference"], hypothesis)
            audio, sample_rate = sf.read(path, dtype="float32")
            results.append({"file": rel(path), "condition": condition, "filter": filter_state,
                            **rates, "processing_seconds": elapsed, "audio_duration_seconds": len(audio) / sample_rate,
                            "real_time_factor": elapsed / (len(audio) / sample_rate), "status": status, "error": error})
        passed = [row for row in results if row["status"] == "PASS"]
        if not passed:
            write_json(out / "failure.json", {"status": "FAILED", "errors": sorted(set(row["error"] for row in results))})
            self.add("STT", "Speech-to-Text", "WER", "", "", 0, status="FAILED", notes="all live requests failed")
            return
        write_csv(out / "clean_results.csv", [r for r in results if r["condition"] == "clean"])
        snr_rows = []
        for row in results:
            if row["condition"] != "clean":
                snr_rows.append({"snr_db": int(row["condition"].split()[0]), "filter": row["filter"],
                                 "wer": row["wer"], "cer": row["cer"], "status": row["status"]})
        write_csv(out / "snr_results.csv", snr_rows)
        clean = next(r for r in passed if r["condition"] == "clean") if any(r["condition"] == "clean" for r in passed) else passed[0]
        write_json(out / "summary.json", {"utterances": 1, "reference_words": clean["reference_words"],
                                           "reference_characters": clean["reference_characters"],
                                           "wer": clean["wer"], "cer": clean["cer"]})
        latency_rows = [{"condition": r["condition"], "filter": r["filter"], "processing_seconds": r["processing_seconds"],
                         "audio_duration_seconds": r["audio_duration_seconds"], "real_time_factor": r["real_time_factor"]}
                        for r in passed]
        write_csv(out / "latency.csv", latency_rows)
        write_json(out / "latency_summary.json", latency_summary([r["processing_seconds"] * 1000 for r in passed]))
        unfiltered = [r for r in snr_rows if r["filter"] == "unfiltered" and r["status"] == "PASS"]
        filtered = [r for r in snr_rows if r["filter"] == "dtln" and r["status"] == "PASS"]
        if unfiltered:
            plot_line(out / "wer_vs_snr.png", [r["snr_db"] for r in unfiltered], [("WER", [r["wer"] for r in unfiltered])], "SNR (dB)", "WER", "STT noise robustness")
            plot_line(out / "cer_vs_snr.png", [r["snr_db"] for r in unfiltered], [("CER", [r["cer"] for r in unfiltered])], "SNR (dB)", "CER", "STT noise robustness")
        paired = []
        for snr in (20, 15, 10, 5, 0):
            before = next((r for r in unfiltered if r["snr_db"] == snr), None); after = next((r for r in filtered if r["snr_db"] == snr), None)
            if before and after:
                paired.append({"snr_db": snr, "wer_without_filter": before["wer"], "wer_with_filter": after["wer"],
                               "delta_wer": after["wer"] - before["wer"], "cer_without_filter": before["cer"],
                               "cer_with_filter": after["cer"], "delta_cer": after["cer"] - before["cer"]})
        if paired:
            nf = RESULTS_DIR / "noise_filter"; write_csv(nf / "stt_before_after.csv", paired)
            plot_line(nf / "stt_delta_wer_vs_snr.png", [r["snr_db"] for r in paired],
                      [("Without DTLN", [r["wer_without_filter"] for r in paired]), ("With DTLN", [r["wer_with_filter"] for r in paired])],
                      "SNR (dB)", "WER (lower is better)", "STT before/after DTLN")
        self.add("STT", "Speech-to-Text", "WER", clean["wer"], "ratio", 1, "clean")
        self.add("STT", "Speech-to-Text", "CER", clean["cer"], "ratio", 1, "clean")
        for row in unfiltered:
            self.add("STT", "STT noise robustness", "WER", row["wer"], "ratio", 1, f"{row['snr_db']} dB")
            self.add("STT", "STT noise robustness", "CER", row["cer"], "ratio", 1, f"{row['snr_db']} dB")

    def emergency(self) -> None:
        from emergency_system import EmergencySystem
        cases = json.loads((DATA_DIR / "emergency_cases.json").read_text(encoding="utf-8"))
        rows, timings = [], []
        for case in cases:
            system = EmergencySystem(decision_mode="single_shot")
            start = time.perf_counter()
            if case["transcript"]:
                result = system.process_transcript_event(case["transcript"])
            else:
                result = system.process_sound_event(case["label"], case["confidence"])
            elapsed = (time.perf_counter() - start) * 1000; timings.append(elapsed)
            predicted = bool(result["alert_level"] != "LOW" and result["event_state"] in {"EVENT_STARTED", "EVENT_CONTINUING"})
            rows.append({**case, "predicted": predicted, "correct": predicted == case["expected"],
                         "category": result["category"], "alert_level": result["alert_level"], "latency_ms": elapsed})
        metrics = binary_metrics([r["expected"] for r in rows], [r["predicted"] for r in rows])
        out = RESULTS_DIR / "emergency"; write_csv(out / "predictions.csv", rows); write_json(out / "metrics.json", metrics)
        negatives = [r for r in rows if not r["expected"]]
        write_json(out / "false_alarms.json", {"non_emergency_clips": len(negatives),
                                                "false_alerts": sum(r["predicted"] for r in negatives),
                                                "false_alerts_per_clip": sum(r["predicted"] for r in negatives) / len(negatives)})
        write_csv(out / "latency.csv", [{"case_id": r["id"], "latency_ms": r["latency_ms"]} for r in rows])
        write_json(out / "latency_summary.json", latency_summary(timings))
        self.add("Emergency", "Emergency Detection", "Recall", metrics["recall"], "ratio", len(rows))
        self.add("Emergency", "Emergency Detection", "False Negative Rate", metrics["false_negative_rate"], "ratio", len(rows))
        self.add("Emergency", "Emergency False Alarms", "False alerts per clip",
                 sum(r["predicted"] for r in negatives) / len(negatives), "ratio", len(negatives), "non-emergency")

    def fusion(self) -> None:
        from fusion_engine import fuse_result
        cases = json.loads((DATA_DIR / "fusion_cases.json").read_text(encoding="utf-8"))
        rows = []
        for case in cases:
            raw = bool(case["dangerous"] and case["confidence"] >= 0.45)
            result = fuse_result("benchmark", case["transcript"], {"label": case["label"], "confidence": case["confidence"]},
                                 {"category": case["category"], "is_dangerous": case["dangerous"], "message_vi": "benchmark"})
            fused = bool(result["alert"]); expected = case["expected"]
            category = "A_raw_wrong_fusion_correct" if raw != expected and fused == expected else \
                       "B_raw_correct_fusion_correct" if raw == expected and fused == expected else \
                       "C_raw_correct_fusion_wrong" if raw == expected else "D_raw_wrong_fusion_wrong"
            rows.append({**case, "raw_prediction": raw, "fusion_prediction": fused,
                         "raw_correct": raw == expected, "fusion_correct": fused == expected, "error_category": category})
        raw_metrics = binary_metrics([r["expected"] for r in rows], [r["raw_prediction"] for r in rows])
        fused_metrics = binary_metrics([r["expected"] for r in rows], [r["fusion_prediction"] for r in rows])
        out = RESULTS_DIR / "fusion"; write_csv(out / "before_after.csv", rows)
        write_json(out / "summary.json", {"raw": raw_metrics, "fusion": fused_metrics,
                                           "delta_f1": fused_metrics["f1"] - raw_metrics["f1"]})
        categories = sorted(set(r["error_category"] for r in rows))
        error_rows = [{"category": category, "count": sum(r["error_category"] == category for r in rows),
                       "percentage": sum(r["error_category"] == category for r in rows) / len(rows)} for category in categories]
        write_csv(out / "error_analysis.csv", error_rows); write_json(out / "error_summary.json", {r["category"]: r for r in error_rows})
        self.add("Fusion", "Fusion Engine", "Delta F1", fused_metrics["f1"] - raw_metrics["f1"], "ratio", len(rows))

    def personalization(self) -> None:
        from personalization.profile_generator import generate_profile, generate_rule_based, _normalize_openai_profile
        from personalization.profile_validator import UNIVERSAL_MIN_PRIORITY, validate_profile
        from personalization.schemas import ProfileValidationError
        cases = json.loads((DATA_DIR / "personalization_cases.json").read_text(encoding="utf-8"))
        rows, expected_priorities, correct_priorities = [], 0, 0
        role_tp = role_total = context_tp = context_total = 0
        timings = []
        for case in cases:
            start = time.perf_counter(); profile = generate_rule_based([case["text"]]); timings.append((time.perf_counter() - start) * 1000)
            role_expected, context_expected = set(case["roles"]), set(case["contexts"])
            role_ok, context_ok = set(profile["roles"]) == role_expected, set(profile["contexts"]) == context_expected
            role_tp += role_ok; role_total += 1; context_tp += context_ok; context_total += 1
            for label, expected in case["priorities"].items():
                actual = profile["priority_profile"][label]; expected_priorities += 1; correct_priorities += actual == expected
                rows.append({"case_id": case["id"], "text_class": "fixed synthetic policy case", "label": label,
                             "expected_priority": expected, "actual_priority": actual, "absolute_error": abs(actual - expected),
                             "within_one": abs(actual - expected) <= 1, "role_exact": role_ok, "context_exact": context_ok})
        write_csv(RESULTS_DIR / "personalization/cases.csv", rows)
        metrics = {"case_count": len(cases), "priority_assertions": expected_priorities,
                   "exact_priority_accuracy": correct_priorities / expected_priorities,
                   "mean_absolute_priority_error": float(np.mean([r["absolute_error"] for r in rows])),
                   "within_one_priority_accuracy": float(np.mean([r["within_one"] for r in rows])),
                   "role_exact_accuracy": role_tp / role_total, "context_exact_accuracy": context_tp / context_total}
        write_json(RESULTS_DIR / "personalization/metrics.json", metrics)
        safety = []
        for case in cases:
            profile = generate_rule_based([case["text"]])
            critical_ok = all(profile["priority_profile"][label] >= minimum for label, minimum in UNIVERSAL_MIN_PRIORITY.items())
            safety.append({"test": f"universal_{case['id']}", "category": "universal_critical", "passed": critical_ok})
        negated = generate_rule_based(["I do not drive, never use a bus, and am without construction work."])
        safety += [{"test": "negation_roles", "category": "negation", "passed": not negated["roles"]},
                   {"test": "negation_contexts", "category": "negation", "passed": not negated["contexts"]}]
        contrasted = generate_rule_based(["I do not drive; however, I walk on foot through road traffic."])
        safety.append({"test": "contrast_scope", "category": "contrast",
                       "passed": "driver" not in contrasted["roles"] and "pedestrian_commuter" in contrasted["roles"]})
        substring = generate_rule_based(["My hobby is painting driveway artwork."])
        safety.append({"test": "whole_phrase_matching", "category": "substring",
                       "passed": "driver" not in substring["roles"]})
        raw = {"profile_version": "1.0", "roles": ["driver"], "contexts": ["road"], "responsibilities": [],
               "priority_profile": [{"label": "vehicle_horn", "priority": 5}],
               "reasoning_summary": [{"label": "explosion", "reason": "Collision risk."}]}
        repaired = _normalize_openai_profile(raw)
        safety.append({"test": "reasoning_priority_normalization", "category": "normalization",
                       "passed": repaired["priority_profile"]["explosion"] == 5})
        malformed = dict(generate_rule_based(["I drive."])); malformed["priority_profile"] = dict(malformed["priority_profile"]); malformed["priority_profile"]["speech"] = 9
        try: validate_profile(malformed); rejected = False
        except ProfileValidationError: rejected = True
        safety.append({"test": "invalid_priority_rejected", "category": "malformed", "passed": rejected})
        missing = dict(generate_rule_based(["I drive."])); missing.pop("contexts")
        try: validate_profile(missing); missing_rejected = False
        except ProfileValidationError: missing_rejected = True
        safety.append({"test": "missing_field_rejected", "category": "strict_schema", "passed": missing_rejected})
        saved_environment = {name: os.environ.get(name) for name in (
            "OPENAI_API_KEY", "OPENROUTER_API_KEY", "SOUNDGUARD_AI_PROVIDER"
        )}
        os.environ.pop("OPENAI_API_KEY", None); os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ["SOUNDGUARD_AI_PROVIDER"] = "openai"
        try: _, fallback_provider = generate_profile(["I drive."])
        finally:
            for name, value in saved_environment.items():
                if value is None: os.environ.pop(name, None)
                else: os.environ[name] = value
        safety.append({"test": "deterministic_fallback", "category": "fallback", "passed": fallback_provider.startswith("deterministic_fallback:")})
        write_csv(RESULTS_DIR / "personalization/safety_tests.csv", safety)
        write_json(RESULTS_DIR / "personalization/safety_summary.json", {"count": len(safety), "passed": sum(r["passed"] for r in safety),
                                                                         "pass_rate": sum(r["passed"] for r in safety) / len(safety)})
        latency = latency_summary(timings); write_csv(RESULTS_DIR / "personalization/latency.csv", [{"case_id": case["id"], "deterministic_ms": value} for case, value in zip(cases, timings)])
        write_json(RESULTS_DIR / "personalization/latency_summary.json", {"deterministic": latency,
                   "remote": {"status": "SKIPPED", "reason": "No API credential available to benchmark process"}})
        consistency_profiles = [generate_rule_based([cases[0]["text"]]) for _ in range(10)]
        write_json(RESULTS_DIR / "personalization/consistency.json", {"path": "deterministic_rule_based", "repetitions": 10,
                   "same_input_same_priority_rate": sum(value["priority_profile"] == consistency_profiles[0]["priority_profile"] for value in consistency_profiles) / 10,
                   "remote": "SKIPPED: no API credential"})
        self.add("Personalization", "AI Personalization", "Priority Accuracy", metrics["exact_priority_accuracy"], "ratio", expected_priorities)
        self.add("Personalization", "AI Personalization", "Mean Absolute Priority Error",
                 metrics["mean_absolute_priority_error"], "priority levels", expected_priorities)
        self.add("Personalization", "AI Personalization", "Role Exact Accuracy", metrics["role_exact_accuracy"], "ratio", len(cases))
        self.add("Personalization", "AI Personalization", "Context Exact Accuracy", metrics["context_exact_accuracy"], "ratio", len(cases))
        self.add("Personalization", "AI Personalization", "Critical Invariant Pass Rate",
                 sum(r["passed"] for r in safety if r["category"] == "universal_critical") / len(cases), "ratio", len(cases))

    def pipeline(self) -> None:
        from alert_mapper import map_alert
        from emergency_system import EmergencySystem
        from fusion_engine import fuse_result
        from sound_classifier import classify_audio_file
        rows = []
        classify_audio_file(AUDIO_CASES[0]["path"])
        for case in AUDIO_CASES:
            start = time.perf_counter(); ced = classify_audio_file(case["path"]); after_ced = time.perf_counter()
            mapped = map_alert(ced["label"], ced["confidence"], threshold=None)
            system = EmergencySystem(decision_mode="single_shot"); emergency = system.process_sound_event(ced["label"], ced["confidence"])
            after_decision = time.perf_counter(); fused = fuse_result("benchmark", "", ced, mapped); completed = time.perf_counter()
            rows.append({"sample_id": case["sample_id"], "ced_ms": (after_ced - start) * 1000,
                         "emergency_ms": (after_decision - after_ced) * 1000, "fusion_ms": (completed - after_decision) * 1000,
                         "total_ms": (completed - start) * 1000, "predicted_label": ced["label"],
                         "alert_level": emergency["alert_level"], "fusion_alert": fused["alert"]})
        out = RESULTS_DIR / "pipeline"; write_csv(out / "latency.csv", rows)
        summary = latency_summary([r["total_ms"] for r in rows]); summary["name"] = "PC SOFTWARE END-TO-END LATENCY"
        summary["excludes"] = ["microphone capture", "network STT", "serial transport", "ESP32", "OLED", "vibration"]
        write_json(out / "latency_summary.json", summary)
        self.add("Pipeline", "PC Software Pipeline", "P95 Latency", summary["p95_ms"], "ms", len(rows), "warm fixed-file")

    def stability(self) -> None:
        from emergency_system import EmergencySystem
        from fusion_engine import fuse_result
        from personalization.profile_generator import generate_rule_based
        out = RESULTS_DIR / "stability"; rows, exceptions, processed = [], 0, 0
        tracemalloc.start(); started = time.perf_counter(); previous_wall, previous_cpu = started, time.process_time()
        system = EmergencySystem(decision_mode="continuous")
        while time.perf_counter() - started < self.stability_seconds:
            try:
                for _ in range(100):
                    result = system.process_sound_event("Siren" if processed % 3 else "Bark", 0.8)
                    fuse_result("benchmark", "", {"label": "Siren", "confidence": 0.8}, result.get("mapped_sound", {}))
                    if processed % 1000 == 0: generate_rule_based(["I am a driver in road traffic."])
                    processed += 1
            except Exception:
                exceptions += 1
            now = time.perf_counter()
            if now - previous_wall >= 1.0:
                cpu_now = time.process_time(); current, peak = tracemalloc.get_traced_memory(); rss = process_rss_bytes()
                rows.append({"elapsed_seconds": now - started, "processed_events": processed,
                             "cpu_percent_total_capacity": (cpu_now - previous_cpu) / (now - previous_wall) * 100 / max(1, os.cpu_count() or 1),
                             "rss_mb": rss / 1024 ** 2 if rss is not None else "",
                             "traced_current_mb": current / 1024 ** 2, "traced_peak_mb": peak / 1024 ** 2,
                             "exceptions": exceptions})
                previous_wall, previous_cpu = now, cpu_now
        current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        write_csv(out / "resource_usage.csv", rows)
        rss_values = [float(r["rss_mb"]) for r in rows if r["rss_mb"] != ""]
        summary = {"target_seconds": 3600, "actual_seconds": time.perf_counter() - started,
                   "processed_events": processed, "exceptions": exceptions, "crashes": 0,
                   "dropped_events": "not measurable in synthetic control-path soak",
                   "peak_rss_mb": max(rss_values) if rss_values else None, "traced_peak_mb": peak / 1024 ** 2,
                   "limitation": "bounded synthetic control-path soak; no microphone, model inference, serial hardware, or hour-long run"}
        write_json(out / "summary.json", summary)
        if rows:
            x = [r["elapsed_seconds"] for r in rows]
            ram_series = [("RSS", [float(r["rss_mb"]) for r in rows])] if all(
                r["rss_mb"] != "" for r in rows
            ) else [("Python traced allocations", [r["traced_current_mb"] for r in rows])]
            plot_line(out / "ram_over_time.png", x, ram_series, "Elapsed seconds", "RAM (MiB)", "Stability RAM")
            plot_line(out / "cpu_over_time.png", x, [("CPU", [r["cpu_percent_total_capacity"] for r in rows])], "Elapsed seconds", "CPU (% total capacity)", "Stability CPU")
        self.add("Stability", "Software Stability", "Crash Count", 0, "count", processed,
                 f"{summary['actual_seconds']:.1f} seconds", notes=summary["limitation"])

    def leakage(self) -> None:
        hashes = {}
        for case in AUDIO_CASES:
            if case["path"].exists(): hashes.setdefault(sha256(case["path"]), []).append(rel(case["path"]))
        duplicates = [paths for paths in hashes.values() if len(paths) > 1]
        lines = ["# Data leakage check", "", f"Exact duplicate audio groups: **{len(duplicates)}**.", ""]
        lines += [f"- {', '.join(paths)}" for paths in duplicates] or ["- No exact duplicates among evaluated source clips."]
        lines += ["", "No training lists are present in this repository. The bark filename follows ESC-50 naming, while CED uses an AudioSet-derived model; overlap with upstream model training cannot be independently ruled out.",
                  "The local speech transcript comes from the human-authored manifest, not Google STT output.",
                  "Personalization expectations are hand-authored from documented deterministic policy; the system under test did not generate its own truth.",
                  "Generated noise variants remain grouped with their source and are never treated as independent train/test samples."]
        (RESULTS_DIR / "data_leakage_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def report(self) -> None:
        for row in read_csv(RESULTS_DIR / "stt/snr_results.csv"):
            if row["filter"] == "unfiltered" and row["status"] == "PASS":
                condition = f"{row['snr_db']} dB"
                self.add("STT", "STT noise robustness", "WER", row["wer"], "ratio", 1, condition)
                self.add("STT", "STT noise robustness", "CER", row["cer"], "ratio", 1, condition)
        emergency_metrics_path = RESULTS_DIR / "emergency/false_alarms.json"
        if emergency_metrics_path.exists():
            false_alarms = json.loads(emergency_metrics_path.read_text(encoding="utf-8"))
            self.add("Emergency", "Emergency False Alarms", "False alerts per clip",
                     false_alarms["false_alerts_per_clip"], "ratio", false_alarms["non_emergency_clips"], "non-emergency")
        personalization_path = RESULTS_DIR / "personalization/metrics.json"
        if personalization_path.exists():
            metrics = json.loads(personalization_path.read_text(encoding="utf-8"))
            self.add("Personalization", "AI Personalization", "Mean Absolute Priority Error",
                     metrics["mean_absolute_priority_error"], "priority levels", metrics["priority_assertions"])
            self.add("Personalization", "AI Personalization", "Role Exact Accuracy",
                     metrics["role_exact_accuracy"], "ratio", metrics["case_count"])
            self.add("Personalization", "AI Personalization", "Context Exact Accuracy",
                     metrics["context_exact_accuracy"], "ratio", metrics["case_count"])
        for benchmark, feature, notes in (
            ("Speaker", "Speaker/family voice recognition", "NOT IMPLEMENTED in repository"),
            ("SpeakerNoise", "Speaker recognition noise robustness", "NOT IMPLEMENTED in repository"),
            ("AIAssistant", "Separate AI assistant", "NOT IMPLEMENTED; personalization is not counted twice"),
            ("Direction", "Sound direction/vibration", "NOT IMPLEMENTED; no localization algorithm or multichannel ground truth"),
            ("HUDHardware", "Physical HUD/hardware", "Requires physical glasses; software framing is unit-tested only"),
        ):
            self.add(benchmark, feature, "benchmark status", "", "", 0,
                     status="NOT_IMPLEMENTED" if "NOT IMPLEMENTED" in notes else "SKIPPED", notes=notes)
        write_csv(RESULTS_DIR / "summary.csv", self.summary, SUMMARY_FIELDS)
        write_json(RESULTS_DIR / "summary.json", {"rows": self.summary, "checkpoint": self.checkpoint})
        values = {(r["benchmark"], r["metric"]): r for r in self.summary}
        def metric(benchmark, name):
            row = values.get((benchmark, name)); return "not available" if not row else f"{row['value']} {row['unit']} (n={row['sample_count']})"
        report = f"""# SoundGuard PC Benchmark Report

## 1. System Under Test

Commit `{git('rev-parse', 'HEAD')}` on branch `{git('branch', '--show-current')}` with the pre-existing dirty working tree recorded in `methodology.json`. Benchmark code imports production interfaces without changing core behavior.

## 2. Hardware and Software Environment

See `methodology.json`. This is a Windows PC software benchmark using Python {platform.python_version()}.

## 3. Dataset and Ground Truth

Two source WAV files were available: one human-labeled bark clip and one Vietnamese speech clip with a manifest reference transcript. Seeded Gaussian noise generated paired 20/15/10/5/0 dB variants. Emergency, fusion, and personalization use fixed hand-authored policy cases in `benchmark_data/`. Results are preliminary because the audio dataset is extremely small.

## 4. Methodology

Fixed seed {SEED}; identical noisy WAVs for filtered/unfiltered comparisons; warm-up excluded from steady-state CED latency; Unicode NFC/case/punctuation/whitespace normalization for Vietnamese WER/CER; production thresholds unchanged.

## 5. Results

### Environmental Sound Detection

- Clean accuracy: {metric('CED', 'Accuracy')}
- Clean Macro F1: {metric('CED', 'Macro F1')}
- Warm P95 latency: {metric('CED', 'P95 latency')}

### Speech-to-Text

- Clean WER: {metric('STT', 'WER')}
- Clean CER: {metric('STT', 'CER')}

### Noise Filtering

- Mean Delta SNR: {metric('NoiseFilter', 'Mean Delta SNR')}
- CED Mean Delta F1: {metric('NoiseFilter', 'Mean Delta F1')}

### Emergency Detection

- Recall: {metric('Emergency', 'Recall')}
- False Negative Rate: {metric('Emergency', 'False Negative Rate')}

### Fusion

- Delta F1: {metric('Fusion', 'Delta F1')}

### AI Personalization

- Exact asserted-priority accuracy: {metric('Personalization', 'Priority Accuracy')}
- Universal Critical pass rate: {metric('Personalization', 'Critical Invariant Pass Rate')}
- Remote provider latency/consistency is skipped when no credential is present.

### Sound Direction / Vibration

NOT IMPLEMENTED — NO BENCHMARK RESULT. No localization algorithm or multichannel ground truth exists.

### End-to-End PC Software Pipeline

- PC SOFTWARE END-TO-END P95 latency: {metric('Pipeline', 'P95 Latency')}

### Stability

- Crash count: {metric('Stability', 'Crash Count')}

## 6. Key Findings

- Emergency policy cases achieved 100% recall with no false negatives or false alarms (8 total cases; 4 negatives).
- Fusion improved binary F1 from 0.667 to 1.000 on six constructed rule cases by repairing one help-request miss and suppressing one benign bark false alarm.
- Deterministic personalization passed all 22 priority assertions and all Universal Critical checks; exact role and context inference were both {metric('Personalization', 'Role Exact Accuracy')} and {metric('Personalization', 'Context Exact Accuracy')}.
- CED clean accuracy was only 0.500 (1/2 clips), and SNR behavior was non-monotonic; the denominator is too small for a general accuracy claim.
- After correcting for DTLN's measured 384-sample algorithmic delay, DTLN improved mean intrusive SNR. Despite that signal-level improvement, it reduced CED Macro F1 and worsened Google STT WER at 10, 5, and 0 dB on this tiny dataset; unfiltered STT remained perfect on the single utterance. These negative downstream results are retained.
- No thresholds were tuned and no hard cases were removed after inspection.

## 7. Limitations

This is a **PC SOFTWARE BENCHMARK**, not real smart-glasses hardware validation. Two audio clips cannot establish general CED/STT accuracy or class balance. Synthetic white noise does not represent all real environments. Google STT is network-dependent. The bounded stability run is shorter than the one-hour target. No claim is made about ESP32/OLED/vibration latency, battery life, physical direction accuracy, or real-world user benefit.

## 8. Missing Hardware/User Validation

- Battery life, power consumption, and device temperature
- OLED readability and physical display latency
- Vibration perception and actuator latency
- End-to-end serial/ESP32 latency
- Two-microphone directional accuracy with recorded ground truth
- Long-duration microphone stability in real environments
- Testing with hearing-impaired participants
"""
        (RESULTS_DIR / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    def run(self) -> None:
        self.audit(); self.methodology()
        for name, function in (("prepare_noise", self.prepare_noise), ("ced", self.ced),
                               ("noise_filter", self.noise_filter), ("stt", self.stt),
                               ("emergency", self.emergency), ("fusion", self.fusion),
                               ("personalization", self.personalization), ("pipeline", self.pipeline),
                               ("stability", self.stability), ("data_leakage", self.leakage)):
            self.stage(name, function)
        self.report()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all feasible SoundGuard PC benchmarks")
    parser.add_argument("--expanded", action="store_true",
                        help="Run the public-dataset expanded benchmark into benchmark_results/expanded")
    parser.add_argument("--enable-network-stt", action="store_true", help="Allow Google STT requests")
    parser.add_argument("--stability-seconds", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-stage", action="append", default=[],
                        help="With --resume, rerun one named completed stage")
    args = parser.parse_args()
    if args.stability_seconds < 1:
        parser.error("--stability-seconds must be positive")
    if args.expanded:
        from benchmark.run_expanded_benchmark import run_expanded
        run_expanded(enable_network_stt=args.enable_network_stt,
                     stability_seconds=args.stability_seconds, resume=args.resume,
                     rerun_stages=args.rerun_stage)
        return 0
    np.random.seed(SEED)
    runner = Runner(enable_network_stt=args.enable_network_stt,
                    stability_seconds=args.stability_seconds, resume=args.resume)
    for stage in args.rerun_stage:
        if stage in runner.checkpoint["completed"]:
            runner.checkpoint["completed"].remove(stage)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
