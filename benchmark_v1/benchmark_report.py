"""Markdown reporting for benchmark results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmark_v1.benchmark_metrics import calculate_metrics


def _pct(value) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def generate_report(results_path, output_path, automated_tests="Not recorded",
                    run_id: str | None = None) -> dict:
    results_path, output_path = Path(results_path), Path(output_path)
    with results_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metrics = calculate_metrics(rows, run_id=run_id)
    scope = (f"Single run: `{run_id}`" if run_id is not None
             else "Aggregate historical report (mixed runs)")
    included = ", ".join(metrics["included_run_ids"]) or "none"
    lines = [
        "# SoundGuard benchmark report", "", f"**Scope:** {scope}", "",
        f"**Included run IDs:** {included}", "", "## Dataset and execution", "",
        f"- Unique WAVs: {metrics['unique_wav_count']}",
        f"- Unique samples: {metrics['unique_sample_count']}",
        f"- Sample-condition rows: {metrics['sample_condition_row_count']}",
        f"- Paired DTLN samples: {metrics['paired_dtln_sample_count']}",
        f"- Completed rows: {metrics['completed_rows']}",
        f"- Failed or incomplete rows: {metrics['failed_rows']}",
        f"- Rows missing sound ground truth: {metrics['missing_sound_ground_truth']}",
        f"- Automated tests: {automated_tests}", "", "## Transcript eligibility", "",
        f"- Speech rows eligible for WER/CER: {metrics['speech_rows_eligible_for_transcript_metrics']}",
        f"- Non-speech rows excluded from transcript metrics: {metrics['non_speech_rows_excluded_from_transcript_metrics']}",
        f"- Speech rows missing required transcript ground truth: {metrics['speech_rows_missing_transcript_ground_truth']}",
        f"- STT-success rows evaluated: {metrics['stt_success_rows_evaluated']}",
        f"- STT-failed rows excluded: {metrics['stt_failed_rows_excluded']}",
        "", "## Transcript metrics by condition", "",
        "| Condition | Rows evaluated | STT failed/excluded | WER | CER | Exact match | Empty rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    condition_labels = {
        "clean_speech_dtln_disabled": "Clean speech, DTLN disabled",
        "noisy_speech_dtln_enabled": "Noisy speech, DTLN enabled",
        "noisy_speech_dtln_disabled": "Noisy speech, DTLN disabled",
    }
    for key, label in condition_labels.items():
        values = metrics["condition_metrics"][key]
        lines.append(f"| {label} | {values['stt_success_rows_evaluated']} | "
                     f"{values['stt_failed_rows_excluded']} | {_pct(values['wer'])} | "
                     f"{_pct(values['cer'])} | {_pct(values['exact_transcript_match'])} | "
                     f"{_pct(values['empty_transcript_rate'])} |")
    lines += ["", "## Optional mixed-condition transcript metrics", "",
        "These values combine different acoustic and DTLN conditions and are not a condition-specific score.", "",
        f"- Mixed-condition WER: {_pct(metrics['wer'])}",
        f"- Mixed-condition CER: {_pct(metrics['cer'])}",
        f"- Mixed-condition exact normalized match: {_pct(metrics['exact_transcript_match'])}",
        f"- Mixed-condition empty transcript rate: {_pct(metrics['empty_transcript_rate'])}",
        "", "## Sound and system metrics", "",
        f"- Exact top-1 CED label accuracy: {_pct(metrics['top1_sound_accuracy'])}",
        f"- Canonical-category accuracy: {_pct(metrics['canonical_category_accuracy'])}",
        "- Interpretation: CED is multi-label; semantically related labels such as `Speech synthesizer` "
        "and `Speech` can differ at exact-label level. Exact top-1 accuracy alone is not overall "
        "sound-recognition quality; canonical-category accuracy is reported separately when ground truth exists.",
        f"- Expected alert-level accuracy: {_pct(metrics['expected_alert_level_accuracy'])}",
        f"- Help-request accuracy: {_pct(metrics['help_request_accuracy'])}",
        f"- VAD correctness: {_pct(metrics['vad_correctness'])}",
        f"- False emergencies: {metrics['false_emergencies']}",
        f"- Missed emergencies: {metrics['missed_emergencies']}", "",
        "## Stage validity", "",
    ]
    if metrics["positive_emergency_rows"] == 0:
        lines[lines.index("## Stage validity"):lines.index("## Stage validity")] = [
            "## Emergency sensitivity limitation", "",
            "This run contains no positive emergency samples. False-emergency behavior was tested, "
            "but missed-emergency sensitivity was not meaningfully evaluated. Zero missed emergencies "
            "does not prove emergency detection accuracy.", "",
        ]
    for stage, counts in metrics["stage_status_counts"].items():
        lines.append(f"- {stage.upper()}: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    lines += ["",
        "## Per-class metrics", "", "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in metrics["per_class"].items():
        lines.append(f"| {label} | {_pct(values['precision'])} | {_pct(values['recall'])} | "
                     f"{_pct(values['f1'])} | {values['support']} |")
    lines += ["", "## Confusion-matrix counts", ""]
    lines += [f"- {key}: {value}" for key, value in metrics["confusion_matrix"].items()] or ["- No category ground truth."]
    lines += ["", "## End-to-end and legacy stage timings", "",
              "Historical `*_latency_seconds` values may include imports and model initialization; "
              "they are retained as startup-inclusive evidence and must not be interpreted as steady-state inference.", ""]
    split_markers = ("model_load_or_cold_start", "inference_latency", "inference_realtime_factor")
    for name, values in metrics["latencies"].items():
        if any(marker in name for marker in split_markers):
            continue
        lines.append(f"- {name}: mean={values['mean']:.4f}, min={values['min']:.4f}, "
                     f"max={values['max']:.4f}, n={values['count']}")
    lines += ["", "## Cold-start and steady-state inference timings", "",
              "First successful calls are reported as model-load/cold-start latency. Only explicit "
              "`*_inference_latency_seconds` fields represent later steady-state calls.", ""]
    found_split = False
    for name, values in metrics["latencies"].items():
        if not any(marker in name for marker in split_markers):
            continue
        found_split = True
        lines.append(f"- {name}: mean={values['mean']:.4f}, min={values['min']:.4f}, "
                     f"max={values['max']:.4f}, n={values['count']}")
    if not found_split:
        lines.append("- No split latency fields are populated in these historical rows.")
    paired = metrics["paired_dtln"]
    lines += ["", "## Paired DTLN comparison", "",
              f"- Improved: {paired.get('improved', 0)}",
              f"- Unchanged: {paired.get('unchanged', 0)}",
              f"- Worsened: {paired.get('worsened', 0)}", "",
              "## Limitations", "",
              "The fixed-file benchmark does not test microphone hardware, bounded queues, or live acoustic drift. "
              "External Google STT requires explicit opt-in and network access. Manual listening and scenario checks "
              "remain necessary. A small, hand-selected dataset does not prove real-world accuracy.", ""]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a SoundGuard benchmark report")
    parser.add_argument("--run-id", help="Include only this run ID; omit for historical aggregate")
    parser.add_argument("--results", default="benchmark_v1/results.csv")
    parser.add_argument("--output", default="benchmark_v1/outputs/benchmark_report.md")
    args = parser.parse_args()
    generate_report(args.results, args.output, run_id=args.run_id)
    print(f"Wrote {args.output} for " + (f"run {args.run_id}" if args.run_id else "all historical runs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
