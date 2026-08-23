"""Analyze only collected, complete, non-pilot SoundGuard Layer 3 sessions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from human_study.study_core import PARTICIPANT_DATA_DIR, STUDY_DIR, STUDY_VERSION
from human_study.validate_session import read_csv, validate_session_dir


METRICS = {
    "speech_accuracy": ("speech", "correct", "mean"),
    "speech_rt_ms": ("speech", "rt_from_stimulus_ms", "mean"),
    "environmental_accuracy": ("environmental", "environment_correct", "mean"),
    "environmental_missed_rate": ("environmental", "missed", "mean"),
    "environmental_rt_ms": ("environmental", "rt_from_stimulus_ms", "mean"),
    "mixed_speech_accuracy": ("mixed", "speech_correct", "mean"),
    "mixed_environment_accuracy": ("mixed", "environment_correct", "mean"),
    "mixed_both_correct_rate": ("mixed", "both_correct", "mean"),
    "mixed_missed_rate": ("mixed", "missed", "mean"),
    "mixed_rt_ms": ("mixed", "rt_from_stimulus_ms", "mean"),
    "critical_recognition_rate": ("critical", "correct", "mean"),
    "critical_missed_rate": ("critical", "missed", "mean"),
    "critical_rt_ms": ("critical", "rt_from_stimulus_ms", "mean"),
}


def numeric(value: str) -> float:
    lowered = str(value).strip().lower()
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    return float(value)


def percentile_bootstrap_ci(values: list[float], seed: int = 20260823) -> list[float] | None:
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(array, size=(10000, len(array)), replace=True), axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def paired_statistics(a: dict[str, float], b: dict[str, float]) -> dict:
    ids = sorted(set(a) & set(b))
    av = [a[item] for item in ids]
    bv = [b[item] for item in ids]
    differences = [right - left for left, right in zip(av, bv)]
    result: dict = {
        "n": len(ids), "participant_ids": ids, "without": av, "with": bv,
        "paired_differences_with_minus_without": differences,
    }
    if not ids:
        return result
    result.update({
        "without_mean": statistics.fmean(av),
        "with_mean": statistics.fmean(bv),
        "difference_mean": statistics.fmean(differences),
        "difference_median": statistics.median(differences),
        "difference_95pct_bootstrap_ci": percentile_bootstrap_ci(differences),
    })
    if len(differences) >= 2 and statistics.stdev(differences) > 0:
        result["effect_size_dz"] = statistics.fmean(differences) / statistics.stdev(differences)
    else:
        result["effect_size_dz"] = None
    nonzero = [value for value in differences if value != 0]
    if len(differences) >= 3 and len(set(differences)) > 1:
        shapiro = stats.shapiro(differences)
        result["normality_shapiro"] = {"statistic": float(shapiro.statistic), "p": float(shapiro.pvalue)}
        if shapiro.pvalue >= 0.05:
            test = stats.ttest_rel(bv, av)
            result["paired_test"] = {
                "name": "paired_t", "statistic": float(test.statistic), "p": float(test.pvalue),
                "selection_rule": "Shapiro-Wilk p >= 0.05",
            }
        elif nonzero:
            test = stats.wilcoxon(bv, av, zero_method="wilcox")
            result["paired_test"] = {
                "name": "wilcoxon_signed_rank", "statistic": float(test.statistic),
                "p": float(test.pvalue), "selection_rule": "Shapiro-Wilk p < 0.05",
            }
        else:
            result["paired_test"] = {"name": "not_testable_all_zero_differences"}
    elif len(differences) < 3:
        result["paired_test"] = {"name": "not_tested_n_below_3"}
    else:
        result["paired_test"] = {"name": "not_testable_constant_differences"}
    return result


def participant_metrics(trials: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = defaultdict(dict)
    valid = [row for row in trials if row.get("trial_valid") == "true"]
    for metric, (block, field, _aggregation) in METRICS.items():
        for condition in ("WITHOUT", "WITH"):
            values = []
            for row in valid:
                if row.get("block") == block and row.get("condition") == condition:
                    try:
                        values.append(numeric(row.get(field, "")))
                    except ValueError:
                        pass
            if values:
                output[metric][condition] = statistics.fmean(values)
    return dict(output)


def discover_real_sessions(data_dir: Path) -> list[Path]:
    if data_dir.name != "main" and "fixtures" in {part.lower() for part in data_dir.parts}:
        raise RuntimeError("Fixture/synthetic directories are forbidden for final analysis.")
    sessions = []
    for path in sorted(data_dir.glob("P*/session.json")):
        session = json.loads(path.read_text(encoding="utf-8"))
        participant = str(session.get("participant_id", ""))
        if session.get("pilot") or not participant.startswith("P") or participant.startswith("PILOT"):
            continue
        sessions.append(path.parent)
    return sessions


def analyze_sessions(
    session_dirs: list[Path], *, allow_test_fixtures: bool = False
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    all_trials: list[dict] = []
    participant_rows: list[dict] = []
    quality: list[dict] = []
    metric_pairs: dict[str, dict[str, dict[str, float]]] = {
        metric: {"WITHOUT": {}, "WITH": {}} for metric in METRICS
    }
    for session_dir in session_dirs:
        session_metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        if session_metadata.get("test_fixture") and not allow_test_fixtures:
            raise RuntimeError(
                f"Synthetic/test fixture is forbidden in real analysis: {session_dir}"
            )
        validation = validate_session_dir(session_dir)
        quality.append(validation)
        if not validation["valid"]:
            continue
        _, trials = read_csv(session_dir / "trials.csv")
        _, questionnaire = read_csv(session_dir / "questionnaire.csv")
        participant = validation["participant_id"]
        all_trials.extend(trials)
        metrics = participant_metrics(trials)
        row = {"participant_id": participant, "study_version": STUDY_VERSION}
        row["invalid_trial_count"] = sum(
            trial.get("trial_valid") != "true" for trial in trials
        )
        row["network_failure_trial_count"] = sum(
            trial.get("network_failure") == "true" for trial in trials
        )
        for metric in METRICS:
            for condition in ("WITHOUT", "WITH"):
                value = metrics.get(metric, {}).get(condition)
                row[f"{metric}_{condition.lower()}"] = "" if value is None else value
                block, field, _aggregation = METRICS[metric]
                row[f"{metric}_{condition.lower()}_n_trials"] = sum(
                    trial.get("trial_valid") == "true"
                    and trial.get("block") == block
                    and trial.get("condition") == condition
                    and trial.get(field, "") != ""
                    for trial in trials
                )
                if value is not None:
                    metric_pairs[metric][condition][participant] = value
        sus = next((item.get("scored_value") for item in questionnaire if item.get("item_id") == "SUS_TOTAL"), "")
        row["sus_total"] = sus
        for item in questionnaire:
            if item.get("instrument") == "SOUNDGUARD_LIKERT":
                row[item["item_id"].lower()] = item.get("response", "")
        participant_rows.append(row)
    statistics_output = {
        "study_version": STUDY_VERSION,
        "included_n": len(participant_rows),
        "analysis_note": "Paired participant-level analysis; WITH minus WITHOUT. Bootstrap CIs are exploratory percentile CIs.",
        "paired_binary_note": "McNemar was not used because A/B use non-identical stimulus sets; binary trials are not one-to-one pairs.",
        "metrics": {
            metric: paired_statistics(values["WITHOUT"], values["WITH"])
            for metric, values in metric_pairs.items()
        },
    }
    sus_scores = [
        float(row["sus_total"]) for row in participant_rows
        if row.get("sus_total") not in {None, ""}
    ]
    likert_distributions = {}
    for index in range(1, 8):
        field = f"sg{index:02d}"
        values = [
            int(float(row[field])) for row in participant_rows
            if row.get(field) not in {None, ""}
        ]
        likert_distributions[field] = {
            "n": len(values),
            "counts": {str(score): values.count(score) for score in range(1, 6)},
            "median": statistics.median(values) if values else None,
        }
    statistics_output["usability"] = {
        "sus": {
            "n": len(sus_scores),
            "mean": statistics.fmean(sus_scores) if sus_scores else None,
            "median": statistics.median(sus_scores) if sus_scores else None,
            "participant_scores": {
                row["participant_id"]: float(row["sus_total"])
                for row in participant_rows if row.get("sus_total") not in {None, ""}
            },
        },
        "soundguard_likert_distributions": likert_distributions,
    }
    return all_trials, participant_rows, statistics_output, quality


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def create_figures(output: Path, participants: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        pairs = []
        for row in participants:
            try:
                pairs.append((row["participant_id"], float(row[f"{metric}_without"]), float(row[f"{metric}_with"])))
            except (KeyError, TypeError, ValueError):
                pass
        if not pairs:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for participant, without, with_value in pairs:
            ax.plot([0, 1], [without, with_value], marker="o", alpha=0.8, label=participant)
        ax.set_xticks([0, 1], ["WITHOUT", "WITH"])
        ax.set_ylabel(metric)
        ax.set_title(f"Individual paired outcomes: {metric}")
        ax.grid(axis="y", alpha=0.25)
        if len(pairs) <= 10:
            ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(figures / f"{metric}.png", dpi=160)
        plt.close(fig)


def write_outputs(output: Path, all_trials: list[dict], participants: list[dict], stats_out: dict, quality: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "LAYER3_RESULTS.csv", all_trials)
    write_csv(output / "participant_summary.csv", participants)
    (output / "statistics.json").write_text(
        json.dumps(stats_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    quality_lines = ["# Layer 3 data quality report", ""]
    for report in quality:
        quality_lines.append(
            f"- {report.get('participant_id', report['session_path'])}: "
            f"{'VALID' if report['valid'] else 'INVALID'}; "
            f"errors={report.get('error_count', 0)} warnings={report.get('warning_count', 0)}"
        )
        for issue in report["issues"]:
            quality_lines.append(f"  - {issue['severity']} `{issue['code']}`: {issue['message']}")
    (output / "data_quality_report.md").write_text("\n".join(quality_lines) + "\n", encoding="utf-8")
    summary = [
        "# SoundGuard Layer 3 results", "",
        f"Study version: **{STUDY_VERSION}**", "",
        f"Included complete valid main-study participants: **{len(participants)}**", "",
        "> The evaluated prototype used its existing network-dependent Google Speech Recognition implementation for live speech-to-text.", "",
        "Network dependency is a current system limitation. Pilot, incomplete, invalid, and technical-failure trials are not silently treated as participant errors.", "",
        "## Paired outcomes", "",
        "Values below are WITH minus WITHOUT. Negative and null results are retained.", "",
    ]
    for metric, result in stats_out["metrics"].items():
        summary.append(
            f"- **{metric}**: n={result['n']}; mean difference="
            f"{result.get('difference_mean', 'not available')}; median difference="
            f"{result.get('difference_median', 'not available')}; test="
            f"{result.get('paired_test', {}).get('name', 'not available')}"
        )
    usability = stats_out["usability"]["sus"]
    summary += [
        "", "## Usability", "",
        f"Standard SUS: n={usability['n']}; mean={usability['mean']}; median={usability['median']}.",
        "Project-specific Likert response distributions are reported separately in `statistics.json`.",
    ]
    summary += [
        "", "McNemar testing was not applied because matched A/B sets are deliberately non-identical and therefore do not provide one-to-one paired binary trials.", "",
    ]
    (output / "LAYER3_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    create_figures(output, participants)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PARTICIPANT_DATA_DIR / "main")
    parser.add_argument("--output", type=Path, default=STUDY_DIR / "results")
    args = parser.parse_args()
    sessions = discover_real_sessions(args.data_dir.resolve())
    if not sessions:
        print("No complete real main-study sessions found. No results were generated.")
        return 2
    all_trials, participants, stats_out, quality = analyze_sessions(sessions)
    if not participants:
        print("No valid complete main-study sessions found. No results were generated.")
        return 2
    write_outputs(args.output.resolve(), all_trials, participants, stats_out, quality)
    print(f"Analyzed {len(participants)} real main-study participants into {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
