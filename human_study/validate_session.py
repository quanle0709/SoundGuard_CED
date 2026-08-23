"""Read-only validation for SoundGuard Layer 3 participant sessions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from human_study.study_core import (
    QUESTIONNAIRE_FIELDS,
    STUDY_VERSION,
    TRIAL_FIELDS,
    build_trial_plan,
    load_config,
    protocol_lock,
    stimulus_asset_bundle_sha256,
)


TRUE_FALSE = {"true", "false"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def add_issue(issues: list[dict], code: str, message: str, severity: str = "error") -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def validate_session_dir(path: Path) -> dict:
    issues: list[dict] = []
    report = {"session_path": str(path), "valid": False, "issues": issues}
    session_path = path / "session.json"
    trials_path = path / "trials.csv"
    questionnaire_path = path / "questionnaire.csv"
    runtime_path = path / "runtime.log"
    for required in (session_path, trials_path, questionnaire_path):
        if not required.is_file():
            add_issue(issues, "missing_file", f"Missing {required.name}")
    if not session_path.is_file():
        return report
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        add_issue(issues, "invalid_session_json", str(exc))
        return report

    report["participant_id"] = session.get("participant_id")
    report["pilot"] = bool(session.get("pilot"))
    if session.get("status") != "complete":
        add_issue(issues, "incomplete_study", f"Session status is {session.get('status')!r}")
    if session.get("study_version") != STUDY_VERSION:
        add_issue(issues, "protocol_version_mismatch", str(session.get("study_version")))
    try:
        base_seed = int(session["protocol_lock"]["base_randomization_seed"])
        current_lock = protocol_lock(base_seed)
        for key in ("study_version", "config_sha256", "manifest_sha256"):
            if session["protocol_lock"].get(key) != current_lock[key]:
                add_issue(issues, "protocol_lock_mismatch", key)
    except (KeyError, TypeError, ValueError) as exc:
        add_issue(issues, "invalid_protocol_lock", str(exc))
        base_seed = load_config()["default_base_seed"]
    if not session.get("test_fixture"):
        try:
            actual_asset_hash = stimulus_asset_bundle_sha256()
            if session.get("stimulus_asset_bundle_sha256") != actual_asset_hash:
                add_issue(issues, "stimulus_asset_mismatch", "Current stimulus files differ from the session lock.")
        except (OSError, ValueError) as exc:
            add_issue(issues, "stimulus_asset_validation_failed", str(exc))

    trial_rows: list[dict[str, str]] = []
    if trials_path.is_file():
        fields, trial_rows = read_csv(trials_path)
        missing_fields = set(TRIAL_FIELDS) - set(fields)
        if missing_fields:
            add_issue(issues, "trial_schema", f"Missing columns: {sorted(missing_fields)}")
        ids = [row.get("trial_id", "") for row in trial_rows]
        duplicates = [key for key, count in Counter(ids).items() if key and count > 1]
        if duplicates:
            add_issue(issues, "duplicate_trials", str(duplicates))
        try:
            expected = build_trial_plan(str(session.get("participant_id")), int(base_seed))
            expected_ids = {trial.trial_id for trial in expected}
            actual_ids = set(ids)
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)
            if missing:
                add_issue(issues, "missing_trials", str(missing))
            if unexpected:
                add_issue(issues, "unexpected_trials", str(unexpected))
            expected_stimuli = {trial.trial_id: trial.row["stimulus_id"] for trial in expected}
            mismatched = [
                row.get("trial_id") for row in trial_rows
                if expected_stimuli.get(row.get("trial_id")) != row.get("stimulus_id")
            ]
            if mismatched:
                add_issue(issues, "randomization_mismatch", str(mismatched))
        except Exception as exc:
            add_issue(issues, "plan_validation_failed", str(exc))

        for row in trial_rows:
            trial_id = row.get("trial_id", "[unknown]")
            try:
                start = float(row.get("stimulus_start_monotonic", ""))
                response = float(row.get("response_timestamp_monotonic", ""))
                rt = float(row.get("rt_from_stimulus_ms", ""))
                if response < start or rt < 0:
                    raise ValueError("negative/impossible order")
                calculated = (response - start) * 1000.0
                if abs(calculated - rt) > 2.0:
                    add_issue(issues, "rt_mismatch", f"{trial_id}: {rt} vs {calculated:.3f}")
            except (TypeError, ValueError) as exc:
                add_issue(issues, "invalid_response_time", f"{trial_id}: {exc}")
            for name in ("trial_valid", "playback_success", "hardware_connected", "network_failure"):
                if row.get(name) not in TRUE_FALSE:
                    add_issue(issues, "invalid_boolean", f"{trial_id}.{name}={row.get(name)!r}")
            if row.get("playback_success") == "false":
                add_issue(issues, "playback_failure", trial_id, severity="warning")
            if row.get("hardware_connected") == "false":
                add_issue(issues, "hardware_disconnect", trial_id, severity="warning")
            if row.get("network_failure") == "true":
                add_issue(issues, "network_STT_failure", trial_id, severity="warning")
            if row.get("trial_valid") == "false":
                add_issue(
                    issues, "invalid_trial_retained",
                    f"{trial_id}: {row.get('invalid_reason')}", severity="warning",
                )

    if questionnaire_path.is_file():
        fields, questionnaire = read_csv(questionnaire_path)
        missing_fields = set(QUESTIONNAIRE_FIELDS) - set(fields)
        if missing_fields:
            add_issue(issues, "questionnaire_schema", f"Missing columns: {sorted(missing_fields)}")
        expected_items = {
            *(f"SUS{index:02d}" for index in range(1, 11)),
            *(f"SG{index:02d}" for index in range(1, 8)),
            "SUS_TOTAL",
        }
        actual_items = {row.get("item_id") for row in questionnaire}
        missing = sorted(expected_items - actual_items)
        if missing:
            add_issue(issues, "missing_questionnaire", str(missing))
        duplicates = [
            key for key, count in Counter(row.get("item_id") for row in questionnaire).items()
            if key and count > 1
        ]
        if duplicates:
            add_issue(issues, "duplicate_questionnaire_items", str(duplicates))

    if any(row.get("condition") == "WITH" for row in trial_rows) and not runtime_path.is_file():
        add_issue(issues, "missing_runtime_log", "WITH condition has no runtime.log")

    report["trial_count"] = len(trial_rows)
    report["error_count"] = sum(issue["severity"] == "error" for issue in issues)
    report["warning_count"] = sum(issue["severity"] == "warning" for issue in issues)
    report["valid"] = report["error_count"] == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Participant session directory")
    parser.add_argument("--json", action="store_true", help="Machine-readable report")
    args = parser.parse_args()
    report = validate_session_dir(args.session.resolve())
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("VALID" if report["valid"] else "INVALID")
        print(f"Session: {report['session_path']}")
        for issue in report["issues"]:
            print(f"{issue['severity'].upper()} {issue['code']}: {issue['message']}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
