"""Software-only tests. All generated observations are TEST/SYNTHETIC."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from human_study.analyze_results import (
    analyze_sessions,
    discover_real_sessions,
    write_outputs,
)
from human_study.run_study import (
    ProductMonitor,
    confirm_preflight_subtitle,
    frozen_product_command,
    reserve_session_dir,
    stt_failure_type,
)
from human_study.study_core import (
    QUESTIONNAIRE_FIELDS,
    STUDY_VERSION,
    TRIAL_FIELDS,
    build_trial_plan,
    calculate_rt_ms,
    condition_order,
    condition_set_map,
    derive_seed,
    load_config,
    load_manifest,
    protocol_lock,
    score_sus,
    stimulus_asset_paths,
)
from human_study.validate_session import validate_session_dir


BASE_SEED = 20260823


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_synthetic_complete_session(parent: Path, participant: str, with_bonus: bool = True) -> Path:
    """Create an explicit TEST/SYNTHETIC fixture outside participant_data."""
    path = parent / participant
    path.mkdir(parents=True)
    plan = build_trial_plan(participant, BASE_SEED)
    participant_seed = derive_seed(participant, BASE_SEED)
    session = {
        "participant_id": participant,
        "pilot": False,
        "test_fixture": True,
        "status": "complete",
        "study_version": STUDY_VERSION,
        "protocol_lock": protocol_lock(BASE_SEED),
        "expected_trial_count": 38,
    }
    (path / "session.json").write_text(json.dumps(session), encoding="utf-8")
    (path / "runtime.log").write_text("TEST/SYNTHETIC runtime\n", encoding="utf-8")
    rows = []
    for index, trial in enumerate(plan):
        start = 1000.0 + index * 10.0
        correct = not (trial.condition == "WITHOUT" and with_bonus and index % 3 == 0)
        row = {field: "" for field in TRIAL_FIELDS}
        row.update({
            "participant_id": participant,
            "study_version": STUDY_VERSION,
            "pilot": "false",
            "block": trial.block,
            "condition": trial.condition,
            "condition_position": trial.condition_position,
            "trial_id": trial.trial_id,
            "trial_index": trial.trial_index,
            "stimulus_id": trial.row["stimulus_id"],
            "stimulus_set": trial.row["stimulus_set"],
            "ground_truth": trial.row["ground_truth"],
            "response": trial.row["correct_response"] if correct else "TEST_WRONG",
            "correct": str(correct).lower(),
            "speech_correct": str(correct).lower() if trial.block in {"speech", "mixed"} else "",
            "environment_correct": str(correct).lower() if trial.block in {"environmental", "mixed"} else "",
            "both_correct": str(correct).lower() if trial.block == "mixed" else "",
            "missed": "false",
            "false_perceived_event": "false",
            "stimulus_start_wall": "2026-01-01T00:00:00+00:00",
            "stimulus_start_monotonic": f"{start:.9f}",
            "response_timestamp_wall": "2026-01-01T00:00:01+00:00",
            "response_timestamp_monotonic": f"{start + 1.0:.9f}",
            "rt_from_stimulus_ms": "1000.000",
            "subtitle_state": "YES" if trial.condition == "WITH" else "NOT_APPLICABLE",
            "displayed_ced": "NONE" if trial.condition == "WITH" else "NOT_APPLICABLE",
            "displayed_alert": "NONE" if trial.condition == "WITH" else "NOT_APPLICABLE",
            "stt_attempted": str(trial.condition == "WITH" and trial.block in {"speech", "mixed"}).lower(),
            "stt_success": "true" if trial.condition == "WITH" and trial.block in {"speech", "mixed"} else "",
            "network_failure": "false",
            "trial_valid": "true",
            "playback_success": "true",
            "hardware_connected": "true",
            "error_class": "" if correct else "participant_error",
            "randomization_seed": participant_seed,
        })
        rows.append(row)
    write_csv(path / "trials.csv", TRIAL_FIELDS, rows)
    questionnaire = []
    for index in range(1, 11):
        questionnaire.append({
            "participant_id": participant, "study_version": STUDY_VERSION,
            "pilot": "false", "instrument": "SUS", "item_id": f"SUS{index:02d}",
            "prompt": "TEST/SYNTHETIC", "response": 3, "scored_value": 2,
            "recorded_at": "2026-01-01T00:00:00Z",
        })
    for index in range(1, 8):
        questionnaire.append({
            "participant_id": participant, "study_version": STUDY_VERSION,
            "pilot": "false", "instrument": "SOUNDGUARD_LIKERT", "item_id": f"SG{index:02d}",
            "prompt": "TEST/SYNTHETIC", "response": 3, "scored_value": 3,
            "recorded_at": "2026-01-01T00:00:00Z",
        })
    questionnaire.append({
        "participant_id": participant, "study_version": STUDY_VERSION,
        "pilot": "false", "instrument": "SUS_TOTAL", "item_id": "SUS_TOTAL",
        "prompt": "TEST/SYNTHETIC", "response": 50, "scored_value": 50,
        "recorded_at": "2026-01-01T00:00:00Z",
    })
    write_csv(path / "questionnaire.csv", QUESTIONNAIRE_FIELDS, questionnaire)
    return path


class StudyCoreTests(unittest.TestCase):
    def test_live_stt_preflight_uses_locked_device_and_product_command(self):
        config = {"frozen_product_command": ["app.py", "--live-stt"]}
        self.assertEqual(
            frozen_product_command(config, "COM9", 1),
            ["app.py", "--live-stt", "--device", "1", "--hud-port", "COM9"],
        )

    def test_stt_failure_classification_requires_production_evidence(self):
        self.assertEqual(stt_failure_type({"stt_timing": [{}]}), "")
        self.assertEqual(
            stt_failure_type({"stt_timing": [], "network_error": True}),
            "network_STT_failure",
        )
        self.assertEqual(
            stt_failure_type({"stt_timing": [], "stt_no_result": True}),
            "recognition_no_result",
        )
        self.assertEqual(stt_failure_type({"stt_timing": []}), "no_STT_evidence")

    def test_preflight_subtitle_uses_persistent_partial_and_always_clears(self):
        class RecordingHUD:
            def __init__(self):
                self.calls = []

            def set_partial_subtitle(self, text):
                self.calls.append(("P", text))

            def set_subtitle(self, text):
                self.calls.append(("S", text))

        for answer in (True, False):
            with self.subTest(answer=answer):
                hud = RecordingHUD()

                def confirm(prompt):
                    self.assertEqual(
                        prompt,
                        "Researcher: is PREFLIGHT visible as a subtitle on the OLED?",
                    )
                    self.assertEqual(hud.calls, [("P", "PREFLIGHT")])
                    return answer

                self.assertEqual(
                    confirm_preflight_subtitle(hud, confirm),
                    answer,
                )
                self.assertEqual(
                    hud.calls,
                    [("P", "PREFLIGHT"), ("S", "")],
                )

    def test_manifest_and_csv_schema(self):
        manifest = load_manifest()
        self.assertEqual(len(manifest), 38)
        self.assertIn("stt_error_type", TRIAL_FIELDS)
        self.assertIn("trial_valid", TRIAL_FIELDS)
        self.assertIn("environment_ground_truth", TRIAL_FIELDS)
        self.assertIn("environment_rt_from_stimulus_ms", TRIAL_FIELDS)

    def test_local_stimulus_preparation_paths_are_required(self):
        required = {path.name for path in stimulus_asset_paths()}
        self.assertTrue(
            {"help_A.wav", "help_B.wav", "RECORDING_PROVENANCE.md"}
            <= required
        )

    def test_randomization_is_reproducible_and_seed_sensitive(self):
        first = [item.row["stimulus_id"] for item in build_trial_plan("P01", BASE_SEED)]
        second = [item.row["stimulus_id"] for item in build_trial_plan("P01", BASE_SEED)]
        changed = [item.row["stimulus_id"] for item in build_trial_plan("P01", BASE_SEED + 1)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_counterbalance_four_sequences(self):
        self.assertEqual(condition_order("P01"), ("WITHOUT", "WITH"))
        self.assertEqual(condition_order("P02"), ("WITH", "WITHOUT"))
        self.assertEqual(condition_set_map("P01"), {"WITHOUT": "A", "WITH": "B"})
        self.assertEqual(condition_set_map("P02"), {"WITH": "A", "WITHOUT": "B"})
        self.assertEqual(condition_set_map("P03"), {"WITHOUT": "B", "WITH": "A"})
        self.assertEqual(condition_set_map("P04"), {"WITH": "B", "WITHOUT": "A"})

    def test_no_duplicate_stimulus_across_conditions(self):
        for participant in (f"P{index:02d}" for index in range(1, 11)):
            plan = build_trial_plan(participant, BASE_SEED)
            ids = [item.row["stimulus_id"] for item in plan]
            self.assertEqual(len(ids), len(set(ids)))

    def test_reaction_time(self):
        self.assertEqual(calculate_rt_ms(10.0, 10.125), "125.000")
        with self.assertRaises(ValueError):
            calculate_rt_ms(2.0, 1.0)

    def test_sus_scoring(self):
        self.assertEqual(score_sus([3] * 10), 50.0)
        self.assertEqual(score_sus([5, 1] * 5), 100.0)
        with self.assertRaises(ValueError):
            score_sus([3] * 9)

    def test_study_version_lock(self):
        self.assertEqual(load_config()["study_version"], STUDY_VERSION)
        lock = protocol_lock(BASE_SEED)
        self.assertEqual(lock["study_version"], STUDY_VERSION)
        self.assertEqual(len(lock["manifest_sha256"]), 64)

    def test_resume_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "P01"
            reserve_session_dir(path, resume=False, force=False)
            with self.assertRaises(FileExistsError):
                reserve_session_dir(path, resume=False, force=False)
            reserve_session_dir(path, resume=True, force=False)

    def test_pilot_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "main"
            root.mkdir()
            pilot = root / "PILOT01"
            pilot.mkdir()
            (pilot / "session.json").write_text(json.dumps({
                "participant_id": "PILOT01", "pilot": True, "status": "complete"
            }), encoding="utf-8")
            self.assertEqual(discover_real_sessions(root), [])

    def test_product_runtime_evidence_distinguishes_network_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = ProductMonitor([], Path(directory) / "runtime.log", 1.0)
            monitor.lines = [
                {"wall": 1.0, "monotonic": 1.0,
                 "line": "STT final error: Speech recognition request failed: connection timed out"},
            ]
            evidence = monitor.evidence_since(0)
            self.assertTrue(evidence["network_error"])
            self.assertEqual(len(evidence["stt_errors"]), 1)


class ValidationAndAnalysisTests(unittest.TestCase):
    def test_complete_synthetic_fixture_validates_and_analyzes_only_when_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "TEST_SYNTHETIC"
            p1 = make_synthetic_complete_session(root, "P01")
            p2 = make_synthetic_complete_session(root, "P02")
            self.assertTrue(validate_session_dir(p1)["valid"])
            with self.assertRaises(RuntimeError):
                analyze_sessions([p1, p2])
            trials, participants, statistics_output, quality = analyze_sessions(
                [p1, p2], allow_test_fixtures=True
            )
            self.assertEqual(len(trials), 76)
            self.assertEqual(len(participants), 2)
            self.assertEqual(statistics_output["included_n"], 2)
            self.assertEqual(statistics_output["usability"]["sus"]["n"], 2)
            self.assertTrue(all(item["valid"] for item in quality))
            output = Path(directory) / "TEST_SYNTHETIC_OUTPUT"
            write_outputs(output, trials, participants, statistics_output, quality)
            self.assertTrue((output / "LAYER3_SUMMARY.md").is_file())
            self.assertTrue((output / "statistics.json").is_file())
            self.assertTrue(any((output / "figures").glob("*.png")))

    def test_incomplete_session_is_flagged_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "TEST_SYNTHETIC"
            path = make_synthetic_complete_session(root, "P01")
            session_path = path / "session.json"
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session["status"] = "incomplete"
            session_path.write_text(json.dumps(session), encoding="utf-8")
            report = validate_session_dir(path)
            self.assertFalse(report["valid"])
            self.assertIn("incomplete_study", {item["code"] for item in report["issues"]})
            with (path / "trials.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 38)

    def test_protocol_mismatch_is_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_synthetic_complete_session(Path(directory) / "TEST_SYNTHETIC", "P01")
            session_path = path / "session.json"
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session["protocol_lock"]["manifest_sha256"] = "0" * 64
            session_path.write_text(json.dumps(session), encoding="utf-8")
            report = validate_session_dir(path)
            self.assertIn("protocol_lock_mismatch", {item["code"] for item in report["issues"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
