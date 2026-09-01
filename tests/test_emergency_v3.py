import io
import json
import os
import queue
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from soundguard import app
from soundguard.audio.audio_pipeline import CEDBranch, PipelineEvent
from soundguard.emergency.emergency_system import EmergencySystem
from soundguard.emergency.emergency_v3 import (
    ENABLE_ENV,
    EmergencyV3Specialist,
    create_emergency_v3_specialist,
    emergency_v3_enabled,
    select_emergency_evidence,
)
from personalization import ProfileManager


class FakeProcess:
    def __init__(self, responses):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("".join(json.dumps(item) + "\n" for item in responses))
        self.terminated = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


class FakeHUD:
    def __init__(self):
        self.alerts = []
        self.subtitles = []
        self.environmental_sounds = []
        self.alert_states = []

    def set_alert(self, value):
        self.alerts.append(value)

    def set_subtitle(self, value):
        self.subtitles.append(value)

    def set_environmental_sound(self, value):
        self.environmental_sounds.append(value)

    def set_alert_state(self, event_label="", help_active=False):
        self.alert_states.append((event_label, help_active))


class EmergencyV3Tests(unittest.TestCase):
    def test_feature_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(emergency_v3_enabled())
            self.assertIsNone(create_emergency_v3_specialist())

    def test_feature_can_be_enabled_explicitly(self):
        with patch.dict(os.environ, {}, clear=True):
            specialist = create_emergency_v3_specialist(explicit=True)
            self.assertIsInstance(specialist, EmergencyV3Specialist)

    def test_model_unavailable_falls_back(self):
        messages = []
        specialist = EmergencyV3Specialist(
            python_path="missing-python.exe", logger=messages.append
        )
        result = specialist.analyze_file("missing.wav")
        self.assertFalse(result["detected"])
        self.assertEqual("v2_fallback", result["source"])
        self.assertEqual(1, len(messages))

    def test_worker_exception_falls_back_and_logs_only_once(self):
        messages = []

        def fail(*args, **kwargs):
            raise OSError("worker failed")

        specialist = EmergencyV3Specialist(
            python_path=Path(__file__), popen_factory=fail, logger=messages.append
        )
        self.assertFalse(specialist.analyze_file("one.wav")["detected"])
        self.assertFalse(specialist.analyze_file("two.wav")["detected"])
        self.assertEqual(1, len(messages))

    def test_valid_safety_detection_becomes_additional_evidence(self):
        result = {"detected": True, "category": "siren", "sound_label": "siren"}
        self.assertEqual(
            ("siren", 1.0, "efficientsed"),
            select_emergency_evidence("dog", 0.9, result),
        )

    def test_non_safety_detection_is_ignored(self):
        result = {"detected": True, "category": "dog", "sound_label": "dog"}
        self.assertEqual(
            ("speech", 0.8, "ced_tiny_v2"),
            select_emergency_evidence("speech", 0.8, result),
        )

    def test_existing_dangerous_ced_evidence_takes_precedence(self):
        result = {"detected": True, "category": "fire", "sound_label": "fire"}
        self.assertEqual(
            ("siren", 0.9, "ced_tiny_v2"),
            select_emergency_evidence("siren", 0.9, result),
        )

    def test_threshold_boundary(self):
        from benchmark.experiments.emergency_v3.efficientsed_worker import THRESHOLD

        self.assertEqual(0.05, THRESHOLD)
        self.assertEqual(
            "efficientsed",
            select_emergency_evidence(
                "dog", 0.9,
                {"detected": 0.05 >= THRESHOLD, "sound_label": "fire"},
            )[2],
        )

    def test_taxonomy_includes_exact_and_rejects_broad_labels(self):
        path = Path("benchmark/experiments/emergency_v3/efficientsed_safety_taxonomy.json")
        entries = {
            entry["raw_label"]: entry
            for entry in json.loads(path.read_text(encoding="utf-8"))["entries"]
        }
        self.assertEqual("EXACT", entries["Glass shatter"]["classification"])
        self.assertEqual("glass_breaking", entries["Glass shatter"]["soundguard_category"])
        for label in ("Animal", "Vehicle", "Breaking", "Crying, sobbing"):
            self.assertIn(entries[label]["classification"], {"TOO_BROAD", "EXCLUDED"})
            self.assertIsNone(entries[label]["soundguard_category"])

    def test_existing_ced_result_is_preserved(self):
        class Specialist:
            def analyze_samples(self, audio, sample_rate):
                return {"detected": True, "category": "siren", "sound_label": "siren"}

        events = queue.Queue()
        branch = CEDBranch(
            queue.Queue(), events, 5.0, 0.0,
            classifier=lambda audio, rate: {"label": "Bark", "confidence": 0.8},
            emergency_v3_specialist=Specialist(),
        )
        branch._classify(PipelineEvent("CED_DUE", "ced", 0, 5, 1, np.zeros(80)))
        result = events.get_nowait().payload
        self.assertEqual("Bark", result["label"])
        self.assertEqual(0.8, result["confidence"])
        self.assertEqual("siren", result["emergency_v3"]["category"])

    def test_existing_duplicate_suppression_remains_active(self):
        clock = iter((1.0, 2.0))
        system = EmergencySystem(decision_mode="single_shot", clock=lambda: next(clock))
        first = system.process_sound_event("siren", 1.0)
        second = system.process_sound_event("siren", 1.0)
        self.assertTrue(first["emitted_this_cycle"])
        self.assertFalse(second["emitted_this_cycle"])

    def test_worker_is_reused_across_chunks(self):
        starts = []
        process = FakeProcess([
            {"ok": True, "detected": False, "category": "fire", "score": 0.01},
            {"ok": True, "detected": False, "category": "fire", "score": 0.02},
        ])

        def start(*args, **kwargs):
            starts.append(args)
            return process

        specialist = EmergencyV3Specialist(
            python_path=Path(__file__), popen_factory=start
        )
        specialist.analyze_file("one.wav")
        specialist.analyze_file("two.wav")
        self.assertEqual(1, len(starts))
        specialist.close()

    def test_stt_and_fusion_receive_legacy_inputs(self):
        captured = {}

        def fuse(timestamp, transcript, sound, alert):
            captured.update(transcript=transcript, sound=sound)
            return {"timestamp": timestamp}

        class Specialist:
            def analyze_file(self, path):
                return {
                    "ok": True, "detected": True, "category": "glass_breaking",
                    "sound_label": "glass breaking", "score": 0.7,
                }

        class System:
            context = "neutral"

            def evaluate(self, **kwargs):
                captured["emergency"] = kwargs
                return {
                    "help_request_detected": False, "alert_level": "HIGH",
                    "alert_text": "alert", "context": "neutral",
                    "event_state": "EVENT_STARTED", "decision_mode": "single_shot",
                    "temporal_confirmation": False, "history_window": [],
                    "missing_cycles": 0, "emitted_this_cycle": True,
                }

        with patch.object(app, "transcribe_audio_file", return_value="unchanged transcript"), \
             patch.object(app, "classify_audio_file", return_value={
                 "label": "Bark", "confidence": 0.8, "top_predictions": []}), \
             patch.object(app, "map_alert", return_value={"message_vi": "", "is_dangerous": False}), \
             patch.object(app, "fuse_result", side_effect=fuse):
            app.run_analysis(
                "audio.wav", "speech.wav", True, System(), FakeHUD(), Specialist()
            )
        self.assertEqual("unchanged transcript", captured["transcript"])
        self.assertEqual({"label": "Bark", "confidence": 0.8}, captured["sound"])
        self.assertEqual("glass breaking", captured["emergency"]["sound_label"])

    def test_personalization_is_unchanged_when_v3_environment_is_set(self):
        before = ProfileManager().load()
        with patch.dict(os.environ, {ENABLE_ENV: "1"}):
            after = ProfileManager().load()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
