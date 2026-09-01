import csv
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.experiments.semantic_mapping import active_rules
from soundguard.detection.sound_taxonomy import SEMANTIC_CED_ALIASES, map_ced_label


ROOT = Path(__file__).resolve().parents[1]


class ImprovementTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING", None)
        os.environ.pop("SOUNDGUARD_SAFE_COMPOUND_EVIDENCE", None)

    def test_runtime_taxonomy_matches_accepted_experiment_rules(self):
        expected = {raw: rule["soundguard_category"] for raw, rule in active_rules().items()}
        self.assertEqual(expected, SEMANTIC_CED_ALIASES)

    def test_broad_labels_are_rejected(self):
        for label in ("Animal", "Vehicle", "Breaking", "Crying, sobbing", "Chink, clink", "Smash, crash"):
            self.assertIsNone(map_ced_label(label), label)

    def test_classifier_mapping_is_default_off_and_opt_in(self):
        from soundguard.detection import sound_classifier
        fake = lambda *args, **kwargs: [
            {"label": "Emergency vehicle", "score": .8}, {"label": "Siren", "score": .7}]
        with patch.object(sound_classifier, "_get_classifier", return_value=fake), \
             patch.object(sound_classifier, "load_audio_for_ced", return_value=([0.0, 0.1], 16000)):
            legacy = sound_classifier.classify_audio_file("ignored.wav")
            self.assertEqual(legacy["label"], "Emergency vehicle")
            self.assertNotIn("raw_label", legacy)
            os.environ["SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING"] = "1"
            improved = sound_classifier.classify_audio_file("ignored.wav")
            self.assertEqual(improved["label"], "siren")
            self.assertEqual(improved["raw_label"], "Emergency vehicle")

    def test_compound_protection_is_default_off_and_opt_in(self):
        from personalization.profile_generator import generate_rule_based
        legacy = generate_rule_based(["A factory-method pattern and baby-blue color."])
        self.assertIn("factory_warehouse_worker", legacy["roles"])
        self.assertIn("parent_infant_caregiver", legacy["roles"])
        os.environ["SOUNDGUARD_SAFE_COMPOUND_EVIDENCE"] = "1"
        improved = generate_rule_based(["A factory-method pattern and baby-blue color."])
        self.assertEqual(improved["roles"], [])

    def test_candidate_metrics_are_frozen_and_no_emergency_fp(self):
        summary = json.loads((ROOT / "benchmark_results/improvement/candidate_summary.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(summary["semantic"]["macro_f1"], .65)
        self.assertEqual(summary["semantic"]["emergency_fp"], 0)
        self.assertLess(summary["best_pooling"]["macro_f1"], summary["baseline"]["macro_f1"])
        self.assertEqual(summary["personalization"]["compound_failures_remaining"], 0)

    def test_raw_stt_routing_wins_fixed_real_noise_ab(self):
        with (ROOT / "benchmark_results/improvement/dtln_routing_comparison.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        raw = next(row for row in rows if row["stt_route"] == "raw")
        dtln = next(row for row in rows if row["stt_route"] == "DTLN")
        self.assertLess(float(raw["stt_wer"]), float(dtln["stt_wer"]))


if __name__ == "__main__":
    unittest.main()
