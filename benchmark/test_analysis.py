import math
import csv
from pathlib import Path
import unittest

import numpy as np

from benchmark.analyze_expanded_benchmark import best_scale, estimate_lag, strip_diacritics, validate_snr_metric
from benchmark.noise import snr_db


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_snr_known_values_and_identity(self):
        result = validate_snr_metric()
        self.assertTrue(result["all_known_cases_pass"])
        self.assertTrue(result["edge_cases"]["identical_is_infinite"])

    def test_lag_estimator_recovers_fixed_delay(self):
        rng = np.random.default_rng(7)
        reference = rng.standard_normal(4096).astype(np.float32)
        delayed = np.concatenate((np.zeros(384, dtype=np.float32), reference))[:len(reference)]
        self.assertEqual(estimate_lag(reference, delayed), 384)

    def test_gain_correction_recovers_scaled_identity(self):
        reference = np.linspace(-1, 1, 1000, dtype=np.float32)
        scaled = reference * 0.25
        gain = best_scale(reference, scaled)
        self.assertAlmostEqual(gain, 4.0, places=5)
        self.assertTrue(math.isinf(snr_db(reference, scaled * gain)))

    def test_strip_diacritics_preserves_base_letters(self):
        self.assertEqual(strip_diacritics("tiếng Việt"), "tieng Viet")

    def test_full_analysis_denominators_when_artifacts_exist(self):
        root = Path(__file__).resolve().parents[1] / "benchmark_results" / "analysis"
        required = {
            "ced/clean_top5_details.csv": 280,
            "ced/glass_breaking_failures.csv": 40,
            "ced/dog_barking_failures.csv": 40,
            "emergency/all_false_negatives.csv": 98,
            "dtln/ced_transition_matrix.csv": 17,
            "dtln/stt_transition_analysis.csv": 150,
        }
        for relative, expected in required.items():
            path = root / relative
            if not path.exists():
                self.skipTest("full analysis artifacts have not been generated")
            with path.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), expected, relative)


if __name__ == "__main__":
    unittest.main()
