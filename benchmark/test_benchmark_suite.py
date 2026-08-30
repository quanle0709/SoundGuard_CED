import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.metrics import binary_metrics, classification_metrics, error_rates, latency_summary
from benchmark.noise import align_to_reference, mix_environmental_noise, mix_white_noise
from benchmark.run_expanded_benchmark import aggregate_errors, bootstrap_ci, target_classification_metrics


class BenchmarkSuiteTests(unittest.TestCase):
    def test_classification_metrics_known_case(self):
        result = classification_metrics(["a", "a", "b"], ["a", "b", "b"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["count"], 3)

    def test_binary_metrics_known_case(self):
        result = binary_metrics([True, True, False, False], [True, False, True, False])
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 1, 1))

    def test_vietnamese_error_rates_preserve_diacritics(self):
        result = error_rates("Xin chào!", "xin chao")
        self.assertEqual(result["reference"], "xin chào")
        self.assertGreater(result["wer"], 0)

    def test_noise_mix_hits_target_without_clipping(self):
        signal = np.sin(np.linspace(0, 50, 16000)).astype(np.float32) * 0.5
        mixed, metadata = mix_white_noise(signal, 10, 42)
        self.assertAlmostEqual(metadata["achieved_snr_db"], 10, places=5)
        self.assertLessEqual(float(np.max(np.abs(mixed))), 1.0)

    def test_noise_mix_is_reproducible(self):
        signal = np.ones(1000, dtype=np.float32) * 0.1
        first, _ = mix_white_noise(signal, 5, 42)
        second, _ = mix_white_noise(signal, 5, 42)
        np.testing.assert_array_equal(first, second)

    def test_latency_summary_percentiles(self):
        result = latency_summary([1, 2, 3, 4, 5])
        self.assertEqual(result["median_ms"], 3)
        self.assertGreaterEqual(result["p95_ms"], result["p90_ms"])

    def test_fixed_filter_delay_is_removed_before_signal_metric(self):
        reference = np.random.default_rng(42).standard_normal(4000).astype(np.float32)
        delayed = np.concatenate([np.zeros(384, dtype=np.float32), reference])[:len(reference)]
        aligned_reference, aligned_degraded, lag = align_to_reference(reference, delayed)
        self.assertEqual(lag, 384)
        np.testing.assert_allclose(aligned_reference, aligned_degraded, atol=1e-6)

    def test_environmental_noise_mix_is_reproducible_and_hits_target(self):
        signal = np.sin(np.linspace(0, 20, 1600)).astype(np.float32) * 0.2
        noise = np.random.default_rng(7).standard_normal(8000).astype(np.float32)
        first, metadata = mix_environmental_noise(signal, noise, 5, 42)
        second, second_metadata = mix_environmental_noise(signal, noise, 5, 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(metadata["start_sample"], second_metadata["start_sample"])
        self.assertAlmostEqual(metadata["achieved_snr_db"], 5, places=4)
        self.assertLessEqual(float(np.max(np.abs(first))), 1.0)

    def test_bootstrap_confidence_interval_is_deterministic(self):
        truths = ["a", "a", "b", "b"]
        predictions = ["a", "b", "b", "b"]
        self.assertEqual(bootstrap_ci(truths, predictions, "accuracy", 50),
                         bootstrap_ci(truths, predictions, "accuracy", 50))

    def test_aggregate_stt_errors_uses_corpus_denominator(self):
        rows = [{"reference_words": 2, "reference_characters": 5, "substitutions": 1,
                 "deletions": 0, "insertions": 0, "character_errors": 1},
                {"reference_words": 8, "reference_characters": 15, "substitutions": 0,
                 "deletions": 1, "insertions": 1, "character_errors": 2}]
        result = aggregate_errors(rows)
        self.assertEqual(result["utterances"], 2)
        self.assertAlmostEqual(result["wer"], 0.3)
        self.assertAlmostEqual(result["cer"], 0.15)

    def test_expanded_macro_f1_uses_declared_target_classes(self):
        result = target_classification_metrics(["a", "a", "b", "b"],
                                               ["a", "outside", "b", "outside"])
        self.assertEqual(result["labels"], ["a", "b", "other"])
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(len(result["per_class"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
