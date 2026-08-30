import json
import math
import unittest
from pathlib import Path

from benchmark.experiments.golden import OUT, capture


class GoldenLegacyTests(unittest.TestCase):
    def test_features_off_match_frozen_outputs(self):
        expected = json.loads(Path(OUT).read_text(encoding="utf-8"))
        actual = capture()
        for expected_row, actual_row in zip(expected.pop("ced"), actual.pop("ced")):
            self.assertEqual(expected_row["path"], actual_row["path"])
            self.assertEqual(expected_row["label"], actual_row["label"])
            self.assertTrue(math.isclose(expected_row["confidence"], actual_row["confidence"], abs_tol=1e-6))
            self.assertEqual([r["label"] for r in expected_row["top_predictions"]],
                             [r["label"] for r in actual_row["top_predictions"]])
            for left, right in zip(expected_row["top_predictions"], actual_row["top_predictions"]):
                self.assertTrue(math.isclose(left["score"], right["score"], abs_tol=1e-6))
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
