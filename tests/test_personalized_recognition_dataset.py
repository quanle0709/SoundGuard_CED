from tools.evaluate_personalized_recognition_dataset import (
    apply_operating_point,
    classification_metrics,
    select_operating_point,
)


def calibration_row(expected, top1, score, margin):
    return {
        "expected_label": expected,
        "top1_label": top1,
        "score": score,
        "runner_up_margin": margin,
        "predicted_label": "UNKNOWN",
        "accepted": False,
        "reason": "uncalibrated",
    }


def test_calibration_uses_known_and_unknown_examples_deterministically():
    rows = [
        calibration_row("A", "A", 0.91, 0.25),
        calibration_row("B", "B", 0.88, 0.20),
        calibration_row("UNKNOWN", "A", 0.58, 0.05),
        calibration_row("UNKNOWN", "B", 0.62, 0.03),
    ]
    first = select_operating_point(rows)
    second = select_operating_point(list(reversed(rows)))
    assert first == second
    assert first["known_correct"] == 2
    assert first["unknown_rejected"] == 2
    assert 0.35 <= first["threshold"] <= 0.95
    assert 0 <= first["margin"] <= 0.30


def test_operating_point_rejection_reason_matches_production_order():
    weak = apply_operating_point(
        calibration_row("A", "A", 0.60, 0.30), threshold=0.70, margin=0.10
    )
    ambiguous = apply_operating_point(
        calibration_row("A", "A", 0.90, 0.05), threshold=0.70, margin=0.10
    )
    assert weak["reason"] == "below_threshold"
    assert ambiguous["reason"] == "ambiguous_margin"


def test_metrics_report_raw_open_set_numerators_and_confusion():
    rows = [
        {**calibration_row("A", "A", 0.9, 0.2), "predicted_label": "A",
         "accepted": True, "reason": "accepted"},
        {**calibration_row("A", "A", 0.6, 0.2), "predicted_label": "UNKNOWN",
         "accepted": False, "reason": "below_threshold"},
        {**calibration_row("UNKNOWN", "A", 0.8, 0.2), "predicted_label": "A",
         "accepted": True, "reason": "accepted"},
        {**calibration_row("UNKNOWN", "B", 0.5, 0.1), "predicted_label": "UNKNOWN",
         "accepted": False, "reason": "below_threshold"},
    ]
    metrics = classification_metrics(rows)
    assert metrics["correct_identification"] == {"numerator": 1, "denominator": 2}
    assert metrics["false_rejection"] == {"numerator": 1, "denominator": 2}
    assert metrics["false_acceptance"] == {"numerator": 1, "denominator": 2}
    assert metrics["unknown_rejection"] == {"numerator": 1, "denominator": 2}
    assert metrics["confusion_matrix"]["A"]["UNKNOWN"] == 1
