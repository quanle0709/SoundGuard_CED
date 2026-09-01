import csv
import tempfile
from pathlib import Path

from benchmark_metrics import calculate_metrics
from benchmark_report import generate_report
from benchmark_runner import RESULT_FIELDS


def row(**values):
    result = {field: "" for field in RESULT_FIELDS}
    result["status"] = "completed"
    result.update(values)
    return result


def write_results(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader(); writer.writerows(rows)


def test_wer_cer_exact_empty_and_missing_truth_exclusion():
    metrics = calculate_metrics([
        row(expected_speech="true", stt_enabled="true", stt_status="success", ground_truth_transcript="xin chao", transcript="xin ban"),
        row(expected_speech="true", stt_enabled="true", stt_status="success", ground_truth_transcript="abc", transcript=""),
        row(expected_speech="true", stt_enabled="true", stt_status="success", ground_truth_transcript="", transcript="invented output"),
        row(expected_speech="false", stt_enabled="false", ground_truth_transcript="", transcript=""),
    ])
    assert metrics["transcript_ground_truth_rows"] == 2
    assert metrics["speech_rows_missing_transcript_ground_truth"] == 1
    assert metrics["non_speech_rows_excluded_from_transcript_metrics"] == 1
    assert metrics["wer"] == 2 / 3
    assert metrics["cer"] > 0 and metrics["exact_transcript_match"] == 0
    assert metrics["empty_transcript_rate"] == .5


def test_confusion_per_class_and_accuracy():
    metrics = calculate_metrics([
        row(ground_truth_sound="Siren", predicted_sound="Siren",
            ground_truth_category="alarm", predicted_category="alarm"),
        row(ground_truth_sound="Alarm", predicted_sound="Siren",
            ground_truth_category="alarm", predicted_category="vehicle"),
        row(ground_truth_sound="Horn", predicted_sound="Horn",
            ground_truth_category="vehicle", predicted_category="vehicle"),
        row(),
    ])
    assert metrics["sound_ground_truth_rows"] == 3
    assert metrics["canonical_category_accuracy"] == 2 / 3
    assert metrics["confusion_matrix"]["alarm -> vehicle"] == 1
    assert metrics["per_class"]["alarm"]["recall"] == .5


def test_alert_help_vad_emergency_latency_and_pairs():
    rows = [
        row(sample_id="a", run_id="r", condition="dtln_disabled", stt_enabled="true", stt_status="success",
            ced_status="success",
            ground_truth_transcript="one two", transcript="bad bad",
            expected_alert_level="HIGH", sound_alert_level="HIGH",
            expected_help_request="true", help_request_detected="true",
            expected_speech="true", vad_detected="true", expected_emergency="false",
            emergency_detected="true", total_latency_seconds="2"),
        row(sample_id="a", run_id="r", condition="dtln_enabled", stt_enabled="true", stt_status="success",
            ground_truth_transcript="one two", transcript="one two",
            expected_speech="true",
            expected_emergency="true", emergency_detected="false", total_latency_seconds="4"),
    ]
    metrics = calculate_metrics(rows)
    assert metrics["expected_alert_level_accuracy"] == 1
    assert metrics["help_request_accuracy"] == 1 and metrics["vad_correctness"] == 1
    assert metrics["false_emergencies"] == 1 and metrics["missed_emergencies"] == 1
    assert metrics["latencies"]["total_latency_seconds"]["mean"] == 3
    assert metrics["paired_dtln"]["improved"] == 1


def test_failed_branch_does_not_invalidate_successful_stage_metrics():
    metrics = calculate_metrics([
        row(status="failed", ced_status="success", vad_status="failed", stt_status="failed",
            ground_truth_sound="Bark", predicted_sound="Bark", ground_truth_category="dog",
            predicted_category="dog", expected_speech="true", errors="vad failed"),
    ])
    assert metrics["completed_rows"] == 0 and metrics["failed_rows"] == 1
    assert metrics["sound_ground_truth_rows"] == 1
    assert metrics["top1_sound_accuracy"] == 1
    assert metrics["canonical_category_accuracy"] == 1
    assert metrics["vad_correctness"] is None


def test_report_creation_and_disclosures():
    with tempfile.TemporaryDirectory() as directory:
        source, output = Path(directory) / "results.csv", Path(directory) / "outputs/report.md"
        write_results(source, [row(status="failed", errors="model unavailable")])
        generate_report(source, output, automated_tests="deterministic suite passed")
        text = output.read_text(encoding="utf-8")
        assert "Failed or incomplete rows: 1" in text
        assert "deterministic suite passed" in text
        assert "small, hand-selected dataset does not prove real-world accuracy" in text


def test_filtering_one_run_id():
    rows = [row(run_id="old", sample_id="a"), row(run_id="stt_02", sample_id="b"),
            row(run_id="stt_02", sample_id="c")]
    metrics = calculate_metrics(rows, run_id="stt_02")
    assert metrics["sample_condition_row_count"] == 2
    assert metrics["included_run_ids"] == ["stt_02"]


def test_aggregate_report_lists_included_run_ids():
    with tempfile.TemporaryDirectory() as directory:
        source, output = Path(directory) / "results.csv", Path(directory) / "report.md"
        write_results(source, [row(run_id="baseline_01"), row(run_id="stt_02")])
        generate_report(source, output)
        text = output.read_text(encoding="utf-8")
        assert "Aggregate historical report (mixed runs)" in text
        assert "Included run IDs:** baseline_01, stt_02" in text


def test_separates_dtln_conditions():
    rows = [
        row(run_id="r", scenario="clean_speech", condition="dtln_disabled",
            expected_speech="true", stt_status="success", ground_truth_transcript="a", transcript="a"),
        row(run_id="r", scenario="noisy_speech", condition="dtln_enabled",
            expected_speech="true", stt_status="success", ground_truth_transcript="a", transcript="a"),
        row(run_id="r", scenario="noisy_speech", condition="dtln_disabled",
            expected_speech="true", stt_status="success", ground_truth_transcript="a", transcript="wrong"),
    ]
    metrics = calculate_metrics(rows)
    assert metrics["condition_metrics"]["clean_speech_dtln_disabled"]["wer"] == 0
    assert metrics["condition_metrics"]["noisy_speech_dtln_enabled"]["wer"] == 0
    assert metrics["condition_metrics"]["noisy_speech_dtln_disabled"]["wer"] == 1


def test_intentional_non_speech_transcript_is_excluded():
    metrics = calculate_metrics([
        row(expected_speech="false", stt_status="skipped_no_speech",
            ground_truth_transcript="", scenario="environmental")])
    assert metrics["non_speech_rows_excluded_from_transcript_metrics"] == 1
    assert metrics["speech_rows_missing_transcript_ground_truth"] == 0
    assert metrics["wer"] is None


def test_no_positive_emergency_disclosure():
    with tempfile.TemporaryDirectory() as directory:
        source, output = Path(directory) / "results.csv", Path(directory) / "report.md"
        write_results(source, [row(run_id="r", expected_emergency="false", emergency_detected="false")])
        generate_report(source, output, run_id="r")
        text = output.read_text(encoding="utf-8")
        assert "no positive emergency samples" in text
        assert "False-emergency behavior was tested" in text
        assert "Zero missed emergencies does not prove" in text


def test_canonical_and_exact_label_reporting_are_distinct():
    with tempfile.TemporaryDirectory() as directory:
        source, output = Path(directory) / "results.csv", Path(directory) / "report.md"
        write_results(source, [row(ced_status="success", ground_truth_sound="Speech",
                                   predicted_sound="Speech synthesizer",
                                   ground_truth_category="speech", predicted_category="speech")])
        metrics = generate_report(source, output)
        text = output.read_text(encoding="utf-8")
        assert metrics["top1_sound_accuracy"] == 0
        assert metrics["canonical_category_accuracy"] == 1
        assert "CED is multi-label" in text and "Exact top-1 accuracy alone" in text


def run_tests():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test(); print(f"{test.__name__}: passed")
    print(f"benchmark metrics tests passed ({len(tests)})")


if __name__ == "__main__": run_tests()
