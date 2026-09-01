import time
import contextlib
import shutil
from io import StringIO

import numpy as np

from live_speech_to_text import (
    RecognitionJob,
    RecognitionWorker,
    RecognitionResult,
    TranscriptDisplay,
    TranscriptTracker,
    process_recognition_result,
    recognize_snapshot,
)
from emergency_system import EmergencySystem


def test_partial_replacement_and_duplicate_suppression():
    tracker = TranscriptTracker()
    assert tracker.accept_partial(1, "xin chào") == "xin chào"
    assert tracker.accept_partial(1, "xin chào") is None
    assert tracker.accept_partial(1, "xin chào tôi đang") == "xin chào tôi đang"
    assert tracker.partials[1] == "xin chào tôi đang"
    assert tracker.history == []


def test_partial_growth_and_regression_stability():
    tracker = TranscriptTracker()
    assert tracker.accept_partial(1, "  em   tên ") == "em tên"
    assert tracker.accept_partial(1, "em tên là Minh") == "em tên là Minh"
    assert tracker.last_stable_partial == "em tên là Minh"
    assert tracker.accept_partial(1, "tên là") is None
    assert tracker.last_stable_partial == "em tên là Minh"
    assert tracker.accept_partial(1, "   ") is None
    assert tracker.last_stable_partial == "em tên là Minh"


def test_equal_word_count_regression_is_rejected():
    tracker = TranscriptTracker()
    assert tracker.accept_partial(1, "em tên là Minh") == "em tên là Minh"
    assert tracker.accept_partial(1, "tên em Minh là") is None
    assert tracker.last_stable_partial == "em tên là Minh"

    assert tracker.accept_partial(1, "em tên thật là Minh") == "em tên thật là Minh"


def test_finalization_and_stale_partial():
    tracker = TranscriptTracker()
    tracker.accept_partial(7, "bản nháp")
    assert tracker.accept_final(7, "bản hoàn chỉnh") == "bản hoàn chỉnh"
    assert tracker.history == ["bản hoàn chỉnh"]
    assert tracker.accept_final(7, "bản hoàn chỉnh") is None
    assert tracker.accept_partial(7, "đến muộn") is None


def test_final_history_survives_partial_reset():
    tracker = TranscriptTracker()
    tracker.accept_partial(1, "em tên là Minh")
    final = tracker.accept_final(1, "Em tên là Minh.")
    assert final == "Em tên là Minh."
    assert tracker.last_stable_partial == "em tên là Minh"
    tracker.mark_final_displayed(1)
    assert tracker.last_stable_partial == ""
    assert tracker.history == ["Em tên là Minh."]
    assert tracker.accept_partial(2, "xin chào") == "xin chào"
    assert tracker.last_stable_partial == "xin chào"


def test_display_clears_only_partial_line():
    stream = StringIO()
    display = TranscriptDisplay(stream)
    display.print_permanent("FINAL: câu trước")
    display.show_partial("em tên là Minh")
    display.print_permanent("FINAL: Em tên là Minh.")
    display.print_permanent("[WAITING FOR SPEECH]")
    output = stream.getvalue()
    assert output.startswith("FINAL: câu trước\n")
    assert "\r\x1b[2KPARTIAL: em tên là Minh" in output
    assert "\r\x1b[2K\nFINAL: Em tên là Minh." in output
    assert "FINAL: Em tên là Minh.\n[WAITING FOR SPEECH]\n" in output


def test_empty_final_and_new_utterance():
    tracker = TranscriptTracker()
    assert tracker.accept_final(1, "") is None
    assert 1 not in tracker.finalized_ids
    assert tracker.accept_final(1, "kết quả đến muộn") == "kết quả đến muộn"
    assert tracker.accept_partial(2, "mới") == "mới"
    assert tracker.accept_final(2, "mới hoàn chỉnh") == "mới hoàn chỉnh"


def test_empty_final_uses_stable_partial_fallback():
    tracker = TranscriptTracker()
    tracker.accept_partial(3, "em tên là Minh")
    assert tracker.accept_final(3, "   ") == "em tên là Minh"
    assert tracker.history == ["em tên là Minh"]
    assert 3 in tracker.finalized_ids


def test_reset():
    tracker = TranscriptTracker()
    tracker.accept_partial(1, "a")
    tracker.accept_final(1, "a")
    tracker.reset()
    assert tracker.partials == {}
    assert tracker.finalized_ids == set()
    assert tracker.history == []


def test_worker_reports_failure_and_continues():
    calls = 0

    def recognizer(audio, use_dtln):
        nonlocal calls
        del audio, use_dtln
        calls += 1
        if calls == 1:
            raise RuntimeError("network unavailable")
        return "đã phục hồi"

    worker = RecognitionWorker(False, recognizer=recognizer)
    worker.start()
    assert worker.submit(RecognitionJob("PARTIAL", 1, np.zeros(512)))
    assert worker.submit(RecognitionJob("FINAL", 1, np.zeros(512)))
    deadline = time.monotonic() + 2
    results = []
    while len(results) < 2 and time.monotonic() < deadline:
        try:
            results.append(worker.results.get(timeout=0.1))
        except Exception:
            pass
    worker.stop()
    assert len(results) == 2
    assert "network unavailable" in results[0].error
    assert results[1].transcript == "đã phục hồi"


class FakeClock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value


def test_final_help_reaches_emergency_and_prints_fields():
    tracker = TranscriptTracker()
    stream = StringIO()
    display = TranscriptDisplay(stream)
    system = EmergencySystem(clock=FakeClock())
    decision = process_recognition_result(
        RecognitionResult("FINAL", 1, transcript="Cứu tôi với"),
        tracker,
        display,
        system,
    )
    assert decision["help_request_detected"] is True
    assert decision["alert_level"] == "CRITICAL"
    assert decision["event_state"] == "EVENT_STARTED"
    assert decision["emitted_this_cycle"] is True
    assert decision["cooldown_active"] is False
    output = stream.getvalue()
    for label in (
        "FINAL: Cứu tôi với", "Help request: True", "Alert level: CRITICAL",
        "Alert message:", "Event state: EVENT_STARTED", "Context: neutral",
        "Cooldown active: False", "Emitted this cycle: True",
    ):
        assert label in output


def test_repeated_help_respects_cooldown():
    tracker = TranscriptTracker()
    display = TranscriptDisplay(StringIO())
    system = EmergencySystem(clock=FakeClock())
    process_recognition_result(
        RecognitionResult("FINAL", 1, transcript="Cứu tôi với"),
        tracker, display, system,
    )
    repeated = process_recognition_result(
        RecognitionResult("FINAL", 2, transcript="Cứu tôi với"),
        tracker, display, system,
    )
    assert repeated["event_state"] == "EVENT_CONTINUING"
    assert repeated["cooldown_active"] is True
    assert repeated["emitted_this_cycle"] is False


def test_repeated_fire_phrase_respects_cooldown_and_keeps_reason():
    tracker = TranscriptTracker()
    stream = StringIO()
    display = TranscriptDisplay(stream)
    system = EmergencySystem(clock=FakeClock())
    first = process_recognition_result(
        RecognitionResult("FINAL", 1, transcript="nhà cháy rồi"),
        tracker, display, system,
    )
    repeated = process_recognition_result(
        RecognitionResult("FINAL", 2, transcript="có đám cháy"),
        tracker, display, system,
    )
    assert first["help_request_detected"] is True
    assert first["alert_text"] == "Phát hiện tình huống cháy"
    assert repeated["event_state"] == "EVENT_CONTINUING"
    assert repeated["cooldown_active"] is True
    assert repeated["emitted_this_cycle"] is False
    assert repeated["alert_text"] == "Phát hiện tình huống cháy"
    assert "Alert message: Phát hiện tình huống cháy" in stream.getvalue()


def test_empty_final_does_not_reach_emergency():
    class RejectEmergency:
        def evaluate(self, *args, **kwargs):
            raise AssertionError("EmergencySystem must not receive an empty FINAL")

    decision = process_recognition_result(
        RecognitionResult("FINAL", 1, transcript=""),
        TranscriptTracker(), TranscriptDisplay(StringIO()), RejectEmergency(),
    )
    assert decision is None


def test_ordinary_final_remains_low():
    decision = process_recognition_result(
        RecognitionResult("FINAL", 1, transcript="Hôm nay trời đẹp"),
        TranscriptTracker(), TranscriptDisplay(StringIO()),
        EmergencySystem(clock=FakeClock()),
    )
    assert decision["help_request_detected"] is False
    assert decision["alert_level"] == "LOW"
    assert decision["emitted_this_cycle"] is False


def test_live_dtln_is_quiet():
    import audio_capture
    import speech_enhancer
    import speech_recognizer

    original_enhance = speech_enhancer.enhance_audio_file
    original_optimize = audio_capture.create_speech_optimized_wav
    original_transcribe = speech_recognizer.transcribe_audio_file
    verbose_values = []

    def fake_enhance(input_wav, output_wav, verbose=True):
        verbose_values.append(verbose)
        shutil.copyfile(input_wav, output_wav)
        return str(output_wav)

    def fake_optimize(raw_wav_path, output_path=None, preserve_tail_ms=0):
        del preserve_tail_ms
        shutil.copyfile(raw_wav_path, output_path)
        return output_path

    try:
        speech_enhancer.enhance_audio_file = fake_enhance
        audio_capture.create_speech_optimized_wav = fake_optimize
        speech_recognizer.transcribe_audio_file = lambda path: "test"
        captured = StringIO()
        with contextlib.redirect_stdout(captured):
            assert recognize_snapshot(
                np.zeros(1600, dtype=np.float32), True,
                utterance_id=1, is_final=True,
            ) == "test"
        assert captured.getvalue() == ""
        assert verbose_values == [False]
    finally:
        speech_enhancer.enhance_audio_file = original_enhance
        audio_capture.create_speech_optimized_wav = original_optimize
        speech_recognizer.transcribe_audio_file = original_transcribe


def run_tests():
    tests = [
        test_partial_replacement_and_duplicate_suppression,
        test_partial_growth_and_regression_stability,
        test_equal_word_count_regression_is_rejected,
        test_finalization_and_stale_partial,
        test_final_history_survives_partial_reset,
        test_display_clears_only_partial_line,
        test_empty_final_and_new_utterance,
        test_empty_final_uses_stable_partial_fallback,
        test_reset,
        test_worker_reports_failure_and_continues,
        test_final_help_reaches_emergency_and_prints_fields,
        test_repeated_help_respects_cooldown,
        test_repeated_fire_phrase_respects_cooldown_and_keeps_reason,
        test_empty_final_does_not_reach_emergency,
        test_ordinary_final_remains_low,
        test_live_dtln_is_quiet,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: passed")
    print("transcript logic tests passed")


if __name__ == "__main__":
    run_tests()
