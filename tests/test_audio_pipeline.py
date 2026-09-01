import contextlib
import io
import queue
import threading
import time
from io import StringIO

import numpy as np

from soundguard.audio.audio_pipeline import (
    CEDChunker,
    MicrophonePipeline,
    PipelineEvent,
    UtteranceTranscriber,
)
from soundguard.emergency.emergency_system import CATEGORY_CONFIG, EmergencySystem
from soundguard.display.display_transport import HUDTransport
from soundguard.speech.live_speech_to_text import (
    RecognitionJob,
    RecognitionResult,
    RecognitionWorker,
    TranscriptDisplay,
)
from soundguard.audio.streaming_audio import AudioFrame, AudioStreamHub


FRAME = np.ones(512, dtype=np.float32)
SILENCE = np.zeros(512, dtype=np.float32)
HELP_TEXT = CATEGORY_CONFIG["help_request"].aliases[0]


class FakeStream:
    opened = 0
    stopped = 0
    closed = 0
    frames = []

    def __init__(self, **kwargs):
        type(self).opened += 1
        self.callback = kwargs["callback"]

    def start(self):
        for samples in type(self).frames:
            self.callback(np.asarray(samples).reshape(-1, 1), 512, None, None)

    def stop(self):
        type(self).stopped += 1

    def close(self):
        type(self).closed += 1


class RecordingHUD:
    def __init__(self):
        self.alerts = []
        self.alert_states = []
        self.environmental_sounds = []
        self.partial_subtitles = []
        self.final_subtitles = []
        self.calls = []
        self.c_frames_sent = 0

    def set_alert(self, text):
        self.alerts.append(text)
        self.calls.append(("alert", text))

    def set_alert_state(self, event_label="", help_active=False):
        self.alert_states.append((event_label, help_active))
        self.calls.append(("alert_state", event_label, help_active))

    def set_subtitle(self, text):
        self.final_subtitles.append(text)
        self.calls.append(("subtitle", text))

    def set_partial_subtitle(self, text):
        self.partial_subtitles.append(text)
        self.calls.append(("partial_subtitle", text))

    def set_environmental_sound(self, text):
        self.environmental_sounds.append(text)
        self.calls.append(("environmental_sound", text))
        self.c_frames_sent += 1

    def set_status(self, text):
        pass

    def update(self):
        pass

    @property
    def counters(self):
        return {"C_frames_sent": self.c_frames_sent}


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def reset_fake_stream(frames):
    FakeStream.opened = FakeStream.stopped = FakeStream.closed = 0
    FakeStream.frames = frames


def vad_from_amplitude(samples, threshold):
    del threshold
    return {"has_speech": bool(np.max(np.abs(samples)) > 0.5)}


def drive_idle_production_audio(pipeline, *, seconds=16.0):
    """Drive the real 512-sample live fan-out with >=15 seconds of audio."""
    frame_count = int(np.ceil(seconds * 16000 / 512))
    pipeline.speech.start()
    pipeline.ced.start()
    try:
        for _ in range(frame_count):
            pipeline.hub._callback(FRAME.reshape(-1, 1), 512, None, None)
            pipeline._pump()
            time.sleep(0.0005)
        deadline = time.time() + 5.0
        while (pipeline.ced.frames_consumed < frame_count or
               pipeline.ced.windows_processed < int(seconds // 5)):
            pipeline._pump()
            if time.time() >= deadline:
                raise AssertionError(pipeline.counters)
            time.sleep(0.005)
        pipeline._pump()
        return frame_count, dict(pipeline.counters)
    finally:
        pipeline.speech.stop(force_final=False)
        pipeline.ced.stop()
        pipeline.speech.join_capture()
        pipeline.ced.join()
        pipeline.speech.stop_recognition()
        pipeline._pump()


def test_transcript_final_does_not_change_sound_history():
    system = EmergencySystem()
    system.process_sound_event("Siren", 0.9)
    before = list(system.history)
    system.process_transcript_event(HELP_TEXT)
    assert list(system.history) == before


def test_ced_event_does_not_change_help_missing_cycles():
    system = EmergencySystem()
    system.process_transcript_event(HELP_TEXT)
    before = system.help_missing_cycles
    system.process_sound_event("unknown", 0.0)
    assert system.help_missing_cycles == before


def test_hud_receives_partial_and_final_subtitles_on_distinct_paths():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
    )
    pipeline._dispatch_recognition(RecognitionResult(
        "PARTIAL", 1, "xin chao", event_sequence=1
    ))
    pipeline._dispatch_recognition(RecognitionResult(
        "FINAL", 1, "xin chao ban", event_sequence=2
    ))
    assert hud.partial_subtitles == ["xin chao"]
    assert hud.final_subtitles == ["xin chao ban"]


def test_hud_routes_only_meaningful_noncritical_ced_to_screen2():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="continuous",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )

    def dispatch(label, confidence):
        pipeline._dispatch_event(PipelineEvent(
            "CED", "ced", 0.0, 1.0, 1,
            {"label": label, "confidence": confidence},
        ))

    dispatch("Static", 0.99)
    assert pipeline.active_ced_display_label == ""
    assert hud.alert_states[-1] == ("", False)
    dispatch("Dog", 0.99)
    assert hud.environmental_sounds[-1] == "DOG"
    dispatch("Siren", 0.99)
    assert pipeline.active_ced_display_label == "DOG"
    dispatch("Siren", 0.99)
    emergency_alert = hud.alert_states[-1]
    assert emergency_alert == ("SIREN", False)
    assert hud.calls[-1] == ("alert_state", "SIREN", False)
    dispatch("Static", 0.99)
    assert hud.alert_states[-1] == emergency_alert
    dispatch("Static", 0.99)
    assert hud.alert_states[-1] == ("", False)
    assert hud.environmental_sounds[-1] == ""


def test_live_stt_dog_routes_to_ced_only_without_speech():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )

    pipeline._dispatch_event(PipelineEvent(
        "CED", "ced", 0.0, 1.0, 1,
        {"label": "Dog", "confidence": 0.99},
    ))

    assert hud.environmental_sounds[-1] == "DOG"
    assert not hud.partial_subtitles
    assert not hud.final_subtitles
    assert hud.alert_states[-1] == ("", False)


def test_live_stt_partial_plus_dog_updates_before_final():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )
    pipeline._dispatch_recognition(RecognitionResult(
        "PARTIAL", 1, "long utterance still active", event_sequence=1
    ))
    pipeline._dispatch_event(PipelineEvent(
        "CED", "ced", 0.0, 1.0, 2,
        {"label": "Dog", "confidence": 0.99},
    ))

    assert pipeline.final_seen is False
    assert hud.partial_subtitles == ["long utterance still active"]
    assert hud.environmental_sounds[-1] == "DOG"


def test_live_stt_ced_worker_consumes_during_long_active_utterance():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        duration=0.064,
        queue_seconds=0.5,
        partial_interval=0.064,
        end_silence_ms=700,
        max_utterance_seconds=10.0,
        pre_roll_ms=0,
        post_roll_ms=0,
        classifier=lambda audio, rate: {
            "label": "Dog", "confidence": 0.99, "top_predictions": []
        },
        vad=lambda samples, threshold: {"has_speech": True},
        recognizer=lambda audio, use_dtln: "long active utterance",
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )
    assert pipeline.ced is not None

    pipeline.speech.start()
    pipeline.ced.start()
    try:
        for _ in range(40):
            pipeline.hub._callback(FRAME.reshape(-1, 1), 512, None, None)
            pipeline._pump()
            time.sleep(0.004)
        deadline = time.time() + 2.0
        while pipeline.ced.windows_processed < 20 and time.time() < deadline:
            pipeline._pump()
            time.sleep(0.005)
    finally:
        pipeline.speech.stop(force_final=False)
        pipeline.ced.stop()
        pipeline.speech.join_capture()
        pipeline.ced.join()
        pipeline.speech.stop_recognition()
        pipeline._pump()

    assert pipeline.final_seen is False
    assert pipeline.counters["ced_frames_consumed"] == 40
    assert pipeline.counters["ced_windows_processed"] == 20
    assert pipeline.counters["ced_frames_dropped"] == 0
    assert pipeline.counters["ced_events_dropped"] == 0
    assert hud.environmental_sounds[-1] == "DOG"


def test_production_windows_dispatch_dog_during_16_seconds_without_speech():
    stream = io.BytesIO()
    hud = HUDTransport(stream=stream, auto_clock_sync=False)
    classifier_calls = []
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        duration=5.0,
        queue_seconds=4.0,
        classifier=lambda audio, rate: (
            classifier_calls.append((audio.size, rate)) or
            {"label": "Dog", "confidence": 0.99, "top_predictions": []}
        ),
        vad=lambda samples, threshold: {"has_speech": False},
        recognizer=lambda audio, use_dtln: (_ for _ in ()).throw(
            AssertionError("STT must not run without speech")
        ),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )

    frame_count, counters = drive_idle_production_audio(pipeline)

    assert classifier_calls[:3] == [(80000, 16000)] * 3
    assert pipeline.final_seen is False
    assert pipeline.last_sequence_displayed["transcript"] == -1
    assert pipeline.active_ced_display_label == "DOG"
    assert counters["raw_frames_captured"] == frame_count
    assert counters["speech_frames_produced"] == 0
    assert counters["ced_frames_produced"] == frame_count
    assert counters["ced_frames_consumed"] == frame_count
    assert counters["ced_frames_dropped"] == 0
    assert counters["ced_windows_processed"] == 3
    assert counters["ced_events_dispatched"] == 3
    assert counters["C_frames_sent"] == 1


def test_production_windows_process_background_without_speech_or_ced_label():
    stream = io.BytesIO()
    hud = HUDTransport(stream=stream, auto_clock_sync=False)
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        duration=5.0,
        queue_seconds=4.0,
        classifier=lambda audio, rate: {
            "label": "Static", "confidence": 0.99, "top_predictions": []
        },
        vad=lambda samples, threshold: {"has_speech": False},
        recognizer=lambda audio, use_dtln: (_ for _ in ()).throw(
            AssertionError("STT must not run without speech")
        ),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )

    frame_count, counters = drive_idle_production_audio(pipeline)

    assert pipeline.final_seen is False
    assert pipeline.active_ced_display_label == ""
    assert counters["raw_frames_captured"] == frame_count
    assert counters["speech_frames_produced"] == 0
    assert counters["ced_frames_produced"] == frame_count
    assert counters["ced_frames_consumed"] == frame_count
    assert counters["ced_frames_dropped"] == 0
    assert counters["ced_windows_processed"] == 3
    assert counters["ced_events_dispatched"] == 3
    assert counters["C_frames_sent"] == 0


def test_ced_ingestion_continues_while_production_window_inference_is_blocked():
    inference_started = threading.Event()
    release_inference = threading.Event()

    def blocked_classifier(audio, rate):
        del audio, rate
        inference_started.set()
        assert release_inference.wait(timeout=5.0)
        return {"label": "Dog", "confidence": 0.99, "top_predictions": []}

    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        duration=5.0,
        queue_seconds=4.0,
        classifier=blocked_classifier,
        vad=lambda samples, threshold: {"has_speech": False},
        recognizer=lambda audio, use_dtln: "",
        display=TranscriptDisplay(StringIO()),
        hud=RecordingHUD(),
        show_all_detected_sounds_on_hud=True,
    )
    frame_count = int(np.ceil(16.0 * 16000 / 512))
    pipeline.speech.start()
    pipeline.ced.start()
    try:
        for _ in range(frame_count):
            pipeline.hub._callback(FRAME.reshape(-1, 1), 512, None, None)
            time.sleep(0.0005)
        assert inference_started.wait(timeout=2.0)
        deadline = time.time() + 2.0
        while pipeline.ced.frames_consumed < frame_count and time.time() < deadline:
            time.sleep(0.005)

        assert pipeline.ced.frames_consumed == frame_count
        assert pipeline.hub.dropped_ced_frames == 0
        assert pipeline.ced.windows_processed == 0
        release_inference.set()
        deadline = time.time() + 5.0
        while pipeline.ced.windows_processed < 3 and time.time() < deadline:
            pipeline._pump()
            time.sleep(0.005)
        pipeline._pump()
        assert pipeline.ced.windows_processed == 3
        assert pipeline.ced_events_dispatched == 3
        assert pipeline.speech.speech_frames_produced == 0
    finally:
        release_inference.set()
        pipeline.speech.stop(force_final=False)
        pipeline.ced.stop()
        pipeline.speech.join_capture()
        pipeline.ced.join()
        pipeline.speech.stop_recognition()
        pipeline._pump()


def test_idle_runtime_counter_report_contains_required_live_fields():
    clock = ManualClock()
    output = StringIO()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        vad=lambda samples, threshold: {"has_speech": False},
        display=TranscriptDisplay(output),
        hud=RecordingHUD(),
        clock=clock,
    )
    pipeline._report_runtime_counters_if_due()
    clock.advance(5.0)
    pipeline._report_runtime_counters_if_due()
    report = output.getvalue()
    for name in (
        "raw_frames_captured", "speech_frames_produced",
        "ced_frames_produced", "ced_frames_consumed",
        "ced_frames_dropped", "ced_windows_processed",
        "ced_events_dispatched", "C_frames_sent",
    ):
        assert f"{name}=0" in report


def test_three_production_mixed_windows_do_not_starve_ced_or_stt():
    stream = io.BytesIO()
    hud = HUDTransport(stream=stream, auto_clock_sync=False)
    scores = iter((0.597129, 0.618249, 0.623165, 0.623165))

    def mixed_classifier(audio, rate):
        assert audio.size in {16000, 80000}
        assert rate == 16000
        dog_score = next(scores, 0.623165)
        return {
            "label": "Speech",
            "confidence": 0.66,
            "top_predictions": [
                {"label": "Speech", "score": 0.66},
                {"label": "Dog", "score": dog_score},
            ],
        }

    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        duration=5.0,
        queue_seconds=4.0,
        partial_interval=1.5,
        max_utterance_seconds=12.0,
        classifier=mixed_classifier,
        vad=lambda samples, threshold: {"has_speech": True},
        recognizer=lambda audio, use_dtln: "spoken words",
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )

    frame_count, counters = drive_idle_production_audio(pipeline)

    assert counters["raw_frames_captured"] == frame_count == 500
    assert counters["speech_frames_produced"] == frame_count
    assert counters["ced_frames_produced"] == frame_count
    assert counters["ced_frames_consumed"] == frame_count
    assert counters["ced_frames_dropped"] == 0
    assert counters["ced_windows_processed"] == 3
    assert counters["ced_windows_dropped"] == 0
    assert counters["ced_window_queue_max_depth"] <= 3
    assert counters["ced_top_k_recoveries"] == 3
    assert counters["ced_held_label_refreshes"] == 2
    assert counters["ced_ttl_expirations"] == 0
    assert counters["C_frames_sent"] == 1
    assert counters["stt_partial_count"] > 0
    assert counters["stt_final_count"] > 0
    assert pipeline.active_ced_display_label == "DOG"


def test_hud_alert_state_uses_independent_sound_and_help_lifecycles():
    system = EmergencySystem(decision_mode="continuous")
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=system,
        display=TranscriptDisplay(StringIO()),
        hud=hud,
    )

    pipeline._print_decision(system.process_sound_event("Siren", 0.99))
    pipeline._print_decision(system.process_sound_event("Siren", 0.99))
    assert hud.alert_states[-1] == ("SIREN", False)

    pipeline._print_decision(system.process_transcript_event(HELP_TEXT))
    assert hud.alert_states[-1] == ("SIREN", True)

    help_only_system = EmergencySystem(decision_mode="continuous")
    help_only_hud = RecordingHUD()
    help_only_pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=help_only_system,
        display=TranscriptDisplay(StringIO()),
        hud=help_only_hud,
    )
    help_only_pipeline._print_decision(
        help_only_system.process_transcript_event(HELP_TEXT)
    )
    assert help_only_hud.alert_states[-1] == ("", True)


def test_noncritical_ced_and_help_overlay_clear_without_stale_state():
    system = EmergencySystem(decision_mode="continuous")
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="continuous",
        emergency_system=system,
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )

    pipeline._print_decision(system.process_transcript_event(HELP_TEXT))
    pipeline._dispatch_event(PipelineEvent(
        "CED", "ced", 0.0, 1.0, 1,
        {"label": "Dog", "confidence": 0.99},
    ))
    assert hud.calls[-2:] == [
        ("alert_state", "", True),
        ("environmental_sound", "DOG"),
    ]

    pipeline._print_decision(system.process_transcript_event("normal words"))
    pipeline._print_decision(system.process_transcript_event("normal words"))
    assert hud.alert_states[-1] == ("", False)
    assert pipeline.active_ced_display_label == "DOG"
    assert hud.environmental_sounds[-1] == "DOG"


def test_dog_survives_repeated_speech_predictions_within_display_ttl():
    clock = ManualClock()
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(
            decision_mode="continuous", clock=clock
        ),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
        clock=clock,
    )

    def ced(label, confidence, top_predictions=None):
        pipeline._dispatch_event(PipelineEvent(
            "CED", "ced", 0.0, 1.0, 1,
            {
                "label": label,
                "confidence": confidence,
                "top_predictions": top_predictions or [],
            },
        ))

    ced("Dog", 0.99)
    pipeline._dispatch_recognition(RecognitionResult(
        "PARTIAL", 1, "speech is still active", event_sequence=2
    ))
    clock.advance(2.0)
    ced("Speech", 0.68, [
        {"label": "Speech", "score": 0.68},
        {"label": "Dog", "score": 0.60},
    ])
    clock.advance(2.0)
    ced("Conversation", 0.67)

    assert pipeline.final_seen is False
    assert pipeline.active_ced_display_label == "DOG"
    assert hud.partial_subtitles[-1] == "speech is still active"
    assert hud.environmental_sounds[-1] == "DOG"


def test_new_meaningful_ced_replaces_held_label_immediately():
    clock = ManualClock()
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(
            decision_mode="continuous", clock=clock
        ),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
        clock=clock,
    )
    for label in ("Dog", "Vehicle horn"):
        pipeline._dispatch_event(PipelineEvent(
            "CED", "ced", 0.0, 1.0, 1,
            {"label": label, "confidence": 0.99},
        ))

    assert pipeline.active_ced_display_label == "HORN"
    assert hud.environmental_sounds[-2:] == ["DOG", "HORN"]


def test_threshold_valid_top_k_environmental_class_recovers_from_speech_top1():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )
    pipeline._dispatch_event(PipelineEvent(
        "CED", "ced", 0.0, 1.0, 1,
        {
            "label": "Speech",
            "confidence": 0.68,
            "top_predictions": [
                {"label": "Speech", "score": 0.68},
                {"label": "Dog", "score": 0.66},
            ],
        },
    ))

    assert pipeline.active_ced_display_label == "DOG"
    assert hud.environmental_sounds[-1] == "DOG"


def test_explicit_event_end_clears_held_ced_immediately():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )
    pipeline._dispatch_event(PipelineEvent(
        "CED", "ced", 0.0, 1.0, 1,
        {"label": "Dog", "confidence": 0.99},
    ))
    pipeline._update_ced_display(
        {"label": "Static", "confidence": 0.99},
        {"event_state": "EVENT_ENDED", "mapped_sound": {"level": "LOW"}},
    )

    assert pipeline.active_ced_display_label == ""
    assert hud.environmental_sounds[-1] == ""


def test_ced_display_ttl_expires_without_new_confirmation():
    clock = ManualClock()
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(
            decision_mode="continuous", clock=clock
        ),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
        ced_display_ttl_seconds=5.5,
        clock=clock,
    )
    pipeline._dispatch_event(PipelineEvent(
        "CED", "ced", 0.0, 1.0, 1,
        {"label": "Dog", "confidence": 0.99},
    ))
    clock.advance(5.49)
    pipeline._pump()
    assert pipeline.active_ced_display_label == "DOG"
    clock.advance(0.02)
    pipeline._pump()

    assert pipeline.active_ced_display_label == ""
    assert pipeline.ced_display_expires_at is None
    assert hud.environmental_sounds[-1] == ""


def test_suppressed_ced_from_clean_state_never_creates_fake_label():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )
    for label in ("Speech", "Conversation", "Static", "background noise"):
        pipeline._dispatch_event(PipelineEvent(
            "CED", "ced", 0.0, 1.0, 1,
            {"label": label, "confidence": 0.99},
        ))

    assert pipeline.active_ced_display_label == ""
    assert not hud.environmental_sounds


def test_speech_with_no_prior_ced_is_subtitle_only():
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous"),
        display=TranscriptDisplay(StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
    )
    pipeline._dispatch_recognition(RecognitionResult(
        "PARTIAL", 1, "subtitle only", event_sequence=1
    ))
    pipeline._dispatch_event(PipelineEvent(
        "CED", "ced", 0.0, 1.0, 2,
        {
            "label": "Speech",
            "confidence": 0.68,
                "top_predictions": [
                    {"label": "Speech", "score": 0.68},
                    {"label": "Dog", "score": 0.50},
                ],
        },
    ))

    assert hud.partial_subtitles == ["subtitle only"]
    assert pipeline.active_ced_display_label == ""
    assert not hud.environmental_sounds


def test_interleaved_evidence_preserves_independent_lifecycles():
    system = EmergencySystem(help_end_grace_cycles=2)
    system.process_sound_event("Siren", 0.9)
    help_started = system.process_transcript_event(HELP_TEXT)
    sound_started = system.process_sound_event("Siren", 0.9)
    assert help_started["event_state"] == "EVENT_STARTED"
    assert sound_started["event_state"] == "EVENT_STARTED"
    assert system.help_active is True
    assert system.active_sound == "siren"
    system.process_transcript_event("mot cau noi binh thuong")
    assert system.help_missing_cycles == 1
    assert list(system.history) == ["siren", "siren"]


def test_delayed_stt_retains_capture_metadata():
    def delayed(audio, use_dtln):
        del audio, use_dtln
        time.sleep(0.02)
        return "complete final"

    worker = RecognitionWorker(False, recognizer=delayed)
    worker.start()
    worker.submit(RecognitionJob("FINAL", 7, FRAME, 10.0, 12.5, 99))
    result = worker.results.get(timeout=1.0)
    worker.stop()
    worker.join()
    assert result.capture_started_at == 10.0
    assert result.capture_ended_at == 12.5
    assert result.event_sequence == 99
    assert result.source == "transcript"


def test_mic_no_speech_still_runs_raw_ced_and_emergency():
    reset_fake_stream([np.full(512, 0.2, dtype=np.float32)] * 3)
    received = []

    def classifier(audio, sample_rate):
        received.append(audio.copy())
        assert sample_rate == 16000
        return {"label": "Siren", "confidence": 0.95, "top_predictions": []}

    output = StringIO()
    pipeline = MicrophonePipeline(
        mode="mic", emergency_system=EmergencySystem(decision_mode="single_shot"),
        duration=0.03, stream_factory=FakeStream, classifier=classifier,
        vad=lambda samples, threshold: {"has_speech": False},
        recognizer=lambda audio, use_dtln: "", display=TranscriptDisplay(output),
        partial_interval=0.032, max_utterance_seconds=0.064,
    )
    assert pipeline.run() == 0
    assert len(received) == 1
    assert np.allclose(received[0], 0.2)
    text = output.getvalue()
    assert "Detected sound: Siren" in text
    assert "Event state: EVENT_STARTED" in text


def test_mic_final_keeps_quiet_last_word_frame():
    last_word = np.full(512, 0.8, dtype=np.float32)
    reset_fake_stream([FRAME, FRAME, last_word, SILENCE, SILENCE])
    recognized = []

    def recognizer(audio, use_dtln):
        del use_dtln
        recognized.append(audio.copy())
        return "sentence final word"

    pipeline = MicrophonePipeline(
        mode="mic", emergency_system=EmergencySystem(decision_mode="single_shot"),
        duration=.15, stream_factory=FakeStream,
        classifier=lambda audio, rate: {"label": "unknown", "confidence": 0},
        vad=vad_from_amplitude, recognizer=recognizer,
        display=TranscriptDisplay(StringIO()), partial_interval=.032,
        end_silence_ms=64, post_roll_ms=0, max_utterance_seconds=1,
        pre_roll_ms=0,
    )
    pipeline.run()
    assert recognized
    assert any(np.array_equal(frame, last_word)
               for frame in np.split(recognized[-1], recognized[-1].size // 512))


def test_mic_timeout_finalizes_latest_complete_snapshot():
    tail = np.full(512, 0.9, dtype=np.float32)
    reset_fake_stream([FRAME, FRAME, tail])
    recognized = []

    def recognizer(audio, use_dtln):
        del use_dtln
        recognized.append(audio.copy())
        return "forced complete"

    pipeline = MicrophonePipeline(
        mode="mic", emergency_system=EmergencySystem(decision_mode="single_shot"),
        duration=.03, stream_factory=FakeStream,
        classifier=lambda audio, rate: {"label": "unknown", "confidence": 0},
        vad=vad_from_amplitude, recognizer=recognizer,
        display=TranscriptDisplay(StringIO()), partial_interval=.032,
        max_utterance_seconds=1, pre_roll_ms=0,
    )
    pipeline.run()
    assert len(recognized) >= 1
    assert np.array_equal(recognized[-1][-512:], tail)


def test_only_one_physical_microphone_stream_is_opened():
    reset_fake_stream([SILENCE])
    hub = AudioStreamHub(stream_factory=FakeStream)
    with hub:
        pass
    hub.close()
    assert FakeStream.opened == 1
    assert FakeStream.stopped == 1
    assert FakeStream.closed == 1


def test_ced_receives_raw_and_never_dtln_audio():
    raw = np.full(512, 0.25, dtype=np.float32)
    reset_fake_stream([raw])
    with AudioStreamHub(stream_factory=FakeStream) as hub:
        ced_frame = hub.ced_frames.get(timeout=0.1)
        speech_frame = hub.speech_frames.get(timeout=0.1)
    simulated_dtln = speech_frame.samples * 0.1
    assert np.array_equal(ced_frame.samples, raw)
    assert not np.array_equal(ced_frame.samples, simulated_dtln)
    assert not np.shares_memory(ced_frame.samples, speech_frame.samples)


def test_sentence_crossing_ced_boundary_is_one_complete_final():
    events = queue.Queue()
    frames = queue.Queue()
    recognized = []

    def recognizer(audio, use_dtln):
        del use_dtln
        recognized.append(audio.copy())
        return "mot cau hoan chinh"

    transcriber = UtteranceTranscriber(
        frames, events, vad_threshold=0.5, use_dtln=False,
        partial_interval=0.064, end_silence_ms=64,
        max_utterance_seconds=1.0, pre_roll_ms=0, post_roll_ms=0,
        vad=vad_from_amplitude, recognizer=recognizer,
    )
    chunker = CEDChunker(chunk_seconds=0.064)
    sequence = [FRAME, FRAME, FRAME, SILENCE, SILENCE]
    for index, samples in enumerate(sequence, 1):
        frame = AudioFrame(samples, False, index * .032 - .032, index * .032, index)
        frames.put(frame)
        chunker.add(frame)
    transcriber.start()
    deadline = time.time() + 1
    while not frames.empty() and time.time() < deadline:
        time.sleep(0.005)
    transcriber.stop()
    transcriber.join_capture()
    transcriber.stop_recognition()
    results = transcriber.drain_results()
    finals = [result for result in results if result.kind == "FINAL"]
    assert len(finals) == 1
    assert finals[0].transcript == "mot cau hoan chinh"
    assert recognized[-1].size == 5 * 512


def test_continuous_branch_produces_partial_and_final():
    events = queue.Queue()
    frames = queue.Queue()
    transcriber = UtteranceTranscriber(
        frames, events, vad_threshold=.5, use_dtln=False,
        partial_interval=.064, end_silence_ms=64,
        max_utterance_seconds=1, pre_roll_ms=0, post_roll_ms=0,
        vad=vad_from_amplitude,
        recognizer=lambda audio, use_dtln: f"words-{audio.size}",
    )
    transcriber.start()
    for index, samples in enumerate([FRAME, FRAME, FRAME, SILENCE, SILENCE], 1):
        frames.put(AudioFrame(samples, False, index*.032-.032, index*.032, index))
    deadline = time.time() + 1
    while not frames.empty() and time.time() < deadline:
        time.sleep(.005)
    transcriber.stop()
    transcriber.join_capture()
    transcriber.stop_recognition()
    kinds = [result.kind for result in transcriber.drain_results()]
    assert "PARTIAL" in kinds
    assert "FINAL" in kinds


def test_display_owner_serializes_partial_and_permanent_output():
    stream = StringIO()
    display = TranscriptDisplay(stream)
    display.show_partial("unfinished")
    threads = [threading.Thread(target=display.print_permanent, args=(f"CED {i}",))
               for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    value = stream.getvalue()
    for index in range(4):
        assert f"CED {index}\n" in value
    assert "PARTIAL: unfinishedCED" not in value


def test_clean_close_leaves_no_pipeline_workers():
    reset_fake_stream([SILENCE])
    output = StringIO()
    pipeline = MicrophonePipeline(
        mode="mic", emergency_system=EmergencySystem(decision_mode="single_shot"),
        duration=.02, stream_factory=FakeStream,
        classifier=lambda audio, rate: {"label": "unknown", "confidence": 0},
        vad=lambda samples, threshold: {"has_speech": False},
        recognizer=lambda audio, use_dtln: "", display=TranscriptDisplay(output),
        partial_interval=.032, max_utterance_seconds=.064,
    )
    pipeline.run()
    assert pipeline.hub._stream is None
    assert not pipeline.speech.thread.is_alive()
    assert not pipeline.speech.worker.thread.is_alive()
    assert pipeline.ced is not None and not pipeline.ced.thread.is_alive()
    assert not pipeline.ced.inference_thread.is_alive()
    assert FakeStream.stopped == FakeStream.closed == 1


def test_queue_overflow_counters_are_independent():
    hub = AudioStreamHub(queue_seconds=.032)
    for index in range(3):
        samples = np.full((512, 1), index, dtype=np.float32)
        hub._callback(samples, 512, None, None)
    assert hub.dropped_speech_frames == 2
    assert hub.dropped_ced_frames == 2


def run_tests():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"{test.__name__}: passed")
    print(f"audio pipeline tests passed ({len(tests)})")


if __name__ == "__main__":
    run_tests()
