import hashlib
import io
import json
from pathlib import Path

import librosa
import numpy as np
import pytest

from audio_pipeline import MicrophonePipeline, PipelineEvent, classify_raw_audio
from emergency_system import EmergencySystem
from hud_awareness import (
    HUD_AWARENESS_THRESHOLD,
    evaluate_ced_for_hud,
    is_speech_only_ced_label,
)
from live_speech_to_text import RecognitionResult, TranscriptDisplay


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results" / "hud_awareness_policy_v2"


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class RecordingHUD:
    def __init__(self):
        self.environmental = []
        self.partials = []
        self.finals = []
        self.alert_states = []
        self.c_frames = 0

    def set_environmental_sound(self, value):
        self.environmental.append(value)
        self.c_frames += 1

    def set_partial_subtitle(self, value):
        self.partials.append(value)

    def set_subtitle(self, value):
        self.finals.append(value)

    def set_alert_state(self, event_label="", help_active=False):
        self.alert_states.append((event_label, help_active))

    def set_status(self, value):
        del value

    def update(self):
        pass

    @property
    def counters(self):
        return {"C_frames_sent": self.c_frames}


def make_pipeline(clock=None):
    clock = clock or ManualClock()
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt",
        emergency_system=EmergencySystem(decision_mode="continuous", clock=clock),
        display=TranscriptDisplay(io.StringIO()),
        hud=hud,
        show_all_detected_sounds_on_hud=True,
        clock=clock,
    )
    return pipeline, hud, clock


def ced_event(label, confidence, top_predictions, sequence=1):
    return PipelineEvent(
        "CED", "ced", (sequence - 1) * 5.0, sequence * 5.0,
        sequence * 10,
        {"label": label, "confidence": confidence,
         "top_predictions": top_predictions},
    )


def test_policy_is_locked_before_heldout_validation_and_matches_runtime():
    policy_bytes = (RESULTS / "locked_policy.json").read_bytes()
    policy = json.loads(policy_bytes)
    validation = json.loads((RESULTS / "heldout_validation.json").read_text())
    assert policy["locked_before_validation"] is True
    assert policy["threshold"] == HUD_AWARENESS_THRESHOLD == 0.52
    assert validation["policy_sha256"] == hashlib.sha256(policy_bytes).hexdigest()


def test_calibration_and_heldout_material_are_disjoint():
    calibration = json.loads(
        (RESULTS / "calibration_records.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (RESULTS / "heldout_validation.json").read_text(encoding="utf-8")
    )["records"]
    calibration_sources = {
        record.get(key) for record in calibration
        for key in ("environmental_source", "speech_source") if record.get(key)
    }
    validation_sources = {
        record.get(key) for record in validation
        for key in ("environmental_source", "speech_source") if record.get(key)
    }
    assert calibration_sources.isdisjoint(validation_sources)
    assert {record.get("environmental_db") for record in calibration
            if record["condition"] == "mixed"} == {-6.0, 0.0, 6.0}


def test_heldout_policy_metrics_are_untouched_and_speech_safe():
    summary = json.loads(
        (RESULTS / "heldout_validation.json").read_text(encoding="utf-8")
    )["summary"]
    assert summary["precision"] == 34 / 35
    assert summary["recall"] == 34 / 40
    assert summary["f1"] == pytest.approx(68 / 75)
    assert summary["by_condition"]["speech_only"]["fp"] == 0
    assert summary["by_condition"]["environment_only"]["tp"] == 16
    assert summary["by_condition"]["mixed"]["tp"] == 18


def test_semantic_speech_suppression_does_not_hide_screaming():
    for label in (
        "Speech", "Conversation", "Speech synthesizer",
        "Male speech, man speaking", "Narration, monologue", "Mantra",
    ):
        assert is_speech_only_ced_label(label)
        assert not evaluate_ced_for_hud(label, 0.99)["accepted"]
    assert not is_speech_only_ced_label("Screaming")
    assert evaluate_ced_for_hud("Screaming", 0.99)["reason"] == "unsupported_or_alert_only"


def test_top1_and_topk_use_the_same_hud_threshold():
    assert evaluate_ced_for_hud("Dog", 0.599)["accepted"]
    assert evaluate_ced_for_hud("Dog", 0.623)["accepted"]
    assert not evaluate_ced_for_hud("Dog", 0.519)["accepted"]
    assert evaluate_ced_for_hud(
        "Vehicle horn, car horn, honking", 0.60
    )["display_label"] == "HORN"


def test_measured_dog_then_three_mixed_windows_refreshes_through_subtitle():
    pipeline, hud, clock = make_pipeline()
    pipeline._dispatch_event(ced_event(
        "Dog", 0.599421, [{"label": "Dog", "score": 0.599421}], 1
    ))
    pipeline._dispatch_recognition(RecognitionResult(
        "PARTIAL", 1, "con cho van dang sua", event_sequence=11
    ))
    scores = (0.597129, 0.618249, 0.623165)
    for index, score in enumerate(scores, start=2):
        clock.advance(5.0)
        pipeline._expire_ced_display_if_needed()
        pipeline._dispatch_event(ced_event(
            "Speech", 0.66,
            [{"label": "Speech", "score": 0.66},
             {"label": "Dog", "score": score}], index,
        ))
        assert pipeline.active_ced_display_label == "DOG"

    assert hud.partials == ["con cho van dang sua"]
    assert pipeline.ced_top_k_recoveries == 3
    assert pipeline.ced_held_label_refreshes == 3
    assert pipeline.ced_ttl_expirations == 0


def test_mixed_from_clean_state_can_coexist_with_subtitle():
    pipeline, hud, _ = make_pipeline()
    pipeline._dispatch_recognition(RecognitionResult(
        "PARTIAL", 1, "subtitle", event_sequence=1
    ))
    pipeline._dispatch_event(ced_event(
        "Speech", 0.655,
        [{"label": "Speech", "score": 0.655},
         {"label": "Dog", "score": 0.597}], 1,
    ))
    assert hud.partials == ["subtitle"]
    assert pipeline.active_ced_display_label == "DOG"


def test_environment_end_clears_label_but_preserves_subtitle():
    pipeline, hud, clock = make_pipeline()
    pipeline._dispatch_recognition(RecognitionResult(
        "PARTIAL", 1, "speech continues", event_sequence=1
    ))
    pipeline._dispatch_event(ced_event(
        "Dog", 0.60, [{"label": "Dog", "score": 0.60}], 1
    ))
    clock.advance(5.51)
    pipeline._pump()
    assert pipeline.active_ced_display_label == ""
    assert hud.environmental[-1] == ""
    assert hud.partials == ["speech continues"]


def test_new_noncritical_class_replaces_and_critical_still_routes_to_alert():
    pipeline, hud, _ = make_pipeline()
    pipeline._dispatch_event(ced_event(
        "Dog", 0.60, [{"label": "Dog", "score": 0.60}], 1
    ))
    pipeline._dispatch_event(ced_event(
        "Vehicle horn", 0.61,
        [{"label": "Vehicle horn", "score": 0.61}], 2,
    ))
    assert hud.environmental[-2:] == ["DOG", "HORN"]

    pipeline._dispatch_event(ced_event(
        "Siren", 0.99, [{"label": "Siren", "score": 0.99}], 3
    ))
    pipeline._dispatch_event(ced_event(
        "Siren", 0.99, [{"label": "Siren", "score": 0.99}], 4
    ))
    assert hud.alert_states[-1] == ("SIREN", False)
    assert pipeline.active_ced_display_label == "HORN"


def test_real_model_reference_mix_has_three_refreshable_production_windows():
    dog, _ = librosa.load(
        ROOT / "benchmark_data/external/esc50/audio/1-100032-A-0.wav",
        sr=16000, mono=True,
    )
    speech, _ = librosa.load(ROOT / "samples/test_speech.wav", sr=16000, mono=True)
    samples = 15 * 16000
    dog = np.resize(dog.astype(np.float32), samples)
    speech = np.resize(speech.astype(np.float32), samples)
    dog /= max(float(np.sqrt(np.mean(np.square(dog)))), 1e-8)
    speech /= max(float(np.sqrt(np.mean(np.square(speech)))), 1e-8)
    mixed = (dog + speech) * 0.5

    pipeline, _, clock = make_pipeline()
    for index in range(3):
        result = classify_raw_audio(mixed[index * 80000:(index + 1) * 80000], 16000)
        clock.advance(5.0 if index else 0.0)
        pipeline._expire_ced_display_if_needed()
        pipeline._dispatch_event(PipelineEvent(
            "CED", "ced", index * 5.0, (index + 1) * 5.0,
            (index + 1) * 10, result,
        ))
        assert result["label"] == "Speech"
        assert pipeline.active_ced_display_label == "DOG"
    assert pipeline.ced_top_k_recoveries == 3
    assert pipeline.ced_ttl_expirations == 0
