import base64
import io
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from personalization.recognition import (
    BoundedRecognitionWorker,
    RecognitionError,
    RecognitionStore,
    TemporalDecisionGate,
    build_centroid,
    open_set_match,
    validate_audio_bytes,
)
from personalization.runtime import PersonalizedRecognitionRuntime
from personalization.web_server import PersonalizationHandler
from soundguard.audio.audio_pipeline import MicrophonePipeline
from soundguard.audio.streaming_audio import AudioStreamHub
from soundguard.emergency.emergency_system import EmergencySystem
from soundguard.speech.live_speech_to_text import RecognitionResult, TranscriptDisplay


def wav_bytes(seconds=1.0, frequency=440.0, sample_rate=16_000):
    samples = 0.15 * np.sin(
        2 * np.pi * frequency * np.arange(int(seconds * sample_rate)) / sample_rate
    )
    stream = io.BytesIO()
    sf.write(stream, samples.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def fake_embed(kind, path):
    del path
    vector = np.array([1.0, 0.0, 0.0]) if kind == "sound" else np.array([0.0, 1.0, 0.0])
    return vector, {
        "model": "test-model", "revision": "fixed", "sha256": "0" * 64,
        "preprocessing": "test-v1",
    }


def test_audio_quality_validation_accepts_pcm_and_rejects_silence():
    _, rate, quality = validate_audio_bytes(wav_bytes(), "sound")
    assert rate == 16_000
    assert quality["quality"] == "valid"
    silent = io.BytesIO()
    sf.write(silent, np.zeros(16_000), 16_000, format="WAV")
    with pytest.raises(RecognitionError, match="silent"):
        validate_audio_bytes(silent.getvalue(), "voice")


def test_centroid_and_open_set_threshold_margin_rejection():
    centroid = build_centroid([np.array([1.0, 0.0]), np.array([0.9, 0.1])])
    profiles = [
        {"id": "a", "display_name": "A", "embedding": centroid, "enabled": True,
         "threshold": 0.7, "margin": 0.1},
        {"id": "b", "display_name": "B", "embedding": np.array([0.98, 0.2]),
         "enabled": True, "threshold": 0.7, "margin": 0.1},
    ]
    ambiguous = open_set_match(np.array([1.0, 0.05]), profiles)
    assert ambiguous["label"] == "UNKNOWN"
    assert ambiguous["reason"] == "ambiguous_margin"
    weak = open_set_match(np.array([0.0, 1.0]), profiles, threshold=0.9)
    assert weak["reason"] == "below_threshold"


def test_temporal_smoothing_and_cooldown():
    now = [10.0]
    gate = TemporalDecisionGate(required=2, window=3, cooldown_seconds=5, clock=lambda: now[0])
    accepted = {"accepted": True, "label": "Door"}
    assert gate.apply(accepted)["reason"] == "temporal_evidence"
    assert gate.apply(accepted)["accepted"] is True
    assert gate.apply(accepted)["reason"] == "cooldown"
    now[0] += 6
    assert gate.apply(accepted)["accepted"] is True


def test_store_lifecycle_keeps_raw_audio_out_of_profile_compatibility_file(tmp_path):
    store = RecognitionStore(tmp_path / "enrollments")
    profile = store.create("sound", "Door chime", contexts=["home"], priority=4)
    store.add_sample("sound", profile["id"], wav_bytes(frequency=440))
    store.add_sample("sound", profile["id"], wav_bytes(frequency=660))
    built = store.build("sound", profile["id"], fake_embed)
    assert built["built"] and built["enabled"]
    assert built["embedding_dimension"] == 3
    assert len(store.matcher_profiles("sound")) == 1
    assert not (tmp_path / "user_profile.json").exists()
    store.update("sound", profile["id"], {"enabled": False})
    assert store.matcher_profiles("sound") == []
    assert store.delete("sound", profile["id"])["deleted"]


def test_voice_consent_and_minimum_sample_contract(tmp_path):
    store = RecognitionStore(tmp_path)
    with pytest.raises(RecognitionError, match="consent"):
        store.create("voice", "Person")
    profile = store.create("voice", "Person", consent_acknowledged=True)
    for frequency in (220, 260):
        store.add_sample("voice", profile["id"], wav_bytes(1.2, frequency))
    with pytest.raises(RecognitionError, match="at least 3"):
        store.build("voice", profile["id"], fake_embed)


def test_bounded_worker_isolates_model_failure_and_counts_it():
    def broken(_):
        raise RuntimeError("model crashed")
    worker = BoundedRecognitionWorker("sound", broken, maxsize=1)
    worker.start()
    worker.submit("one", object())
    result = worker.results.get(timeout=2).result
    worker.stop()
    assert result["label"] == "UNKNOWN"
    assert result["reason"] == "worker_error"
    assert worker.counters["failures"] == 1


def test_runtime_fails_open_when_model_client_raises(tmp_path):
    class BrokenClient:
        def embed(self, *_):
            raise RuntimeError("offline")
        def close(self):
            pass
    store = RecognitionStore(tmp_path)
    profile = store.create("sound", "Bell")
    for frequency in (440, 660):
        store.add_sample("sound", profile["id"], wav_bytes(frequency=frequency))
    store.build("sound", profile["id"], fake_embed)
    runtime = PersonalizedRecognitionRuntime(
        familiar_sounds=True, store=store, client=BrokenClient()
    )
    runtime.submit_sound(np.ones(48_000, dtype=np.float32))
    deadline = time.time() + 2
    results = []
    while time.time() < deadline and not results:
        results = runtime.drain_sound_results()
        time.sleep(0.01)
    runtime.close()
    assert results[0]["label"] == "UNKNOWN"
    assert results[0]["reason"] == "worker_error"


def test_runtime_with_no_built_profiles_closes_without_starting_workers(tmp_path):
    class Client:
        def close(self): pass
    runtime = PersonalizedRecognitionRuntime(
        familiar_sounds=True, familiar_voices=True,
        store=RecognitionStore(tmp_path), client=Client(),
    )
    assert not runtime.sound_enabled and not runtime.voice_enabled
    runtime.close()


def test_single_microphone_callback_fans_out_only_when_enabled():
    enabled = [False]
    hub = AudioStreamHub(familiar_sound_enabled=lambda: enabled[0])
    frame = np.ones((512, 1), dtype=np.float32)
    hub._callback(frame, 512, None, None)
    assert hub.familiar_sound_frames.empty()
    enabled[0] = True
    hub._callback(frame, 512, None, None)
    assert hub.familiar_sound_frames.qsize() == 1
    assert hub.counters["raw_frames_captured"] == 2


def test_live_enabled_check_is_cached_and_never_reads_storage_in_callback():
    class Store:
        calls = 0
        def matcher_profiles(self, kind):
            self.calls += 1
            return [{"id": "x"}] if kind == "sound" else []
    class Client:
        def close(self): pass
    store = Store()
    runtime = PersonalizedRecognitionRuntime(
        familiar_sounds=True, store=store, client=Client()
    )
    for _ in range(100):
        assert runtime.sound_enabled
    assert store.calls == 1


class RecordingHUD:
    def __init__(self):
        self.final_subtitles = []
        self.environmental_sounds = []
        self.counters = {"C_frames_sent": 0}
    def set_subtitle(self, value): self.final_subtitles.append(value)
    def set_partial_subtitle(self, value): pass
    def set_environmental_sound(self, value): self.environmental_sounds.append(value)
    def set_alert_state(self, *args): pass
    def set_status(self, value): pass
    def update(self): pass


def test_speaker_prefix_is_display_only_and_help_fusion_receives_raw_transcript():
    class Runtime:
        familiar_sounds = False
        sound_enabled = False
        counters = {}
        def submit_voice(self, *_): return True
        def take_voice_result(self, *_args, **_kwargs):
            return {"accepted": True, "label": "Lan"}
        def drain_sound_results(self): return []
    system = EmergencySystem(decision_mode="continuous")
    seen = []
    original = system.process_transcript_event
    def capture(text, **kwargs):
        seen.append(text)
        return original(text, **kwargs)
    system.process_transcript_event = capture
    hud = RecordingHUD()
    pipeline = MicrophonePipeline(
        mode="live-stt", emergency_system=system,
        display=TranscriptDisplay(io.StringIO()), hud=hud,
        personalized_runtime=Runtime(),
    )
    pipeline._dispatch_recognition(RecognitionResult("FINAL", 1, "help me", event_sequence=2))
    assert hud.final_subtitles == ["[Lan] help me"]
    assert seen == ["help me"]


def test_http_api_preserves_start_and_adds_local_enrollment(tmp_path):
    class FakeClient:
        python_path = Path(__file__)
        loaded = False
        embed = staticmethod(fake_embed)
    store = RecognitionStore(tmp_path)
    with patch("personalization.web_server.RECOGNITION_STORE", store), patch(
        "personalization.web_server.EMBEDDING_CLIENT", FakeClient()
    ):
        server = ThreadingHTTPServer(("127.0.0.1", 0), PersonalizationHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def post(path, payload):
            req = urllib.request.Request(
                base + path, json.dumps(payload).encode(),
                {"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, json.load(response)
        def mutate(path, method, payload=None):
            data = None if payload is None else json.dumps(payload).encode()
            req = urllib.request.Request(
                base + path, data, {"Content-Type": "application/json"}, method=method
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, json.load(response)
        try:
            assert post("/api/start", {})[0] == 200
            status, created = post("/api/familiar-sounds", {"display_name": "Bell"})
            assert status == 201
            profile_id = created["profile"]["id"]
            payload = {"audio_base64": base64.b64encode(wav_bytes()).decode()}
            _, first = post(f"/api/familiar-sounds/{profile_id}/samples", payload)
            second = {"audio_base64": base64.b64encode(wav_bytes(frequency=660)).decode()}
            assert post(f"/api/familiar-sounds/{profile_id}/samples", second)[0] == 201
            assert post(f"/api/familiar-sounds/{profile_id}/build", {})[1]["profile"]["built"]
            tested = post("/api/familiar-sounds/test", payload)[1]["result"]
            assert tested["label"] == "Bell"
            assert mutate(
                f"/api/familiar-sounds/{profile_id}", "PATCH", {"enabled": False}
            )[1]["profile"]["enabled"] is False
            assert mutate(
                f"/api/familiar-sounds/{profile_id}/samples/{first['sample']['id']}",
                "DELETE",
            )[0] == 200
            with urllib.request.urlopen(base + "/api/recognition", timeout=3) as response:
                listed = json.load(response)
            assert listed["familiar_sounds"][0]["display_name"] == "Bell"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
