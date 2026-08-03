import contextlib
import queue
import threading
import time
from io import StringIO

import numpy as np

from audio_pipeline import CEDChunker, MicrophonePipeline, UtteranceTranscriber
from emergency_system import CATEGORY_CONFIG, EmergencySystem
from live_speech_to_text import RecognitionJob, RecognitionWorker, TranscriptDisplay
from streaming_audio import AudioFrame, AudioStreamHub


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


def reset_fake_stream(frames):
    FakeStream.opened = FakeStream.stopped = FakeStream.closed = 0
    FakeStream.frames = frames


def vad_from_amplitude(samples, threshold):
    del threshold
    return {"has_speech": bool(np.max(np.abs(samples)) > 0.5)}


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
