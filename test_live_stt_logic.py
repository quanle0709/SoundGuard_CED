import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

from audio_capture import create_speech_optimized_wav
from live_speech_to_text import LiveSpeechStateMachine
from streaming_audio import StreamingAudioInput


FRAME = np.ones(512, dtype=np.float32)
SILENCE = np.zeros(512, dtype=np.float32)


def kinds(events):
    return [event.kind for event in events]


def test_waiting_and_pre_roll():
    machine = LiveSpeechStateMachine(pre_roll_ms=64)
    assert machine.process_frame(SILENCE, False) == []
    events = machine.process_frame(FRAME, True)
    assert kinds(events) == ["SPEECH_STARTED"]
    assert len(machine.utterance) == 2
    assert np.array_equal(machine.utterance[0], SILENCE)


def test_continuation_and_partial():
    machine = LiveSpeechStateMachine(pre_roll_ms=0, partial_interval=0.064)
    machine.process_frame(FRAME, True)
    events = machine.process_frame(FRAME, True)
    assert kinds(events) == ["PARTIAL_DUE"]
    assert events[0].audio is not None
    assert events[0].audio.size == 1024


def test_exact_silence_finalization_and_reset():
    machine = LiveSpeechStateMachine(pre_roll_ms=0, end_silence_ms=64, post_roll_ms=0)
    machine.process_frame(FRAME, True)
    assert machine.process_frame(SILENCE, False) == []
    events = machine.process_frame(SILENCE, False)
    assert kinds(events) == ["FINAL_DUE"]
    assert machine.speaking is False
    assert machine.utterance == []
    assert events[0].utterance_id == 1
    assert machine.process_frame(FRAME, True)[0].utterance_id == 2


def test_boundary_speech_frame_is_in_final_snapshot():
    boundary = np.full(512, 0.25, dtype=np.float32)
    machine = LiveSpeechStateMachine(pre_roll_ms=0, end_silence_ms=64, post_roll_ms=0)
    machine.process_frame(FRAME, True)
    machine.process_frame(boundary, True)
    machine.process_frame(SILENCE, False)
    final = machine.process_frame(SILENCE, False)[0]
    assert final.kind == "FINAL_DUE"
    assert np.array_equal(final.audio[512:1024], boundary)
    assert final.audio.size == 4 * 512


def test_post_roll_frames_are_retained():
    tail_one = np.full(512, 0.10, dtype=np.float32)
    tail_two = np.full(512, 0.20, dtype=np.float32)
    machine = LiveSpeechStateMachine(pre_roll_ms=0, end_silence_ms=64, post_roll_ms=64)
    machine.process_frame(FRAME, True)
    machine.process_frame(SILENCE, False)
    assert machine.process_frame(SILENCE, False) == []
    assert machine.process_frame(tail_one, False) == []
    final = machine.process_frame(tail_two, False)[0]
    assert final.kind == "FINAL_DUE"
    assert np.array_equal(final.audio[-1024:-512], tail_one)
    assert np.array_equal(final.audio[-512:], tail_two)


def test_final_snapshot_is_newer_than_latest_partial():
    machine = LiveSpeechStateMachine(
        pre_roll_ms=0, end_silence_ms=64, post_roll_ms=64,
        partial_interval=0.064,
    )
    machine.process_frame(FRAME, True)
    partial = machine.process_frame(FRAME, True)[0]
    machine.process_frame(SILENCE, False)
    machine.process_frame(SILENCE, False)
    machine.process_frame(SILENCE, False)
    final = machine.process_frame(SILENCE, False)[0]
    assert partial.kind == "PARTIAL_DUE"
    assert final.kind == "FINAL_DUE"
    assert final.audio.size >= partial.audio.size
    assert np.array_equal(final.audio[:partial.audio.size], partial.audio)


def test_live_right_tail_preserves_quiet_final_audio():
    sample_rate = 16000
    audio = np.zeros(sample_rate, dtype=np.float32)
    audio[:8000] = 0.4 * np.sin(2 * np.pi * 220 * np.arange(8000) / sample_rate)
    audio[12000:15000] = 0.0001
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "raw.wav"
        output = Path(directory) / "optimized.wav"
        legacy_output = Path(directory) / "legacy.wav"
        sf.write(raw, audio, sample_rate, subtype="FLOAT")
        create_speech_optimized_wav(raw, output_path=legacy_output)
        create_speech_optimized_wav(raw, output_path=output, preserve_tail_ms=500)
        legacy, _ = sf.read(legacy_output, dtype="float32")
        optimized, _ = sf.read(output, dtype="float32")
    assert legacy.size < 10000
    assert optimized.size > 15900
    assert optimized.size > legacy.size + 5000


def test_max_duration_finalization():
    machine = LiveSpeechStateMachine(
        pre_roll_ms=0,
        partial_interval=0.032,
        max_utterance_seconds=0.064,
    )
    machine.process_frame(FRAME, True)
    events = machine.process_frame(FRAME, True)
    assert "FINAL_DUE" in kinds(events)


def test_ring_buffer_is_bounded():
    machine = LiveSpeechStateMachine(pre_roll_ms=64)
    for _ in range(20):
        machine.process_frame(SILENCE, False)
    assert len(machine.pre_roll) == 2


def test_audio_queue_overflow_keeps_latest():
    audio = StreamingAudioInput(queue_seconds=0.032)
    audio._enqueue(np.zeros(512, dtype=np.float32))
    audio._enqueue(np.ones(512, dtype=np.float32))
    assert audio.dropped_frames == 1
    assert np.array_equal(audio.get().samples, np.ones(512, dtype=np.float32))
    assert audio._stream is None


def run_tests():
    tests = [
        test_waiting_and_pre_roll,
        test_continuation_and_partial,
        test_exact_silence_finalization_and_reset,
        test_boundary_speech_frame_is_in_final_snapshot,
        test_post_roll_frames_are_retained,
        test_final_snapshot_is_newer_than_latest_partial,
        test_live_right_tail_preserves_quiet_final_audio,
        test_max_duration_finalization,
        test_ring_buffer_is_bounded,
        test_audio_queue_overflow_keeps_latest,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: passed")
    print("live STT logic tests passed")


if __name__ == "__main__":
    run_tests()
