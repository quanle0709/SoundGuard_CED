import csv
import hashlib
import shutil
import struct
import tempfile
import wave
from datetime import datetime
from pathlib import Path

from benchmark_runner import (EVENT_FIELDS, MANIFEST_FIELDS, RESULT_FIELDS,
                              FixedFileBenchmark, migrate_results,
                              resolve_manifest_path, validate_manifest)
from emergency_system import EmergencySystem


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def wav(path, seconds=.05, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(sample_rate)
        stream.writeframes(struct.pack("<h", 1000) * int(sample_rate * seconds))


def sample(name="a", scenario="clean_speech", **values):
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(sample_id=name, relative_path=f"{scenario}/{name}.wav", scenario=scenario,
               context="neutral", cycle_index="0")
    row.update(values)
    return row


class Harness:
    def __init__(self, rows, *, events=True, transcript="", enable_stt=None,
                 has_speech=None):
        self.temp = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp.name) / "SoundGuard_CED"
        self.root = self.project_root / "benchmark_v1"
        for row in rows: wav(resolve_manifest_path(row["relative_path"], self.project_root))
        write_csv(self.root / "manifest.csv", MANIFEST_FIELDS, rows)
        write_csv(self.root / "results.csv", RESULT_FIELDS, [])
        write_csv(self.root / "events.csv", EVENT_FIELDS, [])
        self.ced_paths = []; self.vad_paths = []; self.vad_formats = []
        self.enhanced = []; self.enhancer_inputs = []
        self.transcriber_paths = []; self.transcriber_path_exists = []
        speech_expected = bool(transcript) if has_speech is None else has_speech
        def classifier(path):
            self.ced_paths.append(Path(path))
            return {"label": "Siren", "confidence": .99, "top_predictions": []}
        def enhancer(source, output, verbose=False):
            del verbose; self.enhancer_inputs.append(Path(source)); self.enhanced.append(Path(output)); shutil.copy2(source, output)
        def vad_detector(path):
            import soundfile as sf
            self.vad_paths.append(Path(path))
            info = sf.info(path); self.vad_formats.append((info.samplerate, info.channels))
            return {"has_speech": speech_expected, "speech_duration": .05, "speech_ratio": 1}
        def transcriber(path):
            self.transcriber_paths.append(Path(path))
            self.transcriber_path_exists.append(Path(path).exists())
            return transcript
        self.runner = FixedFileBenchmark(
            self.root, project_root=self.project_root,
            enable_stt=bool(transcript) if enable_stt is None else enable_stt,
            write_events=events,
            classifier=classifier, vad=vad_detector,
            enhancer=enhancer, transcriber=transcriber)

    def close(self): self.temp.cleanup()


def test_backup_and_migrate_empty_existing_csv():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.csv"
        write_csv(path, ["sample_id", "old"], [])
        backup = migrate_results(path, require_empty=True, clock=lambda: datetime(2026, 1, 2, 3, 4, 5))
        assert backup and backup.exists()
        assert backup.read_text(encoding="utf-8").startswith("sample_id,old")
        assert next(csv.reader(path.open(encoding="utf-8"))) == RESULT_FIELDS


def test_project_root_relative_sample_path():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory) / "SoundGuard_CED"
        expected = project / "samples" / "test.wav"
        wav(expected)
        resolved = resolve_manifest_path("samples/test.wav", project)
        assert resolved == expected.resolve()
        validate_manifest([sample(relative_path="samples/test.wav")], project)


def test_benchmark_directory_misresolution_regression():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory) / "SoundGuard_CED"
        benchmark = project / "benchmark_v1"
        expected = project / "test_outputs" / "clip.wav"
        wav(expected); benchmark.mkdir(parents=True)
        resolved = resolve_manifest_path("test_outputs/clip.wav", project)
        assert resolved == expected.resolve()
        assert resolved != (benchmark / "test_outputs/clip.wav").resolve()


def test_missing_wav_error_shows_fully_resolved_path():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory) / "SoundGuard_CED"
        expected = (project / "samples" / "missing.wav").resolve()
        try:
            validate_manifest([sample(relative_path="samples/missing.wav")], project)
        except ValueError as exc:
            assert str(expected) in str(exc)
        else:
            raise AssertionError("Missing WAV was accepted")


def test_windows_forward_and_backslash_manifest_paths():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory) / "SoundGuard_CED"
        expected = project / "samples" / "test.wav"
        wav(expected)
        assert resolve_manifest_path("samples/test.wav", project) == expected.resolve()
        assert resolve_manifest_path(r"samples\test.wav", project) == expected.resolve()


def test_paths_escaping_project_root_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory) / "SoundGuard_CED"
        try:
            resolve_manifest_path("../outside.wav", project)
        except ValueError as exc:
            assert "escapes project root" in str(exc)
        else:
            raise AssertionError("Escaping path was accepted")


def test_migration_preserves_populated_csv_data():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.csv"
        write_csv(path, ["sample_id", "notes"], [{"sample_id": "kept", "notes": "human"}])
        migrate_results(path)
        rows = read_csv(path)
        assert len(rows) == 1 and rows[0]["sample_id"] == "kept" and rows[0]["notes"] == "human"


def test_one_row_per_sample_condition_and_paired_dtln():
    harness = Harness([sample("noise", "noisy_speech")])
    try:
        rows = harness.runner.run("pair")
        assert len(rows) == 2
        assert {r["condition"] for r in rows} == {"dtln_enabled", "dtln_disabled"}
        assert {(r["sample_id"], r["run_id"]) for r in rows} == {("noise", "pair")}
    finally: harness.close()


def test_optional_event_log_writing():
    harness = Harness([sample() ], events=False)
    try:
        harness.runner.run("noevents")
        assert read_csv(harness.root / "events.csv") == []
    finally: harness.close()


def test_raw_wav_reaches_ced_and_dtln_never_reaches_ced():
    harness = Harness([sample("noise", "noisy_speech")], transcript="hello")
    try:
        harness.runner.run("routing")
        source = (harness.project_root / "noisy_speech/noise.wav").resolve()
        assert harness.ced_paths == [source, source]
        assert len(harness.enhanced) == 1 and harness.enhanced[0] != source
    finally: harness.close()


def test_44100_wav_uses_temporary_speech_branch_and_preserves_source():
    import soundfile as sf
    harness = Harness([sample("noise", "noisy_speech")], transcript="hello")
    try:
        source = harness.project_root / "noisy_speech/noise.wav"
        wav(source, sample_rate=44100)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = harness.runner.run("resample")
        assert harness.ced_paths == [source.resolve(), source.resolve()]
        assert len(harness.vad_paths) == 2
        assert all(path != source.resolve() for path in harness.vad_paths)
        assert harness.vad_formats == [(16000, 1), (16000, 1)]
        assert len(harness.enhancer_inputs) == 1
        assert harness.enhancer_inputs[0] in harness.vad_paths
        assert harness.transcriber_paths[0] == harness.enhanced[0]
        assert harness.transcriber_paths[1] in harness.vad_paths
        assert harness.transcriber_path_exists == [True, True]
        assert all(row["speech_branch_path_kind"] == "temporary_16khz_mono" for row in rows)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        assert all(not path.exists() for path in harness.vad_paths + harness.enhanced)
    finally: harness.close()


def test_skipped_no_speech_does_not_invoke_stt():
    harness = Harness([sample("quiet", "environmental")], enable_stt=True,
                      has_speech=False)
    try:
        result = harness.runner.run("quiet")[0]
        assert result["stt_status"] == "skipped_no_speech"
        assert harness.transcriber_paths == []
    finally: harness.close()


def test_speech_conversion_failure_preserves_successful_ced_result():
    import benchmark_runner
    harness = Harness([sample("broken", "environmental")])
    original = benchmark_runner.prepare_speech_branch_wav
    benchmark_runner.prepare_speech_branch_wav = lambda source: (_ for _ in ()).throw(
        RuntimeError("synthetic conversion failure"))
    try:
        result = harness.runner.run("conversion_failure")[0]
        assert result["status"] == "failed"
        assert result["speech_normalization_status"] == "failed"
        assert result["vad_status"] == "unavailable"
        assert result["ced_status"] == "success"
        assert result["predicted_sound"] == "Siren"
    finally:
        benchmark_runner.prepare_speech_branch_wav = original
        harness.close()


def test_independent_sound_and_help_lifecycles():
    system = EmergencySystem()
    system.process_sound_event("Siren", .99)
    history = list(system.history)
    system.process_transcript_event("cứu tôi với")
    assert list(system.history) == history
    missing = system.help_missing_cycles
    system.process_sound_event("unknown", 0)
    assert system.help_missing_cycles == missing


def test_sequence_ordering_reset_and_two_of_three_voting():
    rows = [
        sample("third", "emergency", sequence_id="seq", cycle_index="3"),
        sample("first", "emergency", sequence_id="seq", cycle_index="1", reset_emergency_before="true"),
        sample("second", "emergency", sequence_id="seq", cycle_index="2"),
        sample("reset", "emergency", sequence_id="seq", cycle_index="4", reset_emergency_before="true"),
    ]
    harness = Harness(rows)
    try:
        result = harness.runner.run("sequence")
        assert [r["sample_id"] for r in result] == ["first", "second", "third", "reset"]
        assert result[0]["sound_event_state"] == "NO_EVENT"
        assert result[1]["sound_event_state"] == "EVENT_STARTED"
        assert result[1]["sound_temporal_confirmation"] == "true"
        assert result[3]["sound_event_state"] == "NO_EVENT"
    finally: harness.close()


def test_resumable_appending():
    harness = Harness([sample()])
    try:
        assert len(harness.runner.run("resume")) == 1
        assert harness.runner.run("resume") == []
        assert len(read_csv(harness.root / "results.csv")) == 1
    finally: harness.close()


def run_tests():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test(); print(f"{test.__name__}: passed")
    print(f"benchmark runner tests passed ({len(tests)})")


if __name__ == "__main__": run_tests()
