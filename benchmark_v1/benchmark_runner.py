"""Reproducible fixed-WAV benchmark runner for SoundGuard.

The raw source path is passed directly to CED. VAD and optional DTLN/STT form
an independent speech branch; this module never opens a microphone.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from math import gcd
from pathlib import Path

from soundguard.emergency.emergency_system import EmergencySystem, match_sound_category

BENCHMARK_ROOT = Path(__file__).resolve().parent
MANIFEST_FIELDS = ["sample_id", "relative_path", "scenario", "context",
                   "ground_truth_sound", "ground_truth_category", "ground_truth_transcript",
                   "expected_speech", "expected_alert_level", "expected_help_request",
                   "expected_emergency", "noise_type", "sequence_id", "cycle_index",
                   "reset_emergency_before", "notes"]
RESULT_FIELDS = MANIFEST_FIELDS + [
    "condition", "run_id", "source_sha256", "audio_duration_seconds",
    "speech_normalization_status", "speech_branch_path_kind",
    "vad_status", "vad_detected", "vad_speech_duration_seconds", "vad_speech_ratio",
    "predicted_sound", "sound_confidence", "top_predictions", "predicted_category",
    "ced_status", "transcript", "stt_enabled", "stt_status", "dtln_enabled", "dtln_status",
    "sound_alert_level", "sound_event_state",
    "sound_temporal_confirmation", "sound_emitted", "transcript_alert_level",
    "transcript_event_state", "transcript_emitted", "help_request_detected",
    "emergency_detected", "vad_latency_seconds", "dtln_latency_seconds",
    "ced_latency_seconds", "stt_latency_seconds", "total_latency_seconds",
    "dtln_realtime_factor", "vad_cold_start", "vad_model_load_or_cold_start_latency_seconds",
    "vad_inference_latency_seconds", "ced_cold_start",
    "ced_model_load_or_cold_start_latency_seconds", "ced_inference_latency_seconds",
    "dtln_cold_start", "dtln_model_load_or_cold_start_latency_seconds",
    "dtln_inference_latency_seconds", "dtln_inference_realtime_factor",
    "stt_cold_start", "stt_model_load_or_cold_start_latency_seconds",
    "stt_inference_latency_seconds", "status", "errors", "completed_at",
]
EVENT_FIELDS = ["sample_id", "condition", "run_id", "sequence_id", "cycle_index",
                "evidence_type", "source", "capture_started_at", "capture_ended_at",
                "event_sequence", "worker_completed_at", "alert_level", "event_state",
                "emitted_this_cycle"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _atomic_write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def migrate_results(path: Path, require_empty: bool = False, backup: bool = True,
                    clock=None) -> Path | None:
    """Back up then migrate results while retaining every existing data row."""
    path = Path(path)
    rows = read_csv(path)
    if require_empty and rows:
        raise ValueError(f"Refusing migration: {path} contains {len(rows)} data rows")
    old_fields = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            old_fields = next(csv.reader(stream), [])
    if old_fields == RESULT_FIELDS:
        return None
    backup_path = None
    if path.exists() and backup:
        now = clock() if clock else dt.datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S_%f")
        backup_path = path.with_name(f"results.pre_schema_{stamp}.csv")
        shutil.copy2(path, backup_path)
    _atomic_write(path, RESULT_FIELDS, rows)
    return backup_path


def ensure_csv(path: Path, fields: list[str]) -> None:
    if not path.exists():
        _atomic_write(path, fields, [])


def parse_bool(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def conditions_for(row: dict) -> list[str]:
    scenario = row.get("scenario", "").strip().lower().replace("-", "_")
    return ["dtln_enabled", "dtln_disabled"] if scenario == "noisy_speech" else ["dtln_disabled"]


def resolve_manifest_path(relative_path: str, project_root: Path,
                          allow_external_paths: bool = False) -> Path:
    """Resolve a manifest audio path against the project, never benchmark_v1."""
    project_root = Path(project_root).resolve()
    # Accept either separator in manifests, including CSVs authored off-Windows.
    candidate_text = str(relative_path or "").strip().replace("\\", os.sep).replace("/", os.sep)
    candidate = Path(candidate_text)
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    if not allow_external_paths and resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"Audio path escapes project root: {resolved}")
    return resolved


def validate_manifest(rows: list[dict], project_root: Path,
                      allow_external_paths: bool = False) -> None:
    seen = set()
    for index, row in enumerate(rows, 2):
        missing = [name for name in ("sample_id", "relative_path") if not row.get(name, "").strip()]
        if missing:
            raise ValueError(f"Manifest row {index} missing: {', '.join(missing)}")
        if row["sample_id"] in seen:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        source = resolve_manifest_path(row["relative_path"], project_root, allow_external_paths)
        if not source.is_file():
            raise ValueError(f"Invalid or missing WAV for {row['sample_id']}: {source}")


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_duration(path: Path) -> float:
    import soundfile as sf
    info = sf.info(path)
    return float(info.frames / info.samplerate)


def prepare_speech_branch_wav(source: Path) -> tuple[Path, Path | None]:
    """Return mono/16 kHz speech input and an optional temporary file to clean."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    source = Path(source)
    info = sf.info(source)
    if info.samplerate == 16000 and info.channels == 1:
        return source, None
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Speech-branch source WAV is empty: {source}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != 16000:
        divisor = gcd(int(sample_rate), 16000)
        mono = resample_poly(mono, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    handle, name = tempfile.mkstemp(prefix="soundguard_speech_", suffix=".wav")
    os.close(handle)
    temporary = Path(name)
    try:
        sf.write(temporary, mono, 16000, subtype="PCM_16")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary, temporary


class FixedFileBenchmark:
    def __init__(self, root=BENCHMARK_ROOT, *, project_root=None,
                 allow_external_paths=False, enable_stt=False, write_events=True,
                 classifier=None, vad=None, enhancer=None, transcriber=None,
                 emergency_factory=EmergencySystem, timer=time.perf_counter):
        self.root = Path(root)
        self.project_root = (Path(project_root) if project_root is not None
                             else self.root.parent).resolve()
        self.allow_external_paths = allow_external_paths
        self.enable_stt, self.write_events = enable_stt, write_events
        if classifier is None:
            from soundguard.detection.sound_classifier import classify_audio_file
            classifier = classify_audio_file
        if vad is None:
            from soundguard.speech.voice_activity_detector import detect_speech
            vad = detect_speech
        if enhancer is None:
            from soundguard.speech.speech_enhancer import enhance_audio_file
            enhancer = enhance_audio_file
        if transcriber is None:
            from soundguard.speech.speech_recognizer import transcribe_audio_file
            transcriber = transcribe_audio_file
        self.classifier, self.vad, self.enhancer = classifier, vad, enhancer
        self.transcriber, self.emergency_factory, self.timer = transcriber, emergency_factory, timer
        self._successful_stage_calls = defaultdict(int)

    def _record_stage_timing(self, row: dict, stage: str, latency: float) -> None:
        cold = self._successful_stage_calls[stage] == 0
        self._successful_stage_calls[stage] += 1
        row[f"{stage}_cold_start"] = str(cold).lower()
        target = (f"{stage}_model_load_or_cold_start_latency_seconds" if cold
                  else f"{stage}_inference_latency_seconds")
        row[target] = latency

    @staticmethod
    def _sequence_key(row: dict) -> str:
        return row.get("sequence_id", "").strip() or f"sample:{row['sample_id']}"

    def run(self, run_id: str | None = None, resume=True) -> list[dict]:
        run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path, results_path = self.root / "manifest.csv", self.root / "results.csv"
        events_path = self.root / "events.csv"
        manifest = read_csv(manifest_path)
        validate_manifest(manifest, self.project_root, self.allow_external_paths)
        migrate_results(results_path, backup=True)
        ensure_csv(events_path, EVENT_FIELDS)
        existing, events = read_csv(results_path), read_csv(events_path)
        completed = {(r.get("sample_id"), r.get("condition"), r.get("run_id"))
                     for r in existing if r.get("status") == "completed"}
        ordered = sorted(enumerate(manifest), key=lambda item: (
            self._sequence_key(item[1]), int(item[1].get("cycle_index") or 0), item[0]))
        systems, new_rows = {}, []
        for _, sample in ordered:
            for condition in conditions_for(sample):
                key = (sample["sample_id"], condition, run_id)
                if resume and key in completed:
                    continue
                sequence = self._sequence_key(sample)
                system_key = (sequence, condition)
                if system_key not in systems or parse_bool(sample.get("reset_emergency_before")):
                    systems[system_key] = self.emergency_factory(
                        context=sample.get("context") or "neutral", decision_mode="continuous")
                row, diagnostic = self._run_one(sample, condition, run_id, systems[system_key])
                existing.append(row); new_rows.append(row); events.extend(diagnostic)
                _atomic_write(results_path, RESULT_FIELDS, existing)
                if self.write_events:
                    _atomic_write(events_path, EVENT_FIELDS, events)
        return new_rows

    def _run_one(self, sample, condition, run_id, emergency):
        source = resolve_manifest_path(sample["relative_path"], self.project_root,
                                       self.allow_external_paths)
        started = self.timer(); errors = []; diagnostics = []
        row = {field: sample.get(field, "") for field in MANIFEST_FIELDS}
        row.update(condition=condition, run_id=run_id, source_sha256=_sha256(source),
                   stt_enabled=str(self.enable_stt).lower(),
                   dtln_enabled=str(condition == "dtln_enabled").lower())
        try:
            duration = _audio_duration(source); row["audio_duration_seconds"] = duration
        except Exception as exc:
            duration = 0.0; errors.append(f"audio: {type(exc).__name__}: {exc}")
        speech_path = None
        temporary_paths = []
        try:
            speech_path, normalized_temp = prepare_speech_branch_wav(source)
            if normalized_temp is not None:
                temporary_paths.append(normalized_temp)
            row["speech_normalization_status"] = "success"
            row["speech_branch_path_kind"] = "temporary_16khz_mono" if normalized_temp else "original_16khz_mono"
        except Exception as exc:
            row["speech_normalization_status"] = "failed"
            errors.append(f"speech_normalization: {type(exc).__name__}: {exc}")
        if speech_path is not None:
          try:
            mark = self.timer(); vad_result = self.vad(speech_path); elapsed = self.timer() - mark
            row["vad_latency_seconds"] = elapsed; self._record_stage_timing(row, "vad", elapsed)
            row["vad_status"] = "success"
            row.update(vad_detected=str(bool(vad_result.get("has_speech"))).lower(),
                       vad_speech_duration_seconds=vad_result.get("speech_duration", ""),
                       vad_speech_ratio=vad_result.get("speech_ratio", ""))
          except Exception as exc:
            row["vad_latency_seconds"] = self.timer() - mark; row["vad_status"] = "failed"
            vad_result = {"has_speech": False}; errors.append(f"vad: {type(exc).__name__}: {exc}")
        else:
            vad_result = {"has_speech": False}; row["vad_status"] = "unavailable"
        dtln_path = None
        if condition == "dtln_enabled":
            if speech_path is None:
                row["dtln_status"] = "unavailable"
            else:
              dtln_path = Path(tempfile.gettempdir()) / f"soundguard_benchmark_{uuid.uuid4().hex}.wav"
              temporary_paths.append(dtln_path)
              try:
                mark = self.timer(); self.enhancer(speech_path, dtln_path, verbose=False)
                elapsed = self.timer() - mark; row["dtln_latency_seconds"] = elapsed
                self._record_stage_timing(row, "dtln", elapsed); row["dtln_status"] = "success"
                row["dtln_realtime_factor"] = (float(row["dtln_latency_seconds"]) / duration
                                                   if duration else "")
                if row.get("dtln_inference_latency_seconds") not in {None, ""}:
                    row["dtln_inference_realtime_factor"] = (elapsed / duration if duration else "")
                speech_path = dtln_path
              except Exception as exc:
                row["dtln_latency_seconds"] = self.timer() - mark; row["dtln_status"] = "failed"
                errors.append(f"dtln: {type(exc).__name__}: {exc}")
        else:
            row["dtln_status"] = "disabled"
        try:
            mark = self.timer(); ced = self.classifier(source); elapsed = self.timer() - mark
            row["ced_latency_seconds"] = elapsed; self._record_stage_timing(row, "ced", elapsed)
            row["ced_status"] = "success"
            row.update(predicted_sound=ced.get("label", "unknown"),
                       sound_confidence=ced.get("confidence", 0.0),
                       top_predictions=json.dumps(ced.get("top_predictions", []), ensure_ascii=False),
                       predicted_category=match_sound_category(ced.get("label", "")) or "none")
            decision = emergency.process_sound_event(row["predicted_sound"], float(row["sound_confidence"]))
            row.update(sound_alert_level=decision["alert_level"], sound_event_state=decision["event_state"],
                       sound_temporal_confirmation=str(decision["temporal_confirmation"]).lower(),
                       sound_emitted=str(decision["emitted_this_cycle"]).lower())
            diagnostics.append(self._event(sample, condition, run_id, "sound", "ced", 1, decision))
        except Exception as exc:
            row["ced_latency_seconds"] = self.timer() - mark; row["ced_status"] = "failed"
            errors.append(f"ced: {type(exc).__name__}: {exc}")
        transcript = ""
        if not self.enable_stt:
            row["stt_status"] = "disabled"
        elif row.get("vad_status") != "success":
            row["stt_status"] = "unavailable"
        elif not vad_result.get("has_speech"):
            row["stt_status"] = "skipped_no_speech"
        else:
            try:
                mark = self.timer(); transcript = self.transcriber(speech_path); elapsed = self.timer() - mark
                row["stt_latency_seconds"] = elapsed; self._record_stage_timing(row, "stt", elapsed)
                row["stt_status"] = "success"
            except Exception as exc:
                row["stt_latency_seconds"] = self.timer() - mark; row["stt_status"] = "failed"
                errors.append(f"stt: {type(exc).__name__}: {exc}")
        row["transcript"] = transcript
        if transcript.strip():
            try:
                decision = emergency.process_transcript_event(transcript)
                row.update(transcript_alert_level=decision["alert_level"],
                           transcript_event_state=decision["event_state"],
                           transcript_emitted=str(decision["emitted_this_cycle"]).lower(),
                           help_request_detected=str(decision["help_request_detected"]).lower())
                diagnostics.append(self._event(sample, condition, run_id, "transcript", "stt", 2, decision))
            except Exception as exc:
                errors.append(f"transcript_event: {type(exc).__name__}: {exc}")
        row["emergency_detected"] = str(row.get("sound_event_state") in {"EVENT_STARTED", "EVENT_CONTINUING"}
                                           or row.get("transcript_event_state") in {"EVENT_STARTED", "EVENT_CONTINUING"}).lower()
        row["total_latency_seconds"] = self.timer() - started
        row["status"] = "failed" if errors else "completed"; row["errors"] = " | ".join(errors)
        row["completed_at"] = dt.datetime.now().isoformat(timespec="seconds")
        for temporary in temporary_paths:
            try: temporary.unlink(missing_ok=True)
            except OSError: pass
        return row, diagnostics

    @staticmethod
    def _event(sample, condition, run_id, evidence, source, sequence, decision):
        now = dt.datetime.now().isoformat(timespec="milliseconds")
        return {"sample_id": sample["sample_id"], "condition": condition, "run_id": run_id,
                "sequence_id": sample.get("sequence_id", ""), "cycle_index": sample.get("cycle_index", ""),
                "evidence_type": evidence, "source": source, "capture_started_at": "",
                "capture_ended_at": "", "event_sequence": sequence, "worker_completed_at": now,
                "alert_level": decision["alert_level"], "event_state": decision["event_state"],
                "emitted_this_cycle": decision["emitted_this_cycle"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--project-root", type=Path,
                        help="Project root for manifest paths; defaults to --root's parent")
    parser.add_argument("--allow-external-paths", action="store_true",
                        help="Explicitly allow manifest audio outside the project root")
    parser.add_argument("--run-id")
    parser.add_argument("--enable-stt", action="store_true",
                        help="Explicitly allow external Google STT calls")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    rows = FixedFileBenchmark(args.root, project_root=args.project_root,
                              allow_external_paths=args.allow_external_paths,
                              enable_stt=args.enable_stt,
                              write_events=not args.no_events).run(args.run_id, not args.no_resume)
    print(f"Completed {len(rows)} new sample-condition rows. Microphone was not opened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
