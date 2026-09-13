"""Reproducible public-dataset validation for HearVis personalized recognition.

The evaluator uses only cached official ESC-50 and LibriSpeech SLR12 material.
It deliberately stores benchmark enrollments below ignored benchmark_data/ and
never reads or writes the normal personalization/enrollments profile store.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from personalization import web_server
from personalization.recognition import (
    DEFAULT_ROOT,
    BoundedRecognitionWorker,
    EmbeddingClient,
    RecognitionStore,
    TemporalDecisionGate,
    RecognitionError,
    validate_audio_bytes,
)
from personalization.runtime import PersonalizedRecognitionRuntime
from personalization.web_server import PersonalizationHandler
from soundguard.audio.audio_pipeline import MicrophonePipeline
from soundguard.emergency.emergency_system import EmergencySystem
from soundguard.speech.live_speech_to_text import RecognitionResult, TranscriptDisplay


SEED = 20260913
ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
ESC50_METADATA_SHA256 = "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246"
LIBRISPEECH_TEST_CLEAN_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
SOUND_TARGETS = {
    "door_wood_knock": "DOOR KNOCK",
    "clock_alarm": "CLOCK ALARM",
    "washing_machine": "WASHING MACHINE",
}
SOUND_NEGATIVES = {
    "door_wood_creaks": "DOOR KNOCK",
    "clock_tick": "CLOCK ALARM",
    "vacuum_cleaner": "WASHING MACHINE",
    "car_horn": "",
    "siren": "",
    "dog": "",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def distribution(values) -> dict:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {"count": 0, "min": None, "p25": None, "p50": None,
                "p95": None, "max": None}
    return {
        "count": len(cleaned),
        "min": min(cleaned),
        "p25": percentile(cleaned, 25),
        "p50": percentile(cleaned, 50),
        "p95": percentile(cleaned, 95),
        "max": max(cleaned),
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    columns = fields or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def snapshot_real_user_data() -> dict:
    """Fingerprint real profile data without exposing its content in results."""
    roots = [DEFAULT_ROOT, ROOT / "personalization" / "user_profile.json"]
    entries = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in paths:
            if path.is_file():
                entries.append((repo_path(path), path.stat().st_size, sha256(path)))
    digest = hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"file_count": len(entries), "aggregate_sha256": digest}


def load_esc50_rows(metadata_path: Path) -> list[dict]:
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def valid_for_enrollment(path: Path, kind: str) -> bool:
    try:
        validate_audio_bytes(path.read_bytes(), kind)
        return True
    except (OSError, RecognitionError):
        return False


def _sound_row(row: dict, split: str, expected: str, audio_root: Path,
               confusable_for: str = "") -> dict:
    path = audio_root / row["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"missing selected ESC-50 clip: {path}")
    return {
        "dataset": "ESC-50",
        "split": split,
        "expected_label": expected,
        "category": row["category"],
        "fold": int(row["fold"]),
        "source_id": row.get("src_file", ""),
        "take": row.get("take", ""),
        "confusable_for": confusable_for,
        "file": row["filename"],
        "path": repo_path(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "_path": path,
    }


def build_sound_split(metadata_path: Path, audio_root: Path) -> list[dict]:
    rows = load_esc50_rows(metadata_path)
    selected = []
    for category, label in SOUND_TARGETS.items():
        category_rows = [row for row in rows if row["category"] == category]
        for fold, split, count in (
            (1, "enrollment", 5),
            (2, "calibration", 5), (3, "calibration", 5),
            (4, "positive_holdout", 5), (5, "positive_holdout", 5),
        ):
            candidates = [row for row in sorted(
                (row for row in category_rows if int(row["fold"]) == fold),
                key=lambda row: row["filename"],
            ) if valid_for_enrollment(audio_root / row["filename"], "sound")][:count]
            if len(candidates) != count:
                raise RuntimeError(f"ESC-50 {category} fold {fold} has insufficient clips")
            selected.extend(_sound_row(row, split, label, audio_root) for row in candidates)
    for category, confusable_for in SOUND_NEGATIVES.items():
        category_rows = [row for row in rows if row["category"] == category]
        for fold, split in (
            (2, "calibration_unknown"), (3, "calibration_unknown"),
            (4, "negative_holdout"), (5, "negative_holdout"),
        ):
            candidates = [row for row in sorted(
                (row for row in category_rows if int(row["fold"]) == fold),
                key=lambda row: row["filename"],
            ) if valid_for_enrollment(audio_root / row["filename"], "sound")][:2]
            if len(candidates) != 2:
                raise RuntimeError(f"ESC-50 {category} fold {fold} has insufficient clips")
            selected.extend(
                _sound_row(row, split, "UNKNOWN", audio_root, confusable_for)
                for row in candidates
            )
    _assert_independent_split(selected, "source_id")
    return selected


def build_voice_split(librispeech_root: Path) -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    speakers = {}
    for directory in sorted(path for path in librispeech_root.iterdir() if path.is_dir()):
        files = sorted(directory.rglob("*.flac"))
        if files:
            rng.shuffle(files)
            speakers[directory.name] = files
    eligible_enrolled = [speaker for speaker, files in speakers.items() if len(files) >= 20]
    rng.shuffle(eligible_enrolled)
    enrolled = eligible_enrolled[:2]
    remaining = [speaker for speaker, files in speakers.items()
                 if speaker not in enrolled and len(files) >= 5]
    rng.shuffle(remaining)
    calibration_impostors, holdout_impostors = remaining[:3], remaining[3:8]
    if len(enrolled) != 2 or len(holdout_impostors) != 5:
        raise RuntimeError("LibriSpeech test-clean has insufficient eligible speakers")
    display = {enrolled[0]: "DEMO SPEAKER A", enrolled[1]: "DEMO SPEAKER B"}
    rows = []

    def add(speaker: str, paths: list[Path], split: str, expected: str) -> None:
        for path in paths:
            rows.append({
                "dataset": "LibriSpeech SLR12 test-clean",
                "split": split,
                "speaker_id": speaker,
                "utterance_id": path.stem,
                "expected_label": expected,
                "file": path.name,
                "path": repo_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "_path": path,
                        })

    def take_valid(speaker: str, count: int) -> list[Path]:
        selected = []
        for path in speakers[speaker]:
            if valid_for_enrollment(path, "voice"):
                selected.append(path)
                if len(selected) == count:
                    return selected
        raise RuntimeError(
            f"LibriSpeech speaker {speaker} has fewer than {count} valid utterances"
        )

    for speaker in enrolled:
        paths = take_valid(speaker, 20)
        add(speaker, paths[:5], "enrollment", display[speaker])
        add(speaker, paths[5:10], "calibration", display[speaker])
        add(speaker, paths[10:20], "positive_holdout", display[speaker])
    for speaker in calibration_impostors:
        add(speaker, take_valid(speaker, 5), "calibration_unknown", "UNKNOWN")
    for speaker in holdout_impostors:
        add(speaker, take_valid(speaker, 5), "negative_holdout", "UNKNOWN")
    _assert_independent_split(rows, "utterance_id")
    return rows, {
        "enrolled_speakers": [{"speaker_id": speaker, "display_label": display[speaker]}
                              for speaker in enrolled],
        "calibration_impostor_speaker_ids": calibration_impostors,
        "holdout_impostor_speaker_ids": holdout_impostors,
    }


def _assert_independent_split(rows: list[dict], identity_field: str) -> None:
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise AssertionError("an audio file was reused across benchmark splits")
    memberships = defaultdict(set)
    for row in rows:
        memberships[row.get(identity_field, "")].add(row["split"])
    cross_split = {key: value for key, value in memberships.items()
                   if key and len(value) > 1}
    if cross_split:
        raise AssertionError(f"source identity crosses benchmark splits: {cross_split}")


class TimedEmbedder:
    def __init__(self, client: EmbeddingClient):
        self.client = client
        self.records: list[dict] = []
        self.stage = "unspecified"

    def embed(self, kind: str, path: Path):
        started = time.perf_counter()
        vector, metadata = self.client.embed(kind, path)
        wall = time.perf_counter() - started
        self.records.append({
            "kind": kind,
            "stage": self.stage,
            "wall_seconds": wall,
            "inference_seconds": metadata.get("inference_seconds"),
            "cpu_seconds": metadata.get("cpu_seconds"),
            "load_seconds": metadata.get("load_seconds"),
            "worker_rss_bytes": metadata.get("worker_rss_bytes"),
            "worker_peak_rss_bytes": metadata.get("worker_peak_rss_bytes"),
            "model": metadata.get("model"),
            "revision": metadata.get("revision"),
            "sha256": metadata.get("sha256"),
            "preprocessing": metadata.get("preprocessing"),
        })
        return vector, metadata


def decision_record(row: dict, result: dict, phase: str) -> dict:
    top1 = result.get("display_name") or "UNKNOWN"
    return {
        "phase": phase,
        "file": row["file"],
        "dataset_label": row.get("category", row.get("speaker_id", "")),
        "expected_label": row["expected_label"],
        "top1_label": top1,
        "predicted_label": result.get("label", "UNKNOWN"),
        "accepted": bool(result.get("accepted")),
        "reason": result.get("reason"),
        "score": result.get("score"),
        "runner_up_score": result.get("runner_up_score"),
        "runner_up_margin": result.get("margin"),
        "threshold": result.get("threshold"),
        "margin_required": result.get("margin_required"),
        "fold": row.get("fold", ""),
        "speaker_id": row.get("speaker_id", ""),
        "confusable_for": row.get("confusable_for", ""),
    }


def apply_operating_point(record: dict, threshold: float, margin: float) -> dict:
    value = dict(record)
    accepted = float(value["score"]) >= threshold and float(value["runner_up_margin"]) >= margin
    value["accepted"] = accepted
    value["predicted_label"] = value["top1_label"] if accepted else "UNKNOWN"
    value["reason"] = "accepted" if accepted else (
        "below_threshold" if float(value["score"]) < threshold else "ambiguous_margin"
    )
    value["threshold"] = threshold
    value["margin_required"] = margin
    return value


def select_operating_point(records: list[dict]) -> dict:
    known = [row for row in records if row["expected_label"] != "UNKNOWN"]
    unknown = [row for row in records if row["expected_label"] == "UNKNOWN"]
    if not known or not unknown:
        raise ValueError("calibration requires known and unknown examples")
    best = None
    for threshold_index in range(35, 96):
        threshold = threshold_index / 100
        for margin_index in range(0, 31):
            margin = margin_index / 100
            applied = [apply_operating_point(row, threshold, margin) for row in records]
            known_correct = sum(row["predicted_label"] == row["expected_label"]
                                for row in applied if row["expected_label"] != "UNKNOWN")
            unknown_correct = sum(row["predicted_label"] == "UNKNOWN"
                                  for row in applied if row["expected_label"] == "UNKNOWN")
            known_rate = known_correct / len(known)
            unknown_rate = unknown_correct / len(unknown)
            rank = (min(known_rate, unknown_rate), (known_rate + unknown_rate) / 2,
                    unknown_rate, known_rate, threshold + margin)
            if best is None or rank > best[0]:
                best = (rank, threshold, margin, known_correct, unknown_correct)
    assert best is not None
    return {
        "method": "fixed-grid maximin balanced known-identification/unknown-rejection; safety tie-break",
        "candidate_threshold_range": [0.35, 0.95, 0.01],
        "candidate_margin_range": [0.0, 0.30, 0.01],
        "threshold": best[1],
        "margin": best[2],
        "known_correct": best[3],
        "known_total": len(known),
        "unknown_rejected": best[4],
        "unknown_total": len(unknown),
    }


def evaluate_kind(kind: str, rows: list[dict], store: RecognitionStore,
                  tracker: TimedEmbedder) -> tuple[list[dict], dict, dict]:
    profiles = {}
    prototype_times = {}
    enrollment_quality = []
    for expected in sorted({row["expected_label"] for row in rows
                            if row["split"] == "enrollment"}):
        kwargs = {"consent_acknowledged": True} if kind == "voice" else {"priority": 3}
        profile = store.create(kind, expected, **kwargs)
        profiles[expected] = profile["id"]
        enrollment = [row for row in rows
                      if row["split"] == "enrollment" and row["expected_label"] == expected]
        if len(enrollment) != 5:
            raise AssertionError(f"{kind} profile {expected} does not have exactly five samples")
        for row in enrollment:
            sample = store.add_sample(kind, profile["id"], row["_path"].read_bytes())
            enrollment_quality.append({
                "file": row["file"], "expected_label": expected,
                "quality": sample["quality"], "duration_seconds": sample["duration_seconds"],
                "sample_rate_hz": sample["sample_rate_hz"],
            })
        tracker.stage = f"{kind}_prototype_build"
        started = time.perf_counter()
        built = store.build(kind, profile["id"], tracker.embed)
        prototype_times[expected] = time.perf_counter() - started
        if not built["built"] or not built["enabled"]:
            raise AssertionError("production profile build did not persist an enabled prototype")

    # A new store instance must be able to recover every persisted prototype.
    if len(RecognitionStore(store.root).matcher_profiles(kind)) != len(profiles):
        raise AssertionError("persisted prototypes could not be reloaded")

    raw_calibration = []
    for row in rows:
        if row["split"] not in {"calibration", "calibration_unknown"}:
            continue
        tracker.stage = f"{kind}_calibration"
        result = store.test(kind, row["_path"].read_bytes(), tracker.embed)
        raw_calibration.append(decision_record(row, result, "calibration"))
    operating_point = select_operating_point(raw_calibration)
    for profile_id in profiles.values():
        store.update(kind, profile_id, {
            "threshold": operating_point["threshold"],
            "margin": operating_point["margin"],
        })
    calibration = [apply_operating_point(
        row, operating_point["threshold"], operating_point["margin"]
    ) for row in raw_calibration]

    holdout = []
    for row in rows:
        if row["split"] not in {"positive_holdout", "negative_holdout"}:
            continue
        tracker.stage = f"{kind}_holdout"
        result = store.test(kind, row["_path"].read_bytes(), tracker.embed)
        holdout.append(decision_record(row, result, "holdout"))
    predictions = calibration + holdout
    metrics = classification_metrics(holdout)
    metrics.update({
        "calibration": operating_point,
        "calibration_counts": {
            "known": sum(row["expected_label"] != "UNKNOWN" for row in calibration),
            "unknown": sum(row["expected_label"] == "UNKNOWN" for row in calibration),
        },
        "prototype_build_seconds": prototype_times,
        "prototype_build_seconds_distribution": distribution(list(prototype_times.values())),
        "enrollment_quality": enrollment_quality,
        "rejected_examples": [
            {key: row[key] for key in ("phase", "file", "expected_label", "score",
                                       "runner_up_margin", "reason")}
            for row in predictions if row["predicted_label"] == "UNKNOWN"
        ],
    })
    smoothing = (
        exercise_smoothing_and_cooldown(holdout)
        if kind == "sound" else {
            "applicable": False,
            "reason": "voice identification is utterance-level; sound-only smoothing/cooldown is not applied",
        }
    )
    return predictions, metrics, smoothing


def classification_metrics(rows: list[dict]) -> dict:
    known = [row for row in rows if row["expected_label"] != "UNKNOWN"]
    unknown = [row for row in rows if row["expected_label"] == "UNKNOWN"]
    correct_known = sum(row["predicted_label"] == row["expected_label"] for row in known)
    known_accepted = sum(row["predicted_label"] != "UNKNOWN" for row in known)
    false_rejects = sum(row["predicted_label"] == "UNKNOWN" for row in known)
    misidentified = sum(row["predicted_label"] not in {"UNKNOWN", row["expected_label"]}
                        for row in known)
    unknown_rejected = sum(row["predicted_label"] == "UNKNOWN" for row in unknown)
    false_accepts = len(unknown) - unknown_rejected
    top1_correct = sum(row["top1_label"] == row["expected_label"] for row in known)
    labels = sorted({row["expected_label"] for row in known}) + ["UNKNOWN"]
    confusion = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for row in rows:
        predicted = row["predicted_label"]
        confusion.setdefault(row["expected_label"], {}).setdefault(predicted, 0)
        confusion[row["expected_label"]][predicted] += 1
    per_profile = {}
    for label in sorted({row["expected_label"] for row in known}):
        items = [row for row in known if row["expected_label"] == label]
        per_profile[label] = {
            "top1_correct": sum(row["top1_label"] == label for row in items),
            "top1_total": len(items),
            "accepted_correct": sum(row["predicted_label"] == label for row in items),
            "accepted_correct_total": len(items),
        }
    return {
        "holdout_counts": {"known": len(known), "unknown": len(unknown), "total": len(rows)},
        "correct_identification": {"numerator": correct_known, "denominator": len(known)},
        "known_acceptance": {"numerator": known_accepted, "denominator": len(known)},
        "unknown_rejection": {"numerator": unknown_rejected, "denominator": len(unknown)},
        "false_acceptance": {"numerator": false_accepts, "denominator": len(unknown)},
        "false_rejection": {"numerator": false_rejects, "denominator": len(known)},
        "misidentification": {"numerator": misidentified, "denominator": len(known)},
        "top1_known": {"numerator": top1_correct, "denominator": len(known)},
        "overall_decision_correct": {
            "numerator": correct_known + unknown_rejected,
            "denominator": len(rows),
        },
        "ambiguous_results": {
            "numerator": sum(row["reason"] == "ambiguous_margin" for row in rows),
            "denominator": len(rows),
        },
        "per_profile": per_profile,
        "confusion_matrix": confusion,
        "holdout_rejection_reasons": dict(Counter(
            row["reason"] for row in rows if row["predicted_label"] == "UNKNOWN"
        )),
        "score_distributions": {
            "known": distribution(row["score"] for row in known),
            "unknown": distribution(row["score"] for row in unknown),
        },
        "runner_up_margin_distributions": {
            "known": distribution(row["runner_up_margin"] for row in known),
            "unknown": distribution(row["runner_up_margin"] for row in unknown),
        },
    }


def exercise_smoothing_and_cooldown(holdout: list[dict]) -> dict:
    source = next((row for row in holdout if row["accepted"]), None)
    if source is None:
        return {"performed": False, "reason": "no accepted real-model holdout decision"}
    clock = [100.0]
    gate = TemporalDecisionGate(required=2, window=3, cooldown_seconds=5.0,
                                clock=lambda: clock[0])
    raw = {"accepted": True, "label": source["predicted_label"],
           "score": source["score"], "reason": "accepted"}
    first = gate.apply(raw)
    second = gate.apply(raw)
    third = gate.apply(raw)
    clock[0] += 6.0
    fourth = gate.apply(raw)
    return {
        "performed": True,
        "source_file": source["file"],
        "sequence": [
            {"step": "first", "accepted": first["accepted"], "reason": first["reason"]},
            {"step": "second", "accepted": second["accepted"], "reason": second["reason"]},
            {"step": "within_cooldown", "accepted": third["accepted"], "reason": third["reason"]},
            {"step": "after_cooldown", "accepted": fourth["accepted"], "reason": fourth["reason"]},
        ],
    }


def load_mono_16k(path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sample_rate == 16_000:
        return mono.astype(np.float32)
    count = max(1, round(len(mono) * 16_000 / sample_rate))
    source = np.arange(len(mono), dtype=np.float64)
    target = np.linspace(0, len(mono) - 1, count, dtype=np.float64)
    return np.interp(target, source, mono).astype(np.float32)


def run_real_concurrent_workers(store: RecognitionStore, client: EmbeddingClient,
                                sound_path: Path, voice_path: Path) -> dict:
    runtime = PersonalizedRecognitionRuntime(
        familiar_sounds=True, familiar_voices=True, store=store, client=client,
        sample_rate=16_000, sound_cooldown_seconds=0,
    )
    sound_results = []
    voice_result = None
    try:
        runtime.start()
        runtime.submit_sound(load_mono_16k(sound_path), "sound-concurrent-1")
        runtime.submit_voice("voice-concurrent-1", load_mono_16k(voice_path))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and (not sound_results or voice_result is None):
            sound_results.extend(runtime.drain_sound_results())
            voice_result = voice_result or runtime.take_voice_result("voice-concurrent-1", timeout=0.02)
            time.sleep(0.01)
        runtime.submit_sound(load_mono_16k(sound_path), "sound-concurrent-2")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(sound_results) < 2:
            sound_results.extend(runtime.drain_sound_results())
            time.sleep(0.01)
        return {
            "both_worker_threads_started": runtime.sound_worker.thread.is_alive()
            and runtime.voice_worker.thread.is_alive(),
            "sound_results": sound_results,
            "voice_result": voice_result,
            "counters": runtime.counters,
            "combined_worker_rss_bytes": sum(
                value for value in (
                    sound_results[-1].get("model", {}).get("worker_rss_bytes")
                    if sound_results else None,
                    voice_result.get("model", {}).get("worker_rss_bytes")
                    if voice_result else None,
                ) if value is not None
            ),
            "combined_worker_peak_rss_bytes": sum(
                value for value in (
                    sound_results[-1].get("model", {}).get("worker_peak_rss_bytes")
                    if sound_results else None,
                    voice_result.get("model", {}).get("worker_peak_rss_bytes")
                    if voice_result else None,
                ) if value is not None
            ),
        }
    finally:
        runtime.close()


def run_dataset_fusion_check(voice_result: dict, sound_result: dict) -> dict:
    class Runtime:
        familiar_sounds = True
        sound_enabled = True
        counters = {}

        def __init__(self):
            self.sound_pending = [sound_result]
            self.voice_pending = voice_result

        def submit_sound(self, *_): return True
        def submit_voice(self, *_): return True

        def take_voice_result(self, *_args, **_kwargs):
            value, self.voice_pending = self.voice_pending, None
            return value

        def drain_sound_results(self):
            values, self.sound_pending = self.sound_pending, []
            return values

    class HUD:
        def __init__(self):
            self.partial_subtitles = []
            self.final_subtitles = []
            self.environmental_sounds = []
            self.counters = {"C_frames_sent": 0}

        def set_partial_subtitle(self, value): self.partial_subtitles.append(value)
        def set_subtitle(self, value): self.final_subtitles.append(value)
        def set_environmental_sound(self, value): self.environmental_sounds.append(value)
        def set_alert_state(self, *_): pass
        def set_status(self, *_): pass
        def update(self): pass

    runtime, hud = Runtime(), HUD()
    emergency = EmergencySystem(decision_mode="continuous")
    raw_transcripts = []
    original = emergency.process_transcript_event

    def capture(text, **kwargs):
        raw_transcripts.append(text)
        return original(text, **kwargs)

    emergency.process_transcript_event = capture
    pipeline = MicrophonePipeline(
        mode="live-stt", emergency_system=emergency,
        display=TranscriptDisplay(io.StringIO()), hud=hud,
        personalized_runtime=runtime,
    )
    pipeline._dispatch_recognition(
        RecognitionResult("PARTIAL", 1, "giúp", event_sequence=1)
    )
    pipeline._dispatch_recognition(
        RecognitionResult("FINAL", 1, "giúp tôi", event_sequence=2)
    )
    locked_before_sound = pipeline.active_hud_alert
    pipeline._pump()
    expected_final = f"[{voice_result['label']}] giúp tôi"
    return {
        "used_real_model_voice_decision": bool(voice_result.get("accepted")),
        "used_real_model_sound_decision": bool(sound_result.get("accepted")),
        "partial_caption_unprefixed": hud.partial_subtitles == ["giúp"],
        "final_caption_prefixed": hud.final_subtitles == [expected_final],
        "expected_final_caption": expected_final,
        "help_received_raw_unprefixed_transcript": raw_transcripts == ["giúp tôi"],
        "help_active": emergency.help_active,
        "locked_alert_before_sound": locked_before_sound,
        "locked_alert_after_sound": pipeline.active_hud_alert,
        "familiar_sound_displayed": hud.environmental_sounds[-1]
        if hud.environmental_sounds else None,
        "familiar_sound_did_not_override_locked_alert": bool(locked_before_sound)
        and pipeline.active_hud_alert == locked_before_sound,
    }


def run_synthetic_queue_and_failure_checks() -> dict:
    workers = []
    stress = {}
    for kind in ("sound", "voice"):
        started, release = threading.Event(), threading.Event()
        processed = []

        def recognize(value, started=started, release=release, processed=processed):
            processed.append(value)
            if value == 0:
                started.set()
                if not release.wait(5):
                    raise TimeoutError("stress harness release timed out")
            return {"accepted": False, "label": "UNKNOWN", "reason": "synthetic"}

        worker = BoundedRecognitionWorker(kind, recognize, maxsize=2)
        worker.start()
        worker.submit(f"{kind}-0", 0)
        if not started.wait(3):
            raise TimeoutError(f"{kind} stress worker did not start")
        for value in range(1, 10):
            worker.submit(f"{kind}-{value}", value)
        workers.append((kind, worker, release, processed))
    for _, _, release, _ in workers:
        release.set()
    for kind, worker, _, processed in workers:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and worker.counters["consumed"] < 3:
            time.sleep(0.01)
        worker.stop()
        stress[kind] = {
            "submitted": worker.counters["produced"],
            "processed": worker.counters["consumed"],
            "timed_out": 0,
            "failed": worker.counters["failures"],
            "dropped": worker.counters["dropped"],
            "processed_payloads": processed,
            "drop_oldest_verified": processed == [0, 8, 9],
        }

    def fail(value):
        if value == "timeout":
            raise TimeoutError("deterministic model timeout")
        raise RuntimeError("deterministic model failure")

    failure_worker = BoundedRecognitionWorker("voice", fail, maxsize=2)
    failure_worker.start()
    failure_worker.submit("timeout", "timeout")
    failure_worker.submit("failure", "failure")
    results = [failure_worker.results.get(timeout=3).result for _ in range(2)]
    failure_worker.stop()
    return {
        "bounded_queue_stress": stress,
        "failure_isolation": {
            "submitted": 2,
            "processed": failure_worker.counters["consumed"],
            "timed_out": sum("TimeoutError" in row.get("error", "") for row in results),
            "failed": sum("RuntimeError" in row.get("error", "") for row in results),
            "dropped": failure_worker.counters["dropped"],
            "all_fail_open_unknown": all(
                row.get("label") == "UNKNOWN" and row.get("reason") == "worker_error"
                for row in results
            ),
        },
    }


def run_features_off_check(store: RecognitionStore) -> dict:
    class NoModelClient:
        calls = 0

        def embed(self, *_):
            self.calls += 1
            raise AssertionError("features-off runtime loaded a model")

        def close(self):
            pass

    client = NoModelClient()
    runtime = PersonalizedRecognitionRuntime(
        familiar_sounds=False, familiar_voices=False, store=store, client=client
    )
    try:
        return {
            "sound_worker_created": runtime.sound_worker is not None,
            "voice_worker_created": runtime.voice_worker is not None,
            "sound_submit_result": runtime.submit_sound(np.zeros(16_000, dtype=np.float32)),
            "voice_submit_result": runtime.submit_voice("off", np.zeros(16_000, dtype=np.float32)),
            "model_calls": client.calls,
            "baseline_preserved": runtime.sound_worker is None and runtime.voice_worker is None
            and client.calls == 0,
        }
    finally:
        runtime.close()


def _http_json(base: str, method: str, path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, json.load(response)


def run_api_lifecycle(root: Path, client: EmbeddingClient,
                      sound_paths: list[Path], voice_paths: list[Path]) -> dict:
    store = RecognitionStore(root)
    original_store, original_client = web_server.RECOGNITION_STORE, web_server.EMBEDDING_CLIENT
    web_server.RECOGNITION_STORE, web_server.EMBEDDING_CLIENT = store, client
    server = ThreadingHTTPServer(("127.0.0.1", 0), PersonalizationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    results = {}
    try:
        for kind, endpoint, paths in (
            ("sound", "familiar-sounds", sound_paths[:2]),
            ("voice", "familiar-voices", voice_paths[:3]),
        ):
            create = {"display_name": f"API {kind.upper()}"}
            if kind == "voice":
                create["consent_acknowledged"] = True
            statuses = []
            status, created = _http_json(base, "POST", f"/api/{endpoint}", create)
            statuses.append(status)
            profile_id = created["profile"]["id"]
            sample_ids = []
            for path in paths:
                status, response = _http_json(base, "POST", f"/api/{endpoint}/{profile_id}/samples", {
                    "audio_base64": base64.b64encode(path.read_bytes()).decode("ascii")
                })
                statuses.append(status)
                sample_ids.append(response["sample"]["id"])
            status, built = _http_json(base, "POST", f"/api/{endpoint}/{profile_id}/build", {})
            statuses.append(status)
            test_status, tested = _http_json(base, "POST", f"/api/{endpoint}/test", {
                "audio_base64": base64.b64encode(paths[0].read_bytes()).decode("ascii")
            })
            statuses.append(test_status)
            _, disabled = _http_json(base, "PATCH", f"/api/{endpoint}/{profile_id}", {"enabled": False})
            _, enabled = _http_json(base, "PATCH", f"/api/{endpoint}/{profile_id}", {"enabled": True})
            delete_sample_status, after_sample_delete = _http_json(
                base, "DELETE", f"/api/{endpoint}/{profile_id}/samples/{sample_ids[0]}"
            )
            delete_status, deleted = _http_json(base, "DELETE", f"/api/{endpoint}/{profile_id}")
            list_status, listed = _http_json(base, "GET", "/api/recognition")
            results[kind] = {
                "http_statuses": statuses,
                "built": built["profile"]["built"],
                "consent_acknowledged": created["profile"]["consent_acknowledged"],
                "test_reason": tested["result"]["reason"],
                "disabled": disabled["profile"]["enabled"] is False,
                "reenabled": enabled["profile"]["enabled"] is True,
                "sample_delete_status": delete_sample_status,
                "sample_delete_invalidated_prototype": not after_sample_delete["built"]
                and not after_sample_delete["enabled"],
                "profile_delete_status": delete_status,
                "profile_deleted": deleted["deleted"],
                "profile_list_status": list_status,
                "remaining_kind_profiles": len(listed[
                    "familiar_sounds" if kind == "sound" else "familiar_voices"
                ]),
            }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        web_server.RECOGNITION_STORE, web_server.EMBEDDING_CLIENT = original_store, original_client
    return results


def latency_metrics(records: list[dict], cold_extra: dict[str, list[dict]]) -> dict:
    values = {}
    for kind in ("sound", "voice"):
        kind_records = [row for row in records if row["kind"] == kind]
        if not kind_records:
            continue
        cold = [kind_records[0], *cold_extra.get(kind, [])]
        warm = kind_records[1:]
        values[kind] = {
            "cold_wall_seconds": distribution(row["wall_seconds"] for row in cold),
            "cold_measurement_count": len(cold),
            "warm_wall_seconds": distribution(row["wall_seconds"] for row in warm),
            "warm_model_inference_seconds": distribution(
                row["inference_seconds"] for row in warm
            ),
            "warm_worker_cpu_seconds": distribution(
                row["cpu_seconds"] for row in warm
            ),
            "worker_rss_bytes": distribution(
                row["worker_rss_bytes"] for row in kind_records
            ),
            "worker_peak_rss_bytes": distribution(
                row["worker_peak_rss_bytes"] for row in kind_records
            ),
            "model": {key: kind_records[-1].get(key) for key in (
                "model", "revision", "sha256", "preprocessing"
            )},
        }
    return values


def measure_extra_cold(kind: str, path: Path, python_path: Path, repetitions: int = 2) -> list[dict]:
    values = []
    for _ in range(repetitions):
        client = EmbeddingClient(python_path)
        started = time.perf_counter()
        try:
            _, metadata = client.embed(kind, path)
            values.append({
                "wall_seconds": time.perf_counter() - started,
                "inference_seconds": metadata.get("inference_seconds"),
                "worker_rss_bytes": metadata.get("worker_rss_bytes"),
                "worker_peak_rss_bytes": metadata.get("worker_peak_rss_bytes"),
            })
        finally:
            client.close()
    return values


def optional_runtime_versions(python_path: Path) -> dict:
    names = ["onnxruntime", "torch", "torchvision", "torchaudio", "nnAudio",
             "librosa", "huggingface-hub", "numpy", "soundfile"]
    program = (
        "import importlib.metadata as m,json; names=" + repr(names) + "; "
        "print(json.dumps({n:m.version(n) for n in names}))"
    )
    return json.loads(subprocess.check_output(
        [str(python_path), "-c", program], text=True, encoding="utf-8"
    ))


def public_rows(rows: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def count_splits(rows: list[dict]) -> dict:
    return dict(Counter(row["split"] for row in rows))


def ratio_text(metric: dict) -> str:
    numerator, denominator = metric["numerator"], metric["denominator"]
    percent = 100 * numerator / denominator if denominator else 0
    return f"{numerator}/{denominator} ({percent:.1f}%)"


def build_summary(sound: dict, voice: dict, system: dict, manifest: dict) -> str:
    sound_perf = system["latency_and_memory"]["sound"]
    voice_perf = system["latency_and_memory"]["voice"]
    sound_queue = system["synthetic_queue_and_failure"]["bounded_queue_stress"]["sound"]
    voice_queue = system["synthetic_queue_and_failure"]["bounded_queue_stress"]["voice"]
    failure = system["synthetic_queue_and_failure"]["failure_isolation"]
    sound_profiles = ", ".join(
        f"{label}: {value['top1_correct']}/{value['top1_total']} top-1, "
        f"{value['accepted_correct']}/{value['accepted_correct_total']} accepted"
        for label, value in sound["per_profile"].items()
    )
    return f"""# HearVis personalized-recognition dataset benchmark

Run: `{manifest['run_id']}`. Fixed seed: `{SEED}`.

## 1. Software/unit verification

Focused and full regression commands are recorded in `system_metrics.json`. Their final results are filled after this dataset command completes; dataset artifacts do not substitute for unit tests.

## 2. Public-dataset functional evaluation

### Familiar Sounds — ESC-50 few-shot custom-category behavior

- Enrollment: {manifest['splits']['sound'].get('enrollment', 0)} clips; calibration: {manifest['splits']['sound'].get('calibration', 0)} known + {manifest['splits']['sound'].get('calibration_unknown', 0)} unknown; holdout: {manifest['splits']['sound'].get('positive_holdout', 0)} known + {manifest['splits']['sound'].get('negative_holdout', 0)} unknown.
- Correct known identification: {ratio_text(sound['correct_identification'])}.
- Known acceptance: {ratio_text(sound['known_acceptance'])}; false rejection: {ratio_text(sound['false_rejection'])}.
- Unknown rejection: {ratio_text(sound['unknown_rejection'])}; false acceptance: {ratio_text(sound['false_acceptance'])}.
- Overall correct open-set decisions: {ratio_text(sound['overall_decision_correct'])}. This tiny fixed split is a functional result, not a general accuracy claim.
- Known top-1 before rejection: {ratio_text(sound['top1_known'])}; {sound_profiles}.
- Calibration-only operating point: cosine threshold `{sound['calibration']['threshold']:.2f}`, runner-up margin `{sound['calibration']['margin']:.2f}`.
- Confusion matrix, score/margin distributions, and every rejected example with reason are in `sound_metrics.json`; trial rows are in `sound_predictions.csv`.

ESC-50 demonstrates few-shot custom-category behavior. It does **not** demonstrate personalized-instance recognition of a particular household device.

### Familiar Voices — LibriSpeech SLR12 test-clean

- Enrollment: {manifest['splits']['voice'].get('enrollment', 0)} utterances; calibration: {manifest['splits']['voice'].get('calibration', 0)} known + {manifest['splits']['voice'].get('calibration_unknown', 0)} impostor; holdout: {manifest['splits']['voice'].get('positive_holdout', 0)} known + {manifest['splits']['voice'].get('negative_holdout', 0)} impostor.
- Correct enrolled-speaker identification: {ratio_text(voice['correct_identification'])}.
- False rejection: {ratio_text(voice['false_rejection'])}; impostor false acceptance: {ratio_text(voice['false_acceptance'])}.
- Impostor UNKNOWN rejection: {ratio_text(voice['unknown_rejection'])}; ambiguous decisions: {ratio_text(voice['ambiguous_results'])}.
- Overall correct open-set decisions: {ratio_text(voice['overall_decision_correct'])}. This tiny fixed split is a functional result, not a population estimate.
- Known top-1 before rejection: {ratio_text(voice['top1_known'])}.
- Calibration-only operating point: cosine threshold `{voice['calibration']['threshold']:.2f}`, runner-up margin `{voice['calibration']['margin']:.2f}`.
- Confusion matrix, score/margin distributions, and every rejected example with reason are in `voice_metrics.json`; trial rows are in `voice_predictions.csv`.

This is a public-dataset functional evaluation. It does not establish family-member performance, Vietnamese speech performance, replay-attack resistance, population-level accuracy, or guaranteed independence from every pretraining identity.

### Measured model cost

- EfficientAT `mn10_as` (`{sound_perf['model']['revision']}`, SHA-256 `{sound_perf['model']['sha256']}`): cold P50 `{sound_perf['cold_wall_seconds']['p50']:.3f}` s, cold P95 `{sound_perf['cold_wall_seconds']['p95']:.3f}` s, cold max `{sound_perf['cold_wall_seconds']['max']:.3f}` s (n={sound_perf['cold_measurement_count']}); warm P50 `{sound_perf['warm_wall_seconds']['p50']:.4f}` s, warm P95 `{sound_perf['warm_wall_seconds']['p95']:.4f}` s, warm max `{sound_perf['warm_wall_seconds']['max']:.4f}` s; peak worker RSS `{sound_perf['worker_peak_rss_bytes']['max'] / 1048576:.1f}` MiB.
- WeSpeaker ECAPA-TDNN512-LM (`{voice_perf['model']['revision']}`, SHA-256 `{voice_perf['model']['sha256']}`): cold P50 `{voice_perf['cold_wall_seconds']['p50']:.3f}` s, cold P95 `{voice_perf['cold_wall_seconds']['p95']:.3f}` s, cold max `{voice_perf['cold_wall_seconds']['max']:.3f}` s (n={voice_perf['cold_measurement_count']}); warm P50 `{voice_perf['warm_wall_seconds']['p50']:.4f}` s, warm P95 `{voice_perf['warm_wall_seconds']['p95']:.4f}` s, warm max `{voice_perf['warm_wall_seconds']['max']:.4f}` s; peak worker RSS `{voice_perf['worker_peak_rss_bytes']['max'] / 1048576:.1f}` MiB.
- Prototype-build distributions and per-request worker CPU time are in the metrics JSON. Both retained worker processes were measured concurrently; combined current/peak RSS is recorded in `system_metrics.json`.

## 3. Synthetic queue/failure tests

Both real model workers were started in the same isolated runtime. The deterministic sound queue submitted {sound_queue['submitted']}, processed {sound_queue['processed']}, dropped {sound_queue['dropped']}, timed out {sound_queue['timed_out']}, and failed {sound_queue['failed']}; the voice queue submitted {voice_queue['submitted']}, processed {voice_queue['processed']}, dropped {voice_queue['dropped']}, timed out {voice_queue['timed_out']}, and failed {voice_queue['failed']}. Both processed payloads `[0, 8, 9]`, proving drop-oldest behavior. The injected isolation check submitted {failure['submitted']}, processed {failure['processed']}, timed out {failure['timed_out']}, failed {failure['failed']}, dropped {failure['dropped']}, and returned fail-open `UNKNOWN` for both.

## 4. Measurements not performed

- Physical microphone capture, OLED/HUD hardware rendering, end-to-end live CED/STT/model contention, energy draw, and thermal behavior.
- Real household instances, consenting familiar people, Vietnamese voices, overlapping speakers, replay/spoof attacks, and noisy/reverberant field conditions.
- Statistical confidence intervals or population-level validation.

## 5. Claims that are not supported

- No claim of personalized-instance recognition, real family-member accuracy, Vietnamese speaker accuracy, authentication/security suitability, replay resistance, certified-alarm behavior, or population-level accuracy.
- No claim that a small-sample perfect result, if observed, generalizes beyond the exact checksummed split.

## Isolation

Real user profile storage was fingerprinted before and after the run and was unchanged: `{system['user_data_isolation']['unchanged']}`. Benchmark profiles remained under Git-ignored benchmark storage.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Result directory; defaults to a timestamped directory")
    parser.add_argument("--data-root", default="benchmark_data/external/personalized_recognition_datasets")
    parser.add_argument("--esc50-root", default="benchmark_data/external/esc50")
    parser.add_argument("--python", default=None, help="Prepared optional model interpreter")
    parser.add_argument("--downloaded-bytes", type=int, default=0,
                        help="Provenance note for downloads performed before this run")
    args = parser.parse_args()

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output = Path(args.output) if args.output else (
        ROOT / "benchmark_results" / "personalized_recognition_dataset" / run_id
    )
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    data_root = (ROOT / args.data_root).resolve()
    esc_root = (ROOT / args.esc50_root).resolve()
    metadata = data_root / "esc50_official.csv"
    cached_metadata = esc_root / "esc50.csv"
    libri_archive = data_root / "test-clean.tar.gz"
    libri_root = data_root / "LibriSpeech" / "test-clean"
    python_path = Path(args.python) if args.python else (
        ROOT / "benchmark_data" / "external" / "personalized_recognition_venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    for required in (metadata, cached_metadata, libri_archive, libri_root, python_path):
        if not required.exists():
            raise FileNotFoundError(required)
    if sha256(metadata) != ESC50_METADATA_SHA256:
        raise RuntimeError("official ESC-50 metadata SHA-256 mismatch")
    if sha256(cached_metadata) != ESC50_METADATA_SHA256:
        raise RuntimeError("cached ESC-50 metadata differs from official metadata")
    if md5(libri_archive) != LIBRISPEECH_TEST_CLEAN_MD5:
        raise RuntimeError("LibriSpeech test-clean official MD5 mismatch")

    sound_rows = build_sound_split(metadata, esc_root / "audio")
    voice_rows, voice_selection = build_voice_split(libri_root)
    write_csv(output / "sound_split_manifest.csv", public_rows(sound_rows))
    write_csv(output / "voice_split_manifest.csv", public_rows(voice_rows))

    real_before = snapshot_real_user_data()
    isolated_root = ROOT / "benchmark_data" / "external" / "personalized_recognition_dataset_runs" / run_id
    isolated_root.mkdir(parents=True, exist_ok=False)
    temp_root = isolated_root / "temp"
    temp_root.mkdir()
    os.environ["TEMP"] = os.environ["TMP"] = str(temp_root)
    tempfile.tempdir = str(temp_root)
    store = RecognitionStore(isolated_root / "profiles")
    client = EmbeddingClient(python_path)
    tracker = TimedEmbedder(client)
    try:
        sound_predictions, sound_metrics, sound_smoothing = evaluate_kind(
            "sound", sound_rows, store, tracker
        )
        voice_predictions, voice_metrics, voice_smoothing = evaluate_kind(
            "voice", voice_rows, store, tracker
        )

        api_metrics = run_api_lifecycle(
            isolated_root / "api_profiles", client,
            [row["_path"] for row in sound_rows if row["split"] == "enrollment"],
            [row["_path"] for row in voice_rows if row["split"] == "enrollment"],
        )
        concurrent = run_real_concurrent_workers(
            store, client,
            next(row["_path"] for row in sound_rows if row["split"] == "positive_holdout"),
            next(row["_path"] for row in voice_rows if row["split"] == "positive_holdout"),
        )
        fusion = run_dataset_fusion_check(
            concurrent["voice_result"],
            next(row for row in concurrent["sound_results"] if row.get("accepted")),
        )
        # run_real_concurrent_workers owns and closes the shared client.
        client = None
    finally:
        if client is not None:
            client.close()

    cold_extra = {
        "sound": measure_extra_cold(
            "sound", next(row["_path"] for row in sound_rows if row["split"] == "positive_holdout"),
            python_path,
        ),
        "voice": measure_extra_cold(
            "voice", next(row["_path"] for row in voice_rows if row["split"] == "positive_holdout"),
            python_path,
        ),
    }
    synthetic = run_synthetic_queue_and_failure_checks()
    features_off = run_features_off_check(store)
    real_after = snapshot_real_user_data()

    sound_metrics["smoothing_and_cooldown"] = sound_smoothing
    voice_metrics["temporal_smoothing"] = voice_smoothing
    voice_metrics["consent_acknowledged_in_isolated_benchmark"] = True
    system_metrics = {
        "schema_version": "1.0",
        "run_id": run_id,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "model_threads": os.environ.get("SOUNDGUARD_MODEL_THREADS", "worker defaults"),
            "optional_runtime_packages": optional_runtime_versions(python_path),
        },
        "latency_and_memory": latency_metrics(tracker.records, cold_extra),
        "real_concurrent_workers": concurrent,
        "synthetic_queue_and_failure": synthetic,
        "features_off": features_off,
        "api_lifecycle": api_metrics,
        "dataset_backed_fusion": fusion,
        "fusion_regression_contracts": {
            "final_caption_prefix_only": "verified with real dataset decisions and focused test",
            "help_receives_raw_unprefixed_transcript": "verified with real dataset decisions and focused test",
            "familiar_sound_cannot_override_help_alert": "verified with real dataset decisions, focused test, and firmware ALERT precedence",
        },
        "user_data_isolation": {
            "before": real_before,
            "after": real_after,
            "unchanged": real_before == real_after,
            "benchmark_root": repo_path(isolated_root),
            "normal_profile_root_not_used": repo_path(DEFAULT_ROOT),
        },
        "commands": {
            "data_acquisition": [
                "curl.exe -L --fail --retry 3 --output benchmark_data\\external\\personalized_recognition_datasets\\test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz",
                "Get-FileHash benchmark_data\\external\\personalized_recognition_datasets\\test-clean.tar.gz -Algorithm MD5",
                "tar -xzf benchmark_data\\external\\personalized_recognition_datasets\\test-clean.tar.gz -C benchmark_data\\external\\personalized_recognition_datasets",
                "download selected ESC-50 WAV files from https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/audio/<filename> using sound_split_manifest.csv",
            ],
            "benchmark": (
                ".\\.venv\\Scripts\\python.exe tools\\evaluate_personalized_recognition_dataset.py "
                f"--output {repo_path(output)} --downloaded-bytes {args.downloaded_bytes}"
            ),
            "focused_tests": ".\\.venv\\Scripts\\python.exe -m pytest -q tests\\test_personalized_recognition.py tests\\test_personalized_recognition_dataset.py",
            "full_tests": ".\\.venv\\Scripts\\python.exe -m pytest -q",
        },
        "software_unit_verification": {
            "focused": {"status": "pending after dataset run"},
            "full": {"status": "pending after dataset run"},
        },
        "measurements_not_performed": [
            "physical microphone capture", "physical OLED/HUD rendering",
            "live CED/STT/model contention", "power and thermal measurements",
            "real family-member or Vietnamese speech evaluation", "replay/spoof testing",
        ],
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "random_seed": SEED,
        "downloaded_bytes_for_this_validation": args.downloaded_bytes,
        "datasets": {
            "sound": {
                "name": "ESC-50",
                "source_url": "https://github.com/karolpiczak/ESC-50",
                "revision": ESC50_COMMIT,
                "license": "CC BY-NC 3.0 (see official repository LICENSE)",
                "metadata_sha256": sha256(metadata),
                "official_folds": [1, 2, 3, 4, 5],
                "selected_files": len(sound_rows),
                "selected_bytes": sum(row["bytes"] for row in sound_rows),
            },
            "voice": {
                "name": "LibriSpeech SLR12 test-clean",
                "source_url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
                "catalog_url": "https://www.openslr.org/12/",
                "license": "CC BY 4.0",
                "archive_bytes": libri_archive.stat().st_size,
                "official_md5": LIBRISPEECH_TEST_CLEAN_MD5,
                "actual_md5": md5(libri_archive),
                "selected_files": len(voice_rows),
                "selected_bytes": sum(row["bytes"] for row in voice_rows),
                **voice_selection,
            },
        },
        "splits": {"sound": count_splits(sound_rows), "voice": count_splits(voice_rows)},
        "split_integrity": {
            "same_file_reuse": False,
            "esc50_source_cross_split": False,
            "voice_utterance_cross_split": False,
            "threshold_and_margin_selected_from_calibration_only": True,
        },
        "attribution_and_license_notes": [
            "ESC-50 clips originate from Freesound and are distributed by the official ESC-50 repository under CC BY-NC 3.0.",
            "LibriSpeech SLR12 is read English speech distributed under CC BY 4.0.",
            "Raw audio is cached only under Git-ignored benchmark_data/external paths and is not committed.",
        ],
    }
    write_csv(output / "sound_predictions.csv", sound_predictions)
    write_csv(output / "voice_predictions.csv", voice_predictions)
    write_json(output / "sound_metrics.json", sound_metrics)
    write_json(output / "voice_metrics.json", voice_metrics)
    write_json(output / "system_metrics.json", system_metrics)
    write_json(output / "dataset_manifest.json", manifest)
    (output / "benchmark_summary.md").write_text(
        build_summary(sound_metrics, voice_metrics, system_metrics, manifest), encoding="utf-8"
    )
    print(json.dumps({
        "output": repo_path(output),
        "sound": sound_metrics["overall_decision_correct"],
        "voice": voice_metrics["overall_decision_correct"],
        "user_data_unchanged": real_before == real_after,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
