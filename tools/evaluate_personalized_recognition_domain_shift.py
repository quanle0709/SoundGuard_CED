"""Independent DCASE/VIVOS domain-shift validation for personalized recognition.

The evaluator never reads the normal enrollment store. It consumes locally cached,
official archives/extractions, selects source-disjoint development and holdout data,
and does not embed holdout audio until a development-only method is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from personalization.recognition import (
    DEFAULT_MARGINS,
    DEFAULT_THRESHOLDS,
    EmbeddingClient,
    RecognitionError,
    RecognitionStore,
    build_centroid,
    l2_normalize,
    open_set_match,
    validate_audio_bytes,
)
from personalization.runtime import PersonalizedRecognitionRuntime
from tools.evaluate_personalized_recognition_dataset import (
    TimedEmbedder,
    classification_metrics,
    distribution,
    md5,
    optional_runtime_versions,
    public_rows,
    ratio_text,
    repo_path,
    run_dataset_fusion_check,
    sha256,
    snapshot_real_user_data,
    write_csv,
    write_json,
)


SEED = 20260913
VIVOS_MD5 = "72972a8b14050f3f11ea7c3debabd7af"
DCASE_DOI = "10.5281/zenodo.2583796"
DCASE_LICENSE = "CC BY-NC 4.0"
TARGETS = {
    "Alarm_bell_ringing": "ALARM / BELL",
    "Blender": "BLENDER",
    "Vacuum_cleaner": "VACUUM CLEANER",
}
CONFUSABLES = {
    "Electric_shaver_toothbrush": "VACUUM CLEANER",
    "Dishes": "ALARM / BELL",
    "Running_water": "BLENDER",
    "Frying": "BLENDER",
}
ALL_NEGATIVES = (*CONFUSABLES, "Speech", "Dog", "Cat")
DEVELOPMENT_SPLITS = {"enrollment", "development_extra", "calibration", "calibration_unknown"}
HOLDOUT_SPLITS = {"positive_holdout", "negative_holdout"}


def _canonical_source(value: object) -> str:
    return str(value or "").replace("\\", "/").strip()


def _scaper_events(path: Path) -> tuple[list[dict], str]:
    """Read the small subset of JAMS/Scaper fields needed for leakage control."""
    document = json.loads(path.read_text(encoding="utf-8"))
    data = []
    for annotation in document.get("annotations", []):
        if str(annotation.get("namespace", "")).lower() == "scaper":
            data.extend(annotation.get("data", []))
    events, background = [], ""
    for item in data:
        value = item.get("value") if isinstance(item.get("value"), dict) else {}
        role = str(value.get("role", ""))
        source = _canonical_source(value.get("source_file"))
        if role == "background":
            background = source
            continue
        if role != "foreground":
            continue
        onset = float(item.get("time", value.get("event_time", 0.0)))
        duration = float(item.get("duration", value.get("event_duration", 0.0)))
        events.append({
            "label": str(value.get("event_label", value.get("label", ""))),
            "source_id": source,
            "onset": onset,
            "offset": onset + duration,
            "snr_db": value.get("snr"),
        })
    return events, background


def discover_dcase_candidates(jams_root: Path, audio_root: Path) -> list[dict]:
    wavs = defaultdict(list)
    for path in audio_root.rglob("*.wav"):
        wavs[path.stem].append(path)
    candidates = []
    for jams_path in sorted(jams_root.rglob("*.jams")):
        matches = wavs.get(jams_path.stem, [])
        if len(matches) != 1:
            continue
        events, background = _scaper_events(jams_path)
        if not events or not background:
            continue
        for index, event in enumerate(events):
            if event["label"] not in {*TARGETS, *ALL_NEGATIVES}:
                continue
            # Crop with a small context pad, but exclude any foreground overlap.
            crop_onset = max(0.0, event["onset"] - 0.10)
            crop_offset = min(10.0, event["offset"] + 0.10)
            overlaps = [
                other for other_index, other in enumerate(events)
                if other_index != index
                and min(crop_offset, other["offset"]) - max(crop_onset, other["onset"]) > 0.0
            ]
            duration = crop_offset - crop_onset
            if overlaps or duration < 0.4 or duration > 15.0 or not event["source_id"]:
                continue
            candidates.append({
                "dataset": "DCASE 2019 Task 4 synthetic strongly annotated",
                "soundscape": matches[0].name,
                "soundscape_path": repo_path(matches[0]),
                "jams_file": jams_path.name,
                "category": event["label"],
                "foreground_source_id": event["source_id"],
                "background_source_id": background,
                "event_onset_seconds": event["onset"],
                "event_offset_seconds": event["offset"],
                "crop_onset_seconds": crop_onset,
                "crop_offset_seconds": crop_offset,
                "snr_db": event["snr_db"],
                "overlapping_foreground_events": 0,
                "_soundscape": matches[0],
            })
    return candidates


def _select_disjoint(candidates: list[dict], count: int, used_fg: set[str],
                     used_bg: set[str], rng: random.Random) -> list[dict]:
    values = list(candidates)
    rng.shuffle(values)
    selected = []
    for row in values:
        foreground, background = row["foreground_source_id"], row["background_source_id"]
        if foreground in used_fg or background in used_bg:
            continue
        selected.append(dict(row))
        used_fg.add(foreground)
        used_bg.add(background)
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"only {len(selected)}/{count} candidates remain after foreground/background disjointness"
    )


def build_sound_split(candidates: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    used_fg: set[str] = set()
    used_bg: set[str] = set()
    rows = []
    by_category = defaultdict(list)
    for row in candidates:
        by_category[row["category"]].append(row)
    for category, label in TARGETS.items():
        for split, count in (
            ("enrollment", 5), ("development_extra", 5),
            ("calibration", 5), ("positive_holdout", 10),
        ):
            chosen = _select_disjoint(by_category[category], count, used_fg, used_bg, rng)
            for row in chosen:
                row.update({"split": split, "expected_label": label, "confusable_for": ""})
                rows.append(row)
    negative_pool = []
    for category in ALL_NEGATIVES:
        for row in by_category[category]:
            value = dict(row)
            value["confusable_for"] = CONFUSABLES.get(category, "")
            negative_pool.append(value)
    for split in ("calibration_unknown", "negative_holdout"):
        chosen = _select_disjoint(negative_pool, 20, used_fg, used_bg, rng)
        for row in chosen:
            row.update({"split": split, "expected_label": "UNKNOWN"})
            rows.append(row)
    if len({row["foreground_source_id"] for row in rows}) != len(rows):
        raise AssertionError("a DCASE foreground source crossed benchmark splits")
    if len({row["background_source_id"] for row in rows}) != len(rows):
        raise AssertionError("a DCASE background source crossed benchmark splits")
    return rows


def materialize_sound_crops(rows: list[dict], crop_root: Path, splits: set[str]) -> None:
    crop_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(row for row in rows if row["split"] in splits):
        if row.get("_path") is not None:
            continue
        samples, sample_rate = sf.read(row["_soundscape"], dtype="float32", always_2d=True)
        start = max(0, round(float(row["crop_onset_seconds"]) * sample_rate))
        stop = min(len(samples), round(float(row["crop_offset_seconds"]) * sample_rate))
        crop = samples[start:stop]
        token = hashlib.sha256(
            f"{row['soundscape']}:{start}:{stop}".encode("utf-8")
        ).hexdigest()[:16]
        path = crop_root / f"{row['split']}_{index:03d}_{token}.wav"
        sf.write(path, crop, sample_rate, subtype="PCM_16")
        _, _, quality = validate_audio_bytes(path.read_bytes(), "sound")
        row.update({
            "file": path.name, "path": repo_path(path), "sha256": sha256(path),
            "bytes": path.stat().st_size, "quality": quality,
            "_path": path,
        })


def build_vivos_split(vivos_root: Path) -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    speakers = {}
    for directory in sorted(path for path in vivos_root.iterdir() if path.is_dir()):
        paths = sorted(directory.glob("*.wav"))
        rng.shuffle(paths)
        valid = []
        for path in paths:
            try:
                validate_audio_bytes(path.read_bytes(), "voice")
                valid.append(path)
            except RecognitionError:
                continue
        if len(valid) >= 20:
            speakers[directory.name] = valid
    speaker_ids = sorted(speakers)
    rng.shuffle(speaker_ids)
    enrolled = speaker_ids[:2]
    calibration_impostors = speaker_ids[2:5]
    holdout_impostors = speaker_ids[5:10]
    if len(enrolled) != 2 or len(holdout_impostors) != 5:
        raise RuntimeError("VIVOS test partition has insufficient eligible speakers")
    display = {
        enrolled[0]: "VIETNAMESE SPEAKER A",
        enrolled[1]: "VIETNAMESE SPEAKER B",
    }
    rows = []

    def add(speaker: str, paths: list[Path], split: str, expected: str) -> None:
        for path in paths:
            rows.append({
                "dataset": "VIVOS test", "split": split,
                "speaker_id": speaker, "utterance_id": path.stem,
                "expected_label": expected, "file": path.name,
                "path": repo_path(path), "sha256": sha256(path),
                "bytes": path.stat().st_size, "_path": path,
            })

    for speaker in enrolled:
        paths = speakers[speaker][:20]
        add(speaker, paths[:5], "enrollment", display[speaker])
        add(speaker, paths[5:10], "calibration", display[speaker])
        add(speaker, paths[10:20], "positive_holdout", display[speaker])
    for speaker in calibration_impostors:
        add(speaker, speakers[speaker][:5], "calibration_unknown", "UNKNOWN")
    for speaker in holdout_impostors:
        add(speaker, speakers[speaker][:5], "negative_holdout", "UNKNOWN")
    files = [row["path"] for row in rows]
    if len(files) != len(set(files)):
        raise AssertionError("a VIVOS utterance was reused across splits")
    return rows, {
        "enrolled_speakers": [
            {"speaker_id": speaker, "display_label": display[speaker]}
            for speaker in enrolled
        ],
        "calibration_impostor_speaker_ids": calibration_impostors,
        "holdout_impostor_speaker_ids": holdout_impostors,
    }


def _prototype(vectors: list[np.ndarray], method: str) -> np.ndarray:
    normal = np.stack([l2_normalize(vector) for vector in vectors])
    if method.startswith("medoid"):
        similarities = normal @ normal.T
        return normal[int(np.argmax(similarities.mean(axis=1)))]
    if method.startswith("trimmed") and len(normal) >= 4:
        similarities = normal @ normal.T
        keep = np.argsort(similarities.mean(axis=1))[1:]
        normal = normal[keep]
    return build_centroid(list(normal))


def _profiles_for(rows: list[dict], vectors: dict[str, np.ndarray], method: str) -> list[dict]:
    use_extra = method.endswith("10")
    allowed = {"enrollment", "development_extra"} if use_extra else {"enrollment"}
    profiles = []
    for label in sorted({row["expected_label"] for row in rows if row["split"] == "enrollment"}):
        items = [row for row in rows if row["expected_label"] == label and row["split"] in allowed]
        expected_count = 10 if use_extra else 5
        if len(items) != expected_count:
            raise AssertionError(f"{label} requires {expected_count} development samples")
        profiles.append({
            "id": label, "display_name": label, "enabled": True, "priority": 3,
            "threshold": 0.0, "margin": 0.0,
            "embedding": _prototype([vectors[row["path"]] for row in items], method),
        })
    return profiles


def _raw_decisions(rows: list[dict], vectors: dict[str, np.ndarray], profiles: list[dict],
                   phase: str) -> list[dict]:
    results = []
    for row in rows:
        match = open_set_match(vectors[row["path"]], profiles, threshold=0.0, margin=0.0)
        results.append({
            "phase": phase, "file": row["file"], "path": row["path"],
            "dataset_label": row.get("category", row.get("speaker_id", "")),
            "expected_label": row["expected_label"],
            "top1_label": match.get("display_name", "UNKNOWN"),
            "score": match.get("score"), "runner_up_score": match.get("runner_up_score"),
            "runner_up_margin": match.get("margin"),
            "confusable_for": row.get("confusable_for", ""),
        })
    return results


def _apply_policy(records: list[dict], thresholds: dict[str, float],
                  margins: dict[str, float]) -> list[dict]:
    applied = []
    for record in records:
        value = dict(record)
        label = value["top1_label"]
        threshold = float(thresholds[label])
        margin = float(margins[label])
        accepted = float(value["score"]) >= threshold and float(value["runner_up_margin"]) >= margin
        value.update({
            "accepted": accepted,
            "predicted_label": label if accepted else "UNKNOWN",
            "reason": "accepted" if accepted else (
                "below_threshold" if float(value["score"]) < threshold else "ambiguous_margin"
            ),
            "threshold": threshold, "margin_required": margin,
        })
        applied.append(value)
    return applied


def _fixed_policy(kind: str, labels: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {label: DEFAULT_THRESHOLDS[kind] for label in labels},
        {label: DEFAULT_MARGINS[kind] for label in labels},
    )


def _calibrate_per_profile(kind: str, raw: list[dict], labels: list[str],
                           baseline: list[dict]) -> dict:
    thresholds, margins, per_profile = {}, {}, {}
    baseline_false_rejects = Counter(
        row["expected_label"] for row in baseline
        if row["expected_label"] != "UNKNOWN" and row["predicted_label"] == "UNKNOWN"
    )
    for label in labels:
        positives = [row for row in raw if row["expected_label"] == label]
        top1_negatives = [
            row for row in raw if row["expected_label"] != label and row["top1_label"] == label
        ]
        best = None
        allowed_rejects = baseline_false_rejects[label] + max(1, round(0.10 * len(positives)))
        for threshold_i in range(35, 96):
            for margin_i in range(0, 31):
                threshold, margin = threshold_i / 100, margin_i / 100
                accepted_positive = sum(
                    row["top1_label"] == label and float(row["score"]) >= threshold
                    and float(row["runner_up_margin"]) >= margin for row in positives
                )
                rejects = len(positives) - accepted_positive
                false_accepts = sum(
                    float(row["score"]) >= threshold
                    and float(row["runner_up_margin"]) >= margin for row in top1_negatives
                )
                if rejects > allowed_rejects:
                    continue
                rank = (-false_accepts, accepted_positive, threshold + margin)
                if best is None or rank > best[0]:
                    best = (rank, threshold, margin, accepted_positive, false_accepts)
        if best is None:
            threshold, margin = DEFAULT_THRESHOLDS[kind], DEFAULT_MARGINS[kind]
            accepted_positive, false_accepts = 0, len(top1_negatives)
        else:
            _, threshold, margin, accepted_positive, false_accepts = best
        thresholds[label], margins[label] = threshold, margin
        per_profile[label] = {
            "threshold": threshold, "margin": margin,
            "accepted_positive": accepted_positive, "positive_total": len(positives),
            "false_accepts": false_accepts, "top1_negative_total": len(top1_negatives),
        }
    return {"thresholds": thresholds, "margins": margins, "per_profile": per_profile}


def embed_rows(rows: list[dict], tracker: TimedEmbedder, kind: str,
               splits: set[str]) -> dict[str, np.ndarray]:
    vectors = {}
    for row in rows:
        if row["split"] not in splits:
            continue
        validate_audio_bytes(row["_path"].read_bytes(), kind)
        tracker.stage = f"{kind}_{row['split']}"
        vector, _ = tracker.embed(kind, row["_path"])
        vectors[row["path"]] = vector
    return vectors


def create_production_profiles(kind: str, rows: list[dict], store: RecognitionStore,
                               tracker: TimedEmbedder) -> dict:
    profile_ids, build_seconds = {}, {}
    for label in sorted({row["expected_label"] for row in rows if row["split"] == "enrollment"}):
        kwargs = {"consent_acknowledged": True} if kind == "voice" else {}
        profile = store.create(kind, label, **kwargs)
        profile_ids[label] = profile["id"]
        enrollment = [
            row for row in rows
            if row["split"] == "enrollment" and row["expected_label"] == label
        ]
        if len(enrollment) != 5:
            raise AssertionError(f"{label} does not have exactly five enrollment samples")
        for row in enrollment:
            store.add_sample(kind, profile["id"], row["_path"].read_bytes())
        tracker.stage = f"{kind}_production_prototype_build"
        started = time.perf_counter()
        built = store.build(kind, profile["id"], tracker.embed)
        build_seconds[label] = time.perf_counter() - started
        if not built["built"] or not built["enabled"]:
            raise AssertionError("production profile build failed")
    persisted = RecognitionStore(store.root).matcher_profiles(kind)
    if len(persisted) != len(profile_ids):
        raise AssertionError("persisted prototype recovery failed")
    return {
        "profile_ids": profile_ids,
        "build_seconds": build_seconds,
        "build_seconds_distribution": distribution(list(build_seconds.values())),
        "persisted_profile_count": len(persisted),
    }


def _metric_value(metrics: dict, name: str) -> int:
    return int(metrics[name]["numerator"])


def calibrate_candidates(kind: str, rows: list[dict], vectors: dict[str, np.ndarray]) -> dict:
    calibration_rows = [
        row for row in rows if row["split"] in {"calibration", "calibration_unknown"}
    ]
    labels = sorted({row["expected_label"] for row in rows if row["split"] == "enrollment"})
    methods = ["mean5"] if kind == "voice" else [
        "mean5", "medoid5", "trimmed5", "mean10", "trimmed10"
    ]
    baseline_profiles = _profiles_for(rows, vectors, "mean5")
    baseline_raw = _raw_decisions(calibration_rows, vectors, baseline_profiles, "calibration")
    fixed_thresholds, fixed_margins = _fixed_policy(kind, labels)
    baseline_predictions = _apply_policy(baseline_raw, fixed_thresholds, fixed_margins)
    baseline_metrics = classification_metrics(baseline_predictions)
    candidates = []
    for method in methods:
        profiles = _profiles_for(rows, vectors, method)
        raw = _raw_decisions(calibration_rows, vectors, profiles, "calibration")
        calibration = _calibrate_per_profile(kind, raw, labels, baseline_predictions)
        predictions = _apply_policy(raw, calibration["thresholds"], calibration["margins"])
        metrics = classification_metrics(predictions)
        known_total = metrics["holdout_counts"]["known"]
        allowed_false_rejects = (
            _metric_value(baseline_metrics, "false_rejection")
            + max(1, round(0.10 * known_total))
        )
        eligible = (
            _metric_value(metrics, "false_acceptance")
            < _metric_value(baseline_metrics, "false_acceptance")
            and _metric_value(metrics, "false_rejection") <= allowed_false_rejects
            and _metric_value(metrics, "overall_decision_correct")
            > _metric_value(baseline_metrics, "overall_decision_correct")
        )
        candidates.append({
            "method": method, "calibration": calibration,
            "metrics": metrics, "predictions": predictions, "eligible": eligible,
            "allowed_false_rejects": allowed_false_rejects,
        })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(
        eligible,
        key=lambda item: (
            _metric_value(item["metrics"], "overall_decision_correct"),
            -_metric_value(item["metrics"], "false_acceptance"),
            -_metric_value(item["metrics"], "false_rejection"),
            item["method"] == "mean5",
        ),
        default=None,
    )
    return {
        "baseline": {
            "method": "mean5", "thresholds": fixed_thresholds,
            "margins": fixed_margins, "metrics": baseline_metrics,
            "predictions": baseline_predictions,
        },
        "candidates": candidates,
        "selected": selected,
        "selection_rule": (
            "development-only: reduce false acceptance, improve total correct, "
            "and cap false-rejection increase at 10 percentage points (minimum one sample)"
        ),
    }


def evaluate_holdout(kind: str, rows: list[dict], vectors: dict[str, np.ndarray],
                     development: dict) -> dict:
    holdout_rows = [row for row in rows if row["split"] in HOLDOUT_SPLITS]
    labels = sorted({row["expected_label"] for row in rows if row["split"] == "enrollment"})
    baseline_profiles = _profiles_for(rows, vectors, "mean5")
    baseline_raw = _raw_decisions(holdout_rows, vectors, baseline_profiles, "final_holdout")
    fixed_thresholds, fixed_margins = _fixed_policy(kind, labels)
    baseline_predictions = _apply_policy(baseline_raw, fixed_thresholds, fixed_margins)
    baseline_metrics = classification_metrics(baseline_predictions)
    selected = development["selected"]
    if selected is None:
        return {
            "baseline_predictions": baseline_predictions,
            "baseline_metrics": baseline_metrics,
            "improved_predictions": [], "improved_metrics": None,
            "selected_development_method": None,
            "algorithm_change_retained": False,
            "retention_reason": "no development candidate met the safety gates",
        }
    improved_profiles = _profiles_for(rows, vectors, selected["method"])
    improved_raw = _raw_decisions(holdout_rows, vectors, improved_profiles, "final_holdout")
    improved_predictions = _apply_policy(
        improved_raw,
        selected["calibration"]["thresholds"],
        selected["calibration"]["margins"],
    )
    improved_metrics = classification_metrics(improved_predictions)
    known_total = baseline_metrics["holdout_counts"]["known"]
    allowed_false_rejects = (
        _metric_value(baseline_metrics, "false_rejection")
        + max(1, round(0.10 * known_total))
    )
    retained = (
        _metric_value(improved_metrics, "false_acceptance")
        < _metric_value(baseline_metrics, "false_acceptance")
        and _metric_value(improved_metrics, "false_rejection") <= allowed_false_rejects
        and _metric_value(improved_metrics, "overall_decision_correct")
        > _metric_value(baseline_metrics, "overall_decision_correct")
    )
    return {
        "baseline_predictions": baseline_predictions,
        "baseline_metrics": baseline_metrics,
        "improved_predictions": improved_predictions,
        "improved_metrics": improved_metrics,
        "selected_development_method": selected,
        "algorithm_change_retained": retained,
        "retention_reason": (
            "untouched holdout improved with lower false acceptance and bounded false rejection"
            if retained else
            "development candidate failed one or more untouched-holdout retention gates"
        ),
        "allowed_false_rejects": allowed_false_rejects,
    }


def tracker_metrics(records: list[dict]) -> dict:
    values = {}
    for kind in ("sound", "voice"):
        subset = [row for row in records if row["kind"] == kind]
        values[kind] = {
            "wall_seconds": distribution(row["wall_seconds"] for row in subset),
            "model_inference_seconds": distribution(row["inference_seconds"] for row in subset),
            "worker_cpu_seconds": distribution(row["cpu_seconds"] for row in subset),
            "worker_rss_bytes": distribution(row["worker_rss_bytes"] for row in subset),
            "worker_peak_rss_bytes": distribution(
                row["worker_peak_rss_bytes"] for row in subset
            ),
            "by_stage": {
                stage: distribution(row["wall_seconds"] for row in subset if row["stage"] == stage)
                for stage in sorted({row["stage"] for row in subset})
            },
            "model": ({
                key: subset[-1].get(key) for key in (
                    "model", "revision", "sha256", "preprocessing"
                )
            } if subset else None),
        }
    return values


def measure_background_preload(store: RecognitionStore, python_path: Path,
                               sound_path: Path, voice_path: Path) -> dict:
    client = EmbeddingClient(python_path)
    runtime = PersonalizedRecognitionRuntime(
        familiar_sounds=True, familiar_voices=True, store=store, client=client,
        sample_rate=16_000, sound_cooldown_seconds=0,
    )
    ready_seconds = {}
    started = time.perf_counter()
    try:
        runtime.start()
        start_return_seconds = time.perf_counter() - started
        initial = client.status()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            states = client.status()
            elapsed = time.perf_counter() - started
            for kind in ("sound", "voice"):
                if states[kind]["status"] == "ready" and kind not in ready_seconds:
                    ready_seconds[kind] = elapsed
            if all(states[kind]["status"] in {"ready", "failed"} for kind in ("sound", "voice")):
                break
            time.sleep(0.02)
        final = client.status()
        post_ready = {}
        for kind, path in (("sound", sound_path), ("voice", voice_path)):
            if final[kind]["status"] != "ready":
                post_ready[kind] = {"performed": False, "reason": final[kind]["error"]}
                continue
            request_started = time.perf_counter()
            _, metadata = client.embed(kind, path)
            post_ready[kind] = {
                "performed": True,
                "wall_seconds": time.perf_counter() - request_started,
                "model_inference_seconds": metadata.get("inference_seconds"),
                "worker_rss_bytes": metadata.get("worker_rss_bytes"),
                "worker_peak_rss_bytes": metadata.get("worker_peak_rss_bytes"),
            }
        return {
            "runtime_start_return_seconds": start_return_seconds,
            "initial_status": initial, "final_status": final,
            "time_to_ready_seconds": ready_seconds,
            "post_ready_first_request": post_ready,
            "worker_process_count": len(client._processes),
            "duplicate_preload_requests_suppressed": (
                not client.preload_async("sound") and not client.preload_async("voice")
            ),
            "runtime_counters": runtime.counters,
        }
    finally:
        runtime.close()


def _previous_cold_metrics() -> dict:
    path = (
        ROOT / "benchmark_results" / "personalized_recognition_dataset"
        / "20260913T191404+0700" / "system_metrics.json"
    )
    if not path.is_file():
        return {"available": False, "path": repo_path(path)}
    value = json.loads(path.read_text(encoding="utf-8"))["latency_and_memory"]
    return {
        "available": True, "path": repo_path(path),
        "sound": value.get("sound", {}).get("cold_wall_seconds"),
        "voice": value.get("voice", {}).get("cold_wall_seconds"),
        "note": "reused unchanged prior run; ESC-50/LibriSpeech were not rerun",
    }


def _split_counts(rows: list[dict]) -> dict:
    return dict(Counter(row["split"] for row in rows))


def _safe_manifest_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            key: value for key, value in row.items()
            if not key.startswith("_") and key != "quality"
        }
        for row in rows
    ]


def _metric_line(metrics: dict) -> str:
    return (
        f"correct {ratio_text(metrics['overall_decision_correct'])}; "
        f"known accepted {ratio_text(metrics['known_acceptance'])}; "
        f"false rejects {ratio_text(metrics['false_rejection'])}; "
        f"unknown rejected {ratio_text(metrics['unknown_rejection'])}; "
        f"false accepts {ratio_text(metrics['false_acceptance'])}; "
        f"misidentified {ratio_text(metrics['misidentification'])}"
    )


def build_summary(manifest: dict, sound: dict, voice: dict, runtime: dict) -> str:
    sound_improved = sound.get("improved_metrics")
    voice_improved = voice.get("improved_metrics")
    sound_improved_text = (
        _metric_line(sound_improved) if sound_improved else "not run; no safe development candidate"
    )
    voice_improved_text = (
        _metric_line(voice_improved) if voice_improved else "not run; no safe development candidate"
    )
    return f"""# HearVis personalized-recognition domain-shift validation

Run `{manifest['run_id']}` used only the official DCASE 2019 Task 4 strongly annotated synthetic subset and the official VIVOS test partition. No ESC-50 or LibriSpeech evaluation was repeated.

## Exact splits

- DCASE sounds: `{manifest['splits']['sound']}`. Every selected event has a distinct original foreground source and background source across enrollment, development, calibration, and final holdout. Holdout audio was embedded once, only after development selection was frozen.
- VIVOS voices: `{manifest['splits']['voice']}`. Two neutral-label enrolled speakers use 5 enrollment, 5 calibration, and 10 final positive utterances each. Three disjoint speakers supply calibration impostors and five other disjoint speakers supply 5 final impostor utterances each.

## DCASE household sounds

- Frozen current method: {_metric_line(sound['baseline_metrics'])}.
- Development-selected method on the same untouched holdout: {sound_improved_text}.
- Retention gate passed: `{sound['algorithm_change_retained']}` -- {sound['retention_reason']}.
- Confusion matrices, per-profile counts, score/margin distributions, raw decisions, calibration grids, crop timestamps, source IDs, and latency are in the sibling JSON/CSV artifacts.

## Vietnamese familiar voices

- Frozen current method: {_metric_line(voice['baseline_metrics'])}.
- Development-selected method: {voice_improved_text}.
- VIVOS is quiet, read Vietnamese speech. This small test does not prove performance for spontaneous family conversation, room or microphone changes, noise, illness, overlapping speakers, or replay attacks.

## Non-blocking model preload

- Runtime start returned in `{runtime['runtime_start_return_seconds']:.4f}` seconds.
- Background time-to-ready: sound `{runtime['time_to_ready_seconds'].get('sound')}` seconds; voice `{runtime['time_to_ready_seconds'].get('voice')}` seconds.
- Final lifecycle states: sound `{runtime['final_status']['sound']['status']}`, voice `{runtime['final_status']['voice']['status']}`; worker processes `{runtime['worker_process_count']}`; duplicate requests suppressed `{runtime['duplicate_preload_requests_suppressed']}`.
- The unchanged prior synchronous cold-start measurements are copied by reference into `runtime_metrics.json`; post-ready first-request latency and worker memory are recorded there. Server startup, capture, CED, STT, HELP, and emergency processing do not wait for these model loads.

## Isolation and limitations

Raw audio, model caches, cropped events, embeddings, and isolated profiles remain under Git-ignored `benchmark_data/external/`. No raw audio or biometric data is included here. Physical microphone/OLED behavior, live CED/STT contention, power, thermals, replay resistance, and real household/family conditions were not measured.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="timestamped result directory")
    parser.add_argument(
        "--dcase-jams-root", required=True,
        help="extracted official DCASE 2019 Scaper JAMS directory",
    )
    parser.add_argument(
        "--dcase-audio-root", required=True,
        help="extracted official DCASE 2019 synthetic soundscape directory",
    )
    parser.add_argument("--dcase-archive", required=True)
    parser.add_argument("--dcase-official-md5", required=True)
    parser.add_argument("--dcase-jams-archive", required=True)
    parser.add_argument("--dcase-jams-official-md5", required=True)
    parser.add_argument(
        "--vivos-archive", default="benchmark_data/external/vivos/vivos.tar.gz"
    )
    parser.add_argument(
        "--vivos-root",
        default="benchmark_data/external/personalized_recognition_domain_shift/vivos/vivos/test/waves",
    )
    parser.add_argument("--python", default=None)
    parser.add_argument("--downloaded-bytes", type=int, default=0)
    args = parser.parse_args()

    def absolute(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    dcase_jams = absolute(args.dcase_jams_root)
    dcase_audio = absolute(args.dcase_audio_root)
    dcase_archive = absolute(args.dcase_archive)
    dcase_jams_archive = absolute(args.dcase_jams_archive)
    vivos_archive = absolute(args.vivos_archive)
    vivos_root = absolute(args.vivos_root)
    python_path = absolute(args.python) if args.python else (
        ROOT / "benchmark_data" / "external" / "personalized_recognition_venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    for path in (
        dcase_jams, dcase_audio, dcase_archive, dcase_jams_archive,
        vivos_archive, vivos_root, python_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_dcase_md5 = md5(dcase_archive)
    if actual_dcase_md5.lower() != args.dcase_official_md5.lower():
        raise RuntimeError("official DCASE archive MD5 mismatch")
    actual_dcase_jams_md5 = md5(dcase_jams_archive)
    if actual_dcase_jams_md5.lower() != args.dcase_jams_official_md5.lower():
        raise RuntimeError("official DCASE JAMS archive MD5 mismatch")
    if md5(vivos_archive) != VIVOS_MD5:
        raise RuntimeError("official VIVOS archive MD5 mismatch")

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output = absolute(args.output) if args.output else (
        ROOT / "benchmark_results" / "personalized_recognition_domain_shift" / run_id
    )
    output.mkdir(parents=True, exist_ok=False)
    isolated_root = (
        ROOT / "benchmark_data" / "external"
        / "personalized_recognition_domain_shift_runs" / run_id
    )
    isolated_root.mkdir(parents=True, exist_ok=False)
    temp_root = isolated_root / "temp"
    temp_root.mkdir()
    os.environ["TEMP"] = os.environ["TMP"] = str(temp_root)
    tempfile.tempdir = str(temp_root)

    candidates = discover_dcase_candidates(dcase_jams, dcase_audio)
    sound_rows = build_sound_split(candidates)
    # Only development material is read before method selection.
    materialize_sound_crops(sound_rows, isolated_root / "sound_crops", DEVELOPMENT_SPLITS)
    voice_rows, voice_selection = build_vivos_split(vivos_root)
    real_before = snapshot_real_user_data()
    store = RecognitionStore(isolated_root / "profiles")
    client = EmbeddingClient(python_path)
    tracker = TimedEmbedder(client)
    try:
        sound_build = create_production_profiles("sound", sound_rows, store, tracker)
        voice_build = create_production_profiles("voice", voice_rows, store, tracker)
        sound_vectors = embed_rows(sound_rows, tracker, "sound", DEVELOPMENT_SPLITS)
        voice_vectors = embed_rows(voice_rows, tracker, "voice", DEVELOPMENT_SPLITS)
        sound_development = calibrate_candidates("sound", sound_rows, sound_vectors)
        voice_development = calibrate_candidates("voice", voice_rows, voice_vectors)

        # The method is now frozen. Materialize/embed untouched final holdout once.
        materialize_sound_crops(sound_rows, isolated_root / "sound_crops", HOLDOUT_SPLITS)
        sound_vectors.update(embed_rows(sound_rows, tracker, "sound", HOLDOUT_SPLITS))
        voice_vectors.update(embed_rows(voice_rows, tracker, "voice", HOLDOUT_SPLITS))
        sound_final = evaluate_holdout("sound", sound_rows, sound_vectors, sound_development)
        voice_final = evaluate_holdout("voice", voice_rows, voice_vectors, voice_development)
    finally:
        client.close()

    # Write manifests only after the untouched holdout has been materialized so every
    # selected crop has its size and checksum recorded.
    write_csv(output / "sound_split_manifest.csv", _safe_manifest_rows(sound_rows))
    write_csv(output / "voice_split_manifest.csv", public_rows(voice_rows))

    sound_accepted = next(
        (row for row in sound_final["baseline_predictions"] if row["accepted"]), None
    )
    voice_accepted = next(
        (row for row in voice_final["baseline_predictions"] if row["accepted"]), None
    )
    fusion = (
        run_dataset_fusion_check(voice_accepted, sound_accepted)
        if sound_accepted and voice_accepted else {
            "performed": False,
            "reason": "no accepted frozen holdout pair was available",
        }
    )
    preload = measure_background_preload(
        store, python_path,
        next(row["_path"] for row in sound_rows if row["split"] == "positive_holdout"),
        next(row["_path"] for row in voice_rows if row["split"] == "positive_holdout"),
    )
    real_after = snapshot_real_user_data()

    for final, development, build in (
        (sound_final, sound_development, sound_build),
        (voice_final, voice_development, voice_build),
    ):
        final["development"] = development
        final["production_profile_build"] = build
    runtime_metrics = {
        "schema_version": "1.0", "run_id": run_id,
        "preload": preload,
        "previous_synchronous_cold_start": _previous_cold_metrics(),
        "embedding_latency_and_memory": tracker_metrics(tracker.records),
        "fusion": fusion,
        "user_data_isolation": {
            "before": real_before, "after": real_after,
            "unchanged": real_before == real_after,
            "isolated_profile_root": repo_path(store.root),
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "optional_runtime_packages": optional_runtime_versions(python_path),
        },
        "queue_drops_during_preload_check": {
            key: value for key, value in preload["runtime_counters"].items()
            if key.endswith("_dropped")
        },
        "physical_tests_not_run": [
            "microphone capture", "OLED/HUD hardware", "live CED/STT CPU contention",
            "power and thermal behavior",
        ],
    }
    manifest = {
        "schema_version": "1.0", "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(), "seed": SEED,
        "downloaded_bytes_for_this_run": args.downloaded_bytes,
        "datasets": {
            "sound": {
                "name": "DCASE 2019 Task 4 synthetic strongly annotated subset",
                "task_url": "https://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments",
                "doi": DCASE_DOI, "license": DCASE_LICENSE,
                "soundscape_archive": {
                    "path": repo_path(dcase_archive),
                    "bytes": dcase_archive.stat().st_size,
                    "official_md5": args.dcase_official_md5.lower(),
                    "actual_md5": actual_dcase_md5,
                },
                "jams_archive": {
                    "path": repo_path(dcase_jams_archive),
                    "bytes": dcase_jams_archive.stat().st_size,
                    "official_md5": args.dcase_jams_official_md5.lower(),
                    "actual_md5": actual_dcase_jams_md5,
                },
                "candidate_clean_events": len(candidates),
                "selected_events": len(sound_rows),
                "selected_crop_bytes": sum(row["bytes"] for row in sound_rows),
            },
            "voice": {
                "name": "VIVOS Vietnamese Speech Corpus for ASR, test partition",
                "record_url": "https://zenodo.org/records/7068130",
                "doi": "10.5281/zenodo.7068130",
                "license": "CC BY-NC-SA 4.0; academic-use notice in corpus README",
                "archive_path": repo_path(vivos_archive),
                "archive_bytes": vivos_archive.stat().st_size,
                "official_md5": VIVOS_MD5, "actual_md5": md5(vivos_archive),
                "corpus_properties": {
                    "train_speakers": 46, "test_speakers": 19,
                    "recording": "quiet environment, high-quality microphone, read sentences",
                },
                "selected_utterances": len(voice_rows),
                "selected_bytes": sum(row["bytes"] for row in voice_rows),
                **voice_selection,
            },
        },
        "splits": {"sound": _split_counts(sound_rows), "voice": _split_counts(voice_rows)},
        "split_integrity": {
            "sound_foreground_source_disjoint": True,
            "sound_background_source_disjoint": True,
            "holdout_from_enrollment_augmentation": False,
            "voice_file_reuse": False,
            "voice_impostor_speakers_disjoint": True,
            "holdout_embedded_after_development_method_frozen": True,
        },
        "privacy": {
            "raw_audio_committed": False, "embeddings_committed": False,
            "real_speaker_names_used": False,
        },
    }

    write_csv(output / "sound_frozen_predictions.csv", sound_final["baseline_predictions"])
    write_csv(output / "sound_improved_predictions.csv", sound_final["improved_predictions"])
    write_csv(output / "voice_frozen_predictions.csv", voice_final["baseline_predictions"])
    write_csv(output / "voice_improved_predictions.csv", voice_final["improved_predictions"])
    write_json(output / "sound_metrics.json", sound_final)
    write_json(output / "voice_metrics.json", voice_final)
    write_json(output / "sound_calibration_results.json", sound_development)
    write_json(output / "voice_calibration_results.json", voice_development)
    write_json(output / "runtime_metrics.json", runtime_metrics)
    write_json(output / "dataset_manifest.json", manifest)
    (output / "reproduction.md").write_text(
        "# Reproduction\n\n"
        "Use the checksummed official archives listed in `dataset_manifest.json`, extract them "
        "under ignored `benchmark_data/external/`, then run:\n\n"
        f"```powershell\n.\\.venv\\Scripts\\python.exe tools\\evaluate_personalized_recognition_domain_shift.py --dcase-jams-root {repo_path(dcase_jams)} --dcase-audio-root {repo_path(dcase_audio)} --dcase-archive {repo_path(dcase_archive)} --dcase-official-md5 {args.dcase_official_md5.lower()} --dcase-jams-archive {repo_path(dcase_jams_archive)} --dcase-jams-official-md5 {args.dcase_jams_official_md5.lower()} --downloaded-bytes {args.downloaded_bytes}\n```\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        build_summary(manifest, sound_final, voice_final, preload), encoding="utf-8"
    )
    print(json.dumps({
        "output": repo_path(output),
        "sound_frozen": sound_final["baseline_metrics"]["overall_decision_correct"],
        "sound_improved": (
            sound_final["improved_metrics"]["overall_decision_correct"]
            if sound_final["improved_metrics"] else None
        ),
        "voice_frozen": voice_final["baseline_metrics"]["overall_decision_correct"],
        "user_data_unchanged": real_before == real_after,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
