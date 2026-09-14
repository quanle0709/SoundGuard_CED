"""Independent UrbanSound8K/VIVOS validation for personalized recognition.

The evaluator never reads the normal enrollment store. It consumes locally cached,
verified archives/extractions, selects fold- and source-disjoint development and
holdout data, and does not embed holdout audio until a development-only method is
frozen.  A prior VIVOS result may be carried forward without re-embedding it.
"""

from __future__ import annotations

import argparse
import csv
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
URBANSOUND8K_MD5 = "9aa69802bbf37fb986f71ec1483a196e"
URBANSOUND8K_DOI = "10.5281/zenodo.1203745"
URBANSOUND8K_LICENSE = "CC BY-NC 3.0"
URBANSOUND8K_CLASS_IDS = {
    "air_conditioner": 0,
    "car_horn": 1,
    "children_playing": 2,
    "dog_bark": 3,
    "drilling": 4,
    "engine_idling": 5,
    "gun_shot": 6,
    "jackhammer": 7,
    "siren": 8,
    "street_music": 9,
}
URBANSOUND8K_CLASSES = set(URBANSOUND8K_CLASS_IDS)
TARGETS = {
    "dog_bark": "DOG BARK",
    "car_horn": "CAR HORN",
    "siren": "SIREN",
}
CONFUSABLES = {
    "engine_idling": "CAR HORN",
    "street_music": "SIREN",
    "children_playing": "SIREN",
    "air_conditioner": "SIREN",
    "drilling": "SIREN",
    "jackhammer": "SIREN",
    "gun_shot": "CAR HORN",
}
ALL_NEGATIVES = tuple(CONFUSABLES)
DEVELOPMENT_SPLITS = {"enrollment", "development_extra", "calibration", "calibration_unknown"}
HOLDOUT_SPLITS = {"positive_holdout", "negative_holdout"}


def load_urbansound8k(metadata_csv: Path, audio_root: Path,
                      expected_rows: int = 8_732) -> tuple[list[dict], dict]:
    """Validate the official inventory and return metadata-linked WAV candidates."""
    with metadata_csv.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = list(csv.DictReader(stream))
    required = {
        "slice_file_name", "fsID", "start", "end", "salience", "fold",
        "classID", "class",
    }
    if len(source_rows) != expected_rows:
        raise RuntimeError(
            f"UrbanSound8K metadata has {len(source_rows)}, expected {expected_rows} rows"
        )
    if not source_rows or not required.issubset(source_rows[0]):
        raise RuntimeError("UrbanSound8K.csv is missing required columns")
    classes = {row["class"] for row in source_rows}
    folds = {int(row["fold"]) for row in source_rows}
    if classes != URBANSOUND8K_CLASSES or folds != set(range(1, 11)):
        raise RuntimeError("UrbanSound8K class or fold inventory does not match v1.0")
    if any(
        int(row["classID"]) != URBANSOUND8K_CLASS_IDS[row["class"]]
        for row in source_rows
    ):
        raise RuntimeError("UrbanSound8K classID mapping does not match v1.0")
    if len({row["slice_file_name"] for row in source_rows}) != len(source_rows):
        raise RuntimeError("UrbanSound8K metadata contains duplicate excerpt filenames")

    candidates, missing = [], []
    for source in source_rows:
        fold = int(source["fold"])
        path = audio_root / f"fold{fold}" / source["slice_file_name"]
        if not path.is_file():
            missing.append(str(path))
        candidates.append({
            "dataset": "UrbanSound8K v1.0",
            "category": source["class"],
            "class_id": int(source["classID"]),
            "fs_id": str(source["fsID"]),
            "fold": fold,
            "salience": int(source["salience"]),
            "source_start_seconds": float(source["start"]),
            "source_end_seconds": float(source["end"]),
            "file": source["slice_file_name"],
            "path": repo_path(path),
            "_path": path,
        })
    if missing:
        raise RuntimeError(
            f"{len(missing)} UrbanSound8K metadata-linked WAV files are missing; first={missing[0]}"
        )
    wav_count = sum(1 for _ in audio_root.rglob("*.wav"))
    if wav_count != expected_rows:
        raise RuntimeError(
            f"UrbanSound8K extraction has {wav_count}, expected {expected_rows} WAV files"
        )
    return candidates, {
        "metadata_rows": len(source_rows),
        "wav_files": wav_count,
        "classes": sorted(classes),
        "folds": sorted(folds),
        "metadata_linked_wavs_missing": 0,
    }


def _sound_quality_status(row: dict, cache: dict[str, tuple[bool, str]]) -> tuple[bool, str]:
    key = row["path"]
    if key not in cache:
        try:
            validate_audio_bytes(row["_path"].read_bytes(), "sound")
            cache[key] = (True, "")
        except RecognitionError as exc:
            cache[key] = (False, str(exc))
    return cache[key]


def _select_disjoint(candidates: list[dict], count: int, used_fsids: set[str],
                     rng: random.Random, *, require_valid: bool = False,
                     quality_cache: dict[str, tuple[bool, str]] | None = None,
                     quality_rejections: dict[str, dict] | None = None) -> list[dict]:
    values = list(candidates)
    rng.shuffle(values)
    values.sort(key=lambda row: (row["salience"] != 1, -(
        row["source_end_seconds"] - row["source_start_seconds"]
    )))
    selected = []
    for row in values:
        source_id = row["fs_id"]
        if source_id in used_fsids:
            continue
        if require_valid:
            if quality_cache is None:
                raise AssertionError("quality-aware selection requires a cache")
            valid, reason = _sound_quality_status(row, quality_cache)
            if not valid:
                if quality_rejections is not None:
                    quality_rejections.setdefault(row["path"], {
                        "file": row["file"], "path": row["path"],
                        "category": row["category"], "fold": row["fold"],
                        "fs_id": row["fs_id"], "reason": reason,
                    })
                continue
        selected.append(dict(row))
        used_fsids.add(source_id)
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"only {len(selected)}/{count} candidates remain after fsID disjointness"
    )


def build_sound_split(candidates: list[dict], *, quality_splits: set[str] | None = None,
                      quality_cache: dict[str, tuple[bool, str]] | None = None,
                      quality_rejections: dict[str, dict] | None = None) -> list[dict]:
    rng = random.Random(SEED)
    quality_splits = quality_splits or set()
    used_fsids: set[str] = set()
    rows = []
    by_category = defaultdict(list)
    for row in candidates:
        by_category[row["category"]].append(row)
    for category, label in TARGETS.items():
        for split, count, folds in (
            ("enrollment", 5, {1, 2}),
            ("development_extra", 5, {3, 4, 5}),
            ("calibration", 10, {3, 4, 5}),
            ("positive_holdout", 10, {6, 7, 8, 9, 10}),
        ):
            pool = [row for row in by_category[category] if row["fold"] in folds]
            chosen = _select_disjoint(
                pool, count, used_fsids, rng,
                require_valid=split in quality_splits,
                quality_cache=quality_cache,
                quality_rejections=quality_rejections,
            )
            for row in chosen:
                row.update({"split": split, "expected_label": label, "confusable_for": ""})
                rows.append(row)

    # Thirty unknowns per phase, balanced across all seven non-enrolled classes.
    # Engine idling and street music receive the two remainder slots.
    negative_counts = {category: 4 for category in ALL_NEGATIVES}
    negative_counts.update({"engine_idling": 5, "street_music": 5})
    for split, folds in (
        ("calibration_unknown", {3, 4, 5}),
        ("negative_holdout", {6, 7, 8, 9, 10}),
    ):
        for category in ALL_NEGATIVES:
            pool = [row for row in by_category[category] if row["fold"] in folds]
            chosen = _select_disjoint(
                pool, negative_counts[category], used_fsids, rng,
                require_valid=split in quality_splits,
                quality_cache=quality_cache,
                quality_rejections=quality_rejections,
            )
            for row in chosen:
                row.update({
                    "split": split, "expected_label": "UNKNOWN",
                    "confusable_for": CONFUSABLES[category],
                })
                rows.append(row)
    if len({row["fs_id"] for row in rows}) != len(rows):
        raise AssertionError("an UrbanSound8K fsID crossed benchmark splits")
    expected_folds = {
        "enrollment": {1, 2},
        "development_extra": {3, 4, 5},
        "calibration": {3, 4, 5},
        "calibration_unknown": {3, 4, 5},
        "positive_holdout": {6, 7, 8, 9, 10},
        "negative_holdout": {6, 7, 8, 9, 10},
    }
    for row in rows:
        if row["fold"] not in expected_folds[row["split"]]:
            raise AssertionError("an UrbanSound8K row crossed the designated fold boundary")
    return rows


def prepare_sound_rows(rows: list[dict], splits: set[str]) -> None:
    """Validate and checksum only the phase whose audio is about to be embedded."""
    for row in (row for row in rows if row["split"] in splits):
        if row.get("sha256"):
            continue
        path = row["_path"]
        raw = path.read_bytes()
        _, _, quality = validate_audio_bytes(raw, "sound")
        row.update({
            "sha256": sha256(path),
            "bytes": path.stat().st_size, "quality": quality,
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
                               sound_path: Path, voice_path: Path | None = None) -> dict:
    enabled_kinds = ["sound", *(["voice"] if voice_path is not None else [])]
    client = EmbeddingClient(python_path)
    runtime = PersonalizedRecognitionRuntime(
        familiar_sounds=True, familiar_voices=voice_path is not None,
        store=store, client=client,
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
            for kind in enabled_kinds:
                if states[kind]["status"] == "ready" and kind not in ready_seconds:
                    ready_seconds[kind] = elapsed
            if all(
                states[kind]["status"] in {"ready", "failed"}
                for kind in enabled_kinds
            ):
                break
            time.sleep(0.02)
        final = client.status()
        post_ready = {}
        paths = {"sound": sound_path, **({"voice": voice_path} if voice_path else {})}
        for kind, path in paths.items():
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
            "enabled_kinds": enabled_kinds,
            "duplicate_preload_requests_suppressed": all(
                not client.preload_async(kind) for kind in enabled_kinds
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


def load_carried_vivos_result(path: Path, voice_rows: list[dict], selection: dict) -> tuple[dict, dict]:
    """Normalize the completed VIVOS evaluation without repeating model inference."""
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("archive_md5") != VIVOS_MD5:
        raise RuntimeError("carried VIVOS result archive MD5 does not match the official corpus")
    if source.get("split_counts") != _split_counts(voice_rows):
        raise RuntimeError("carried VIVOS split counts do not match the deterministic manifest")
    if source.get("selection") != selection:
        raise RuntimeError("carried VIVOS speaker selection does not match the deterministic manifest")
    if source.get("real_user_data_unchanged") is not True:
        raise RuntimeError("carried VIVOS result did not preserve real user data")
    final = dict(source["final"])
    final["development"] = source["development"]
    final["production_profile_build"] = source["production_profile_build"]
    final["evaluation_carried_forward"] = True
    final["source_result"] = repo_path(path)
    return final, source["runtime"]["voice"]


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

Run `{manifest['run_id']}` used UrbanSound8K v1.0 and carried forward the completed official VIVOS test-partition evaluation. No DCASE, ESC-50, LibriSpeech, or VIVOS inference was repeated.

## Exact splits

- UrbanSound8K sounds: `{manifest['splits']['sound']}`. Enrollment uses folds 1-2, development uses folds 3-5, and final holdout uses folds 6-10. Every selected clip has a distinct `fsID` across all phases. Holdout audio was embedded once, only after development selection was frozen.
- VIVOS voices: `{manifest['splits']['voice']}`. Two neutral-label enrolled speakers use 5 enrollment, 5 calibration, and 10 final positive utterances each. Three disjoint speakers supply calibration impostors and five other disjoint speakers supply 5 final impostor utterances each.

## UrbanSound8K familiar sounds

- Frozen current method: {_metric_line(sound['baseline_metrics'])}.
- Development-selected method on the same untouched holdout: {sound_improved_text}.
- Retention gate passed: `{sound['algorithm_change_retained']}` -- {sound['retention_reason']}.
- Confusion matrices, per-profile counts, score/margin distributions, raw decisions, calibration grids, folds, `fsID` values, and latency are in the sibling JSON/CSV artifacts.

## Vietnamese familiar voices

- Frozen current method: {_metric_line(voice['baseline_metrics'])}.
- Development-selected method: {voice_improved_text}. These metrics were copied from the completed checksummed VIVOS run; no voice embeddings were recomputed.
- VIVOS is quiet, read Vietnamese speech. This small test does not prove performance for spontaneous family conversation, room or microphone changes, noise, illness, overlapping speakers, or replay attacks.

## Non-blocking model preload

- Runtime start returned in `{runtime['runtime_start_return_seconds']:.4f}` seconds.
- Background sound-model time-to-ready: `{runtime['time_to_ready_seconds'].get('sound')}` seconds.
- Final lifecycle state: sound `{runtime['final_status']['sound']['status']}`; worker processes `{runtime['worker_process_count']}`; duplicate requests suppressed `{runtime['duplicate_preload_requests_suppressed']}`. Voice preload was not repeated because no voice-path code changed after its successful two-model smoke test.
- The unchanged prior synchronous cold-start measurements are copied by reference into `runtime_metrics.json`; post-ready first-request latency and worker memory are recorded there. Server startup, capture, CED, STT, HELP, and emergency processing do not wait for these model loads.

## Isolation and limitations

Raw audio, model caches, embeddings, and isolated profiles remain under Git-ignored `benchmark_data/external/`. No raw audio or biometric data is included here. Physical microphone/OLED behavior, live CED/STT contention, power, thermals, replay resistance, and real household/family conditions were not measured.

This is a contract-specific open-set personalization split, not UrbanSound8K's recommended ten-fold cross-validation protocol. Its metrics must not be compared with published UrbanSound8K classifier results.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="timestamped result directory")
    parser.add_argument(
        "--urbansound-root",
        default="benchmark_data/external/personalized_recognition_domain_shift/urbansound8k/UrbanSound8K",
        help="extracted UrbanSound8K directory containing audio/ and metadata/",
    )
    parser.add_argument(
        "--urbansound-archive",
        default="benchmark_data/external/personalized_recognition_domain_shift/urbansound8k/UrbanSound8K.tar.gz",
    )
    parser.add_argument(
        "--urbansound-download-url",
        default="https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1",
    )
    parser.add_argument(
        "--allow-verified-repackage", action="store_true",
        help=(
            "allow a reputable mirror/repackage whose inventory passes all checks; "
            "the manifest will explicitly report that the official archive MD5 was not verified"
        ),
    )
    parser.add_argument(
        "--vivos-archive", default="benchmark_data/external/vivos/vivos.tar.gz"
    )
    parser.add_argument(
        "--vivos-root",
        default="benchmark_data/external/personalized_recognition_domain_shift/vivos/vivos/test/waves",
    )
    parser.add_argument(
        "--vivos-result",
        default=(
            "benchmark_data/external/personalized_recognition_domain_shift_runs/"
            "vivos_partial_1789307364/result.json"
        ),
        help="completed VIVOS result to carry forward without repeating embeddings",
    )
    parser.add_argument("--python", default=None)
    parser.add_argument("--downloaded-bytes", type=int, default=0)
    args = parser.parse_args()

    def absolute(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    urbansound_root = absolute(args.urbansound_root)
    urbansound_archive = absolute(args.urbansound_archive)
    vivos_archive = absolute(args.vivos_archive)
    vivos_root = absolute(args.vivos_root)
    vivos_result = absolute(args.vivos_result)
    python_path = absolute(args.python) if args.python else (
        ROOT / "benchmark_data" / "external" / "personalized_recognition_venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    for path in (
        urbansound_root, urbansound_archive, vivos_archive, vivos_root,
        vivos_result, python_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_urbansound_md5 = md5(urbansound_archive)
    official_archive_verified = actual_urbansound_md5 == URBANSOUND8K_MD5
    if not official_archive_verified and not args.allow_verified_repackage:
        raise RuntimeError("UrbanSound8K archive MD5 does not match the official v1.0 archive")
    if not official_archive_verified and "zenodo.org" in args.urbansound_download_url:
        raise RuntimeError("a repackaged archive must not be labeled as an official Zenodo download")
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

    metadata_csv = urbansound_root / "metadata" / "UrbanSound8K.csv"
    audio_root = urbansound_root / "audio"
    candidates, urban_inventory = load_urbansound8k(metadata_csv, audio_root)
    quality_cache: dict[str, tuple[bool, str]] = {}
    quality_rejections: dict[str, dict] = {}
    sound_rows = build_sound_split(
        candidates,
        quality_splits=DEVELOPMENT_SPLITS,
        quality_cache=quality_cache,
        quality_rejections=quality_rejections,
    )
    # Only development material is read before method selection.
    prepare_sound_rows(sound_rows, DEVELOPMENT_SPLITS)
    voice_rows, voice_selection = build_vivos_split(vivos_root)
    voice_final, carried_voice_runtime = load_carried_vivos_result(
        vivos_result, voice_rows, voice_selection
    )
    real_before = snapshot_real_user_data()
    store = RecognitionStore(isolated_root / "profiles")
    client = EmbeddingClient(python_path)
    tracker = TimedEmbedder(client)
    try:
        sound_build = create_production_profiles("sound", sound_rows, store, tracker)
        sound_vectors = embed_rows(sound_rows, tracker, "sound", DEVELOPMENT_SPLITS)
        sound_development = calibrate_candidates("sound", sound_rows, sound_vectors)

        # The method is now frozen. Apply the same production quality gate to a
        # metadata-preordered holdout, then validate/embed the accepted clips once.
        development_identity = [
            (row["split"], row["path"]) for row in sound_rows
            if row["split"] in DEVELOPMENT_SPLITS
        ]
        development_file_metadata = {
            row["path"]: {
                key: row[key] for key in ("sha256", "bytes", "quality")
            }
            for row in sound_rows if row["split"] in DEVELOPMENT_SPLITS
        }
        sound_rows = build_sound_split(
            candidates,
            quality_splits=DEVELOPMENT_SPLITS | HOLDOUT_SPLITS,
            quality_cache=quality_cache,
            quality_rejections=quality_rejections,
        )
        if development_identity != [
            (row["split"], row["path"]) for row in sound_rows
            if row["split"] in DEVELOPMENT_SPLITS
        ]:
            raise AssertionError("quality-gating holdout changed the frozen development split")
        for row in sound_rows:
            if row["split"] in DEVELOPMENT_SPLITS:
                row.update(development_file_metadata[row["path"]])
        prepare_sound_rows(sound_rows, HOLDOUT_SPLITS)
        sound_vectors.update(embed_rows(sound_rows, tracker, "sound", HOLDOUT_SPLITS))
        sound_final = evaluate_holdout("sound", sound_rows, sound_vectors, sound_development)
    finally:
        client.close()

    # Write manifests only after untouched holdout embedding so each selected file
    # has its size and checksum recorded without pre-reading holdout audio.
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
    )
    real_after = snapshot_real_user_data()

    sound_final["development"] = sound_development
    sound_final["production_profile_build"] = sound_build
    runtime_metrics = {
        "schema_version": "2.0", "run_id": run_id,
        "preload": preload,
        "previous_synchronous_cold_start": _previous_cold_metrics(),
        "embedding_latency_and_memory": tracker_metrics(tracker.records),
        "carried_vivos_embedding_latency_and_memory": carried_voice_runtime,
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
        "schema_version": "2.0", "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(), "seed": SEED,
        "downloaded_bytes_for_this_run": args.downloaded_bytes,
        "datasets": {
            "sound": {
                "name": "UrbanSound8K v1.0",
                "project_url": "https://urbansounddataset.weebly.com/urbansound8k.html",
                "record_url": "https://zenodo.org/records/1203745",
                "download_url_used": args.urbansound_download_url,
                "doi": URBANSOUND8K_DOI, "license": URBANSOUND8K_LICENSE,
                "archive": {
                    "path": repo_path(urbansound_archive),
                    "bytes": urbansound_archive.stat().st_size,
                    "official_md5": URBANSOUND8K_MD5,
                    "actual_md5": actual_urbansound_md5,
                    "official_archive_md5_verified": official_archive_verified,
                    "verified_repackage_allowed": args.allow_verified_repackage,
                },
                "inventory": urban_inventory,
                "candidate_metadata_rows": len(candidates),
                "selected_clips": len(sound_rows),
                "selected_bytes": sum(row["bytes"] for row in sound_rows),
                "quality_rejections": list(quality_rejections.values()),
                "profiles": {
                    label: [
                        {
                            "file": row["file"], "fs_id": row["fs_id"],
                            "fold": row["fold"], "salience": row["salience"],
                        }
                        for row in sound_rows
                        if row["split"] == "enrollment" and row["expected_label"] == label
                    ]
                    for label in sorted(TARGETS.values())
                },
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
                "evaluation_carried_forward": True,
                "source_result": repo_path(vivos_result),
                **voice_selection,
            },
        },
        "splits": {"sound": _split_counts(sound_rows), "voice": _split_counts(voice_rows)},
        "split_integrity": {
            "sound_fs_id_disjoint": True,
            "sound_enrollment_folds": [1, 2],
            "sound_development_folds": [3, 4, 5],
            "sound_holdout_folds": [6, 7, 8, 9, 10],
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
    write_json(output / "voice_calibration_results.json", voice_final["development"])
    write_json(output / "runtime_metrics.json", runtime_metrics)
    write_json(output / "dataset_manifest.json", manifest)
    (output / "reproduction.md").write_text(
        "# Reproduction\n\n"
        "Use the verified UrbanSound8K archive/extraction and completed VIVOS result listed "
        "in `dataset_manifest.json`, keep them under ignored `benchmark_data/external/`, "
        "then run:\n\n"
        f"```powershell\n.\\.venv\\Scripts\\python.exe tools\\evaluate_personalized_recognition_domain_shift.py --urbansound-root {repo_path(urbansound_root)} --urbansound-archive {repo_path(urbansound_archive)} --urbansound-download-url \"{args.urbansound_download_url}\" --vivos-result {repo_path(vivos_result)} --downloaded-bytes {args.downloaded_bytes}\n```\n",
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
