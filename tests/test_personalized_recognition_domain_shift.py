import json
import random
from collections import Counter

import numpy as np
import soundfile as sf

from tools.evaluate_personalized_recognition_domain_shift import (
    ALL_NEGATIVES,
    TARGETS,
    build_sound_split,
    calibrate_candidates,
    load_carried_vivos_result,
    load_urbansound8k,
    _select_disjoint,
)


def test_urbansound_split_is_exact_fold_bounded_and_fs_id_disjoint():
    candidates = []
    for fold in range(1, 11):
        for category in [*TARGETS, *ALL_NEGATIVES]:
            for index in range(20):
                candidates.append({
                    "category": category,
                    "fs_id": f"{fold}-{category}-{index}",
                    "fold": fold,
                    "salience": 1 if index % 2 == 0 else 2,
                    "source_start_seconds": 0.0,
                    "source_end_seconds": float(index + 1),
                    "file": f"{fold}-{category}-{index}.wav",
                    "path": f"fold{fold}/{fold}-{category}-{index}.wav",
                })
    rows = build_sound_split(candidates)
    assert Counter(row["split"] for row in rows) == {
        "enrollment": 15,
        "development_extra": 15,
        "calibration": 30,
        "positive_holdout": 30,
        "calibration_unknown": 30,
        "negative_holdout": 30,
    }
    assert len({row["fs_id"] for row in rows}) == len(rows)
    assert all(row["salience"] == 1 for row in rows if row["split"] == "enrollment")
    assert {row["fold"] for row in rows if row["split"] == "enrollment"} <= {1, 2}
    assert {
        row["fold"] for row in rows if row["split"] in {
            "development_extra", "calibration", "calibration_unknown"
        }
    } <= {3, 4, 5}
    assert {
        row["fold"] for row in rows if row["split"] in {
            "positive_holdout", "negative_holdout"
        }
    } <= {6, 7, 8, 9, 10}
    assert set(ALL_NEGATIVES) <= {
        row["category"] for row in rows if row["split"] == "calibration_unknown"
    }


def test_urbansound_inventory_links_all_metadata_rows_to_expected_folds(tmp_path):
    audio = tmp_path / "audio"
    metadata = tmp_path / "UrbanSound8K.csv"
    rows = []
    for fold, category in enumerate(sorted({*TARGETS, *ALL_NEGATIVES}), start=1):
        directory = audio / f"fold{fold}"
        directory.mkdir(parents=True)
        name = f"{fold}-0-0-0.wav"
        sf.write(directory / name, np.ones(16_000, dtype=np.float32) * 0.02, 16_000)
        rows.append(
            f"{name},{fold},0,1,1,{fold},{fold - 1},{category}"
        )
    metadata.write_text(
        "slice_file_name,fsID,start,end,salience,fold,classID,class\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    found, inventory = load_urbansound8k(metadata, audio, expected_rows=10)
    assert len(found) == 10
    assert inventory["metadata_rows"] == inventory["wav_files"] == 10
    assert inventory["folds"] == list(range(1, 11))
    assert set(inventory["classes"]) == {*TARGETS, *ALL_NEGATIVES}


def test_quality_aware_selection_skips_invalid_candidate(tmp_path):
    quiet = tmp_path / "quiet.wav"
    valid = tmp_path / "valid.wav"
    sf.write(quiet, np.zeros(16_000, dtype=np.float32), 16_000)
    sf.write(valid, np.ones(16_000, dtype=np.float32) * 0.02, 16_000)
    rows = [
        {
            "category": "siren", "fs_id": path.stem, "fold": 2,
            "salience": 1, "source_start_seconds": 0.0,
            "source_end_seconds": 2.0 if path == quiet else 1.0, "file": path.name,
            "path": str(path), "_path": path,
        }
        for path in (quiet, valid)
    ]
    cache, rejected = {}, {}
    selected = _select_disjoint(
        rows, 1, set(), random.Random(1), require_valid=True,
        quality_cache=cache, quality_rejections=rejected,
    )
    assert selected[0]["file"] == "valid.wav"
    assert rejected[str(quiet)]["reason"] == "audio is silent or too quiet"


def test_calibration_uses_hard_negatives_and_can_reduce_false_acceptance():
    rows = []
    vectors = {}
    labels = ("SPEAKER A", "SPEAKER B")
    for label_index, label in enumerate(labels):
        basis = np.array([1.0, 0.0]) if label_index == 0 else np.array([0.0, 1.0])
        for split, count in (("enrollment", 5), ("calibration", 5)):
            for index in range(count):
                path = f"{label}-{split}-{index}"
                rows.append({
                    "split": split,
                    "expected_label": label,
                    "path": path,
                    "file": path,
                })
                vectors[path] = basis
    for index in range(4):
        path = f"unknown-{index}"
        rows.append({
            "split": "calibration_unknown",
            "expected_label": "UNKNOWN",
            "path": path,
            "file": path,
        })
        vectors[path] = np.array([0.75, 0.66])
    result = calibrate_candidates("voice", rows, vectors)
    assert result["baseline"]["metrics"]["false_acceptance"]["numerator"] == 4
    assert result["selected"] is not None
    assert result["selected"]["metrics"]["false_acceptance"]["numerator"] == 0


def test_vivos_result_is_carried_forward_only_when_provenance_and_split_match(tmp_path):
    rows = [
        {"split": "enrollment"},
        {"split": "positive_holdout"},
    ]
    selection = {"enrolled_speakers": [{"speaker_id": "A"}]}
    source = {
        "archive_md5": "72972a8b14050f3f11ea7c3debabd7af",
        "split_counts": {"enrollment": 1, "positive_holdout": 1},
        "selection": selection,
        "production_profile_build": {"persisted_profile_count": 1},
        "development": {"baseline": {}},
        "final": {"baseline_metrics": {}, "baseline_predictions": []},
        "runtime": {"voice": {"wall_seconds": {"count": 2}}},
        "real_user_data_unchanged": True,
    }
    path = tmp_path / "vivos_result.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    final, runtime = load_carried_vivos_result(path, rows, selection)
    assert final["evaluation_carried_forward"] is True
    assert final["production_profile_build"]["persisted_profile_count"] == 1
    assert runtime["wall_seconds"]["count"] == 2
