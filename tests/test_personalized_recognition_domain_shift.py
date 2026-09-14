import json
from collections import Counter

import numpy as np
import soundfile as sf

from tools.evaluate_personalized_recognition_domain_shift import (
    ALL_NEGATIVES,
    TARGETS,
    build_sound_split,
    calibrate_candidates,
    discover_dcase_candidates,
)


def test_dcase_split_is_exact_and_source_background_disjoint():
    candidates = []
    for category in [*TARGETS, *ALL_NEGATIVES]:
        for index in range(60):
            candidates.append({
                "category": category,
                "foreground_source_id": f"{category}/foreground-{index}.wav",
                "background_source_id": f"{category}/background-{index}.wav",
                "soundscape": f"{category}-{index}.wav",
            })
    rows = build_sound_split(candidates)
    assert Counter(row["split"] for row in rows) == {
        "enrollment": 15,
        "development_extra": 15,
        "calibration": 15,
        "positive_holdout": 30,
        "calibration_unknown": 20,
        "negative_holdout": 20,
    }
    assert len({row["foreground_source_id"] for row in rows}) == len(rows)
    assert len({row["background_source_id"] for row in rows}) == len(rows)


def test_dcase_discovery_uses_strong_timestamps_and_rejects_overlap(tmp_path):
    audio = tmp_path / "audio"
    jams = tmp_path / "jams"
    audio.mkdir()
    jams.mkdir()
    sf.write(audio / "clean.wav", np.ones(160_000, dtype=np.float32) * 0.02, 16_000)
    sf.write(audio / "overlap.wav", np.ones(160_000, dtype=np.float32) * 0.02, 16_000)

    def write(name, values):
        (jams / f"{name}.jams").write_text(json.dumps({
            "annotations": [{
                "namespace": "scaper",
                "data": [{
                    "time": 0,
                    "duration": 10,
                    "value": {
                        "role": "background",
                        "source_file": f"bg/{name}.wav",
                    },
                }, *values],
            }],
        }), encoding="utf-8")

    target = {
        "time": 1.0,
        "duration": 2.0,
        "value": {
            "role": "foreground",
            "event_label": "Blender",
            "source_file": "fg/blender-a.wav",
            "snr": 12,
        },
    }
    write("clean", [target])
    write("overlap", [target, {
        "time": 2.0,
        "duration": 1.0,
        "value": {
            "role": "foreground",
            "event_label": "Dishes",
            "source_file": "fg/dishes-a.wav",
            "snr": 8,
        },
    }])
    found = discover_dcase_candidates(jams, audio)
    assert len(found) == 1
    assert found[0]["soundscape"] == "clean.wav"
    assert found[0]["event_onset_seconds"] == 1.0
    assert found[0]["event_offset_seconds"] == 3.0


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
