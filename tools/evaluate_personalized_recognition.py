"""Deterministic, leakage-aware local evaluation for both embedding paths."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from personalization.recognition import EmbeddingClient, build_centroid, open_set_match


def embed_many(client, kind, paths):
    values, metadata, timings = [], None, []
    for path in paths:
        vector, metadata = client.embed(kind, path)
        values.append(vector)
        timings.append(float(metadata["inference_seconds"]))
    return values, metadata, timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=None)
    args = parser.parse_args()
    client = EmbeddingClient(args.python)
    result = {"schema_version": "1.0", "created_at_unix": time.time()}
    try:
        esc_root = ROOT / "benchmark_data" / "external" / "esc50"
        with (esc_root / "esc50.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        classes = ("door_wood_knock", "dog")
        sound_profiles, sound_times, sound_metadata = [], [], None
        for category in classes:
            selected = sorted(
                (row for row in rows if row["category"] == category and
                 (esc_root / "audio" / row["filename"]).is_file()),
                key=lambda row: (int(row["fold"]), row["filename"]),
            )
            enrollment = [esc_root / "audio" / next(
                row["filename"] for row in selected if int(row["fold"]) == fold
            ) for fold in (1, 2, 3)]
            vectors, sound_metadata, timings = embed_many(client, "sound", enrollment)
            sound_times.extend(timings)
            sound_profiles.append({
                "id": category, "display_name": category, "enabled": True,
                "embedding": build_centroid(vectors), "threshold": 0.72, "margin": 0.08,
            })
        sound_trials = []
        for expected in (*classes, "car_horn"):
            candidates = sorted(
                (row for row in rows if row["category"] == expected and
                 int(row["fold"]) in (4, 5) and
                 (esc_root / "audio" / row["filename"]).is_file()),
                key=lambda row: (int(row["fold"]), row["filename"]),
            )[:2]
            for row in candidates:
                vector, metadata = client.embed("sound", esc_root / "audio" / row["filename"])
                sound_times.append(float(metadata["inference_seconds"]))
                decision = open_set_match(vector, sound_profiles)
                sound_trials.append({
                    "file": row["filename"], "expected": expected if expected in classes else "UNKNOWN",
                    "predicted": decision["label"], "accepted": decision["accepted"],
                    "score": decision["score"], "margin": decision["margin"],
                    "reason": decision["reason"],
                })
        vivos = ROOT / "benchmark_data" / "external" / "vivos" / "selected"
        speakers = ("VIVOSDEV03", "VIVOSDEV04")
        voice_profiles, voice_times, voice_metadata = [], [], None
        held_out = {}
        for speaker in speakers:
            paths = sorted((vivos / speaker).glob("*.wav"))
            vectors, voice_metadata, timings = embed_many(client, "voice", paths[:3])
            voice_times.extend(timings)
            held_out[speaker] = paths[3:5]
            voice_profiles.append({
                "id": speaker, "display_name": speaker, "enabled": True,
                "embedding": build_centroid(vectors), "threshold": 0.72, "margin": 0.06,
            })
        held_out["UNKNOWN"] = sorted((vivos / "VIVOSDEV05").glob("*.wav"))[:2]
        voice_trials = []
        for expected, paths in held_out.items():
            for path in paths:
                vector, metadata = client.embed("voice", path)
                voice_times.append(float(metadata["inference_seconds"]))
                decision = open_set_match(vector, voice_profiles)
                voice_trials.append({
                    "file": path.name, "expected": expected,
                    "predicted": decision["label"], "accepted": decision["accepted"],
                    "score": decision["score"], "margin": decision["margin"],
                    "reason": decision["reason"],
                })
        result.update({
            "sound": {"dataset": "ESC-50", "enrollment_folds": [1, 2, 3],
                      "test_folds": [4, 5], "trials": sound_trials,
                      "metadata": sound_metadata, "inference_seconds": sound_times},
            "voice": {"dataset": "VIVOS selected", "enrollment_utterances_per_speaker": 3,
                      "trials": voice_trials, "metadata": voice_metadata,
                      "inference_seconds": voice_times},
        })
        for section in ("sound", "voice"):
            timings = result[section]["inference_seconds"]
            trials = result[section]["trials"]
            result[section]["summary"] = {
                "trials": len(trials),
                "correct": sum(item["predicted"] == item["expected"] for item in trials),
                "unknown_trials": sum(item["expected"] == "UNKNOWN" for item in trials),
                "false_accepts": sum(item["expected"] == "UNKNOWN" and item["accepted"] for item in trials),
                "median_inference_seconds": statistics.median(timings),
                "max_inference_seconds": max(timings),
            }
    finally:
        client.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key]["summary"] for key in ("sound", "voice")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
