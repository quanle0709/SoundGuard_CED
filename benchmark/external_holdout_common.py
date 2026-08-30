"""Frozen methodology helpers for the 2026-08-20 external holdout.

This module contains only pre-prediction decisions.  It must not be edited in
response to external predictions.  The preparation script content-addresses
this file in the final external freeze manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter
from scipy.signal import resample_poly, stft


ROOT = Path(__file__).resolve().parents[1]
DATE_TAG = "20260820"
SEED = 42

DOWNLOAD_DIR = ROOT / "benchmark_data" / "external" / "computer_holdout_downloads"
EXTRACT_DIR = ROOT / "benchmark_data" / "external" / "computer_holdout_audio"
FSD_AUDIO_DIR = EXTRACT_DIR / "FSD50K.eval_audio"
DCASE_AUDIO_DIR = EXTRACT_DIR / "TUT-rare-sound-events-2017-evaluation"
FSD_META_DIR = ROOT / "benchmark_data" / "external" / "emergency_v3_final"

DECONTAM_DIR = ROOT / "benchmark_results" / f"external_decontamination_{DATE_TAG}"
RESULTS_DIR = ROOT / "benchmark_results" / f"external_results_{DATE_TAG}"
BASELINE_DIR = ROOT / "benchmark_results" / f"frozen_baseline_{DATE_TAG}"

FSD_ARCHIVES = {
    "FSD50K.eval_audio.z01": {
        "url": "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01?download=1",
        "md5": "3090670eaeecc013ca1ff84fe4442aeb",
        "bytes": 3_221_225_472,
    },
    "FSD50K.eval_audio.zip": {
        "url": "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip?download=1",
        "md5": "6fa47636c3a3ad5c7dfeba99f2637982",
        "bytes": 3_037_675_767,
    },
}
FSD_AUXILIARY_ARCHIVES = {
    "FSD50K.doc.zip": "3516162b82dc2945d3e7feba0904e800",
    "FSD50K.ground_truth.zip": "ca27382c195e37d2269c4c866dd73485",
    "FSD50K.metadata.zip": "b9ea0c829a411c1d42adb9da539ed237",
}
DCASE_ARCHIVE = "TUT-rare-sound-events-2017-evaluation.source_data_events.zip"
DCASE_MD5 = "5c037df9c1012411a1f2c1325fe20163"

# Preregistered dataset-to-capability mapping.  FSD50K Crackle is intentionally
# absent: it is not synonymous with fire.  Broad Alarm is absent for the same
# reason.  DCASE gunshot is a real emergency but was outside the requested
# baby/glass target scope and is neither a target nor a negative here.
FSD_TARGET_MAPPING = {
    "Fire": "fire",
    "Siren": "siren",
    "Vehicle_horn_and_car_horn_and_honking": "vehicle_horn",
}
DCASE_TARGET_MAPPING = {
    "babycry": "baby_crying",
    "glassbreak": "glass_breaking",
}
TARGET_CLASSES = (
    "glass_breaking",
    "vehicle_horn",
    "fire",
    "siren",
    "baby_crying",
)

FSD_HARD_NEGATIVE_LABELS = {
    "Dog",
    "Bark",
    "Door",
    "Doorbell",
    "Knock",
    "Sliding_door",
    "Speech",
    "Male_speech_and_man_speaking",
    "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking",
    "Speech_synthesizer",
    "Music",
    "Musical_instrument",
    "Engine",
    "Engine_starting",
    "Bell",
    "Bicycle_bell",
    "Church_bell",
    "Cowbell",
}

# Any of these labels makes a clip unsuitable as a known non-emergency.  This
# guard is deliberately broader than the five scored target classes.
FSD_NEGATIVE_GUARD_LABELS = {
    "Alarm",
    "Boom",
    "Crack",
    "Crackle",
    "Crying_and_sobbing",
    "Explosion",
    "Fire",
    "Fireworks",
    "Glass",
    "Gunshot_and_gunfire",
    "Screaming",
    "Shatter",
    "Shout",
    "Siren",
    "Vehicle_horn_and_car_horn_and_honking",
    "Yell",
}

MAPPING_AUDIT = [
    {
        "source_dataset": "FSD50K_eval",
        "source_label": source,
        "soundguard_label": target,
        "decision": "INCLUDE_TARGET",
        "reason": "direct ontology concept with a real frozen SoundGuard V3 capability",
    }
    for source, target in FSD_TARGET_MAPPING.items()
] + [
    {
        "source_dataset": "DCASE2017_source_events",
        "source_label": source,
        "soundguard_label": target,
        "decision": "INCLUDE_TARGET",
        "reason": "direct event class with a real frozen SoundGuard V3 capability",
    }
    for source, target in DCASE_TARGET_MAPPING.items()
] + [
    {
        "source_dataset": "FSD50K_eval",
        "source_label": "Crackle",
        "soundguard_label": "",
        "decision": "EXCLUDE_AMBIGUOUS_MAPPING",
        "reason": "generic crackle is not sufficient evidence of fire",
    },
    {
        "source_dataset": "FSD50K_eval",
        "source_label": "Alarm",
        "soundguard_label": "",
        "decision": "EXCLUDE_AMBIGUOUS_MAPPING",
        "reason": "broad alarm is not specific to fire/smoke or another frozen target",
    },
    {
        "source_dataset": "FSD50K_eval",
        "source_label": "Glass",
        "soundguard_label": "",
        "decision": "EXCLUDE_AMBIGUOUS_MAPPING",
        "reason": "glass presence does not necessarily mean glass breaking",
    },
    {
        "source_dataset": "FSD50K_eval",
        "source_label": "Shatter",
        "soundguard_label": "",
        "decision": "OUTSIDE_PREREGISTERED_FSD_TARGETS",
        "reason": "baby/glass external targets were preregistered against DCASE source events",
    },
    {
        "source_dataset": "FSD50K_eval",
        "source_label": "Crying_and_sobbing",
        "soundguard_label": "",
        "decision": "EXCLUDE_AMBIGUOUS_MAPPING",
        "reason": "does not establish that the speaker is a baby",
    },
    {
        "source_dataset": "DCASE2017_source_events",
        "source_label": "gunshot",
        "soundguard_label": "gunshot",
        "decision": "OUTSIDE_PREREGISTERED_TARGETS_NOT_NEGATIVE",
        "reason": "valid emergency category, but outside requested DCASE baby/glass scope",
    },
]

MANIFEST_FIELDS = (
    "sample_id",
    "audio_path",
    "source_dataset",
    "original_label",
    "original_labels",
    "mapped_soundguard_label",
    "is_target",
    "is_hard_negative",
    "duration",
    "split",
    "included",
    "exclusion_reason",
    "matched_dataset",
    "matched_id",
    "raw_sha256",
    "canonical_pcm_sha256",
    "fingerprint_decision",
    "fingerprint_match_id",
    "fingerprint_evidence",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 - archive integrity, not security
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (list(rows[0]) if rows else MANIFEST_FIELDS))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if target != resolved and resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {info.filename}")
        bundle.extractall(destination)


def esc50_freesound_ids() -> set[str]:
    rows = read_csv(ROOT / "benchmark_data" / "external" / "esc50" / "esc50.csv")
    if len(rows) != 2000:
        raise RuntimeError(f"Expected 2,000 official ESC-50 metadata rows, found {len(rows)}")
    ids = set()
    for row in rows:
        filename = row.get("filename", "")
        match = re.match(r"\d+-(\d+)-", filename)
        if match:
            ids.add(match.group(1))
    if len(ids) != 1524:
        raise RuntimeError(f"Expected 1,524 unique ESC-50 Freesound IDs, found {len(ids)}")
    return ids


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(int(source_rate), sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, int(source_rate) // divisor)
    return np.asarray(mono, dtype=np.float32)


def canonical_pcm_hash(path: Path, sample_rate: int = 16_000) -> str:
    mono = load_mono(path, sample_rate)
    pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    digest = hashlib.sha256()
    digest.update(f"mono_s16le_{sample_rate}_{len(pcm)}\0".encode("ascii"))
    digest.update(pcm.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LandmarkFingerprint:
    duration: float
    hashes: tuple[tuple[int, int], ...]  # (packed landmark hash, anchor frame)


def landmark_fingerprint(path: Path, sample_rate: int = 8_000) -> LandmarkFingerprint:
    """Return a Shazam-style, non-embedding acoustic landmark fingerprint."""
    audio = load_mono(path, sample_rate)
    duration = len(audio) / sample_rate
    if len(audio) < 1024 or float(np.max(np.abs(audio), initial=0.0)) < 1e-5:
        return LandmarkFingerprint(duration, ())
    audio = audio / max(float(np.max(np.abs(audio))), 1e-8)
    _, _, spectrum = stft(
        audio,
        fs=sample_rate,
        nperseg=1024,
        noverlap=768,
        boundary=None,
        padded=False,
    )
    db = 20.0 * np.log10(np.abs(spectrum) + 1e-8)
    first_bin = max(1, int(150 / (sample_rate / 1024)))
    last_bin = min(db.shape[0], int(3500 / (sample_rate / 1024)))
    db = db[first_bin:last_bin]
    local = db == maximum_filter(db, size=(9, 5), mode="nearest")
    floor = float(np.median(db) + 12.0)
    peaks_by_time: dict[int, list[int]] = defaultdict(list)
    peak_f, peak_t = np.nonzero(local & (db >= floor))
    by_frame: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for freq, frame in zip(peak_f.tolist(), peak_t.tolist()):
        by_frame[frame].append((float(db[freq, frame]), freq + first_bin))
    for frame, values in by_frame.items():
        peaks_by_time[frame] = [freq for _, freq in sorted(values, reverse=True)[:6]]

    landmarks: list[tuple[int, int]] = []
    frames = sorted(peaks_by_time)
    for anchor_pos, anchor_frame in enumerate(frames):
        targets: list[tuple[int, int]] = []
        for target_frame in frames[anchor_pos + 1 :]:
            delta = target_frame - anchor_frame
            if delta > 40:
                break
            if delta >= 2:
                targets.extend((target_frame, freq) for freq in peaks_by_time[target_frame])
            if len(targets) >= 8:
                break
        for anchor_freq in peaks_by_time[anchor_frame]:
            for target_frame, target_freq in targets[:8]:
                delta = target_frame - anchor_frame
                packed = (anchor_freq << 20) | (target_freq << 8) | delta
                landmarks.append((packed, anchor_frame))
    return LandmarkFingerprint(duration, tuple(landmarks))


def build_reference_index(
    references: list[tuple[str, Path]],
) -> tuple[dict[int, list[tuple[str, int]]], dict[str, LandmarkFingerprint]]:
    index: dict[int, list[tuple[str, int]]] = defaultdict(list)
    fingerprints: dict[str, LandmarkFingerprint] = {}
    for ref_id, path in references:
        fingerprint = landmark_fingerprint(path)
        fingerprints[ref_id] = fingerprint
        for packed, frame in fingerprint.hashes:
            index[packed].append((ref_id, frame))
    return index, fingerprints


def match_landmarks(
    candidate: LandmarkFingerprint,
    index: dict[int, list[tuple[str, int]]],
    references: dict[str, LandmarkFingerprint],
) -> dict:
    offsets: dict[str, Counter] = defaultdict(Counter)
    total_matches: Counter = Counter()
    for packed, candidate_frame in candidate.hashes:
        for ref_id, reference_frame in index.get(packed, ()):
            offsets[ref_id][candidate_frame - reference_frame] += 1
            total_matches[ref_id] += 1
    if not total_matches:
        return {"decision": "NO_MATCH", "method": "landmark_v1"}
    best_id = max(total_matches, key=lambda ref: max(offsets[ref].values()))
    dominant_offset, dominant_hits = offsets[best_id].most_common(1)[0]
    all_hits = total_matches[best_id]
    denominator = max(1, min(len(candidate.hashes), len(references[best_id].hashes)))
    consistency = dominant_hits / all_hits
    coverage = dominant_hits / denominator
    evidence = {
        "method": "landmark_v1",
        "matched_reference": best_id,
        "dominant_offset_frames": dominant_offset,
        "dominant_offset_seconds": round(dominant_offset * 0.032, 6),
        "dominant_hits": dominant_hits,
        "all_shared_hash_hits": all_hits,
        "offset_consistency": consistency,
        "minimum_fingerprint_coverage": coverage,
        "candidate_landmarks": len(candidate.hashes),
        "reference_landmarks": len(references[best_id].hashes),
    }
    if dominant_hits >= 40 and consistency >= 0.70 and coverage >= 0.20:
        evidence["decision"] = "EXCLUDE_HIGH_CONFIDENCE_NEAR_DUPLICATE"
    elif dominant_hits >= 20 and consistency >= 0.40 and coverage >= 0.10:
        evidence["decision"] = "FLAG_FOR_REVIEW"
    else:
        evidence["decision"] = "NO_MATCH"
    return evidence


def verify_aligned_waveform_ncc(
    candidate_path: Path,
    reference_path: Path,
    dominant_offset_frames: int,
    sample_rate: int = 8_000,
) -> dict:
    """Verify a landmark match with aligned normalized waveform correlation."""
    candidate = load_mono(candidate_path, sample_rate)
    reference = load_mono(reference_path, sample_rate)
    predicted_reference_lag = -int(round(dominant_offset_frames * 0.032 * sample_rate))

    def score(reference_lag: int) -> tuple[float, int]:
        reference_start = max(0, reference_lag)
        candidate_start = max(0, -reference_lag)
        overlap = min(len(candidate) - candidate_start, len(reference) - reference_start)
        if overlap <= 0:
            return 0.0, 0
        left = candidate[candidate_start : candidate_start + overlap].astype(np.float64)
        right = reference[reference_start : reference_start + overlap].astype(np.float64)
        left -= left.mean()
        right -= right.mean()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return (float(np.dot(left, right) / denominator) if denominator else 0.0), overlap

    radius = int(round(0.1 * sample_rate))
    coarse = range(predicted_reference_lag - radius, predicted_reference_lag + radius + 1, 8)
    best_lag = max(coarse, key=lambda lag: abs(score(lag)[0]))
    fine = range(best_lag - 8, best_lag + 9)
    best_lag = max(fine, key=lambda lag: abs(score(lag)[0]))
    ncc, overlap = score(best_lag)
    return {
        "aligned_overlap_seconds": overlap / sample_rate,
        "aligned_waveform_ncc": ncc,
        "verified_reference_lag_seconds": best_lag / sample_rate,
        "alignment_search_radius_seconds": 0.1,
    }


def development_reference_audio() -> list[tuple[str, Path]]:
    references: list[tuple[str, Path]] = []
    groups = (
        ROOT / "benchmark_data" / "external" / "esc50" / "audio",
        ROOT / "benchmark_data" / "external" / "vivos" / "selected",
        ROOT / "benchmark_data" / "external" / "demand" / "selected",
        ROOT / "samples",
    )
    for group in groups:
        if not group.is_dir():
            continue
        for path in sorted(group.rglob("*.wav")):
            references.append((path.relative_to(ROOT).as_posix(), path))
    return references


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def manifest_sha256s(paths: list[Path]) -> dict[str, str]:
    return {relative(path): sha256_file(path) for path in paths}
