"""Acquire, decontaminate, and freeze the external holdout before prediction.

The script never invokes a model.  Its final freeze is created only after
archive verification, exact-ID filtering, exact content hashes, and the
conservative acoustic-landmark check have completed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.external_holdout_common import (
    BASELINE_DIR,
    DATE_TAG,
    DCASE_ARCHIVE,
    DCASE_AUDIO_DIR,
    DCASE_MD5,
    DCASE_TARGET_MAPPING,
    DECONTAM_DIR,
    DOWNLOAD_DIR,
    EXTRACT_DIR,
    FSD_ARCHIVES,
    FSD_AUXILIARY_ARCHIVES,
    FSD_AUDIO_DIR,
    FSD_HARD_NEGATIVE_LABELS,
    FSD_META_DIR,
    FSD_NEGATIVE_GUARD_LABELS,
    FSD_TARGET_MAPPING,
    MANIFEST_FIELDS,
    MAPPING_AUDIT,
    RESULTS_DIR,
    ROOT,
    SEED,
    build_reference_index,
    canonical_pcm_hash,
    development_reference_audio,
    esc50_freesound_ids,
    landmark_fingerprint,
    manifest_sha256s,
    match_landmarks,
    md5_file,
    read_csv,
    relative,
    safe_extract_zip,
    sha256_file,
    verify_aligned_waveform_ncc,
    write_csv,
    write_json,
)
from benchmark.split_zip import extract_split_zip


def now() -> str:
    return datetime.now().astimezone().isoformat()


def file_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()


def verify_archives(require_fsd: bool) -> dict:
    archives = []
    dcase_path = DOWNLOAD_DIR / DCASE_ARCHIVE
    if not dcase_path.is_file():
        raise FileNotFoundError(dcase_path)
    dcase_actual = md5_file(dcase_path)
    if dcase_actual != DCASE_MD5:
        raise RuntimeError(f"DCASE MD5 mismatch: {dcase_actual}")
    archives.append(
        {
            "dataset": "DCASE2017",
            "official_record": "https://zenodo.org/records/1160455",
            "source_url": "https://zenodo.org/record/1160455/files/"
            + DCASE_ARCHIVE,
            "archive_name": DCASE_ARCHIVE,
            "archive_size_bytes": dcase_path.stat().st_size,
            "local_file_mtime": file_mtime(dcase_path),
            "md5_expected": DCASE_MD5,
            "md5_actual": dcase_actual,
            "verified": True,
        }
    )
    for name, metadata in FSD_ARCHIVES.items():
        path = DOWNLOAD_DIR / name
        if not path.is_file():
            if require_fsd:
                raise FileNotFoundError(path)
            continue
        size_ok = path.stat().st_size == metadata["bytes"]
        if require_fsd and not size_ok:
            raise RuntimeError(
                f"FSD50K size mismatch for {name}: {path.stat().st_size} != {metadata['bytes']}"
            )
        actual = md5_file(path) if size_ok else "NOT_COMPUTED_INCOMPLETE_FILE"
        verified = actual == metadata["md5"]
        if require_fsd and not verified:
            raise RuntimeError(f"FSD50K MD5 mismatch for {name}: {actual}")
        archives.append(
            {
                "dataset": "FSD50K_eval",
                "official_record": "https://zenodo.org/records/4060432",
                "source_url": metadata["url"],
                "archive_name": name,
                "archive_size_bytes": path.stat().st_size,
                "local_file_mtime": file_mtime(path),
                "md5_expected": metadata["md5"],
                "md5_actual": actual,
                "verified": verified,
            }
        )
    for name, expected in FSD_AUXILIARY_ARCHIVES.items():
        path = FSD_META_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = md5_file(path)
        if actual != expected:
            raise RuntimeError(f"FSD50K auxiliary archive MD5 mismatch for {name}: {actual}")
        archives.append(
            {
                "dataset": "FSD50K_eval",
                "official_record": "https://zenodo.org/records/4060432",
                "source_url": f"https://zenodo.org/records/4060432/files/{name}?download=1",
                "archive_name": name,
                "archive_size_bytes": path.stat().st_size,
                "local_file_mtime": file_mtime(path),
                "md5_expected": expected,
                "md5_actual": actual,
                "verified": True,
            }
        )
    with zipfile.ZipFile(FSD_META_DIR / "FSD50K.metadata.zip") as bundle:
        clip_info = json.loads(bundle.read("FSD50K.metadata/eval_clips_info_FSD50K.json"))
    license_counts = Counter(entry["license"] for entry in clip_info.values())
    supporting_files = []
    for name in ("DCASE_LICENSE.txt", "DCASE_FREESOUNDCREDITS.txt"):
        path = FSD_META_DIR / name
        if path.is_file():
            supporting_files.append(
                {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    integrity_incidents = []
    quarantine = DOWNLOAD_DIR / "FSD50K.eval_audio.z01.corrupt-md5-3b4f938d3b828931baf8ea14d8a8ac53"
    if quarantine.is_file():
        integrity_incidents.append(
            {
                "dataset": "FSD50K_eval",
                "file": relative(quarantine),
                "size_bytes": quarantine.stat().st_size,
                "observed_md5": "3b4f938d3b828931baf8ea14d8a8ac53",
                "expected_md5": FSD_ARCHIVES["FSD50K.eval_audio.z01"]["md5"],
                "decision": "QUARANTINED_NOT_USED",
                "resolution": "full byte-range redownload to a separate file; replacement only after official MD5 matched",
            }
        )
    return {
        "verified_at": now(),
        "archives": archives,
        "fsd50k_eval_clip_license_counts": dict(sorted(license_counts.items())),
        "supporting_license_and_credit_files": supporting_files,
        "download_integrity_incidents": integrity_incidents,
    }


def extract_dcase() -> None:
    verify_archives(require_fsd=False)
    if DCASE_AUDIO_DIR.is_dir() and list(DCASE_AUDIO_DIR.rglob("*.wav")):
        print(f"DCASE already extracted: {DCASE_AUDIO_DIR}")
        return
    safe_extract_zip(DOWNLOAD_DIR / DCASE_ARCHIVE, EXTRACT_DIR)
    wavs = list(DCASE_AUDIO_DIR.rglob("*.wav"))
    if len(wavs) != 83:
        raise RuntimeError(f"Expected 83 DCASE event WAVs after extraction, found {len(wavs)}")
    print(f"Extracted DCASE event audio: {len(wavs)} WAVs")


def extract_fsd() -> None:
    verify_archives(require_fsd=True)
    existing = list(FSD_AUDIO_DIR.rglob("*.wav")) if FSD_AUDIO_DIR.is_dir() else []
    if len(existing) == 10_231:
        print(f"FSD50K already extracted: {len(existing)} WAVs")
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Extracting verified FSD50K split archive with per-entry CRC checks...")
    extract_split_zip(
        [
            DOWNLOAD_DIR / "FSD50K.eval_audio.z01",
            DOWNLOAD_DIR / "FSD50K.eval_audio.zip",
        ],
        EXTRACT_DIR,
    )
    wavs = list(FSD_AUDIO_DIR.rglob("*.wav"))
    if len(wavs) != 10_231:
        raise RuntimeError(f"Expected 10,231 FSD50K eval WAVs, found {len(wavs)}")
    print(f"Extracted FSD50K eval audio: {len(wavs)} WAVs")


def fsd_ground_truth() -> list[dict[str, str]]:
    archive = FSD_META_DIR / "FSD50K.ground_truth.zip"
    with zipfile.ZipFile(archive) as bundle:
        stream = io.TextIOWrapper(bundle.open("FSD50K.ground_truth/eval.csv"), encoding="utf-8")
        rows = list(csv.DictReader(stream))
    if len(rows) != 10_231:
        raise RuntimeError(f"Expected 10,231 FSD50K eval rows, found {len(rows)}")
    return rows


def find_fsd_audio(sample_id: str) -> Path:
    direct = FSD_AUDIO_DIR / f"{sample_id}.wav"
    if direct.is_file():
        return direct
    matches = list(FSD_AUDIO_DIR.rglob(f"{sample_id}.wav"))
    if len(matches) != 1:
        raise RuntimeError(f"Could not uniquely resolve FSD50K audio {sample_id}: {matches}")
    return matches[0]


def base_row(
    sample_id: str,
    path: Path,
    source_dataset: str,
    original_label: str,
    original_labels: str,
    mapped_label: str,
    is_target: bool,
    is_negative: bool,
    decision: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "audio_path": relative(path),
        "source_dataset": source_dataset,
        "original_label": original_label,
        "original_labels": original_labels,
        "mapped_soundguard_label": mapped_label,
        "is_target": is_target,
        "is_hard_negative": is_negative,
        "duration": float(sf.info(path).duration),
        "split": "official_evaluation",
        "included": decision == "ELIGIBLE",
        "exclusion_reason": "" if decision == "ELIGIBLE" else decision,
        "matched_dataset": "",
        "matched_id": "",
        "raw_sha256": "",
        "canonical_pcm_sha256": "",
        "fingerprint_decision": "NOT_RUN",
        "fingerprint_match_id": "",
        "fingerprint_evidence": "",
        "selection_decision": decision,
    }


def build_population() -> list[dict]:
    esc_ids = esc50_freesound_ids()
    rows: list[dict] = []
    for source in fsd_ground_truth():
        sample_id = source["fname"]
        labels = source["labels"].split(",")
        label_set = set(labels)
        mapped = sorted({FSD_TARGET_MAPPING[label] for label in label_set if label in FSD_TARGET_MAPPING})
        if len(mapped) > 1:
            decision = "FLAG_FOR_REVIEW_MULTIPLE_TARGET_LABELS"
            mapped_label = ";".join(mapped)
            target = True
            negative = False
        elif mapped:
            decision = "ELIGIBLE"
            mapped_label = mapped[0]
            target = True
            negative = False
        elif label_set & FSD_HARD_NEGATIVE_LABELS and not label_set & FSD_NEGATIVE_GUARD_LABELS:
            decision = "ELIGIBLE"
            mapped_label = "non_emergency"
            target = False
            negative = True
        else:
            decision = "OUTSIDE_PREREGISTERED_TARGET_OR_HARD_NEGATIVE_SCOPE"
            mapped_label = ""
            target = negative = False
        row = base_row(
            sample_id,
            find_fsd_audio(sample_id),
            "FSD50K_eval",
            labels[0],
            source["labels"],
            mapped_label,
            target,
            negative,
            decision,
        )
        if sample_id in esc_ids:
            row.update(
                included=False,
                exclusion_reason="exact_freesound_id_overlap_with_ESC50",
                matched_dataset="ESC-50",
                matched_id=sample_id,
            )
        rows.append(row)

    dcase_root = DCASE_AUDIO_DIR / "data" / "source_data" / "events"
    for path in sorted(dcase_root.glob("*/*.wav")):
        source_label = path.parent.name
        sample_id = path.stem
        mapped_label = DCASE_TARGET_MAPPING.get(source_label, "")
        decision = "ELIGIBLE" if mapped_label else "OUTSIDE_PREREGISTERED_TARGETS_NOT_NEGATIVE"
        row = base_row(
            sample_id,
            path,
            "DCASE2017_source_events",
            source_label,
            source_label,
            mapped_label,
            bool(mapped_label),
            False,
            decision,
        )
        if sample_id in esc_ids:
            row.update(
                included=False,
                exclusion_reason="exact_freesound_id_overlap_with_ESC50",
                matched_dataset="ESC-50",
                matched_id=sample_id,
            )
        rows.append(row)
    if len([row for row in rows if row["source_dataset"] == "DCASE2017_source_events"]) != 83:
        raise RuntimeError("DCASE population is not 83 source events")
    return rows


def hash_and_fingerprint(rows: list[dict]) -> None:
    references = development_reference_audio()
    print(f"Hashing {len(references)} development reference files...")
    raw_refs: dict[str, str] = {}
    pcm_refs: dict[str, str] = {}
    for index, (ref_id, path) in enumerate(references, 1):
        raw_refs.setdefault(sha256_file(path), ref_id)
        pcm_refs.setdefault(canonical_pcm_hash(path), ref_id)
        if index % 100 == 0:
            print(f"Reference exact hashes {index}/{len(references)}", flush=True)

    candidates = [row for row in rows if row["included"]]
    fingerprint_candidates = []
    for index, row in enumerate(candidates, 1):
        path = ROOT / row["audio_path"]
        row["raw_sha256"] = sha256_file(path)
        row["canonical_pcm_sha256"] = canonical_pcm_hash(path)
        match = raw_refs.get(row["raw_sha256"]) or pcm_refs.get(row["canonical_pcm_sha256"])
        if match:
            row.update(
                included=False,
                exclusion_reason="exact_audio_hash_duplicate",
                matched_dataset="SoundGuard_development_audio",
                matched_id=match,
                fingerprint_decision="NOT_REQUIRED_EXACT_HASH_MATCH",
            )
        else:
            fingerprint_candidates.append(row)
        if index % 100 == 0:
            print(f"Candidate exact hashes {index}/{len(candidates)}", flush=True)

    print(f"Building landmark index for {len(references)} development files...")
    reference_index, reference_fingerprints = build_reference_index(references)
    for index, row in enumerate(fingerprint_candidates, 1):
        path = ROOT / row["audio_path"]
        evidence = match_landmarks(
            landmark_fingerprint(path), reference_index, reference_fingerprints
        )
        if evidence["decision"] in {
            "EXCLUDE_HIGH_CONFIDENCE_NEAR_DUPLICATE",
            "FLAG_FOR_REVIEW",
        }:
            reference_path = ROOT / evidence["matched_reference"]
            evidence.update(
                verify_aligned_waveform_ncc(
                    path, reference_path, int(evidence["dominant_offset_frames"])
                )
            )
            if evidence["decision"] == "EXCLUDE_HIGH_CONFIDENCE_NEAR_DUPLICATE" and not (
                evidence["aligned_overlap_seconds"] >= 1.0
                and abs(evidence["aligned_waveform_ncc"]) >= 0.85
            ):
                evidence["decision"] = "FLAG_FOR_REVIEW"
        decision = evidence["decision"]
        row["fingerprint_decision"] = decision
        row["fingerprint_match_id"] = evidence.get("matched_reference", "")
        row["fingerprint_evidence"] = json.dumps(evidence, separators=(",", ":"))
        if decision == "EXCLUDE_HIGH_CONFIDENCE_NEAR_DUPLICATE":
            row.update(
                included=False,
                exclusion_reason="high_confidence_acoustic_fingerprint_duplicate",
                matched_dataset="SoundGuard_development_audio",
                matched_id=evidence["matched_reference"],
            )
        if index % 50 == 0:
            print(f"Acoustic fingerprints {index}/{len(fingerprint_candidates)}", flush=True)


def count_summary(rows: list[dict]) -> dict:
    def key(row: dict) -> tuple[str, str, str]:
        role = "target" if row["is_target"] else "hard_negative" if row["is_hard_negative"] else "outside_scope"
        label = row["mapped_soundguard_label"] or row["original_label"]
        return row["source_dataset"], label, role

    details = []
    for group in sorted({key(row) for row in rows}):
        selected = [row for row in rows if key(row) == group]
        reasons = Counter(row["exclusion_reason"] for row in selected if not row["included"])
        details.append(
            {
                "dataset": group[0],
                "class": group[1],
                "role": group[2],
                "pre_decontamination_n": len(selected),
                "post_decontamination_n": sum(bool(row["included"]) for row in selected),
                "excluded_n": sum(not bool(row["included"]) for row in selected),
                "exclusion_reasons": dict(reasons),
            }
        )
    return {
        "terminology": "Decontaminated external holdout derived from official evaluation data",
        "official_population": {
            dataset: sum(row["source_dataset"] == dataset for row in rows)
            for dataset in sorted({row["source_dataset"] for row in rows})
        },
        "exact_id_overlap_all_official_rows": {
            dataset: sum(
                row["source_dataset"] == dataset
                and row["exclusion_reason"] == "exact_freesound_id_overlap_with_ESC50"
                for row in rows
            )
            for dataset in sorted({row["source_dataset"] for row in rows})
        },
        "benchmark_eligible_before_decontamination": sum(
            row["selection_decision"] == "ELIGIBLE" for row in rows
        ),
        "final_included": sum(bool(row["included"]) for row in rows),
        "details": details,
    }


def freeze(rows: list[dict], provenance: dict) -> Path:
    ambiguous = [row for row in rows if row["fingerprint_decision"] == "FLAG_FOR_REVIEW"]
    multi_target = [
        row for row in rows if row["selection_decision"] == "FLAG_FOR_REVIEW_MULTIPLE_TARGET_LABELS"
    ]
    if ambiguous or multi_target:
        review = DECONTAM_DIR / "overlap_checks" / "manual_review_required.json"
        write_json(review, {"fingerprint_flags": ambiguous, "multiple_target_flags": multi_target})
        raise RuntimeError(
            f"Scientific review gate: {len(ambiguous)} ambiguous fingerprint matches and "
            f"{len(multi_target)} multi-target mappings; no final freeze was created"
        )

    included = [row for row in rows if row["included"]]
    excluded = [row for row in rows if not row["included"]]
    included_csv = DECONTAM_DIR / "included_manifests" / "external_included_manifest.csv"
    included_json = DECONTAM_DIR / "included_manifests" / "external_included_manifest.json"
    excluded_csv = DECONTAM_DIR / "excluded_manifests" / "external_excluded_manifest.csv"
    excluded_json = DECONTAM_DIR / "excluded_manifests" / "external_excluded_manifest.json"
    population_csv = DECONTAM_DIR / "overlap_checks" / "full_official_population_audit.csv"
    summary_path = DECONTAM_DIR / "decontamination_summary.json"
    mapping_csv = DECONTAM_DIR / "freeze_manifest" / "class_mapping_audit.csv"
    provenance_path = DECONTAM_DIR / "source_provenance" / "official_sources.json"

    write_csv(included_csv, included, MANIFEST_FIELDS)
    write_json(included_json, included)
    write_csv(excluded_csv, excluded, MANIFEST_FIELDS)
    write_json(excluded_json, excluded)
    write_csv(population_csv, rows, [*MANIFEST_FIELDS, "selection_decision"])
    write_json(summary_path, count_summary(rows))
    write_csv(mapping_csv, MAPPING_AUDIT)
    write_json(provenance_path, provenance)

    artifacts = [
        included_csv,
        included_json,
        excluded_csv,
        excluded_json,
        population_csv,
        summary_path,
        mapping_csv,
        provenance_path,
    ]
    freeze_path = DECONTAM_DIR / "freeze_manifest" / "final_external_freeze_manifest.json"
    payload = {
        "schema_version": 1,
        "frozen_at": now(),
        "prediction_state_at_freeze": "NOT_STARTED",
        "terminology": "Decontaminated external holdout derived from the official FSD50K/DCASE evaluation data",
        "methodology_statement": (
            "Contamination auditing and all sample/mapping decisions were completed before prediction. "
            "Exact ESC-50 provenance overlaps were removed, exact audio hashes and conservative "
            "non-embedding acoustic landmarks were checked against available SoundGuard development "
            "audio, and model/thresholds were frozen before external evaluation. No post-hoc tuning is allowed."
        ),
        "baseline_manifest": relative(BASELINE_DIR / "baseline_manifest.json"),
        "baseline_manifest_sha256": sha256_file(BASELINE_DIR / "baseline_manifest.json"),
        "frozen_baseline": {
            "policy": "V2 OR EfficientSED approved-safety evidence",
            "ced_model": "mispeech/ced-tiny",
            "ced_revision": "ace276d29dd0bb3f3517b0fa8cf300738c409019",
            "semantic_mapping": "soundguard_taxonomy_v1",
            "efficientsed_checkpoint_sha256": "92a5c1dc5aa21620e262420c4cc657d2f699bdfa20a748de506c354a2fe2770a",
            "efficientsed_threshold": 0.05,
            "target_classes": list(FSD_TARGET_MAPPING.values())
            + list(DCASE_TARGET_MAPPING.values()),
        },
        "selection": {
            "seed": SEED,
            "hard_negative_sampling": "none; all qualifying FSD50K eval clips",
            "hard_negative_labels": sorted(FSD_HARD_NEGATIVE_LABELS),
            "negative_guard_labels": sorted(FSD_NEGATIVE_GUARD_LABELS),
            "model_output_used_for_selection": False,
        },
        "duplicate_checks": {
            "exact_id": "Freesound ID compared with all 2,000 official ESC-50 metadata rows",
            "raw_hash": "SHA256 of source file bytes",
            "canonical_hash": "SHA256 of mono 16 kHz signed-16-bit decoded PCM without peak normalization or trimming",
            "fingerprint": {
                "method": "landmark_v1: local STFT spectral peaks with time-offset-consistent pair hashes, verified by aligned waveform NCC",
                "auto_exclude": "dominant_hits >= 40 AND offset_consistency >= 0.70 AND minimum_coverage >= 0.20 AND aligned overlap >= 1.0 s AND |NCC| >= 0.85",
                "flag_for_review": "dominant_hits >= 20 AND offset_consistency >= 0.40 AND minimum_coverage >= 0.10",
                "embedding_only_exclusion": False,
            },
            "available_development_reference_scope": [
                "benchmark_data/external/esc50/audio",
                "benchmark_data/external/vivos/selected",
                "benchmark_data/external/demand/selected",
                "samples",
            ],
        },
        "artifact_sha256": manifest_sha256s(artifacts),
        "evaluator_code_sha256": manifest_sha256s(
            [
                ROOT / "benchmark" / "external_holdout_common.py",
                ROOT / "benchmark" / "download_datasets.py",
                ROOT / "benchmark" / "prepare_external_holdout.py",
                ROOT / "benchmark" / "run_external_holdout.py",
                ROOT / "benchmark" / "report_external_holdout.py",
                ROOT / "benchmark" / "split_zip.py",
                ROOT / "benchmark" / "test_external_holdout.py",
                ROOT / "soundguard/detection/sound_classifier.py",
                ROOT / "soundguard/detection/sound_taxonomy.py",
                ROOT / "soundguard/emergency/emergency_system.py",
                ROOT / "soundguard/emergency/emergency_v3.py",
                ROOT / "benchmark" / "experiments" / "emergency_v3" / "efficientsed_worker.py",
                ROOT / "benchmark" / "experiments" / "emergency_v3" / "efficientsed_safety_taxonomy.json",
            ]
        ),
        "freeze_rule": "No sample, label mapping, model, threshold, preprocessing, policy, or metric decision may change in response to external results.",
    }
    if (RESULTS_DIR / "inference_started.json").exists():
        raise RuntimeError("Inference marker already exists; refusing to claim a pre-prediction freeze")
    write_json(freeze_path, payload)
    print(f"FINAL PRE-PREDICTION FREEZE: {freeze_path}")
    return freeze_path


def prepare_and_freeze() -> Path:
    provenance = verify_archives(require_fsd=True)
    if len(list(FSD_AUDIO_DIR.rglob("*.wav"))) != 10_231:
        raise RuntimeError("FSD50K audio is not fully extracted")
    if len(list(DCASE_AUDIO_DIR.rglob("*.wav"))) != 83:
        raise RuntimeError("DCASE audio is not fully extracted")
    rows = build_population()
    hash_and_fingerprint(rows)
    return freeze(rows, provenance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-dcase-only", action="store_true")
    parser.add_argument("--extract-fsd", action="store_true")
    parser.add_argument("--prepare-and-freeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not any((args.extract_dcase_only, args.extract_fsd, args.prepare_and_freeze)):
        raise SystemExit("Choose --extract-dcase-only, --extract-fsd, or --prepare-and-freeze")
    if args.extract_dcase_only:
        extract_dcase()
    if args.extract_fsd:
        extract_fsd()
    if args.prepare_and_freeze:
        prepare_and_freeze()


if __name__ == "__main__":
    main()
