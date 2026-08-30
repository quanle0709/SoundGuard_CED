"""Calibrate and validate the presentation-only SoundGuard HUD policy.

Run calibration and validation as separate commands. Validation refuses to run
until a locked calibration policy exists, preventing held-out tuning.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa
import numpy as np

from audio_pipeline import classify_raw_audio
from hud_awareness import SUPPORTED_HUD_LABELS, evaluate_ced_for_hud


ESC_ROOT = ROOT / "benchmark_data" / "external" / "esc50"
VIVOS_ROOT = ROOT / "benchmark_data" / "external" / "vivos" / "selected"
FSD_AUDIO_ROOT = (ROOT / "benchmark_data" / "external" /
                  "computer_holdout_audio" / "FSD50K.eval_audio")
FSD_GROUND_TRUTH = (ROOT / "benchmark_data" / "external" /
                    "emergency_v3_final" / "FSD50K.ground_truth.zip")
OUTPUT_ROOT = ROOT / "benchmark_results" / "hud_awareness_policy_v2"
POLICY_PATH = OUTPUT_ROOT / "locked_policy.json"

CALIBRATION_FOLDS = {1, 2, 3}
VALIDATION_FOLDS = {4, 5}
CALIBRATION_SPEAKERS = {f"VIVOSDEV{index:02d}" for index in range(1, 13)}
VALIDATION_SPEAKERS = {f"VIVOSDEV{index:02d}" for index in range(13, 20)}
MIXTURE_LEVELS_DB = (-6.0, 0.0, 6.0)  # environmental level relative to speech
THRESHOLD_CANDIDATES = tuple(round(0.40 + index * 0.005, 3) for index in range(81))

TARGETS = {
    "dog": ("DOG", "ANIMAL"),
    "car_horn": ("HORN",),
    "door_wood_knock": ("DOOR",),
    "crying_baby": ("BABY",),
}
MIXTURE_TARGETS = ("dog", "car_horn", "door_wood_knock", "crying_baby")
BACKGROUND_LABEL_TERMS = (
    "Rain", "Wind", "Engine", "Vacuum_cleaner", "Washing_machine",
    "Clock", "Computer_keyboard", "Typing", "Air_conditioning",
    "Mechanical_fan", "Traffic_noise", "Water",
)
def select_prediction(top_predictions: list[dict], threshold: float) -> str:
    eligible = []
    for prediction in top_predictions:
        score = float(prediction.get("score", 0.0))
        evaluation = evaluate_ced_for_hud(
            str(prediction.get("label", "")), score, threshold=threshold
        )
        if evaluation["accepted"]:
            eligible.append((score, evaluation["display_label"]))
    return max(eligible, default=(0.0, ""))[1]


def fit_window(path: Path, samples: int = 80000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if not audio.size:
        raise ValueError(f"Empty calibration audio: {path}")
    return np.resize(audio, samples)


def unit_rms(audio: np.ndarray) -> np.ndarray:
    value = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return audio / max(value, 1e-8)


def mix_audio(environment: np.ndarray, speech: np.ndarray,
              environmental_db: float) -> np.ndarray:
    ratio = 10.0 ** (environmental_db / 20.0)
    mixed = unit_rms(speech) + ratio * unit_rms(environment)
    peak = float(np.max(np.abs(mixed)))
    return np.asarray(mixed / max(peak, 1e-8), dtype=np.float32)


def esc_rows() -> list[dict]:
    with (ESC_ROOT / "esc50.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def select_esc_files(categories: set[str], folds: set[int], per_fold: int = 2,
                     offset: int = 0):
    grouped: dict[tuple[str, int], list[Path]] = {}
    for row in esc_rows():
        category, fold = row["category"], int(row["fold"])
        path = ESC_ROOT / "audio" / row["filename"]
        if category in categories and fold in folds and path.exists():
            grouped.setdefault((category, fold), []).append(path)
    selected = []
    for (category, fold), paths in sorted(grouped.items()):
        for path in sorted(paths)[offset:offset + per_fold]:
            selected.append((category, fold, path))
    return selected


def select_speech_files(speakers: set[str], offset: int = 0) -> list[Path]:
    selected = []
    for speaker in sorted(speakers):
        paths = sorted((VIVOS_ROOT / speaker).glob("*.wav"))
        if len(paths) > offset:
            selected.append(paths[offset])
    return selected


def select_fsd_backgrounds(phase: str, limit: int = 30,
                           offset: int = 0) -> list[tuple[str, Path]]:
    with zipfile.ZipFile(FSD_GROUND_TRUTH) as bundle:
        text = bundle.read("FSD50K.ground_truth/eval.csv").decode("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    eligible = []
    for row in sorted(rows, key=lambda item: int(item["fname"])):
        identifier = int(row["fname"])
        split = identifier % 10
        if phase == "calibration" and split >= 6:
            continue
        if phase == "validation" and split < 6:
            continue
        labels = row["labels"].split(",")
        if not any(term in labels for term in BACKGROUND_LABEL_TERMS):
            continue
        joined = " ".join(labels).lower()
        if any(term in joined for term in (
                "dog", "bark", "animal", "horn", "door", "baby", "cry",
                "speech", "conversation", "siren", "fire", "gunshot",
                "explosion", "scream", "glass")):
            continue
        path = FSD_AUDIO_ROOT / f"{identifier}.wav"
        if path.exists():
            eligible.append((row["labels"], path))
        if len(eligible) >= offset + limit:
            break
    selected = eligible[offset:offset + limit]
    if len(selected) < limit:
        raise RuntimeError(f"Only found {len(selected)} FSD50K {phase} negatives")
    return selected


def infer_record(identifier: str, condition: str, expected: tuple[str, ...],
                 audio: np.ndarray, **metadata) -> dict:
    started = datetime.datetime.now(datetime.timezone.utc)
    monotonic_started = __import__("time").perf_counter()
    result = classify_raw_audio(audio, 16000)
    duration = __import__("time").perf_counter() - monotonic_started
    return {
        "id": identifier,
        "condition": condition,
        "expected": list(expected),
        "inference_started_utc": started.isoformat(),
        "inference_seconds": duration,
        "top_predictions": result.get("top_predictions", []),
        **metadata,
    }


def build_records(phase: str) -> list[dict]:
    calibration = phase == "calibration"
    folds = CALIBRATION_FOLDS if calibration else VALIDATION_FOLDS
    speakers = CALIBRATION_SPEAKERS if calibration else VALIDATION_SPEAKERS
    records = []

    source_offset = 0 if calibration else 2
    positives = select_esc_files(set(TARGETS), folds, offset=source_offset)
    for category, fold, path in positives:
        records.append(infer_record(
            f"{phase}:environment:{path.stem}", "environment_only",
            TARGETS[category], fit_window(path), category=category, fold=fold,
            environmental_source=str(path.relative_to(ROOT)),
        ))

    speech_files = select_speech_files(speakers, offset=0 if calibration else 1)
    for path in speech_files:
        records.append(infer_record(
            f"{phase}:speech:{path.stem}", "speech_only", (), fit_window(path),
            speech_source=str(path.relative_to(ROOT)),
        ))

    backgrounds = select_fsd_backgrounds(
        phase, offset=0 if calibration else 30
    )
    for labels, path in backgrounds:
        records.append(infer_record(
            f"{phase}:background:{path.stem}", "background_only", (),
            fit_window(path), source_labels=labels,
            environmental_source=str(path.relative_to(ROOT)),
        ))

    rng = np.random.default_rng(7301 if calibration else 7302)
    records.append(infer_record(
        f"{phase}:background:silence", "background_only", (),
        np.zeros(80000, dtype=np.float32), synthetic="silence",
    ))
    records.append(infer_record(
        f"{phase}:background:static", "background_only", (),
        rng.normal(0.0, 0.05, 80000).astype(np.float32), synthetic="white_noise",
    ))

    mixture_environment = select_esc_files(
        set(MIXTURE_TARGETS), folds, per_fold=1,
        offset=0 if calibration else 4,
    )
    for index, (category, fold, path) in enumerate(mixture_environment):
        speech_path = speech_files[index % len(speech_files)]
        environment = fit_window(path)
        speech = fit_window(speech_path)
        for level in MIXTURE_LEVELS_DB:
            records.append(infer_record(
                f"{phase}:mixed:{path.stem}:{speech_path.stem}:{level:+.0f}dB",
                "mixed", TARGETS[category],
                mix_audio(environment, speech, level), category=category,
                fold=fold, environmental_db=level,
                environmental_source=str(path.relative_to(ROOT)),
                speech_source=str(speech_path.relative_to(ROOT)),
            ))
    return records


def metrics(records: list[dict], threshold: float) -> dict:
    tp = fp = fn = 0
    for record in records:
        selected = select_prediction(record["top_predictions"], threshold)
        record.setdefault("selection_by_threshold", {})[str(threshold)] = selected
        expected = set(record["expected"])
        if selected and selected in expected:
            tp += 1
        elif selected:
            fp += 1
            if expected:
                fn += 1
        elif expected:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def summarize(records: list[dict], threshold: float) -> dict:
    result = metrics(records, threshold)
    by_condition = {}
    for condition in sorted({record["condition"] for record in records}):
        subset = [record for record in records if record["condition"] == condition]
        by_condition[condition] = metrics(subset, threshold)
    result["by_condition"] = by_condition
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_calibration() -> None:
    records = build_records("calibration")
    candidates = [metrics(records, threshold) for threshold in THRESHOLD_CANDIDATES]
    chosen = max(candidates, key=lambda item: (item["f1"], item["precision"],
                                                item["threshold"]))
    locked = {
        "policy": "SoundGuard noncritical HUD awareness v2",
        "locked_before_validation": True,
        "metric": "sample-level micro F1; wrong labels count as FP+FN",
        "threshold_scope": "single global threshold across supported HUD labels",
        "threshold": chosen["threshold"],
        "calibration_folds": sorted(CALIBRATION_FOLDS),
        "calibration_speakers": sorted(CALIBRATION_SPEAKERS),
        "mixture_environment_db_relative_to_speech": list(MIXTURE_LEVELS_DB),
        "supported_labels": sorted(SUPPORTED_HUD_LABELS),
        "calibration_summary": summarize(records, chosen["threshold"]),
    }
    write_json(OUTPUT_ROOT / "calibration_records.json", records)
    write_json(OUTPUT_ROOT / "calibration_threshold_sweep.json", candidates)
    write_json(POLICY_PATH, locked)
    print(json.dumps(locked, indent=2))


def run_validation() -> None:
    if not POLICY_PATH.exists():
        raise RuntimeError("Run calibration first; locked_policy.json is required")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not policy.get("locked_before_validation"):
        raise RuntimeError("Refusing validation with an unlocked policy")
    threshold = float(policy["threshold"])
    records = build_records("validation")
    for record in records:
        record["selected_hud_label"] = select_prediction(
            record["top_predictions"], threshold
        )
    output = {
        "policy_sha256": __import__("hashlib").sha256(
            POLICY_PATH.read_bytes()
        ).hexdigest(),
        "threshold": threshold,
        "validation_folds": sorted(VALIDATION_FOLDS),
        "validation_speakers": sorted(VALIDATION_SPEAKERS),
        "summary": summarize(records, threshold),
        "records": records,
    }
    write_json(OUTPUT_ROOT / "heldout_validation.json", output)
    print(json.dumps({key: output[key] for key in output if key != "records"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibration", "validation"))
    args = parser.parse_args()
    if args.phase == "calibration":
        run_calibration()
    else:
        run_validation()


if __name__ == "__main__":
    main()
