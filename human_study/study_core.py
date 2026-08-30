"""Protocol-locked planning, scoring, and data contracts for Layer 3."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = Path(__file__).resolve().parent
CONFIG_PATH = STUDY_DIR / "study_config.json"
MANIFEST_PATH = STUDY_DIR / "stimuli_manifest.csv"
PARTICIPANT_DATA_DIR = STUDY_DIR / "participant_data"
STUDY_VERSION = "L3_PROTOCOL_V1"
BLOCKS = ("speech", "environmental", "mixed", "critical")
CONDITIONS = ("WITHOUT", "WITH")

TRIAL_FIELDS = (
    "participant_id", "study_version", "pilot", "block", "condition",
    "condition_position", "trial_id", "trial_index", "stimulus_id",
    "stimulus_set", "ground_truth", "environment_ground_truth", "response", "speech_response",
    "environment_response", "correct", "speech_correct",
    "environment_correct", "both_correct", "missed", "false_perceived_event",
    "stimulus_start_wall", "stimulus_start_monotonic", "hud_timestamp",
    "hud_timestamp_source", "response_timestamp_wall",
    "response_timestamp_monotonic", "rt_from_stimulus_ms", "rt_from_hud_ms",
    "hud_observation_timestamp_monotonic", "speech_response_timestamp_monotonic",
    "environment_response_timestamp_monotonic", "speech_rt_from_stimulus_ms",
    "environment_rt_from_stimulus_ms",
    "displayed_ced", "displayed_alert", "subtitle_state", "coexistence_state",
    "stt_attempted", "stt_success", "stt_error_type", "stt_latency_ms",
    "network_failure", "trial_valid", "invalid_reason", "playback_success",
    "hardware_connected", "error_class", "randomization_seed",
)

QUESTIONNAIRE_FIELDS = (
    "participant_id", "study_version", "pilot", "instrument", "item_id",
    "prompt", "response", "scored_value", "recorded_at",
)

SUS_ITEMS = (
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
)

LIKERT_ITEMS = (
    "The subtitles were easy to read.",
    "The environmental sound labels were easy to understand.",
    "The safety alerts were noticeable.",
    "SoundGuard improved my awareness of environmental sounds.",
    "SoundGuard was distracting.",
    "SoundGuard felt useful.",
    "I would consider using a similar device in daily life.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stimulus_asset_paths(rows: Iterable[dict[str, str]] | None = None) -> list[Path]:
    rows = list(rows) if rows is not None else load_manifest()
    paths = {
        ROOT / value
        for row in rows
        for field in ("source_path", "secondary_source_path")
        if (value := row.get(field, "").strip())
    }
    paths.add(STUDY_DIR / "stimuli" / "RECORDING_PROVENANCE.md")
    return sorted(paths, key=lambda item: str(item).lower())


def stimulus_asset_bundle_sha256(rows: Iterable[dict[str, str]] | None = None) -> str:
    digest = hashlib.sha256()
    for path in stimulus_asset_paths(rows):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def tracked_worktree_clean() -> bool:
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "HEAD", "--"], cwd=ROOT,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        return unstaged == 0 and staged == 0
    except OSError:
        return False


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("study_version") != STUDY_VERSION:
        raise ValueError(
            f"Configuration version {config.get('study_version')!r} does not "
            f"match tooling version {STUDY_VERSION!r}."
        )
    return config


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_manifest(rows)
    return rows


def validate_manifest(rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    required = {
        "stimulus_id", "block", "type", "ground_truth", "duration_seconds",
        "stimulus_set", "source_path", "source", "license_provenance",
        "playback_level", "question", "choices", "correct_response", "notes",
    }
    if not rows:
        raise ValueError("Stimulus manifest is empty.")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Stimulus manifest lacks columns: {sorted(missing)}")
    ids = [row["stimulus_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Stimulus manifest contains duplicate stimulus IDs.")
    for row in rows:
        if row["block"] not in BLOCKS:
            raise ValueError(f"Invalid block for {row['stimulus_id']}: {row['block']}")
        if row["stimulus_set"] not in {"A", "B"}:
            raise ValueError(f"Invalid set for {row['stimulus_id']}")
        if row["correct_response"] not in row["choices"].split("|"):
            raise ValueError(f"Correct response is not a choice: {row['stimulus_id']}")
    expected = {"speech": 6, "environmental": 6, "mixed": 4, "critical": 3}
    for block, per_set in expected.items():
        for stimulus_set in ("A", "B"):
            count = sum(
                row["block"] == block and row["stimulus_set"] == stimulus_set
                for row in rows
            )
            if count != per_set:
                raise ValueError(
                    f"Expected {per_set} {block} stimuli in set {stimulus_set}; got {count}."
                )


def validate_participant_id(value: str, *, pilot: bool = False) -> str:
    participant_id = value.strip().upper()
    if pilot:
        if not re.fullmatch(r"PILOT\d{2}", participant_id):
            raise ValueError("Pilot IDs must be PILOT01, PILOT02, ...")
    elif not re.fullmatch(r"P(?:0[1-9]|10)", participant_id):
        raise ValueError("Main-study ID must be P01 through P10.")
    return participant_id


def participant_number(participant_id: str) -> int:
    match = re.search(r"(\d+)$", participant_id)
    if not match:
        raise ValueError(f"Participant ID has no numeric suffix: {participant_id}")
    return int(match.group(1))


def condition_order(participant_id: str) -> tuple[str, str]:
    number = participant_number(participant_id)
    return CONDITIONS if number % 2 else tuple(reversed(CONDITIONS))


def condition_set_map(participant_id: str) -> dict[str, str]:
    """Four-sequence balance of condition order and stimulus set.

    P01: WITHOUT/A -> WITH/B; P02: WITH/A -> WITHOUT/B;
    P03: WITHOUT/B -> WITH/A; P04: WITH/B -> WITHOUT/A, then repeat.
    """
    number = participant_number(participant_id)
    order = condition_order(participant_id)
    first_set = "A" if (number - 1) % 4 < 2 else "B"
    second_set = "B" if first_set == "A" else "A"
    return {order[0]: first_set, order[1]: second_set}


def derive_seed(participant_id: str, base_seed: int, study_version: str = STUDY_VERSION) -> int:
    payload = f"{study_version}|{participant_id}|{int(base_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class PlannedTrial:
    condition: str
    condition_position: int
    block: str
    trial_index: int
    row: dict[str, str]

    @property
    def trial_id(self) -> str:
        return f"{self.condition_position}-{self.block}-{self.trial_index:02d}"


def build_trial_plan(
    participant_id: str,
    base_seed: int,
    rows: list[dict[str, str]] | None = None,
) -> list[PlannedTrial]:
    rows = rows or load_manifest()
    order = condition_order(participant_id)
    set_map = condition_set_map(participant_id)
    seed = derive_seed(participant_id, base_seed)
    plan: list[PlannedTrial] = []
    for condition_position, condition in enumerate(order, start=1):
        stimulus_set = set_map[condition]
        for block_index, block in enumerate(BLOCKS):
            candidates = [
                row for row in rows
                if row["block"] == block and row["stimulus_set"] == stimulus_set
            ]
            rng = random.Random(seed + condition_position * 1000 + block_index)
            rng.shuffle(candidates)
            for trial_index, row in enumerate(candidates, start=1):
                plan.append(PlannedTrial(
                    condition, condition_position, block, trial_index, row
                ))
    return plan


def calculate_rt_ms(start_monotonic: float | None, end_monotonic: float | None) -> str:
    if start_monotonic is None or end_monotonic is None:
        return ""
    if end_monotonic < start_monotonic:
        raise ValueError("Response timestamp precedes start timestamp.")
    return f"{(end_monotonic - start_monotonic) * 1000.0:.3f}"


def score_sus(responses: Iterable[int]) -> float:
    values = [int(value) for value in responses]
    if len(values) != 10 or any(value < 1 or value > 5 for value in values):
        raise ValueError("SUS requires exactly ten responses from 1 to 5.")
    contributions = [
        value - 1 if index % 2 == 0 else 5 - value
        for index, value in enumerate(values)
    ]
    return sum(contributions) * 2.5


def protocol_lock(base_seed: int) -> dict:
    return {
        "study_version": STUDY_VERSION,
        "git_commit": git_commit(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "base_randomization_seed": int(base_seed),
        "tracked_worktree_clean": tracked_worktree_clean(),
    }


def session_directory(participant_id: str, pilot: bool) -> Path:
    parent = PARTICIPANT_DATA_DIR / ("pilot" if pilot else "main")
    return parent / participant_id
