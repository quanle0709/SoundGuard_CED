"""Strict V1 profile schema without adding a new runtime dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .sound_labels import SUPPORTED_SOUND_LABELS, canonicalize_label


PROFILE_VERSION = "1.0"
VALID_ROLES = frozenset({
    "driver",
    "motorcyclist_cyclist",
    "pedestrian_commuter",
    "construction_worker",
    "factory_warehouse_worker",
    "parent_infant_caregiver",
    "pregnant_expectant_mother",
    "older_adult_independent",
    "student",
    "healthcare_care_worker",
})
VALID_CONTEXTS = frozenset({
    "road", "public_transport", "school", "home", "construction_site",
    "factory_warehouse", "healthcare", "outdoors", "workplace",
})
ALLOWED_FIELDS = frozenset({
    "profile_version", "roles", "contexts", "responsibilities",
    "priority_profile", "reasoning_summary",
})


class ProfileValidationError(ValueError):
    pass


@dataclass(frozen=True)
class UserProfile:
    profile_version: str
    roles: tuple[str, ...]
    contexts: tuple[str, ...]
    responsibilities: tuple[str, ...]
    priority_profile: dict[str, int]
    reasoning_summary: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["roles"] = list(self.roles)
        value["contexts"] = list(self.contexts)
        value["responsibilities"] = list(self.responsibilities)
        return value


def _string_list(value: Any, field: str, allowed: frozenset[str] | None = None) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProfileValidationError(f"{field} must be a list of strings")
    cleaned = tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
    if allowed is not None:
        invalid = sorted(set(cleaned) - allowed)
        if invalid:
            raise ProfileValidationError(f"invalid {field}: {', '.join(invalid)}")
    return cleaned


def parse_profile(payload: Any) -> UserProfile:
    """Reject malformed output, unknown keys, conflicts, and unsupported labels."""
    if not isinstance(payload, dict):
        raise ProfileValidationError("profile must be a JSON object")
    unknown = set(payload) - ALLOWED_FIELDS
    missing = ALLOWED_FIELDS - set(payload)
    if unknown:
        raise ProfileValidationError(f"unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ProfileValidationError(f"missing fields: {', '.join(sorted(missing))}")
    if payload["profile_version"] != PROFILE_VERSION:
        raise ProfileValidationError(f"profile_version must be {PROFILE_VERSION}")

    priorities = payload["priority_profile"]
    if not isinstance(priorities, dict) or not priorities:
        raise ProfileValidationError("priority_profile must be a non-empty object")
    validated_priorities: dict[str, int] = {}
    source_for: dict[str, str] = {}
    for raw_label, raw_priority in priorities.items():
        if not isinstance(raw_label, str):
            raise ProfileValidationError("priority labels must be strings")
        label = canonicalize_label(raw_label)
        if label not in SUPPORTED_SOUND_LABELS:
            raise ProfileValidationError(f"unsupported sound label: {raw_label}")
        if isinstance(raw_priority, bool) or not isinstance(raw_priority, int) or not 1 <= raw_priority <= 5:
            raise ProfileValidationError(f"priority for {raw_label} must be an integer from 1 to 5")
        if label in validated_priorities:
            qualifier = "conflicting" if validated_priorities[label] != raw_priority else "duplicated"
            raise ProfileValidationError(
                f"{qualifier} priorities for {source_for[label]} and {raw_label}"
            )
        validated_priorities[label] = raw_priority
        source_for[label] = raw_label

    reasoning = payload["reasoning_summary"]
    if not isinstance(reasoning, dict):
        raise ProfileValidationError("reasoning_summary must be an object")
    validated_reasons: dict[str, str] = {}
    for raw_label, reason in reasoning.items():
        label = canonicalize_label(raw_label)
        if label not in validated_priorities:
            raise ProfileValidationError(f"reasoning label is not prioritized: {raw_label}")
        if not isinstance(reason, str) or not reason.strip():
            raise ProfileValidationError(f"reason for {raw_label} must be text")
        validated_reasons[label] = reason.strip()[:300]

    return UserProfile(
        profile_version=PROFILE_VERSION,
        roles=_string_list(payload["roles"], "roles", VALID_ROLES),
        contexts=_string_list(payload["contexts"], "contexts", VALID_CONTEXTS),
        responsibilities=_string_list(payload["responsibilities"], "responsibilities"),
        priority_profile=validated_priorities,
        reasoning_summary=validated_reasons,
    )
