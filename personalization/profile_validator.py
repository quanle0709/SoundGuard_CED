"""Deterministic safety policy applied after every generator."""

from __future__ import annotations

from typing import Any

from .schemas import UserProfile, parse_profile


# Current SoundGuard CED categories collapse related detector labels into a
# smaller runtime vocabulary. This mapping makes the requested safety policy
# explicit and keeps its enforcement independent from the LLM prompt.
UNIVERSAL_CRITICAL_INPUTS = {
    "smoke_alarm": "smoke_alarm",
    "fire_alarm": "smoke_alarm",
    "carbon_monoxide_alarm": "smoke_alarm",
    "emergency_evacuation_alarm": "siren",
    "emergency_alarm": "siren",
    "emergency_siren": "siren",
    "severe_crash": "explosion",
    "impact": "explosion",
    "gunshot": "gunshot",
    "fire": "fire",
}
UNIVERSAL_MIN_PRIORITY = {
    runtime_label: 5 for runtime_label in set(UNIVERSAL_CRITICAL_INPUTS.values())
}


def validate_profile(payload: Any) -> UserProfile:
    parsed = parse_profile(payload)
    priorities = dict(parsed.priority_profile)
    reasons = dict(parsed.reasoning_summary)
    for label, minimum in UNIVERSAL_MIN_PRIORITY.items():
        if priorities.get(label, 1) < minimum:
            priorities[label] = minimum
            reasons.setdefault(label, "Universal life-safety minimum enforced by SoundGuard.")
    return UserProfile(
        profile_version=parsed.profile_version,
        roles=parsed.roles,
        contexts=parsed.contexts,
        responsibilities=parsed.responsibilities,
        priority_profile=priorities,
        reasoning_summary=reasons,
    )
