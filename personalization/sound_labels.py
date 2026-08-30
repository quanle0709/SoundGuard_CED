"""Single mapping from user/CED labels to SoundGuard's supported categories."""

from __future__ import annotations

import re


# These are deliberately aligned with emergency_system.CATEGORY_CONFIG.  The
# personalization runtime never invents a category the frozen alert core cannot
# process.
SUPPORTED_SOUND_LABELS = frozenset({
    "gunshot",
    "explosion",
    "fire",
    "smoke_alarm",
    "glass_breaking",
    "screaming",
    "siren",
    "vehicle_horn",
    "baby_crying",
    "door_activity",
    "dog_barking",
    "speech",
})

LABEL_ALIASES: dict[str, str] = {
    "gun shot": "gunshot",
    "crash": "explosion",
    "impact": "explosion",
    "severe crash": "explosion",
    "collision": "explosion",
    "fire alarm": "smoke_alarm",
    "carbon monoxide alarm": "smoke_alarm",
    "co alarm": "smoke_alarm",
    "emergency alarm": "siren",
    "evacuation alarm": "siren",
    "emergency siren": "siren",
    "car horn": "vehicle_horn",
    "horn": "vehicle_horn",
    "honk": "vehicle_horn",
    "baby cry": "baby_crying",
    "baby crying": "baby_crying",
    "infant crying": "baby_crying",
    "glass breaking": "glass_breaking",
    "shatter": "glass_breaking",
    "scream": "screaming",
    "shouting": "screaming",
    "shouted warning": "screaming",
    "doorbell": "door_activity",
    "door knock": "door_activity",
    "knock": "door_activity",
    "dog bark": "dog_barking",
    "bark": "dog_barking",
    "conversation": "speech",
    "name call": "speech",
    "public announcement": "speech",
}


def normalize_label(label: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", str(label).strip().lower())
    return re.sub(r"\s+", " ", value).strip()


def canonicalize_label(label: str) -> str | None:
    """Return a supported SoundGuard category, or None for arbitrary labels."""
    normalized = normalize_label(label)
    underscored = normalized.replace(" ", "_")
    if underscored in SUPPORTED_SOUND_LABELS:
        return underscored
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]
    # CED labels can contain descriptive suffixes. Prefer the longest alias so
    # "emergency siren" wins over "siren"-like shorter descriptions.
    for alias in sorted(LABEL_ALIASES, key=len, reverse=True):
        if alias in normalized:
            return LABEL_ALIASES[alias]
    return None
