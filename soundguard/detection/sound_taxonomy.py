"""Opt-in coarse SoundGuard taxonomy for fine-grained CED labels."""

from __future__ import annotations

import os


SEMANTIC_CED_ENV = "SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING"

# Exact and strong aliases accepted by the offline gate. Broad labels such as
# Animal, Vehicle, Breaking, and Crying are deliberately absent.
SEMANTIC_CED_ALIASES: dict[str, str] = {
    "Baby cry": "baby_crying", "Baby cry, infant cry": "baby_crying",
    "Bark": "dog_barking", "Bow-wow": "dog_barking", "Dog": "dog_barking",
    "Knock": "door_activity", "Fire": "fire", "Crackle": "fire",
    "Glass": "glass_breaking", "Shatter": "glass_breaking", "Siren": "siren",
    "Civil defense siren": "siren", "Police car (siren)": "siren",
    "Ambulance (siren)": "siren", "Emergency vehicle": "siren",
    "Vehicle horn": "vehicle_horn", "Toot": "vehicle_horn",
    "Honk": "vehicle_horn", "Air horn": "vehicle_horn",
}


def semantic_ced_enabled() -> bool:
    return os.environ.get(SEMANTIC_CED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def map_ced_label(raw_label: str) -> str | None:
    return SEMANTIC_CED_ALIASES.get(str(raw_label))
