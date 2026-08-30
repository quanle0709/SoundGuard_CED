"""Presentation-only acceptance policy for Screen 2 environmental awareness."""

from __future__ import annotations

import unicodedata

from display_transport import get_ced_display_label
from emergency_system import get_category_level, match_sound_category


# Locked by benchmark/calibrate_hud_awareness.py before held-out validation.
# This threshold is independent of every EmergencySystem threshold.
HUD_AWARENESS_THRESHOLD = 0.52

HUD_CATEGORY_LABELS = {
    "dog_barking": "DOG",
    "vehicle_horn": "HORN",
    "door_activity": "DOOR",
    "baby_crying": "BABY",
}
SUPPORTED_HUD_LABELS = frozenset({"DOG", "ANIMAL", "HORN", "DOOR", "BABY"})
ANIMAL_ONTOLOGY_LABELS = frozenset({
    "animal", "domestic animals pets", "pets", "wild animals",
})
SPEECH_ONLY_TERMS = (
    "speech", "conversation", "narration", "monologue", "mantra",
    "synthetic speech", "speech synthesizer", "whispering", "babbling",
    "chatter", "hubbub",
)


def normalize_ontology_label(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(label or ""))
    ascii_label = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(
        "".join(character if character.isalnum() else " "
                for character in ascii_label).split()
    )


def is_speech_only_ced_label(label: str) -> bool:
    """Suppress speech ontology descendants without suppressing screams."""
    normalized = normalize_ontology_label(label)
    if match_sound_category(label) == "speech":
        return True
    return any(term in normalized for term in SPEECH_ONLY_TERMS)


def _resolve_hud_label(label: str) -> tuple[str, str | None]:
    if is_speech_only_ced_label(label):
        return "", "speech"
    category = match_sound_category(label)
    if category in HUD_CATEGORY_LABELS:
        return HUD_CATEGORY_LABELS[category], category
    if normalize_ontology_label(label) in ANIMAL_ONTOLOGY_LABELS:
        return "ANIMAL", "animal"
    display_label = get_ced_display_label(label)
    if display_label in SUPPORTED_HUD_LABELS:
        return display_label, category
    return "", category


def evaluate_ced_for_hud(label: str, score: float,
                         context: str = "neutral", *,
                         threshold: float = HUD_AWARENESS_THRESHOLD) -> dict:
    """Evaluate one model candidate for noncritical Screen 2 presentation."""
    value = float(score)
    display_label, category = _resolve_hud_label(label)
    if category == "speech":
        return {
            "accepted": False, "reason": "speech_suppressed",
            "label": label, "score": value, "display_label": "",
            "category": category, "threshold": None,
        }
    if not display_label:
        return {
            "accepted": False, "reason": "unsupported_or_alert_only",
            "label": label, "score": value, "display_label": "",
            "category": category, "threshold": None,
        }
    if category in HUD_CATEGORY_LABELS and get_category_level(category, context) in {
            "HIGH", "CRITICAL"}:
        return {
            "accepted": False, "reason": "alert_priority",
            "label": label, "score": value, "display_label": display_label,
            "category": category, "threshold": threshold,
        }
    accepted = value >= threshold
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "below_hud_threshold",
        "label": label,
        "score": value,
        "display_label": display_label,
        "category": category,
        "threshold": threshold,
    }
