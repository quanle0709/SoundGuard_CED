"""Stateful emergency decisions built on top of the existing audio pipeline."""

from __future__ import annotations

import datetime
import re
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable

LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
PRIORITY_LEVEL = {1: "LOW", 2: "LOW", 3: "MEDIUM", 4: "HIGH", 5: "CRITICAL"}
CONTEXTS = ("indoor", "outdoor", "neutral")


@dataclass(frozen=True)
class CategoryConfig:
    aliases: tuple[str, ...]
    message_vi: str
    threshold: float
    level: str
    cooldown_seconds: float
    priority: int
    context_threshold_adjustments: dict[str, float] = field(default_factory=dict)
    context_level_adjustments: dict[str, int] = field(default_factory=dict)


# Single source of truth for aliases, thresholds, severity, context, cooldowns,
# and stable category priority.
CATEGORY_CONFIG: dict[str, CategoryConfig] = {
    "help_request": CategoryConfig(
        ("giúp tôi", "cứu tôi", "cứu với", "gọi giúp", "gọi cứu hộ",
         "gọi công an", "gọi cấp cứu", "có ai không"),
        "Phát hiện yêu cầu trợ giúp", 0.0, "CRITICAL", 20.0, 0),
    "gunshot": CategoryConfig(("gunshot",), "Tiếng nổ nguy hiểm", 0.45,
                               "CRITICAL", 120.0, 1,
                               {"outdoor": -0.05}),
    "explosion": CategoryConfig(("explosion",), "Tiếng nổ nguy hiểm", 0.45,
                                 "CRITICAL", 120.0, 2,
                                 {"outdoor": -0.05}),
    "fire": CategoryConfig(("fire crackling", "fire"), "Có thể có tiếng lửa",
                            0.50, "HIGH", 90.0, 3,
                            {"indoor": -0.05, "outdoor": -0.05},
                            {"indoor": 1}),
    "smoke_alarm": CategoryConfig(("smoke alarm",), "Còi báo động", 0.50,
                                   "HIGH", 60.0, 4, {"indoor": -0.05},
                                   {"indoor": 1}),
    "glass_breaking": CategoryConfig(("glass breaking", "shatter"), "Kính vỡ",
                                      0.55, "HIGH", 60.0, 5,
                                      {"indoor": -0.05}),
    "screaming": CategoryConfig(("screaming", "shout"), "Tiếng la hét", 0.55,
                                 "HIGH", 45.0, 6,
                                 {"indoor": -0.05, "outdoor": -0.05}),
    "siren": CategoryConfig(("siren", "alarm"), "Còi báo động", 0.55,
                             "HIGH", 45.0, 7,
                             {"outdoor": -0.05, "indoor": 0.05},
                             {"indoor": -1}),
    "vehicle_horn": CategoryConfig(("vehicle horn", "car horn", "honk"),
                                    "Còi xe", 0.60, "MEDIUM", 20.0, 8,
                                    {"outdoor": -0.05, "indoor": 0.05},
                                    {"outdoor": 1, "indoor": -1}),
    "baby_crying": CategoryConfig(("baby cry", "crying"), "Tiếng khóc", 0.60,
                                   "MEDIUM", 30.0, 9, {"indoor": -0.05},
                                   {"indoor": 1}),
    "door_activity": CategoryConfig(("doorbell", "knock"), "Có người ở cửa",
                                     0.65, "LOW", 15.0, 10,
                                     {"indoor": -0.05, "outdoor": 0.05},
                                     {"indoor": 1}),
    "dog_barking": CategoryConfig(("bark", "dog"), "Tiếng chó sủa", 0.65,
                                   "LOW", 20.0, 11),
    "speech": CategoryConfig(("speech", "conversation"), "Speech", 0.70,
                             "LOW", 10.0, 12),
}

FIRE_EMERGENCY_PATTERNS = (
    re.compile(r"\bcháy(?:\s+\w+){0,3}\s+nhà\b"),
    re.compile(r"\bnhà(?:\s+\w+){0,3}\s+cháy\b"),
    re.compile(r"\b(?:có|đang)\s+cháy\b"),
    re.compile(r"\bcháy\s+rồi\b"),
    re.compile(r"\bcó\s+đám\s+cháy\b"),
    re.compile(r"\bgọi\s+(?:cứu\s+hỏa|chữa\s+cháy)\b"),
)
FIRE_EMERGENCY_MESSAGE = "Phát hiện tình huống cháy"


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text or "").strip().lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value)


def get_category_level(category: str, context: str = "neutral") -> str:
    config = CATEGORY_CONFIG[category]
    target = LEVEL_ORDER[config.level] + config.context_level_adjustments.get(context, 0)
    target = max(0, min(3, target))
    return next(level for level, value in LEVEL_ORDER.items() if value == target)


def get_category_threshold(category: str, context: str = "neutral") -> float:
    config = CATEGORY_CONFIG[category]
    value = config.threshold + config.context_threshold_adjustments.get(context, 0.0)
    return max(0.0, min(1.0, value))


def match_sound_category(label: str) -> str | None:
    normalized = normalize_text(label)
    for category, config in CATEGORY_CONFIG.items():
        if category != "help_request" and any(alias in normalized for alias in config.aliases):
            return category
    return None


def classify_help_request(transcript: str) -> tuple[bool, str]:
    normalized = normalize_text(transcript)
    if not normalized:
        return False, ""
    if any(pattern.search(normalized) for pattern in FIRE_EMERGENCY_PATTERNS):
        return True, FIRE_EMERGENCY_MESSAGE
    tokens = normalized.split()
    for phrase in CATEGORY_CONFIG["help_request"].aliases:
        phrase_tokens = normalize_text(phrase).split()
        width = len(phrase_tokens)
        if any(tokens[index:index + width] == phrase_tokens
               for index in range(len(tokens) - width + 1)):
            return True, CATEGORY_CONFIG["help_request"].message_vi
    return False, ""


def detect_help_request(transcript: str) -> bool:
    return classify_help_request(transcript)[0]


def evaluate_sound(label: str, confidence: float, context: str = "neutral") -> dict:
    if context not in CONTEXTS:
        raise ValueError(f"Unsupported context: {context}")
    category = match_sound_category(label)
    value = float(confidence)
    if category is None:
        return {"matched": False, "category": "unknown", "message_vi": "",
                "is_dangerous": False, "confidence": value, "threshold": None,
                "level": "LOW"}
    config = CATEGORY_CONFIG[category]
    threshold = get_category_threshold(category, context)
    level = get_category_level(category, context)
    if value < threshold:
        return {"matched": False, "category": "low_confidence",
                "candidate_category": category, "message_vi": "",
                "is_dangerous": False, "confidence": value,
                "threshold": threshold, "level": "LOW"}
    return {"matched": True, "category": category,
            "message_vi": config.message_vi,
            "is_dangerous": LEVEL_ORDER[level] >= LEVEL_ORDER["MEDIUM"],
            "confidence": value, "threshold": threshold, "level": level}


class EmergencySystem:
    def __init__(self, context: str = "neutral", decision_mode: str = "continuous",
                 help_end_grace_cycles: int = 2,
                 clock: Callable[[], float] = time.monotonic,
                 priority_provider: Callable[[str, str], int | None] | None = None):
        if context not in CONTEXTS:
            raise ValueError(f"Unsupported context: {context}")
        if decision_mode not in {"continuous", "single_shot"}:
            raise ValueError(f"Unsupported decision mode: {decision_mode}")
        if help_end_grace_cycles < 1:
            raise ValueError("help_end_grace_cycles must be at least 1")
        self.context = context
        self.decision_mode = decision_mode
        self.help_end_grace_cycles = help_end_grace_cycles
        self.clock = clock
        self.priority_provider = priority_provider
        self.history: deque[str | None] = deque(maxlen=3)
        self.active_sound: str | None = None
        self.help_active = False
        self.help_missing_cycles = 0
        self.active_help_message = CATEGORY_CONFIG["help_request"].message_vi
        self.last_emitted: dict[str, float] = {}

    def _get_level(self, category: str) -> str:
        """Use personalization only when an explicit provider was injected."""
        default = get_category_level(category, self.context)
        if self.priority_provider is None:
            return default
        try:
            priority = self.priority_provider(category, self.context)
        except Exception:
            return default
        if isinstance(priority, bool) or priority not in PRIORITY_LEVEL:
            return default
        return PRIORITY_LEVEL[priority]

    def _can_emit(self, category: str, now: float) -> bool:
        previous = self.last_emitted.get(category)
        if previous is not None and now - previous < CATEGORY_CONFIG[category].cooldown_seconds:
            return False
        self.last_emitted[category] = now
        return True

    def _confirmed_sound(self) -> str | None:
        counts = Counter(item for item in self.history if item is not None)
        candidates = [category for category, count in counts.items()
                      if count >= 2 and LEVEL_ORDER[self._get_level(category)]
                      >= LEVEL_ORDER["MEDIUM"]]
        if not candidates:
            return None
        return min(candidates, key=lambda category: (
            -LEVEL_ORDER[self._get_level(category)],
            -counts[category], CATEGORY_CONFIG[category].priority))

    def _base_result(self, timestamp: str | None) -> dict:
        return {
            "timestamp": timestamp or datetime.datetime.now().isoformat(timespec="seconds"),
            "context": self.context,
            "decision_mode": self.decision_mode,
            "history_window": list(self.history),
            "missing_cycles": sum(item is None for item in self.history),
            "help_missing_cycles": self.help_missing_cycles,
        }

    def process_sound_event(self, sound_label: str, sound_confidence: float,
                            timestamp: str | None = None) -> dict:
        """Advance only sound voting/lifecycle state for one CED window."""
        now = self.clock()
        mapped = evaluate_sound(sound_label, sound_confidence, self.context)
        observed = mapped["category"] if mapped["matched"] else None
        self.history.append(observed)
        confirmed = (
            self._confirmed_sound() if self.decision_mode == "continuous"
            else observed if observed and LEVEL_ORDER[
                self._get_level(observed)] >= LEVEL_ORDER["MEDIUM"]
            else None
        )
        previous = self.active_sound
        self.active_sound = confirmed
        if confirmed:
            state = "EVENT_CONTINUING" if previous == confirmed else "EVENT_STARTED"
            category = confirmed
        elif previous:
            state, category = "EVENT_ENDED", previous
        else:
            state, category = "NO_EVENT", None
        active = state in {"EVENT_STARTED", "EVENT_CONTINUING"}
        level = self._get_level(category) if category else "LOW"
        actionable = category is not None and LEVEL_ORDER[level] >= LEVEL_ORDER["MEDIUM"]
        emitted = bool(active and actionable and self._can_emit(category, now))
        config = CATEGORY_CONFIG.get(category) if category else None
        return {
            **self._base_result(timestamp),
            "evidence_type": "sound",
            "temporal_confirmation": bool(
                self.decision_mode == "continuous" and confirmed and active and actionable
            ),
            "category": category or "none",
            "alert_level": level,
            "event_state": state,
            "alert": emitted,
            "alert_text": config.message_vi if config and active and actionable else "",
            "emitted_this_cycle": emitted,
            "help_request_detected": False,
            "sound_label": sound_label,
            "sound_confidence": float(sound_confidence),
            "mapped_sound": mapped,
        }

    def process_transcript_event(self, transcript: str,
                                 timestamp: str | None = None) -> dict:
        """Advance only help-request lifecycle state for one accepted FINAL."""
        if not normalize_text(transcript):
            raise ValueError("Transcript events must contain an accepted non-empty FINAL.")
        now = self.clock()
        detected, message = classify_help_request(transcript)
        if detected:
            state = "EVENT_CONTINUING" if self.help_active else "EVENT_STARTED"
            self.help_active = True
            self.help_missing_cycles = 0
            self.active_help_message = message
        elif self.help_active:
            self.help_missing_cycles += 1
            if self.help_missing_cycles >= self.help_end_grace_cycles:
                self.help_active = False
                self.help_missing_cycles = 0
                state = "EVENT_ENDED"
            else:
                state = "EVENT_CONTINUING"
        else:
            state = "NO_EVENT"
        active = state in {"EVENT_STARTED", "EVENT_CONTINUING"}
        category = "help_request" if active or state == "EVENT_ENDED" else None
        emitted = bool(active and detected and self._can_emit("help_request", now))
        return {
            **self._base_result(timestamp),
            "evidence_type": "transcript",
            "temporal_confirmation": False,
            "category": category or "none",
            "alert_level": "CRITICAL" if category else "LOW",
            "event_state": state,
            "alert": emitted,
            "alert_text": self.active_help_message if active else "",
            "emitted_this_cycle": emitted,
            "help_request_detected": detected,
            "sound_label": "unknown",
            "sound_confidence": 0.0,
            "mapped_sound": evaluate_sound("unknown", 0.0, self.context),
        }

    def evaluate(self, transcript: str, sound_label: str, sound_confidence: float,
                 timestamp: str | None = None) -> dict:
        """Compatibility API for legacy fixed-cycle callers.

        New streaming code uses the evidence-specific APIs above. This method
        intentionally preserves the original combined-cycle behavior.
        """
        now = self.clock()
        mapped = evaluate_sound(sound_label, sound_confidence, self.context)
        observed = mapped["category"] if mapped["matched"] else None
        self.history.append(observed)
        help_detected, help_message = classify_help_request(transcript)

        if self.decision_mode == "continuous":
            confirmed = self._confirmed_sound()
        else:
            confirmed = observed if observed and LEVEL_ORDER[
                self._get_level(observed)] >= LEVEL_ORDER["MEDIUM"] else None
        previous_sound = self.active_sound
        self.active_sound = confirmed

        if help_detected:
            state = "EVENT_CONTINUING" if self.help_active else "EVENT_STARTED"
            self.help_active = True
            self.help_missing_cycles = 0
            self.active_help_message = help_message
            category = "help_request"
        elif self.help_active:
            self.help_missing_cycles += 1
            category = "help_request"
            if self.help_missing_cycles >= self.help_end_grace_cycles:
                self.help_active = False
                self.help_missing_cycles = 0
                state = "EVENT_ENDED"
            else:
                state = "EVENT_CONTINUING"
        elif confirmed:
            category = confirmed
            state = "EVENT_CONTINUING" if previous_sound == confirmed else "EVENT_STARTED"
        elif previous_sound:
            category = previous_sound
            state = "EVENT_ENDED"
        else:
            category = None
            state = "NO_EVENT"

        active = state in {"EVENT_STARTED", "EVENT_CONTINUING"}
        level = self._get_level(category) if category else "LOW"
        actionable = category is not None and LEVEL_ORDER[level] >= LEVEL_ORDER["MEDIUM"]
        emitted = bool(active and actionable and self._can_emit(category, now))
        config = CATEGORY_CONFIG.get(category) if category else None
        alert_message = (
            self.active_help_message
            if category == "help_request"
            else config.message_vi if config else ""
        )
        temporal = bool(
            self.decision_mode == "continuous"
            and confirmed is not None
            and category == confirmed
            and active
            and actionable
        )
        return {
            "timestamp": timestamp or datetime.datetime.now().isoformat(timespec="seconds"),
            "context": self.context,
            "decision_mode": self.decision_mode,
            "temporal_confirmation": temporal,
            "history_window": list(self.history),
            "missing_cycles": sum(item is None for item in self.history),
            "category": category or "none", "alert_level": level,
            "event_state": state, "alert": emitted,
            "alert_text": alert_message if active and actionable else "",
            "emitted_this_cycle": emitted,
            "help_request_detected": help_detected,
            "help_missing_cycles": self.help_missing_cycles,
            "sound_label": sound_label, "sound_confidence": float(sound_confidence),
            "mapped_sound": mapped,
        }
