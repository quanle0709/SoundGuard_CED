from emergency_system import evaluate_sound


DEFAULT_THRESHOLD = 0.45


def map_alert(
    label: str,
    confidence: float,
    threshold: float | None = DEFAULT_THRESHOLD,
    context: str = "neutral",
):
    """Map a CED result using the centralized category configuration.

    The optional legacy threshold remains supported as an extra minimum.
    Normal application calls use the configured per-category threshold.
    """
    result = evaluate_sound(label, confidence, context)
    if threshold is not None and float(confidence) < threshold:
        return {
            **result,
            "matched": False,
            "category": "low_confidence",
            "message_vi": "",
            "is_dangerous": False,
        }
    return result
