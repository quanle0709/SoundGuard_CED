import datetime
from soundguard.emergency.emergency_system import detect_help_request

def fuse_result(timestamp, transcript, sound_result, alert_result):
    transcript_text = transcript or ""
    help_request_detected = detect_help_request(transcript_text)

    sound_label = sound_result.get("label", "") if isinstance(sound_result, dict) else ""
    sound_confidence = float(sound_result.get("confidence", 0.0)) if isinstance(sound_result, dict) else 0.0

    if not isinstance(alert_result, dict):
        alert_result = {}

    category = str(alert_result.get("category", "")).strip().lower()
    is_dangerous = bool(alert_result.get("is_dangerous", False))
    if sound_confidence < 0.45:
        is_dangerous = False

    if help_request_detected:
        alert_level = "CRITICAL"
    elif category in {"bark", "dog", "speech", "conversation"}:
        alert_level = "LOW"
    elif is_dangerous:
        alert_level = "HIGH"
    else:
        alert_level = "LOW"

    if alert_level == "LOW":
        alert = False
        alert_text = ""
    else:
        alert = True
        alert_text = alert_result.get("message_vi", "") or ""

    return {
        "timestamp": timestamp or datetime.datetime.now().isoformat(timespec="seconds"),
        "transcript": transcript_text,
        "sound_label": sound_label,
        "sound_confidence": sound_confidence,
        "sound_category": alert_result.get("category", "unknown"),
        "alert_level": alert_level,
        "alert": alert,
        "alert_text": alert_text,
        "help_request_detected": help_request_detected,
    }
