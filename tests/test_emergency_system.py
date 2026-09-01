from soundguard.emergency.alert_mapper import map_alert
from soundguard.emergency.emergency_system import (
    CATEGORY_CONFIG,
    EmergencySystem,
    detect_help_request,
    evaluate_sound,
)


class FakeClock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def cycle(system, label="unknown", confidence=0.0, transcript=""):
    return system.evaluate(transcript, label, confidence, timestamp="test")


def test_history_and_voting():
    system = EmergencySystem(clock=FakeClock())
    first = cycle(system, "Siren", 0.90)
    assert first["event_state"] == "NO_EVENT"
    assert first["history_window"] == ["siren"]
    second = cycle(system)
    assert second["history_window"] == ["siren", None]
    assert second["missing_cycles"] == 1
    third = cycle(system, "Siren", 0.90)
    assert third["event_state"] == "EVENT_STARTED"
    assert third["temporal_confirmation"] is True

    # Old evidence expires: detections separated by empty cycles cannot combine.
    isolated = EmergencySystem(clock=FakeClock())
    cycle(isolated, "Siren", 0.90)
    cycle(isolated)
    cycle(isolated)
    result = cycle(isolated, "Siren", 0.90)
    assert result["history_window"] == [None, None, "siren"]
    assert result["event_state"] == "NO_EVENT"


def test_temporal_confirmation_requires_current_quorum():
    silent = EmergencySystem(clock=FakeClock())
    cycle(silent)
    cycle(silent)
    no_event = cycle(silent)
    assert no_event["history_window"] == [None, None, None]
    assert no_event["event_state"] == "NO_EVENT"
    assert no_event["temporal_confirmation"] is False

    one_detection = EmergencySystem(clock=FakeClock())
    cycle(one_detection, "Siren", 0.90)
    cycle(one_detection)
    result = cycle(one_detection)
    assert result["history_window"] == ["siren", None, None]
    assert result["temporal_confirmation"] is False

    confirmed = EmergencySystem(clock=FakeClock())
    cycle(confirmed, "Siren", 0.90)
    cycle(confirmed)
    result = cycle(confirmed, "Siren", 0.90)
    assert result["history_window"] == ["siren", None, "siren"]
    assert result["temporal_confirmation"] is True
    ended = cycle(confirmed)
    assert ended["event_state"] == "EVENT_ENDED"
    assert ended["temporal_confirmation"] is False

    low = EmergencySystem(clock=FakeClock())
    cycle(low, "Dog bark", 0.99)
    low_result = cycle(low, "Dog bark", 0.99)
    assert low_result["event_state"] == "NO_EVENT"
    assert low_result["temporal_confirmation"] is False

    help_override = EmergencySystem(clock=FakeClock())
    cycle(help_override, "Siren", 0.90)
    cycle(help_override)
    help_result = cycle(
        help_override,
        "Siren",
        0.90,
        transcript="giúp tôi",
    )
    assert help_result["category"] == "help_request"
    assert help_result["temporal_confirmation"] is False


def test_sound_lifecycle():
    system = EmergencySystem(clock=FakeClock())
    cycle(system, "Glass breaking", 0.90)
    started = cycle(system, "Glass breaking", 0.90)
    assert started["event_state"] == "EVENT_STARTED"
    assert started["emitted_this_cycle"] is True
    continuing = cycle(system)
    assert continuing["event_state"] == "EVENT_CONTINUING"
    assert continuing["emitted_this_cycle"] is False
    ended = cycle(system)
    assert ended["event_state"] == "EVENT_ENDED"
    assert ended["emitted_this_cycle"] is False
    assert cycle(system)["event_state"] == "NO_EVENT"


def test_low_is_observational_only():
    system = EmergencySystem(context="neutral", clock=FakeClock())
    cycle(system, "Dog bark", 0.99)
    result = cycle(system, "Dog bark", 0.99)
    assert result["history_window"] == ["dog_barking", "dog_barking"]
    assert result["event_state"] == "NO_EVENT"
    assert result["emitted_this_cycle"] is False

    # A LOW observation cannot replace an actionable active event.
    cycle(system, "Siren", 0.99)
    active = cycle(system, "Siren", 0.99)
    assert active["category"] == "siren"
    after_low = cycle(system, "Dog bark", 0.99)
    assert after_low["category"] == "siren"
    assert after_low["event_state"] == "EVENT_CONTINUING"


def test_single_shot():
    system = EmergencySystem(decision_mode="single_shot", clock=FakeClock())
    result = cycle(system, "Glass breaking", 0.90)
    assert result["event_state"] == "EVENT_STARTED"
    assert result["decision_mode"] == "single_shot"
    assert result["temporal_confirmation"] is False
    assert result["emitted_this_cycle"] is True


def test_help_lifecycle_and_grace():
    clock = FakeClock()
    system = EmergencySystem(clock=clock, help_end_grace_cycles=2)
    started = cycle(system, transcript="giúp tôi")
    assert started["category"] == "help_request"
    assert started["alert_level"] == "CRITICAL"
    assert started["event_state"] == "EVENT_STARTED"
    assert started["temporal_confirmation"] is False
    assert started["emitted_this_cycle"] is True
    grace = cycle(system)
    assert grace["event_state"] == "EVENT_CONTINUING"
    assert grace["help_missing_cycles"] == 1
    resumed = cycle(system, transcript="cứu với")
    assert resumed["event_state"] == "EVENT_CONTINUING"
    assert resumed["help_missing_cycles"] == 0
    cycle(system)
    ended = cycle(system)
    assert ended["event_state"] == "EVENT_ENDED"
    assert ended["emitted_this_cycle"] is False


def test_cooldowns_are_per_category():
    assert CATEGORY_CONFIG["help_request"].cooldown_seconds != CATEGORY_CONFIG["gunshot"].cooldown_seconds
    assert CATEGORY_CONFIG["vehicle_horn"].cooldown_seconds != CATEGORY_CONFIG["fire"].cooldown_seconds

    clock = FakeClock()
    help_system = EmergencySystem(clock=clock)
    assert cycle(help_system, transcript="giúp tôi")["emitted_this_cycle"] is True
    clock.advance(19)
    assert cycle(help_system, transcript="giúp tôi")["emitted_this_cycle"] is False
    clock.advance(1)
    assert cycle(help_system, transcript="giúp tôi")["emitted_this_cycle"] is True

    # Category clocks are independent.
    single = EmergencySystem(decision_mode="single_shot", clock=clock)
    assert cycle(single, "Gunshot", 0.99)["emitted_this_cycle"] is True
    assert cycle(single, "Vehicle horn", 0.99)["emitted_this_cycle"] is True


def test_context_thresholds_and_levels():
    indoor_door = evaluate_sound("Doorbell", 0.61, "indoor")
    neutral_door = evaluate_sound("Doorbell", 0.61, "neutral")
    assert indoor_door["matched"] is True
    assert indoor_door["level"] == "MEDIUM"
    assert neutral_door["matched"] is False
    assert evaluate_sound("Vehicle horn", 0.56, "outdoor")["level"] == "HIGH"
    assert evaluate_sound("Vehicle horn", 0.99, "indoor")["level"] == "LOW"


def test_mapping_and_help_detection():
    mapped = map_alert("Car horn", 0.80, context="neutral")
    assert mapped["category"] == "vehicle_horn"
    assert mapped["level"] == "MEDIUM"
    assert detect_help_request("Xin hãy GỌI CẤP CỨU ngay") is True
    assert detect_help_request("Hôm nay có nguy hiểm không?") is False
    assert detect_help_request("Tôi thấy cháy") is False


def test_vietnamese_fire_emergency_phrases():
    detected = (
        "nhà cháy rồi",
        "cháy nhà",
        "có đám cháy",
        "gọi cứu hỏa",
        "gọi chữa cháy",
        "CÓ CHÁY!",
        "đang cháy",
        "cháy rồi",
    )
    for phrase in detected:
        assert detect_help_request(phrase) is True, phrase

    ordinary = (
        "tôi thấy cháy",
        "món cơm bị cháy cạnh",
        "chiếc lá có màu đỏ cháy",
        "máy chạy rồi",
        "tôi gọi về nhà",
    )
    for phrase in ordinary:
        assert detect_help_request(phrase) is False, phrase

    clock = FakeClock()
    system = EmergencySystem(clock=clock)
    first = cycle(system, transcript="nhà cháy rồi")
    assert first["help_request_detected"] is True
    assert first["category"] == "help_request"
    assert first["alert_level"] == "CRITICAL"
    assert first["event_state"] == "EVENT_STARTED"
    assert first["emitted_this_cycle"] is True
    assert first["alert_text"] == "Phát hiện tình huống cháy"

    repeated = cycle(system, transcript="có đám cháy")
    assert repeated["event_state"] == "EVENT_CONTINUING"
    assert repeated["emitted_this_cycle"] is False
    assert repeated["alert_text"] == "Phát hiện tình huống cháy"


def run_tests():
    tests = [
        test_history_and_voting,
        test_temporal_confirmation_requires_current_quorum,
        test_sound_lifecycle,
        test_low_is_observational_only,
        test_single_shot,
        test_help_lifecycle_and_grace,
        test_cooldowns_are_per_category,
        test_context_thresholds_and_levels,
        test_mapping_and_help_detection,
        test_vietnamese_fire_emergency_phrases,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: passed")
    print("emergency system tests passed")


if __name__ == "__main__":
    run_tests()
