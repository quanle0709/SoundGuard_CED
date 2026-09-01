import datetime
from pathlib import Path

from display_transport import crc16_ccitt, encode_hud_frame


ROOT = Path(__file__).resolve().parents[1]
HUD_CPP = (ROOT / "firmware" / "src" / "hud.cpp").read_text(encoding="utf-8")
HUD_H = (ROOT / "firmware" / "src" / "hud.h").read_text(encoding="utf-8")
PROTOCOL = (ROOT / "firmware" / "src" / "protocol.cpp").read_text(
    encoding="utf-8"
)

LEGACY_FRAME_TYPES = frozenset("SPAECT")


def _between(source: str, start: str, end: str) -> str:
    first = source.index(start)
    return source[first:source.index(end, first)]


def _local_hhmm(epoch: int, offset_minutes: int) -> str:
    local_epoch = epoch + offset_minutes * 60
    return datetime.datetime.fromtimestamp(
        local_epoch, tz=datetime.timezone.utc
    ).strftime("%H:%M")


def test_k_is_additive_to_the_preimplementation_legacy_namespace():
    assert "K" not in LEGACY_FRAME_TYPES
    for frame_type in LEGACY_FRAME_TYPES:
        assert f"type_ == '{frame_type}'" in PROTOCOL
        assert f"value != '{frame_type}'" in PROTOCOL
    assert "type_ == 'K'" in PROTOCOL
    assert "value != 'K'" in PROTOCOL
    assert "synchronizeClock(payload_)" in PROTOCOL


def test_initial_clock_is_invalid_and_home_uses_placeholder_only_until_sync():
    assert "bool clockSynchronized_ = false" in HUD_H
    formatter = _between(
        HUD_CPP,
        "bool SoundGuardHUD::formatCurrentTime(",
        "void SoundGuardHUD::renderHomeScreen()",
    )
    assert "if (!clockSynchronized_)" in formatter
    assert 'std::memcpy(timeBuffer, "--:--"' in formatter
    assert '"15:30"' not in HUD_CPP


def test_local_time_math_covers_offsets_advancement_and_no_double_offset():
    epoch = 1787356800
    assert _local_hhmm(epoch, 420) == "07:00"
    assert _local_hhmm(epoch, -300) == "19:00"
    assert _local_hhmm(epoch, 330) == "05:30"
    assert _local_hhmm(epoch + 60, 330) == "05:31"
    assert _local_hhmm(epoch, 330) != _local_hhmm(epoch + 330 * 60, 330)

    formatter = _between(
        HUD_CPP,
        "bool SoundGuardHUD::formatCurrentTime(",
        "void SoundGuardHUD::renderHomeScreen()",
    )
    assert formatter.count("clockUtcOffsetMinutes_") == 1
    assert "time(nullptr)" in formatter
    assert "gmtime_r" in formatter
    assert "getLocalTime" not in formatter


def test_valid_clock_sets_utc_system_clock_then_retains_offset():
    synchronizer = _between(
        HUD_CPP,
        "bool SoundGuardHUD::synchronizeClock(",
        "bool SoundGuardHUD::formatCurrentTime(",
    )
    assert "parseClockPayload(value" in synchronizer
    assert "settimeofday(&clockValue, nullptr)" in synchronizer
    assert synchronizer.index("settimeofday") < synchronizer.index(
        "clockUtcOffsetMinutes_ = utcOffsetMinutes"
    )
    assert synchronizer.index("settimeofday") < synchronizer.index(
        "clockSynchronized_ = true"
    )


def test_malformed_clock_validation_is_bounded_and_preserves_previous_good_clock():
    parser = _between(
        HUD_CPP,
        "bool SoundGuardHUD::parseClockPayload(",
        "bool SoundGuardHUD::synchronizeClock(",
    )
    assert "value.indexOf(',', separator + 1)" in parser
    assert "isUnsignedDecimal(epochText)" in parser
    assert "isSignedDecimal(offsetText)" in parser
    assert "kClockEpochMin" in parser and "kClockEpochMax" in parser
    assert "kClockOffsetMinutesMin" in parser
    assert "kClockOffsetMinutesMax" in parser

    synchronizer = _between(
        HUD_CPP,
        "bool SoundGuardHUD::synchronizeClock(",
        "bool SoundGuardHUD::formatCurrentTime(",
    )
    validation_return = synchronizer.index(
        "if (!parseClockPayload(value, epoch, utcOffsetMinutes)) return false;"
    )
    assert validation_return < synchronizer.index("clockUtcOffsetMinutes_ =")
    assert validation_return < synchronizer.index("clockSynchronized_ = true")


def test_clock_frame_crc_is_identical_to_legacy_frame_crc_guarantee():
    valid = encode_hud_frame("clock", "1787356800,330")
    body = valid[2:-2]
    expected_crc = int.from_bytes(valid[-2:], "big")
    assert crc16_ccitt(body) == expected_crc

    corrupted = bytearray(valid)
    corrupted[7] ^= 0x01
    assert crc16_ccitt(corrupted[2:-2]) != int.from_bytes(corrupted[-2:], "big")
    assert "if (receivedCrc_ == crc_) applyFrame();" in PROTOCOL


def test_k_is_background_only_for_home_subtitle_and_alert_ownership():
    synchronizer = _between(
        HUD_CPP,
        "bool SoundGuardHUD::synchronizeClock(",
        "bool SoundGuardHUD::formatCurrentTime(",
    )
    for protected_state in (
        "state_.subtitle", "state_.environmentalSound", "state_.criticalEvent",
        "state_.helpActive", "state_.alert", "subtitlePageIndex_",
        "subtitlePageStartedMs_",
    ):
        assert protected_state not in synchronizer
    assert "homeWasVisible = desiredScreen() == Screen::Home" in synchronizer
    assert "if (homeWasVisible)" in synchronizer


def test_home_redraw_remains_minute_value_driven():
    update = _between(
        HUD_CPP,
        "void SoundGuardHUD::update()",
        "void SoundGuardHUD::appendWrappedToken(",
    )
    assert "lastHomeClockCheckMs_ < 1000" in update
    assert "std::memcmp(timeBuffer, lastHomeTime_" in update

    synchronizer = _between(
        HUD_CPP,
        "bool SoundGuardHUD::synchronizeClock(",
        "bool SoundGuardHUD::formatCurrentTime(",
    )
    assert "!previouslySynchronized" in synchronizer
    assert "std::memcmp(previousTime, synchronizedTime" in synchronizer
