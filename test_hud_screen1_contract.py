from pathlib import Path


ROOT = Path(__file__).resolve().parent
HUD = (ROOT / "firmware" / "src" / "hud.cpp").read_text(encoding="utf-8")


def test_home_layout_remains_the_completed_design():
    assert 'homeText_.drawStr(0, 5, "soundguard")' in HUD
    assert "homeText_.drawStr(0, 32, timeBuffer)" in HUD
    assert 'homeText_.drawStr(44, 31, "Ready")' in HUD


def test_home_has_safe_unsynchronized_time_and_no_design_placeholder():
    assert 'std::memcpy(timeBuffer, "--:--"' in HUD
    assert '"15:30"' not in HUD


def test_home_clock_redraw_is_minute_value_driven():
    assert "std::memcmp(timeBuffer, lastHomeTime_" in HUD
    assert "lastHomeClockCheckMs_ < 1000" in HUD
