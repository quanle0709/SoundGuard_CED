from pathlib import Path


ROOT = Path(__file__).resolve().parent
HUD_HEADER = (ROOT / "firmware" / "src" / "hud.h").read_text(encoding="utf-8")
HUD_SOURCE = (ROOT / "firmware" / "src" / "hud.cpp").read_text(encoding="utf-8")
PROTOCOL_SOURCE = (ROOT / "firmware" / "src" / "protocol.cpp").read_text(
    encoding="utf-8"
)
MAIN_SOURCE = (ROOT / "firmware" / "src" / "main.cpp").read_text(encoding="utf-8")


def test_vietnamese_subtitles_use_utf8_font_and_pixel_width():
    assert "u8g2_font_unifont_t_vietnamese1" in HUD_SOURCE
    assert "text_.drawUTF8" in HUD_SOURCE
    assert "text_.getUTF8Width(candidate.c_str())" in HUD_SOURCE
    assert "appendWrappedToken" in HUD_SOURCE


def test_final_pages_are_two_lines_for_two_seconds_without_blocking_delay():
    assert "kVisibleSubtitleLines = 2" in HUD_HEADER
    assert "kSubtitlePageDurationMs = 2000" in HUD_HEADER
    assert "subtitlePageIndex_ * kVisibleSubtitleLines" in HUD_SOURCE
    assert "now - subtitlePageStartedMs_" in HUD_SOURCE
    assert "delay(2000" not in HUD_SOURCE
    assert "delay(2000" not in MAIN_SOURCE


def test_partial_and_final_subtitles_use_distinct_protocol_frames():
    assert "type_ == 'S'" in PROTOCOL_SOURCE
    assert "type_ == 'P'" in PROTOCOL_SOURCE
    assert "setSubtitle(payload_)" in PROTOCOL_SOURCE
    assert "setPartialSubtitle(payload_)" in PROTOCOL_SOURCE


def test_ced_is_measured_clipped_and_right_aligned():
    assert "fitCedLabel(state_.environmentalSound, maxWidth)" in HUD_SOURCE
    assert "homeText_.getUTF8Width(label.c_str())" in HUD_SOURCE
    assert "HUD_OLED_WIDTH - kCedRightMargin - width" in HUD_SOURCE
    assert "max(kCedRightMargin" in HUD_SOURCE


def test_alert_priority_and_home_fallback_are_explicit():
    render = HUD_SOURCE[HUD_SOURCE.index("void SoundGuardHUD::render()") :]
    alert_branch = render.index("case Screen::Alert")
    subtitle_branch = render.index("case Screen::Subtitle")
    home_branch = render.index("case Screen::Home")
    assert alert_branch < subtitle_branch < home_branch
    assert 'state_.subtitle = "";' in HUD_SOURCE


def test_lopaka_placeholders_are_not_rendered():
    assert 'drawStr(1, 17, "Text")' not in HUD_SOURCE
    assert 'drawStr(51, 6, "CED")' not in HUD_SOURCE
    assert 'drawStr(1, 19, "")' not in HUD_SOURCE
