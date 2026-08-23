from pathlib import Path


ROOT = Path(__file__).resolve().parent
HUD_HEADER = (ROOT / "firmware" / "src" / "hud.h").read_text(encoding="utf-8")
HUD_SOURCE = (ROOT / "firmware" / "src" / "hud.cpp").read_text(encoding="utf-8")
PROTOCOL_SOURCE = (ROOT / "firmware" / "src" / "protocol.cpp").read_text(
    encoding="utf-8"
)


def _function_body(signature: str, next_signature: str) -> str:
    start = HUD_SOURCE.index(signature)
    end = HUD_SOURCE.index(next_signature, start)
    return HUD_SOURCE[start:end]


def test_alert_renderer_has_required_three_regions_and_help_is_conditional():
    body = _function_body(
        "void SoundGuardHUD::renderAlertScreen()",
        "void SoundGuardHUD::render()",
    )
    assert 'state_.helpActive ? "SOS" : "ALERT"' in body
    assert 'drawCenteredText("HELP", kAlertHelpBaselineY' in body
    assert 'if (state_.helpActive)' in body
    assert 'drawCenteredText("! ALERT !", kAlertFooterBaselineY' in body
    assert "subtitleLines_" not in body
    assert "environmentalSound" not in body


def test_alert_text_is_measured_centered_and_overflow_safe():
    body = _function_body(
        "void SoundGuardHUD::drawCenteredText",
        "void SoundGuardHUD::renderSubtitleScreen()",
    )
    assert "homeText_.getUTF8Width(visible.c_str())" in body
    assert "(HUD_OLED_WIDTH - width) / 2" in body
    assert "kCenteredTextMargin" in body
    assert "visible = overflowFallback" in body


def test_alert_baselines_match_screen3_contract():
    assert "kAlertEventBaselineY = 7" in HUD_HEADER
    assert "kAlertHelpBaselineY = 19" in HUD_HEADER
    assert "kAlertFooterBaselineY = 31" in HUD_HEADER


def test_structured_alert_state_is_atomic_and_has_protocol_type():
    setter = _function_body(
        "void SoundGuardHUD::setAlertState",
        "void SoundGuardHUD::setAlert(",
    )
    assert "state_.helpActive = value[0] == '1'" in setter
    assert "state_.criticalEvent = value.length() > 2" in setter
    assert "type_ == 'E'" in PROTOCOL_SOURCE
    assert "setAlertState(payload_)" in PROTOCOL_SOURCE


def test_alert_immediately_overrides_but_does_not_destroy_subtitle_state():
    render = HUD_SOURCE[HUD_SOURCE.index("void SoundGuardHUD::render()") :]
    setter = _function_body(
        "void SoundGuardHUD::setAlertState",
        "void SoundGuardHUD::setAlert(",
    )
    assert render.index("case Screen::Alert") < render.index("case Screen::Subtitle")
    assert "render();" in setter
    assert "state_.subtitle" not in setter
    assert "resumeSubtitleAfterAlert(alertWasActive)" in setter


def test_screen3_has_no_placeholders_lowercase_footer_or_blocking_delay():
    assert '"Text"' not in HUD_SOURCE
    assert '"! alert !"' not in HUD_SOURCE
    assert "delay(" not in HUD_SOURCE
