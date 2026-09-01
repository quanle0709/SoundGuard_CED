from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUD = (ROOT / "firmware" / "src" / "hud.cpp").read_text(encoding="utf-8")
PROTOCOL = (ROOT / "firmware" / "src" / "protocol.cpp").read_text(
    encoding="utf-8"
)
PIPELINE = (ROOT / "audio_pipeline.py").read_text(encoding="utf-8")


def _between(source: str, start: str, end: str) -> str:
    first = source.index(start)
    return source[first:source.index(end, first)]


def _expected_screen(*, alert=False, subtitle=False, ced=False):
    if alert:
        return "ALERT"
    if subtitle or ced:
        return "SUBTITLE"
    return "HOME"


def test_final_state_decision_table():
    assert _expected_screen(alert=True, subtitle=True, ced=True) == "ALERT"
    assert _expected_screen(subtitle=True, ced=True) == "SUBTITLE"
    assert _expected_screen(subtitle=True) == "SUBTITLE"
    assert _expected_screen(ced=True) == "SUBTITLE"
    assert _expected_screen() == "HOME"

    policy = _between(
        HUD,
        "SoundGuardHUD::Screen SoundGuardHUD::desiredScreen()",
        "void SoundGuardHUD::resumeSubtitleAfterAlert",
    )
    assert policy.index("alertActive()") < policy.index("state_.subtitle")
    assert "state_.environmentalSound" in policy
    assert "return Screen::Home" in policy


def test_stt_and_ced_updates_under_alert_do_not_redraw_another_screen():
    final_setter = _between(
        HUD, "void SoundGuardHUD::setSubtitle(",
        "void SoundGuardHUD::setPartialSubtitle(",
    )
    partial_setter = _between(
        HUD, "void SoundGuardHUD::setPartialSubtitle(",
        "void SoundGuardHUD::setEnvironmentalSound(",
    )
    ced_setter = _between(
        HUD, "void SoundGuardHUD::setEnvironmentalSound(",
        "bool SoundGuardHUD::alertActive()",
    )
    for setter in (final_setter, partial_setter, ced_setter):
        assert "if (!alertActive()) render();" in setter


def test_alert_clear_preserves_underlying_state_and_resets_page_interval():
    alert_setter = _between(
        HUD, "void SoundGuardHUD::setAlertState(",
        "void SoundGuardHUD::setAlert(",
    )
    resume = _between(
        HUD, "void SoundGuardHUD::resumeSubtitleAfterAlert(",
        "void SoundGuardHUD::setAlertState(",
    )
    assert "state_.subtitle" not in alert_setter
    assert "state_.environmentalSound" not in alert_setter
    assert "subtitlePageStartedMs_ = millis()" in resume
    assert alert_setter.count("render();") == 1


def test_alert_transport_is_ordered_before_underlying_ced_update():
    dispatch = _between(
        PIPELINE,
        "    def _dispatch_event(self, event: PipelineEvent)",
        "    def _pump(self)",
    )
    assert dispatch.index("self._print_decision(decision)") < dispatch.index(
        "self._update_ced_display(result, decision)"
    )


def test_e_frame_is_distinct_atomic_and_malformed_payload_is_ignored():
    for frame_type in "SPAECT":
        assert f"value != '{frame_type}'" in PROTOCOL or f"type_ == '{frame_type}'" in PROTOCOL
    assert "receivedCrc_ == crc_" in PROTOCOL
    assert "if (++received_ == length_)" in PROTOCOL
    assert "setAlertState(payload_)" in PROTOCOL

    alert_setter = _between(
        HUD, "void SoundGuardHUD::setAlertState(",
        "void SoundGuardHUD::setAlert(",
    )
    validation = alert_setter[:alert_setter.index('state_.alert = ""')]
    assert "validHelpFlag" in validation
    assert "validSeparator" in validation
    assert "hasCondition" in validation
    assert "return" in validation


def test_full_transition_chain_has_no_competing_render_path():
    chain = [
        _expected_screen(),
        _expected_screen(subtitle=True),
        _expected_screen(alert=True, subtitle=True),
        _expected_screen(subtitle=True),
        _expected_screen(),
    ]
    assert chain == ["HOME", "SUBTITLE", "ALERT", "SUBTITLE", "HOME"]
    render = HUD[HUD.index("void SoundGuardHUD::render()") :]
    assert render.count("switch (desiredScreen())") == 1
