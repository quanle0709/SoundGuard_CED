"""Source contracts for benchmark-only ESP32-C3 transport instrumentation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = (ROOT / "firmware" / "src" / "protocol.cpp").read_text(
    encoding="utf-8"
)
PROTOCOL_HEADER = (ROOT / "firmware" / "src" / "protocol.h").read_text(
    encoding="utf-8"
)
HUD = (ROOT / "firmware" / "src" / "hud.cpp").read_text(encoding="utf-8")


def test_benchmark_ack_is_one_bounded_write_followed_by_flush():
    assert "char ackLine[96];" in PROTOCOL
    assert "output.write(" in PROTOCOL
    assert "output.flush();" in PROTOCOL
    ack_section = PROTOCOL.split("char ackLine[96];", 1)[1].split(
        "if (innerType == 'D')", 1
    )[0]
    assert 'output.print("SGACK|")' not in ack_section
    assert "accepted != expected" in ack_section


def test_diagnostics_remain_inside_benchmark_envelope_only():
    product_dispatch = PROTOCOL.split(
        "void HUDProtocolParser::applyProductFrame", 1
    )[1].split("void HUDProtocolParser::applyBenchmarkFrame", 1)[0]
    assert "'D'" not in product_dispatch
    assert "innerType != 'Q' && innerType != 'D'" in PROTOCOL
    assert 'output.print("SGDIAG|")' in PROTOCOL


def test_parser_diagnostics_do_not_relax_crc_gate():
    assert "if (receivedCrc_ == crc_) applyFrame();" in PROTOCOL
    assert "++crcFailures_;" in PROTOCOL
    assert "++parserRecoveries_;" in PROTOCOL
    assert "uint32_t validFrames_ = 0;" in PROTOCOL_HEADER


def test_display_timing_wrapper_keeps_synchronous_oled_commit():
    assert "void SoundGuardHUD::commitDisplay()" in HUD
    assert "display_.display();" in HUD
    assert HUD.count("commitDisplay();") == 3
