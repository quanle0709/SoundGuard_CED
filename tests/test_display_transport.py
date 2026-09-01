import io
import struct
import sys
import time
import unicodedata
from types import SimpleNamespace

from app import HUD_TEST_SUBTITLES
from display_transport import (
    CLOCK_SYNC_INTERVAL_SECONDS,
    HUDTransport,
    crc16_ccitt,
    encode_clock_payload,
    encode_hud_frame,
    encode_alert_state_payload,
    get_alert_display_state,
    get_ced_display_label,
    get_screen2_ced_label,
    system_utc_offset_minutes,
)


def decode_frame(frame: bytes) -> tuple[str, str]:
    assert frame[:2] == b"SG"
    version, message_type, length = struct.unpack(">BcH", frame[2:6])
    assert version == 1
    body, expected_crc = frame[2:-2], struct.unpack(">H", frame[-2:])[0]
    assert crc16_ccitt(body) == expected_crc
    payload = frame[6:6 + length]
    assert len(payload) == length
    return message_type.decode("ascii"), payload.decode("utf-8")


def decode_frames(data: bytes) -> list[tuple[str, str]]:
    frames = []
    position = 0
    while position < len(data):
        length = struct.unpack(">H", data[position + 4:position + 6])[0]
        frame_end = position + 8 + length
        frames.append(decode_frame(data[position:frame_end]))
        position = frame_end
    return frames


class ManualClock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_all_vietnamese_glyph_groups_survive_utf8_frame():
    text = (
        "ă â ê ô ơ ư Ă Â Ê Ô Ơ Ư đ Đ "
        "á à ả ã ạ ấ ầ ẩ ẫ ậ ế ề ể ễ ệ "
        "ố ồ ổ ỗ ộ ớ ờ ở ỡ ợ ứ ừ ử ữ ự"
    )
    kind, decoded = decode_frame(encode_hud_frame("subtitle", text))
    assert kind == "S"
    assert decoded == text


def test_transport_normalizes_decomposed_vietnamese_to_nfc():
    stream = io.BytesIO()
    transport = HUDTransport(stream=stream)
    decomposed = unicodedata.normalize("NFD", "Tôi đang ở ngoài đường.")
    transport.set_subtitle(decomposed)
    _, decoded = decode_frame(stream.getvalue())
    assert decoded == unicodedata.normalize("NFC", decomposed)


def test_subtitle_and_alert_have_distinct_types():
    assert decode_frame(encode_hud_frame("subtitle", "Xin chào"))[0] == "S"
    assert decode_frame(encode_hud_frame("alert", "CẢNH BÁO NGUY HIỂM"))[0] == "A"


def test_partial_final_and_environment_frames_have_distinct_types():
    assert decode_frame(encode_hud_frame("partial_subtitle", "live"))[0] == "P"
    assert decode_frame(encode_hud_frame("subtitle", "final"))[0] == "S"
    assert decode_frame(encode_hud_frame("environmental_sound", "HORN"))[0] == "C"
    assert decode_frame(encode_hud_frame("alert_state", "0|SIREN"))[0] == "E"
    assert decode_frame(encode_hud_frame("clock", "1787356800,420"))[0] == "K"


def test_clock_payload_supports_positive_negative_and_non_hour_offsets():
    assert encode_clock_payload(1787356800.99, 420) == "1787356800,420"
    assert encode_clock_payload(1787356800, -300) == "1787356800,-300"
    assert encode_clock_payload(1787356800, 330) == "1787356800,330"


def test_clock_sync_uses_fresh_dynamic_host_epoch_and_offset():
    stream = io.BytesIO()
    wall = ManualClock(1787356800.25)
    monotonic = ManualClock(100.0)
    offsets = []

    def dynamic_offset(epoch):
        offsets.append(epoch)
        return 330 if epoch < 1787357400 else -210

    transport = HUDTransport(
        stream=stream,
        auto_clock_sync=True,
        wall_clock=wall,
        monotonic_clock=monotonic,
        utc_offset_provider=dynamic_offset,
    )
    wall.advance(CLOCK_SYNC_INTERVAL_SECONDS)
    monotonic.advance(CLOCK_SYNC_INTERVAL_SECONDS - 1)
    transport.update()
    monotonic.advance(1)
    transport.update()

    assert decode_frames(stream.getvalue()) == [
        ("K", "1787356800,330"),
        ("K", "1787357400,-210"),
    ]
    assert offsets == [1787356800.25, 1787357400.25]


def test_default_clock_payload_is_sourced_from_current_host_os():
    stream = io.BytesIO()
    before = int(time.time())
    HUDTransport(stream=stream, auto_clock_sync=True)
    after = int(time.time())

    kind, payload = decode_frame(stream.getvalue())
    epoch_text, offset_text = payload.split(",")
    epoch = int(epoch_text)
    assert kind == "K"
    assert before <= epoch <= after
    assert int(offset_text) == system_utc_offset_minutes(epoch)


def test_clock_is_not_resent_before_ten_minute_deadline():
    stream = io.BytesIO()
    wall = ManualClock(1787356800)
    monotonic = ManualClock(0)
    transport = HUDTransport(
        stream=stream, auto_clock_sync=True, wall_clock=wall,
        monotonic_clock=monotonic, utc_offset_provider=lambda _epoch: 420,
    )
    for _ in range(599):
        wall.advance(1)
        monotonic.advance(1)
        transport.update()
    assert decode_frames(stream.getvalue()) == [("K", "1787356800,420")]

    wall.advance(1)
    monotonic.advance(1)
    transport.update()
    assert decode_frames(stream.getvalue())[-1] == ("K", "1787357400,420")


def test_clock_payload_rejects_unreasonable_epoch_and_offset():
    for epoch, offset in (
        (946684799, 0), (4102444800, 0),
        (1787356800, -841), (1787356800, 841),
    ):
        try:
            encode_clock_payload(epoch, offset)
        except ValueError:
            pass
        else:
            raise AssertionError((epoch, offset))


def test_owned_serial_reconnect_sends_a_fresh_clock(monkeypatch):
    instances = []
    wall = ManualClock(1787356800)
    monotonic = ManualClock(10)

    class ReconnectingSerial:
        def __init__(self, *, port, baudrate, timeout):
            self.port = port
            self.baudrate = baudrate
            self.timeout = timeout
            self.dtr = self.rts = True
            self.frames = bytearray()
            self.fail_writes = False
            instances.append(self)

        def open(self):
            pass

        def write(self, data):
            if self.fail_writes:
                raise OSError("disconnected")
            self.frames.extend(data)

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "serial", SimpleNamespace(Serial=ReconnectingSerial)
    )
    transport = HUDTransport(
        "COM5", reset_delay=0, wall_clock=wall,
        monotonic_clock=monotonic, utc_offset_provider=lambda _epoch: 420,
    )
    assert decode_frames(bytes(instances[0].frames)) == [
        ("K", "1787356800,420")
    ]

    instances[0].fail_writes = True
    transport.set_status("LIVE")
    wall.advance(5)
    monotonic.advance(2)
    transport.update()

    assert len(instances) == 2
    assert decode_frames(bytes(instances[1].frames)) == [
        ("K", "1787356805,420")
    ]


def test_alert_state_payload_covers_critical_help_and_inactive_cases():
    assert encode_alert_state_payload("siren", False) == "0|SIREN"
    assert encode_alert_state_payload("siren", True) == "1|SIREN"
    assert encode_alert_state_payload("", True) == "1|"
    assert encode_alert_state_payload("", False) == ""


def test_alert_display_state_reuses_compact_ced_mapping_without_inferring_help():
    assert get_alert_display_state("glass_breaking", False) == ("GLASS", False)
    assert get_alert_display_state("siren", False) == ("SIREN", False)
    assert get_alert_display_state("speech", False) == ("EVENT", False)
    assert get_alert_display_state(None, True) == ("", True)


def test_ced_display_mapping_uses_actual_project_labels():
    expected = {
        "Vehicle horn": "HORN",
        "car_horn": "HORN",
        "Dog": "DOG",
        "baby_crying": "BABY",
        "Glass": "GLASS",
        "Siren": "SIREN",
        "Fire": "FIRE",
        "smoke_alarm": "ALARM",
        "door_activity": "DOOR",
        "Screaming": "SCREAM",
    }
    assert {label: get_ced_display_label(label) for label in expected} == expected


def test_background_silence_static_and_speech_are_suppressed():
    for label in ("Static", "background noise", "Silence", "noise",
                  "Speech", "Conversation", "Speech synthesizer",
                  "Male speech, man speaking", "Narration, monologue",
                  "Mantra", "unknown"):
        assert get_ced_display_label(label) == ""
    assert get_ced_display_label("Screaming") == "SCREAM"


def test_critical_ced_is_not_downgraded_to_screen2_corner():
    for level in ("HIGH", "CRITICAL"):
        decision = {"mapped_sound": {"level": level}}
        assert get_screen2_ced_label("Siren", decision) == ""
    assert get_screen2_ced_label(
        "Vehicle horn", {"mapped_sound": {"level": "MEDIUM"}}
    ) == "HORN"


def test_every_deterministic_hud_sample_round_trips():
    for sample in HUD_TEST_SUBTITLES:
        assert decode_frame(encode_hud_frame("subtitle", sample))[1] == sample


def test_transport_does_not_resend_identical_normalized_text():
    stream = io.BytesIO()
    transport = HUDTransport(stream=stream)
    decomposed = unicodedata.normalize("NFD", "Mẹ nhớ áo mưa.")

    transport.set_subtitle(decomposed)
    transport.set_subtitle("Mẹ nhớ áo mưa.")
    transport.set_alert("CÒI XE")
    transport.set_alert("CÒI XE")

    assert stream.getvalue() == (
        encode_hud_frame("subtitle", "Mẹ nhớ áo mưa.")
        + encode_hud_frame("alert", "CÒI XE")
    )


def test_new_partial_lifecycle_allows_identical_final_to_be_resent():
    stream = io.BytesIO()
    transport = HUDTransport(stream=stream)

    transport.set_subtitle("same words")
    transport.set_partial_subtitle("")
    transport.set_subtitle("same words")

    assert stream.getvalue() == (
        encode_hud_frame("subtitle", "same words")
        + encode_hud_frame("partial_subtitle", "")
        + encode_hud_frame("subtitle", "same words")
    )


def test_serial_control_lines_are_inactive_before_port_open(monkeypatch):
    events = []

    class FakeSerial:
        def __init__(self, *, port, baudrate, timeout):
            events.append(("construct", port, baudrate, timeout))
            self.dtr = True
            self.rts = True
            self.port = port
            self.frames = bytearray()

        def open(self):
            events.append(("open", self.port, self.dtr, self.rts))

        def close(self):
            events.append(("close", self.dtr, self.rts))

        def write(self, data):
            self.frames.extend(data)

        def flush(self):
            pass

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=FakeSerial))

    transport = HUDTransport(
        "COM5", reset_delay=0,
        wall_clock=lambda: 1787356800,
        monotonic_clock=lambda: 0,
        utc_offset_provider=lambda _epoch: -300,
    )
    assert decode_frames(bytes(transport.stream.frames)) == [
        ("K", "1787356800,-300")
    ]
    transport.close()

    assert events == [
        ("construct", None, 115200, 0),
        ("open", "COM5", False, False),
        ("close", False, False),
    ]
