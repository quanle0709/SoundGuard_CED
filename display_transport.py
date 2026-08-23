"""UTF-8-safe transport boundary for the SoundGuard glasses HUD."""

from __future__ import annotations

import datetime
import struct
import time
import unicodedata
from typing import BinaryIO, Callable

MAGIC = b"SG"
VERSION = 1
MAX_PAYLOAD_BYTES = 1024
CLOCK_SYNC_INTERVAL_SECONDS = 10 * 60
CLOCK_RECONNECT_INTERVAL_SECONDS = 2.0
CLOCK_EPOCH_MIN = 946684800  # 2000-01-01T00:00:00Z
CLOCK_EPOCH_MAX = 4102444799  # 2099-12-31T23:59:59Z
CLOCK_UTC_OFFSET_MINUTES_MIN = -14 * 60
CLOCK_UTC_OFFSET_MINUTES_MAX = 14 * 60

MESSAGE_TYPES = {
    "subtitle": b"S",
    "partial_subtitle": b"P",
    "alert": b"A",
    "alert_state": b"E",
    "environmental_sound": b"C",
    "status": b"T",
    "clock": b"K",
}

_SUPPRESSED_CED_LABELS = {
    "", "unknown", "static", "background", "background noise",
    "silence", "noise", "white noise", "pink noise",
    "environmental noise", "speech", "conversation",
}

_SPEECH_ONLY_CED_TERMS = {
    "speech", "conversation", "narration", "monologue", "mantra",
    "synthetic speech", "speech synthesizer", "whispering", "babbling",
    "chatter", "hubbub",
}

_CED_DISPLAY_LABELS = {
    "vehicle horn": "HORN",
    "car horn": "HORN",
    "honk": "HORN",
    "toot": "HORN",
    "air horn": "HORN",
    "dog barking": "DOG",
    "bark": "DOG",
    "bow wow": "DOG",
    "dog": "DOG",
    "baby crying": "BABY",
    "baby cry": "BABY",
    "baby cry infant cry": "BABY",
    "crying": "BABY",
    "glass breaking": "GLASS",
    "glass": "GLASS",
    "shatter": "GLASS",
    "siren": "SIREN",
    "civil defense siren": "SIREN",
    "police car siren": "SIREN",
    "ambulance siren": "SIREN",
    "emergency vehicle": "SIREN",
    "fire": "FIRE",
    "fire crackling": "FIRE",
    "crackle": "FIRE",
    "smoke alarm": "ALARM",
    "alarm": "ALARM",
    "door activity": "DOOR",
    "doorbell": "DOOR",
    "knock": "DOOR",
    "screaming": "SCREAM",
    "scream": "SCREAM",
    "shout": "SCREAM",
    "gunshot": "GUNSHOT",
    "explosion": "EXPLOSION",
}

_ALERT_ONLY_CED_DISPLAY_LABELS = {
    "ALARM", "EXPLOSION", "FIRE", "GLASS", "GUNSHOT", "SCREAM", "SIREN",
}


def normalize_hud_text(text: str) -> str:
    """Normalize display text without changing STT recognition behavior."""
    return unicodedata.normalize("NFC", str(text)).replace("\r", " ").replace("\n", " ")


def _normalize_ced_label(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(label or ""))
    ascii_label = normalized.encode("ascii", "ignore").decode("ascii").lower()
    compact = "".join(character if character.isalnum() else " "
                      for character in ascii_label)
    return " ".join(compact.split())


def get_ced_display_label(label: str) -> str:
    """Return a compact ASCII presentation label, or blank when not useful."""
    normalized = _normalize_ced_label(label)
    if (normalized in _SUPPRESSED_CED_LABELS or
            any(term in normalized for term in _SPEECH_ONLY_CED_TERMS)):
        return ""
    return _CED_DISPLAY_LABELS.get(normalized, normalized.upper())


def get_screen2_ced_label(label: str, decision: dict | None = None) -> str:
    """Keep high-priority emergency evidence out of Screen 2's small corner."""
    mapped_sound = (decision or {}).get("mapped_sound") or {}
    if mapped_sound.get("level") in {"HIGH", "CRITICAL"}:
        return ""
    display_label = get_ced_display_label(label)
    if display_label in _ALERT_ONLY_CED_DISPLAY_LABELS:
        return ""
    return display_label


def get_alert_display_state(active_sound: str | None,
                            help_active: bool) -> tuple[str, bool]:
    """Map the emergency system's existing independent active lifecycles."""
    event_label = get_ced_display_label(active_sound or "")
    if active_sound and not event_label:
        event_label = "EVENT"
    return event_label, bool(help_active)


def encode_alert_state_payload(event_label: str, help_active: bool) -> str:
    """Encode Screen 3 state atomically; blank means ALERT is inactive."""
    label = get_ced_display_label(event_label)
    if not label and not help_active:
        return ""
    return f"{'1' if help_active else '0'}|{label}"


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def system_utc_offset_minutes(epoch: float) -> int:
    """Return the host OS local UTC offset active at ``epoch``."""
    offset = datetime.datetime.fromtimestamp(
        epoch, tz=datetime.timezone.utc
    ).astimezone().utcoffset()
    if offset is None:
        raise RuntimeError("Host OS did not provide a local UTC offset")
    offset_seconds = int(offset.total_seconds())
    if offset_seconds % 60:
        raise RuntimeError("Host UTC offset is not representable in whole minutes")
    return offset_seconds // 60


def encode_clock_payload(epoch: float, utc_offset_minutes: int) -> str:
    """Encode ``<UTC Unix epoch seconds>,<local UTC offset minutes>``."""
    epoch_seconds = int(epoch)
    offset_minutes = int(utc_offset_minutes)
    if not CLOCK_EPOCH_MIN <= epoch_seconds <= CLOCK_EPOCH_MAX:
        raise ValueError("Clock epoch is outside the supported 2000-2099 range")
    if not (CLOCK_UTC_OFFSET_MINUTES_MIN <= offset_minutes <=
            CLOCK_UTC_OFFSET_MINUTES_MAX):
        raise ValueError("Clock UTC offset is outside the supported +/-14-hour range")
    return f"{epoch_seconds},{offset_minutes}"


def encode_hud_frame(message_type: str, text: str) -> bytes:
    try:
        type_byte = MESSAGE_TYPES[message_type]
    except KeyError as exc:
        raise ValueError(f"Unknown HUD message type: {message_type}") from exc
    payload = normalize_hud_text(text).encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"HUD payload exceeds {MAX_PAYLOAD_BYTES} UTF-8 bytes")
    body = bytes((VERSION,)) + type_byte + struct.pack(">H", len(payload)) + payload
    return MAGIC + body + struct.pack(">H", crc16_ccitt(body))


class HUDTransport:
    """Length-framed serial writer; incomplete frames are ignored by firmware."""

    def __init__(self, port: str | None = None, baudrate: int = 115200, *,
                 stream: BinaryIO | None = None, reset_delay: float = 1.5,
                 auto_clock_sync: bool | None = None,
                 clock_sync_interval: float = CLOCK_SYNC_INTERVAL_SECONDS,
                 wall_clock: Callable[[], float] = time.time,
                 monotonic_clock: Callable[[], float] = time.monotonic,
                 utc_offset_provider: Callable[[float], int] =
                 system_utc_offset_minutes) -> None:
        self.port = port
        self.baudrate = baudrate
        self._owns_stream = stream is None and port is not None
        self._reset_delay = reset_delay
        self._auto_clock_sync = (
            self._owns_stream if auto_clock_sync is None else auto_clock_sync
        )
        if clock_sync_interval <= 0:
            raise ValueError("Clock synchronization interval must be positive")
        self._clock_sync_interval = float(clock_sync_interval)
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._utc_offset_provider = utc_offset_provider
        self._next_clock_sync_at = 0.0
        self._next_reconnect_at = 0.0
        self._last_sent: dict[str, str] = {}
        self._sent_frame_counts: dict[str, int] = {}
        self._warning_emitted = False
        self._closed = False
        if stream is not None:
            self.stream = stream
        elif port is None:
            self.stream = None
        else:
            self.stream = None
            self._open_serial(reset_delay)
        if self.stream is not None and self._auto_clock_sync:
            self.sync_clock(force=True)

    @property
    def enabled(self) -> bool:
        return self.stream is not None

    def send(self, message_type: str, text: str) -> bool:
        if self.stream is None:
            return False
        normalized = normalize_hud_text(text)
        if self._last_sent.get(message_type) == normalized:
            return False
        try:
            self.stream.write(encode_hud_frame(message_type, normalized))
            flush = getattr(self.stream, "flush", None)
            if flush is not None:
                flush()
        except Exception as exc:
            self._disable(f"HUD serial write failed: {exc}")
            return False
        self._last_sent[message_type] = normalized
        self._sent_frame_counts[message_type] = (
            self._sent_frame_counts.get(message_type, 0) + 1
        )
        return True

    @property
    def counters(self) -> dict[str, int]:
        return {
            "C_frames_sent": self._sent_frame_counts.get(
                "environmental_sound", 0
            )
        }

    def sync_clock(self, *, force: bool = False) -> bool:
        """Send a current OS-derived CLOCK frame when due."""
        if self.stream is None:
            return False
        now_monotonic = self._monotonic_clock()
        if not force and now_monotonic < self._next_clock_sync_at:
            return False
        try:
            epoch = self._wall_clock()
            payload = encode_clock_payload(
                epoch, self._utc_offset_provider(epoch)
            )
        except (OverflowError, OSError, TypeError, ValueError,
                RuntimeError) as exc:
            self._next_clock_sync_at = now_monotonic + min(
                60.0, self._clock_sync_interval
            )
            self._warn(f"HUD clock synchronization skipped: {exc}")
            return False
        if force:
            self._last_sent.pop("clock", None)
        if not self.send("clock", payload):
            return False
        self._next_clock_sync_at = now_monotonic + self._clock_sync_interval
        return True

    def update(self) -> None:
        """Run non-blocking periodic clock maintenance from the app loop."""
        if self.stream is None:
            if (not self._closed and self._owns_stream and
                    self.port is not None and
                    self._monotonic_clock() >= self._next_reconnect_at):
                self._next_reconnect_at = (
                    self._monotonic_clock() + CLOCK_RECONNECT_INTERVAL_SECONDS
                )
                try:
                    self._open_serial(0)
                except Exception as exc:
                    self._warn(f"HUD serial reconnect failed: {exc}")
                    return
                self._last_sent.clear()
                if self._auto_clock_sync:
                    self.sync_clock(force=True)
            return
        if self._auto_clock_sync:
            self.sync_clock()

    def set_subtitle(self, text: str) -> None:
        self.send("subtitle", text)
        self._last_sent.pop("partial_subtitle", None)

    def set_partial_subtitle(self, text: str) -> None:
        # A partial frame marks a new live utterance, so an identical later
        # final still has to reach firmware and start a fresh page lifecycle.
        self._last_sent.pop("subtitle", None)
        self.send("partial_subtitle", text)

    def set_alert(self, text: str) -> None:
        self.send("alert", text)

    def set_alert_state(self, event_label: str = "",
                        help_active: bool = False) -> None:
        self.send(
            "alert_state",
            encode_alert_state_payload(event_label, help_active),
        )

    def set_environmental_sound(self, text: str) -> None:
        self.send("environmental_sound", text)

    def set_status(self, text: str) -> None:
        self.send("status", text)

    def close(self) -> None:
        self._closed = True
        stream = self.stream
        self.stream = None
        if self._owns_stream and stream is not None:
            try:
                stream.close()
            except Exception as exc:
                self._warn(f"HUD serial close failed: {exc}")

    def _open_serial(self, reset_delay: float) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "HUD serial output requires pyserial: python -m pip install pyserial"
            ) from exc
        # ESP32 DevKit boards wire DTR/RTS to EN/GPIO0 for auto-programming.
        # Configure both lines inactive before opening the port so a normal
        # HUD message connection does not reset the running firmware.
        serial_port = serial.Serial(port=None, baudrate=self.baudrate, timeout=0)
        serial_port.dtr = False
        serial_port.rts = False
        serial_port.port = self.port
        try:
            serial_port.open()
        except Exception:
            try:
                serial_port.close()
            except Exception:
                pass
            raise
        self.stream = serial_port
        if reset_delay > 0:
            time.sleep(reset_delay)

    def _warn(self, message: str) -> None:
        if not self._warning_emitted:
            print(f"HUD warning: {message}")
            self._warning_emitted = True

    def _disable(self, message: str) -> None:
        stream = self.stream
        self.stream = None
        self._next_reconnect_at = (
            self._monotonic_clock() + CLOCK_RECONNECT_INTERVAL_SECONDS
        )
        self._warn(message)
        if self._owns_stream and stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def __enter__(self) -> "HUDTransport":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
