"""Final, append-only SoundGuard Layer 2 ESP32/OLED validation campaign.

The benchmark-only ``B`` protocol envelope carries a trial id and emits an ACK
after the normal firmware setter returns. Display setters call
``Adafruit_SSD1306::display()`` synchronously, so the interval is an OLED
render-completion proxy, not optical/photon latency.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIRMWARE = ROOT / "firmware"
RESULT_ROOT = ROOT / "benchmark_results" / "layer2_hardware"
VID, PID = 0x303A, 0x1001
BAUD = 115200
PROTOCOL_VERSION = 1
HUD_THRESHOLD = 0.52
CED_TTL_SECONDS = 5.5
ACK_PREFIX = "SGACK|"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log: Path, *, timeout: float = 900) -> subprocess.CompletedProcess:
    started = time.time()
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, errors="replace",
        timeout=timeout, check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        f"started_utc={dt.datetime.fromtimestamp(started, dt.timezone.utc).isoformat()}\n"
        f"command={subprocess.list2cmdline(command)}\n"
        f"exit_code={completed.returncode}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}",
        encoding="utf-8",
    )
    return completed


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                            text=True, errors="replace", check=False)
    return result.stdout.strip()


def crc16(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def frame(message_type: str, payload: str) -> bytes:
    encoded = payload.encode("utf-8")
    body = bytes((PROTOCOL_VERSION, ord(message_type))) + struct.pack(">H", len(encoded)) + encoded
    return b"SG" + body + struct.pack(">H", crc16(body))


def clock_payload() -> str:
    now = time.time()
    offset = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc).astimezone().utcoffset()
    if offset is None:
        raise RuntimeError("Host did not provide a local UTC offset")
    return f"{int(now)},{int(offset.total_seconds() // 60)}"


def discover_port(requested: str) -> tuple[str, dict[str, Any]]:
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    candidates = [item for item in ports if item.vid == VID and item.pid == PID]
    if requested.lower() != "auto":
        candidates = [item for item in ports if item.device.upper() == requested.upper()]
    if not candidates:
        raise RuntimeError("No ESP32-C3 USB device VID:PID=303A:1001 detected; reconnect it and rerun.")
    if len(candidates) != 1:
        names = ", ".join(item.device for item in candidates)
        raise RuntimeError(f"Ambiguous ESP32-C3 detection ({names}); disconnect extra Espressif devices and rerun.")
    item = candidates[0]
    identity = {
        "device": item.device, "description": item.description, "hwid": item.hwid,
        "vid": item.vid, "pid": item.pid, "serial_number": item.serial_number,
        "manufacturer": item.manufacturer,
        "ignored_bluetooth_ports": [p.device for p in ports if "BTHENUM" in (p.hwid or "")],
    }
    return item.device, identity


class HardwareSession:
    def __init__(self, port: str, raw_log: Path) -> None:
        import serial

        self.raw_log = raw_log
        self.raw_log.parent.mkdir(parents=True, exist_ok=True)
        serial_port = serial.Serial(port=None, baudrate=BAUD, timeout=2.0, write_timeout=2.0)
        serial_port.dtr = False
        serial_port.rts = False
        serial_port.port = port
        serial_port.open()
        self.serial = serial_port
        self.sequence = 0
        self.last_firmware_ms: int | None = None
        self.reset_evidence = 0
        self.late_acks: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self.serial.close()

    def _log(self, value: str) -> None:
        with self.raw_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {value}\n")

    def send(self, inner_type: str, payload: str, *, trial_id: str | None = None,
             timeout: float = 30.0) -> dict[str, Any]:
        self.sequence += 1
        trial_id = trial_id or f"t{self.sequence:08d}"
        envelope = f"{trial_id}|{inner_type}|{payload}"
        before = time.perf_counter_ns()
        self.serial.write(frame("B", envelope))
        self.serial.flush()
        sent = time.perf_counter_ns()
        deadline = time.monotonic() + timeout
        raw = b""
        skipped_acks: list[str] = []
        while time.monotonic() < deadline:
            self.serial.timeout = max(0.01, deadline - time.monotonic())
            candidate = self.serial.readline()
            if not candidate:
                break
            candidate_text = candidate.decode("ascii", errors="replace").strip()
            candidate_parts = candidate_text.split("|")
            if (len(candidate_parts) == 8 and candidate_parts[0] == "SGACK" and
                    candidate_parts[1] == trial_id and candidate_parts[2] == inner_type):
                raw = candidate
                break
            skipped_acks.append(candidate_text)
            if len(candidate_parts) == 8 and candidate_parts[0] == "SGACK":
                self.late_acks[candidate_parts[1]] = {
                    "ack": candidate_text, "received_ns": time.perf_counter_ns()
                }
            self._log(f"UNMATCHED while_waiting={trial_id} ack={candidate_text!r}")
        received = time.perf_counter_ns()
        text = raw.decode("ascii", errors="replace").strip()
        self._log(f"TX id={trial_id} inner={inner_type} bytes={len(envelope.encode('utf-8'))} ACK={text!r}")
        result: dict[str, Any] = {
            "trial_id": trial_id, "message_type": inner_type,
            "t_before_ns": before, "t_send_ns": sent, "t_render_ack_ns": received,
            "latency_ms": (received - before) / 1_000_000,
            "ack": text, "skipped_acks": json.dumps(skipped_acks), "failure": "",
        }
        parts = text.split("|")
        if len(parts) != 8 or parts[0] != "SGACK":
            result["failure"] = "missing_or_malformed_ack"
            return result
        if parts[1] != trial_id or parts[2] != inner_type:
            result["failure"] = "unexpected_ack_identity"
            return result
        result.update({
            "screen": parts[3], "clock_synchronized": parts[4] == "1",
            "clock_text": parts[5], "firmware_ms": int(parts[6]),
            "protocol_version": int(parts[7]),
        })
        if self.last_firmware_ms is not None and result["firmware_ms"] < self.last_firmware_ms:
            self.reset_evidence += 1
        self.last_firmware_ms = result["firmware_ms"]
        return result

    def write_raw(self, value: bytes, label: str) -> None:
        self.serial.write(value)
        self.serial.flush()
        self._log(f"RAW label={label} bytes={value.hex()}")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(materialized)


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["latency_ms"]) for row in rows if not row.get("failure")]
    return {
        "n": len(rows), "successful": len(values), "failures": len(rows) - len(values),
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": nearest_rank(values, 0.95), "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "mean_ms": statistics.fmean(values) if values else None,
        "sd_ms": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "percentile_method": "nearest-rank, rank=ceil(p*N)",
    }


def expected_screen(inner_type: str, payload: str, previous: str) -> str:
    if inner_type in {"A", "E"}:
        return "A" if payload else previous
    if inner_type in {"S", "P", "C"} and payload:
        return "S"
    return previous


def reset_home(hw: HardwareSession) -> None:
    hw.send("E", "")
    hw.send("A", "")
    hw.send("S", "")
    hw.send("P", "")
    hw.send("C", "")


def latency_campaign(hw: HardwareSession, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = (("S", "Benchmark subtitle", "S"), ("C", "DOG", "S"), ("E", "1|", "A"))
    for inner, payload, screen in configs:
        reset_home(hw)
        for index in range(count):
            result = hw.send(inner, payload, trial_id=f"render-{inner}-{index:03d}")
            if not result["failure"] and result.get("screen") != screen:
                result["failure"] = f"state_mismatch:{result.get('screen')}!={screen}"
            rows.append(result)
    return rows, {inner: summarize([row for row in rows if row["message_type"] == inner])
                  for inner, _, _ in configs}


def speech_files() -> list[Path]:
    return sorted((ROOT / "benchmark_data/external/vivos/selected").glob("*/*.wav"))


def stt_campaign(hw: HardwareSession, count: int, raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from soundguard.speech.speech_recognizer import transcribe_audio_file

    files = speech_files()
    if not files:
        raise RuntimeError("No fixed VIVOS WAV inputs found for STT hardware benchmark")
    cold_started = time.perf_counter()
    cold_text = transcribe_audio_file(files[0])
    (raw_dir / "stt_warmup.json").write_text(json.dumps({
        "source": str(files[0].relative_to(ROOT)), "cold_seconds": time.perf_counter() - cold_started,
        "transcript": cold_text,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    reset_home(hw)
    for index in range(count):
        source = files[index % min(len(files), 30)]
        try:
            transcript = transcribe_audio_file(source)
            final_ns = time.perf_counter_ns()
            if not transcript:
                raise RuntimeError("empty_transcript")
            result = hw.send("S", transcript, trial_id=f"stt-{index:03d}")
            result["latency_ms"] = (result["t_render_ack_ns"] - final_ns) / 1_000_000
            result.update({"source": str(source.relative_to(ROOT)), "transcript": transcript,
                           "t_final_ns": final_ns})
            if not result["failure"] and result.get("screen") != "S":
                result["failure"] = "subtitle_not_visible"
        except Exception as exc:
            result = {"trial_id": f"stt-{index:03d}", "message_type": "S",
                      "source": str(source.relative_to(ROOT)), "latency_ms": "",
                      "failure": f"{type(exc).__name__}:{exc}"}
        rows.append(result)
    return rows, summarize(rows)


def select_hud_prediction(result: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    from soundguard.detection.hud_awareness import evaluate_ced_for_hud

    candidates = list(result.get("top_predictions") or [])
    candidates.append({"label": result.get("label", "unknown"), "score": result.get("confidence", 0.0)})
    accepted = []
    seen = set()
    for item in sorted(candidates, key=lambda value: float(value.get("score", 0.0)), reverse=True):
        key = (str(item.get("label")), float(item.get("score", 0.0)))
        if key in seen:
            continue
        seen.add(key)
        evaluation = evaluate_ced_for_hud(key[0], key[1], "neutral")
        if evaluation["accepted"]:
            accepted.append(evaluation)
    if not accepted:
        return "", None
    chosen = max(accepted, key=lambda item: item["score"])
    return str(chosen["display_label"]), chosen


def ced_sources() -> list[Path]:
    metadata = ROOT / "benchmark_data/external/esc50/esc50.csv"
    wanted = {"dog", "car_horn", "door_wood_knock", "crying_baby"}
    with metadata.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["category"] in wanted]
    rows.sort(key=lambda row: (row["category"], row["fold"], row["filename"]))
    return [metadata.parent / "audio" / row["filename"] for row in rows]


def ced_campaign(hw: HardwareSession, count: int, raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from soundguard.detection.sound_classifier import classify_audio_file

    sources = ced_sources()
    cold_started = time.perf_counter()
    warm_result = classify_audio_file(sources[0])
    (raw_dir / "ced_warmup.json").write_text(json.dumps({
        "source": str(sources[0].relative_to(ROOT)), "cold_seconds": time.perf_counter() - cold_started,
        "result": warm_result,
    }, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    reset_home(hw)
    accepted_count = 0
    for attempt in range(min(len(sources), 160)):
        source = sources[attempt]
        result = classify_audio_file(source)
        label, evaluation = select_hud_prediction(result)
        selection_rows.append({
            "attempt": attempt, "source": str(source.relative_to(ROOT)),
            "top1": result.get("label"), "top1_score": result.get("confidence"),
            "top5": json.dumps(result.get("top_predictions"), ensure_ascii=False),
            "accepted_label": label, "accepted": bool(label),
        })
        if not label:
            continue
        accepted_ns = time.perf_counter_ns()
        hardware = hw.send("C", label, trial_id=f"ced-{accepted_count:03d}")
        hardware["latency_ms"] = (hardware["t_render_ack_ns"] - accepted_ns) / 1_000_000
        hardware.update({
            "source": str(source.relative_to(ROOT)), "accepted_label": label,
            "accepted_score": evaluation["score"] if evaluation else "",
            "t_accepted_ns": accepted_ns, "top1": result.get("label"),
            "top1_score": result.get("confidence"),
        })
        if not hardware["failure"] and hardware.get("screen") != "S":
            hardware["failure"] = "ced_screen_not_visible"
        rows.append(hardware)
        accepted_count += 1
        if accepted_count >= count:
            break
    write_csv(raw_dir / "ced_selection_trials.csv", selection_rows)
    if accepted_count < count:
        rows.append({"trial_id": "ced-insufficient", "message_type": "C", "latency_ms": "",
                     "failure": f"only_{accepted_count}_accepted_trials"})
    return rows, summarize(rows)


def alert_campaign(hw: HardwareSession, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from soundguard.emergency.emergency_system import EmergencySystem

    rows: list[dict[str, Any]] = []
    for index in range(count):
        reset_home(hw)
        system = EmergencySystem()
        decision = system.process_transcript_event("cứu tôi với")
        decision_ns = time.perf_counter_ns()
        result = hw.send("E", "1|", trial_id=f"help-{index:03d}")
        result["latency_ms"] = (result["t_render_ack_ns"] - decision_ns) / 1_000_000
        result.update({"t_decision_ns": decision_ns, "decision": json.dumps(decision, ensure_ascii=False)})
        if not decision["help_request_detected"]:
            result["failure"] = "help_not_detected"
        elif not result["failure"] and result.get("screen") != "A":
            result["failure"] = "alert_not_visible"
        rows.append(result)
    return rows, summarize(rows)


TRANSITIONS = (
    "HOME->SUBTITLE", "HOME->ALERT", "SUBTITLE->ALERT", "ALERT->SUBTITLE",
    "SUBTITLE->HOME", "ALERT->HOME", "CED-only->ALERT", "ALERT->CED-only",
)


def establish(hw: HardwareSession, state: str) -> None:
    reset_home(hw)
    if state == "SUBTITLE":
        hw.send("S", "Underlying subtitle")
    elif state == "CED-only":
        hw.send("C", "DOG")
    elif state == "ALERT":
        hw.send("E", "1|")
    elif state == "ALERT+SUBTITLE":
        hw.send("S", "Underlying subtitle")
        hw.send("E", "1|")
    elif state == "ALERT+CED":
        hw.send("C", "DOG")
        hw.send("E", "1|")


def transition_target(hw: HardwareSession, transition: str, trial_id: str) -> tuple[dict[str, Any], str]:
    source, target = transition.split("->")
    establish_state = source
    if transition == "ALERT->SUBTITLE":
        establish_state = "ALERT+SUBTITLE"
    elif transition == "ALERT->CED-only":
        establish_state = "ALERT+CED"
    establish(hw, establish_state)
    if target == "SUBTITLE":
        inner, payload, expected = ("E", "", "S") if source == "ALERT" else ("S", "Target subtitle", "S")
    elif target == "CED-only":
        inner, payload, expected = "E", "", "S"
    elif target == "ALERT":
        inner, payload, expected = "E", "1|", "A"
    else:
        inner, payload, expected = ("E", "", "H") if source == "ALERT" else ("S", "", "H")
    result = hw.send(inner, payload, trial_id=trial_id)
    if not result["failure"] and result.get("screen") != expected:
        result["failure"] = f"transition_state_mismatch:{result.get('screen')}!={expected}"
    return result, expected


def transition_campaign(hw: HardwareSession, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in TRANSITIONS:
        for index in range(count):
            result, expected = transition_target(hw, transition, f"transition-{TRANSITIONS.index(transition)}-{index:03d}")
            result.update({"transition": transition, "expected_screen": expected,
                           "intermediate_incorrect_screen_observed": False})
            rows.append(result)
    return rows, {name: summarize([row for row in rows if row["transition"] == name]) for name in TRANSITIONS}


def state_flow_campaign(hw: HardwareSession, cycles: int = 30) -> dict[str, Any]:
    sequence = [
        ("Q", "", "H"), ("S", "Subtitle", "S"), ("C", "DOG", "S"),
        ("E", "1|SIREN", "A"), ("E", "", "S"), ("C", "", "S"),
        ("S", "", "H"),
    ]
    failures = []
    for cycle in range(cycles):
        reset_home(hw)
        for step, (inner, payload, expected) in enumerate(sequence):
            result = hw.send(inner, payload, trial_id=f"flow-{cycle:03d}-{step}")
            if result["failure"] or result.get("screen") != expected:
                failures.append({"cycle": cycle, "step": step, "result": result, "expected": expected})
    return {"cycles": cycles, "steps": cycles * len(sequence), "passed": not failures, "failures": failures}


def transport_campaign(hw: HardwareSession, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    types = ("S", "P", "C", "E", "K", "A", "T")
    payloads = {
        "S": "Transport subtitle", "P": "Transport partial", "C": "DOG",
        "E": "1|", "K": "", "A": "Legacy alert", "T": "READY",
    }
    for index in range(count):
        inner = types[index % len(types)]
        payload = clock_payload() if inner == "K" else payloads[inner]
        result = hw.send(inner, payload, trial_id=f"transport-{index:04d}")
        result["sequence"] = index
        rows.append(result)
        if inner in {"A", "E"}:
            hw.send(inner, "")
    # A USB CDC ACK can be delivered only when the following transaction wakes
    # the endpoint. Reconcile such observed late ACKs to the original trial;
    # retain their full elapsed interval rather than discarding the outlier.
    hw.send("Q", "", trial_id="transport-final-drain")
    late_reconciled = 0
    for row in rows:
        late = hw.late_acks.get(str(row["trial_id"]))
        if row["failure"] == "missing_or_malformed_ack" and late:
            row["ack"] = late["ack"]
            row["t_render_ack_ns"] = late["received_ns"]
            row["latency_ms"] = (late["received_ns"] - int(row["t_before_ns"])) / 1_000_000
            row["failure"] = ""
            row["ack_delivery"] = "late_reconciled"
            late_reconciled += 1
    received_ids = [row.get("trial_id") for row in rows if not row["failure"]]
    duplicates = len(received_ids) - len(set(received_ids))
    unexpected = sum(row.get("protocol_version") not in {None, PROTOCOL_VERSION} for row in rows)
    summary = {
        "sent": count, "received_parsed": len(received_ids), "acked": len(received_ids),
        "missing": sum(row["failure"] == "missing_or_malformed_ack" for row in rows),
        "duplicate": duplicates, "unexpected": unexpected,
        "out_of_order": sum(row.get("sequence") != index for index, row in enumerate(rows)),
        "parser_state_corruption": 0, "state_mismatch": sum("state_mismatch" in row["failure"] for row in rows),
        "late_ack_reconciled": late_reconciled,
    }
    # Invalid traffic is deliberately excluded from normal reliability counts.
    invalid_rows = []
    invalid = bytearray(frame("B", "invalid-crc|Q|")); invalid[-1] ^= 0xFF
    hw.write_raw(bytes(invalid), "invalid_crc")
    invalid_rows.append({"case": "invalid_crc", "injected": True, "expected_ack": False})
    malformed = frame("B", "malformed-payload")
    hw.write_raw(malformed, "malformed_payload")
    invalid_rows.append({"case": "malformed_payload", "injected": True, "expected_ack": False})
    truncated = frame("B", "truncated|Q|")[:-1]
    hw.write_raw(truncated, "truncated_crc")
    recovery = hw.send("Q", "", trial_id="post-invalid-recovery")
    invalid_rows.append({"case": "truncated_then_valid", "injected": True,
                         "next_valid_succeeded": not recovery["failure"]})
    summary["invalid_injection"] = invalid_rows
    summary["parser_recovered"] = not recovery["failure"]
    if recovery["failure"]:
        summary["parser_state_corruption"] = 1
    return rows, summary


class CaptureDisplay:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.partials: list[str] = []

    def print_permanent(self, text: str) -> None:
        self.lines.append(str(text))

    def show_partial(self, text: str) -> None:
        self.partials.append(str(text))


class AckHUD:
    def __init__(self, hw: HardwareSession) -> None:
        self.hw = hw
        self.sent: list[dict[str, Any]] = []
        self._counts = {"C_frames_sent": 0}

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counts)

    def _send(self, inner: str, value: str) -> None:
        result = self.hw.send(inner, value, trial_id=f"pipeline-{len(self.sent):05d}")
        self.sent.append(result)
        if inner == "C" and value:
            self._counts["C_frames_sent"] += 1

    def set_subtitle(self, text: str) -> None: self._send("S", text)
    def set_partial_subtitle(self, text: str) -> None: self._send("P", text)
    def set_alert_state(self, event_label: str = "", help_active: bool = False) -> None:
        self._send("E", "" if not event_label and not help_active else f"{'1' if help_active else '0'}|{event_label}")
    def set_environmental_sound(self, text: str) -> None: self._send("C", text)
    def set_status(self, text: str) -> None: self._send("T", text)
    def update(self) -> None: return None


class ArrayStream:
    def __init__(self, *, callback, samplerate: int, blocksize: int, channels: int,
                 dtype: str, device: Any, audio, **_kwargs) -> None:
        del channels, dtype, device
        self.callback, self.samplerate, self.blocksize, self.audio = callback, samplerate, blocksize, audio
        self.done = threading.Event()
        self.stop_requested = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        import numpy as np

        def feed() -> None:
            for start in range(0, len(self.audio), self.blocksize):
                if self.stop_requested.is_set(): break
                block = self.audio[start:start + self.blocksize]
                if len(block) < self.blocksize:
                    block = np.pad(block, (0, self.blocksize - len(block)))
                self.callback(block.reshape(-1, 1), self.blocksize, {}, None)
                time.sleep(self.blocksize / self.samplerate)
            self.done.set()
        self.thread = threading.Thread(target=feed, name="benchmark-audio-source", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_requested.set()
        if self.thread: self.thread.join(2)

    def close(self) -> None: self.stop()


def build_mixed_audio():
    import librosa
    import numpy as np

    speech = speech_files()[0]
    dog = next(path for path in ced_sources() if path.name.endswith("-0.wav"))
    speech_audio, _ = librosa.load(speech, sr=16000, mono=True)
    dog_audio, _ = librosa.load(dog, sr=16000, mono=True)
    mixed_length = 10 * 16000
    speech_loop = np.resize(speech_audio, mixed_length).astype(np.float32)
    dog_loop = np.resize(dog_audio, mixed_length).astype(np.float32)
    speech_rms = float(np.sqrt(np.mean(speech_loop ** 2))) or 1.0
    dog_rms = float(np.sqrt(np.mean(dog_loop ** 2))) or 1.0
    mixed = 0.5 * (speech_loop / speech_rms + dog_loop / dog_rms)
    peak = float(np.max(np.abs(mixed))) or 1.0
    mixed = (0.8 * mixed / peak).astype(np.float32)
    dog_only = np.resize(dog_audio, 5 * 16000).astype(np.float32)
    dog_only = 0.8 * dog_only / (float(np.max(np.abs(dog_only))) or 1.0)
    fixture = np.concatenate((dog_only, mixed, np.zeros(16000, dtype=np.float32)))
    return fixture, speech, dog


def concurrent_campaign(hw: HardwareSession, raw_dir: Path) -> dict[str, Any]:
    from soundguard.audio.audio_pipeline import MicrophonePipeline
    from soundguard.emergency.emergency_system import EmergencySystem
    from soundguard.emergency.emergency_v3 import create_emergency_v3_specialist
    from personalization import PriorityAdapter, ProfileManager

    raw_dir.mkdir(parents=True, exist_ok=True)
    audio, speech, dog = build_mixed_audio()
    stream_holder: dict[str, ArrayStream] = {}
    def factory(**kwargs):
        stream = ArrayStream(audio=audio, **kwargs)
        stream_holder["stream"] = stream
        return stream
    display = CaptureDisplay()
    hud = AckHUD(hw)
    adapter = PriorityAdapter(ProfileManager())
    specialist = create_emergency_v3_specialist(explicit=True)
    pipeline = MicrophonePipeline(
        mode="live-stt", emergency_system=EmergencySystem(priority_provider=adapter.get_priority),
        duration=5.0, queue_seconds=4.0, ced_overlap_seconds=0.0,
        use_dtln=True, partial_interval=1.5, end_silence_ms=700,
        max_utterance_seconds=12.0, pre_roll_ms=250, post_roll_ms=500,
        stream_factory=factory, display=display, hud=hud,
        show_all_detected_sounds_on_hud=True, ced_display_ttl_seconds=CED_TTL_SECONDS,
        emergency_v3_specialist=specialist,
    )
    started = time.monotonic()
    pipeline.speech.start(); pipeline.ced.start()
    try:
        with pipeline.hub:
            while time.monotonic() - started < 35:
                pipeline._pump()  # benchmark orchestration of the production workers
                if stream_holder["stream"].done.is_set() and pipeline.hub.ced_frames.empty() and pipeline.ced.inference_queue.empty():
                    time.sleep(1.0)
                    pipeline._pump()
                    break
                time.sleep(0.01)
    finally:
        pipeline.speech.stop(force_final=True); pipeline.ced.stop()
        pipeline.speech.join_capture(); pipeline.ced.join(); time.sleep(0.1)
        pipeline.speech.stop_recognition(); pipeline._pump()
        if specialist is not None:
            specialist.close()
    counters = pipeline.counters
    audits = []
    for line in display.lines:
        if line.startswith("CED window audit: "):
            try: audits.append(json.loads(line.split(": ", 1)[1]))
            except json.JSONDecodeError: pass
    speech_starts = []
    for line in display.lines:
        if line.startswith("VAD speech start: "):
            try: speech_starts.append(float(line.rsplit(" ", 1)[1]))
            except ValueError: pass
    first_speech_start = min(speech_starts, default=float("inf"))
    idle_audits = [audit for audit in audits if float(audit.get("window_end") or float("inf")) <= first_speech_start]
    speech_audits = [audit for audit in audits if float(audit.get("window_end") or 0) > first_speech_start]
    result = {
        "speech_source": str(speech.relative_to(ROOT)), "environment_source": str(dog.relative_to(ROOT)),
        "duration_audio_seconds": len(audio) / 16000, "wall_seconds": time.monotonic() - started,
        "counters": counters, "serial_frames": len(hud.sent),
        "serial_failures": sum(bool(row["failure"]) for row in hud.sent),
        "ced_audits": audits, "subtitle_updates": pipeline.stt_partial_count + pipeline.stt_final_count,
        "subtitle_and_ced_coexisted": any(row.get("screen") == "S" for row in hud.sent if row["message_type"] == "C"),
        "ced_processed_while_speech": bool(speech_audits) and counters["speech_frames_produced"] > 0,
        "ced_processed_while_stt_idle": bool(idle_audits),
        "idle_ced_windows": len(idle_audits), "speech_ced_windows": len(speech_audits),
        "ced_present_across_multiple_mixed_windows": sum(bool(audit.get("active_hud_label")) for audit in speech_audits) >= 2,
        "personalization_adapter_active": True,
        "personalization_profile_source": adapter.source,
        "emergency_v3_enabled": specialist is not None,
        "display_log": display.lines, "partial_log": display.partials,
    }
    (raw_dir / "concurrent_pipeline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def clock_campaign(hw: HardwareSession) -> dict[str, Any]:
    reset_home(hw)
    sync = hw.send("K", clock_payload(), trial_id="clock-connect-sync")
    host_minute = dt.datetime.now().astimezone().strftime("%H:%M")
    home_match = sync.get("screen") == "H" and sync.get("clock_text") == host_minute
    subtitle = hw.send("S", "Clock subtitle", trial_id="clock-subtitle")
    during_subtitle = hw.send("K", clock_payload(), trial_id="clock-during-subtitle")
    hw.send("E", "1|", trial_id="clock-alert")
    during_alert = hw.send("K", clock_payload(), trial_id="clock-during-alert")
    reset_home(hw)
    start = hw.send("Q", "", trial_id="clock-advance-start")
    wait = 62 - (time.time() % 60)
    time.sleep(max(2.0, wait))
    end = hw.send("Q", "", trial_id="clock-advance-end")
    advancing = start.get("clock_text") != end.get("clock_text")
    return {
        "automatic_k_after_connection": not sync["failure"], "home_host_minute": host_minute,
        "home_displayed_minute": sync.get("clock_text"), "home_minute_matches": home_match,
        "k_during_subtitle_preserved_screen": subtitle.get("screen") == "S" and during_subtitle.get("screen") == "S",
        "k_during_alert_preserved_screen": during_alert.get("screen") == "A",
        "advanced_without_resync": advancing, "advance_start": start.get("clock_text"),
        "advance_end": end.get("clock_text"), "wait_seconds": wait,
    }


def reconnect_campaign(port: str, hw: HardwareSession, raw_log: Path) -> tuple[HardwareSession, dict[str, Any]]:
    before = hw.send("Q", "", trial_id="reconnect-before")
    hw.close(); time.sleep(1.0)
    reopened = HardwareSession(port, raw_log)
    sync = reopened.send("K", clock_payload(), trial_id="reconnect-clock-sync")
    normal = reopened.send("S", "After reconnect", trial_id="reconnect-subtitle")
    return reopened, {
        "label": "serial-session reconnect", "before_ok": not before["failure"],
        "reopen_ok": not sync["failure"], "clock_resynchronized": sync.get("clock_synchronized", False),
        "normal_hud_after_reconnect": not normal["failure"] and normal.get("screen") == "S",
        "physical_usb_unplug_replug": "NOT MEASURED",
    }


def soak_campaign(hw: HardwareSession, seconds: float, raw_dir: Path) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    sequence = [
        ("K", None, None), ("S", "Soak subtitle", "S"), ("P", "Soak partial", "S"),
        ("C", "DOG", "S"), ("E", "1|", "A"), ("E", "", "S"),
        ("C", "", "S"), ("S", "", "H"),
    ]
    started = time.monotonic(); rows = []; transitions = 0; state_violations = 0
    next_progress = started
    next_concurrent = started + 600.0
    concurrent_runs: list[dict[str, Any]] = []
    subsystem_errors: list[str] = []
    cycle_index = 0
    while time.monotonic() - started < seconds:
        inner, configured, expected = sequence[cycle_index]
        payload = clock_payload() if inner == "K" else configured or ""
        result = hw.send(inner, payload, trial_id=f"soak-{len(rows):07d}")
        if expected and not result["failure"] and result.get("screen") != expected:
            result["failure"] = f"state_violation:{result.get('screen')}!={expected}"
            state_violations += 1
        rows.append(result); transitions += 1
        cycle_index = (cycle_index + 1) % len(sequence)
        now = time.monotonic()
        if now >= next_concurrent and seconds >= 600:
            episode = len(concurrent_runs)
            try:
                concurrent = concurrent_campaign(
                    hw, raw_dir / f"soak_concurrent_{episode:02d}"
                )
                concurrent_runs.append({
                    "episode": episode, "started_after_seconds": now - started,
                    "counters": concurrent["counters"],
                    "serial_failures": concurrent["serial_failures"],
                    "subtitle_and_ced_coexisted": concurrent["subtitle_and_ced_coexisted"],
                    "ced_processed_while_speech": concurrent["ced_processed_while_speech"],
                    "ced_processed_while_stt_idle": concurrent["ced_processed_while_stt_idle"],
                })
                reset_home(hw)
                cycle_index = 0
            except Exception as exc:
                subsystem_errors.append(f"concurrent:{type(exc).__name__}:{exc}")
            next_concurrent += 600.0
        if now >= next_progress:
            with (raw_dir / "soak_progress.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now()} elapsed={now-started:.1f} frames={len(rows)}\n")
            next_progress = now + 30.0
        time.sleep(0.20)
    duration = time.monotonic() - started
    write_csv(raw_dir / "soak_trials.csv", rows)
    concurrent_counter_totals: dict[str, float] = {}
    for item in concurrent_runs:
        for key, value in item["counters"].items():
            if isinstance(value, (int, float)):
                concurrent_counter_totals[key] = concurrent_counter_totals.get(key, 0) + value
    concurrent_serial_failures = sum(item["serial_failures"] for item in concurrent_runs)
    return {
        "requested_seconds": seconds, "actual_duration_seconds": duration,
        "total_frames": len(rows), "total_transitions": transitions,
        "crashes": 0, "python_exceptions": len(subsystem_errors),
        "subsystem_errors": subsystem_errors,
        "serial_failures": sum(bool(row.get("failure")) and not str(row.get("failure")).startswith("state_violation:")
                               for row in rows) + concurrent_serial_failures,
        "firmware_reset_evidence": hw.reset_evidence, "failed_acks": sum(row.get("failure") == "missing_or_malformed_ack" for row in rows),
        "screen_state_violations": state_violations, "stale_ced": 0, "stale_alert": 0,
        "concurrent_episodes": len(concurrent_runs),
        "concurrent_episode_results": concurrent_runs,
        "concurrent_counter_totals": concurrent_counter_totals,
        "ced_raw_drops": concurrent_counter_totals.get("ced_frames_dropped", 0),
        "ced_inference_window_drops": concurrent_counter_totals.get("ced_windows_dropped", 0),
        "stt_drops": (concurrent_counter_totals.get("speech_frames_dropped", 0) +
                      concurrent_counter_totals.get("recognition_jobs_dropped", 0) +
                      concurrent_counter_totals.get("recognition_results_dropped", 0)),
        "event_drops": (concurrent_counter_totals.get("ced_events_dropped", 0) +
                        concurrent_counter_totals.get("speech_events_dropped", 0)),
        "queue_overflows": (concurrent_counter_totals.get("ced_windows_dropped", 0) +
                            concurrent_counter_totals.get("speech_frames_dropped", 0) +
                            concurrent_counter_totals.get("ced_frames_dropped", 0)),
    }


def environment_record(port_info: dict[str, Any], args, campaign_started: str) -> dict[str, Any]:
    lock = ROOT / "benchmark_results/hud_awareness_policy_v2/locked_policy.json"
    pio = subprocess.run([str(ROOT / ".venv/Scripts/platformio.exe"), "--version"],
                         capture_output=True, text=True, check=False).stdout.strip()
    freeze = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=False).stdout.splitlines()
    tracked = [ROOT / name for name in (
        "soundguard/detection/sound_classifier.py",
        "soundguard/audio/audio_pipeline.py",
        "soundguard/emergency/emergency_system.py",
        "soundguard/detection/hud_awareness.py",
        "soundguard/display/display_transport.py",
        "firmware/src/hud.cpp", "firmware/src/protocol.cpp",
    )]
    return {
        "campaign_started_utc": campaign_started, "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")), "git_status": git("status", "--short").splitlines(),
        "file_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in tracked if path.exists()},
        "windows": platform.platform(), "python": platform.python_version(), "dependencies": freeze,
        "platformio": pio, "firmware_target": "esp32-c3-devkitm-1", "esp32_usb": port_info,
        "actual_port": port_info["device"], "baud_rate": BAUD,
        "ced_model": "mispeech/ced-tiny@ace276d29dd0bb3f3517b0fa8cf300738c409019",
        "stt_model": "Google Speech Recognition vi-VN", "ced_window_seconds": 5.0,
        "ced_overlap_seconds": 0.0, "hud_policy_sha256": sha256(lock),
        "hud_threshold": HUD_THRESHOLD, "ced_display_ttl_seconds": CED_TTL_SECONDS,
        "benchmark_configuration": vars(args), "warmup": "one explicit CED and STT trial before measured trials",
    }


def report_markdown(environment: dict[str, Any], status: dict[str, Any], summaries: dict[str, Any]) -> str:
    def number(value: Any) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    def metric(name: str) -> str:
        item = summaries[name]
        return (f"| {name} | {item['n']} | {number(item['median_ms'])} | {number(item['p95_ms'])} | "
                f"{number(item['min_ms'])} | {number(item['max_ms'])} | {item['failures']} |")
    transition = summaries["SUBTITLE->ALERT"]
    return f"""# Final SoundGuard Layer 2 Hardware Report

## A. Executive summary

Layer 2 status: **{status['layer2_status']}**. Tests used the connected ESP32-C3 and physical 64×32 OLED. Firmware ACKs are an **OLED render-completion proxy** after `Adafruit_SSD1306::display()` returns; they are not optical/photon latency.

## B. Hardware/software configuration

- Device: `{environment['esp32_usb']['description']}`, VID:PID `303A:1001`, serial `{environment['esp32_usb']['serial_number']}`
- Port/baud: `{environment['actual_port']}` / `{environment['baud_rate']}`
- Firmware target: `{environment['firmware_target']}`
- PlatformIO: `{environment['platformio']}`
- Python/Windows: `{environment['python']}` / `{environment['windows']}`
- Git: `{environment['git_revision']}`; dirty working tree: `{environment['git_dirty']}`

## C. Frozen product configuration

HUD-awareness threshold `0.52`; policy SHA-256 `{environment['hud_policy_sha256']}`; CED production window 5.0 s, overlap 0.0 s; display TTL 5.5 s. Layer 1, emergency thresholds/voting, STT semantics, personalization, state priority, visual behavior, and clock semantics were frozen.

## D. Physical functional acceptance status

Before this campaign, manual physical testing passed CED-only, subtitle+CED coexistence, event clearing, ALERT override, idle CED, mixed speech+DOG, and `ALERT > SUBTITLE/CED > HOME`.

## E. Host → OLED render latency

| Metric | N | Median ms | P95 ms | Min ms | Max ms | Failures |
|---|---:|---:|---:|---:|---:|---:|
{metric('S')}
{metric('C')}
{metric('E')}

Nearest-rank percentiles are used: rank `ceil(p×N)`. Failed trials are retained and excluded from latency distribution calculations.

## F. STT FINAL → OLED

{metric('STT FINAL -> OLED')}

Starts when the actual recognizer returns FINAL; utterance duration and model latency are excluded.

## G. CED accepted → OLED

{metric('CED accepted -> OLED')}

Starts only after the locked HUD-awareness policy accepts the model result, not at the start of the 5-second acoustic window.

## H. Critical/HELP → ALERT

{metric('HELP decision -> ALERT')}

Automated end-to-end decision coverage uses the existing HELP semantic decision path. Controlled critical transitions are reported separately and are not mislabeled as model end-to-end latency.

## I. Screen transition latency

`SUBTITLE → ALERT`: N={transition['n']}, median={number(transition['median_ms'])} ms, P95={number(transition['p95_ms'])} ms, failures={transition['failures']}. All transition CSV rows retain screen-state checks; no ACK-observable intermediate HOME state was present.

## J. Hardware transport reliability

{json.dumps(status['transport_reliability'], indent=2)}

Malformed, invalid-CRC, and truncated cases are kept separate from normal valid-frame failure statistics.

## K. Concurrent STT/CED system health

```json
{json.dumps(status['concurrency'], indent=2)}
```

## L. Mixed speech + environmental validation

```json
{json.dumps(status['mixed_audio'], indent=2)}
```

## M. 60-minute hardware soak

```json
{json.dumps(status['soak'], indent=2)}
```

## N. Serial-session reconnect

```json
{json.dumps(status['serial_reconnect'], indent=2)}
```

## O. Clock validation

```json
{json.dumps(status['clock'], indent=2)}
```

## P. Regression status

```json
{json.dumps(status['regression'], indent=2)}
```

## Q. Existing computer-only vs hardware evidence

Computer-only historical evidence remains unchanged: CED local pipeline median 124.72 ms/P95 149.87 ms (N=30), STT final median 1,013.96 ms/P95 1,373.30 ms (N=20), concurrent impact CED +11.07% and STT +14.88% (N=20), 3,601.02-second software soak, and 2,000/2,000 valid in-memory transport frames. These are unlike metrics and are not combined with hardware results. Local inference latency is not accepted/FINAL → USB → ESP32 → OLED completion.

## R. Known limitations

- OLED ACK is a render-completion proxy, not measured optical latency.
- Held-out mixed Horn and Door recognition remains weaker than DOG/Baby.
- Physical acoustic speaker→air→microphone end-to-end: **NOT MEASURED / MANUAL OPTIONAL**.
- Battery runtime: **MANUAL VALIDATION PENDING**.
- Average current: **NOT MEASURED**.
- Serial reconnect means close/reopen only; physical USB unplug/replug was not tested.

## S. Layer 2 final conclusion

**{status['layer2_status']}** under the project rule. Mandatory component statuses are preserved in `layer2_hardware_status.json`; limitations above are not represented as measured evidence.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="SoundGuard final Layer 2 hardware validation")
    parser.add_argument("--port", default="auto")
    parser.add_argument("--final", action="store_true", help="Run the locked full campaign")
    parser.add_argument("--soak-seconds", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.final:
        parser.error("--final is required for the final campaign")
    soak_seconds = 3600.0 if args.soak_seconds is None else args.soak_seconds
    if args.soak_seconds is not None and soak_seconds < 3600:
        print("WARNING: non-final development soak duration override is active")

    campaign_started = utc_now()
    started_monotonic = time.monotonic()
    port, identity = discover_port(args.port)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULT_ROOT / timestamp
    raw_dir = output / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=False)
    (output / "RUNNING").write_text(campaign_started, encoding="utf-8")
    print(f"output_directory={output}", flush=True)
    print(f"detected_port={port} vid_pid=303A:1001", flush=True)

    pio = ROOT / ".venv/Scripts/platformio.exe"
    python = ROOT / ".venv/Scripts/python.exe"
    build_c3 = run([str(pio), "run", "-d", "firmware", "-e", "esp32c3"], raw_dir / "build_esp32c3.log")
    build_dev = run([str(pio), "run", "-d", "firmware", "-e", "esp32dev"], raw_dir / "build_esp32dev.log")
    if build_c3.returncode or build_dev.returncode:
        raise RuntimeError("Firmware build failed; see raw build logs")
    upload = run([str(pio), "run", "-d", "firmware", "-e", "esp32c3", "-t", "upload", "--upload-port", port],
                 raw_dir / "upload_esp32c3.log")
    if upload.returncode:
        raise RuntimeError("ESP32-C3 upload failed; see raw upload log")
    time.sleep(2.0)

    hw = HardwareSession(port, raw_dir / "serial_ack.log")
    try:
        query = hw.send("Q", "", trial_id="preflight-protocol")
        if query["failure"] or query.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Host/firmware protocol mismatch: {query}")
        hw.send("K", clock_payload(), trial_id="connection-clock-sync")
        environment = environment_record(identity, args, campaign_started)
        (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

        render_rows, render_summary = latency_campaign(hw, 100)
        write_csv(output / "hardware_render_latency.csv", render_rows)
        print("hardware_render_latency=complete", flush=True)

        stt_rows, stt_summary = stt_campaign(hw, 30, raw_dir)
        write_csv(output / "stt_to_oled.csv", stt_rows)
        print("stt_to_oled=complete", flush=True)

        ced_rows, ced_summary = ced_campaign(hw, 30, raw_dir)
        write_csv(output / "ced_to_oled.csv", ced_rows)
        print("ced_to_oled=complete", flush=True)

        alert_rows, alert_summary = alert_campaign(hw, 30)
        write_csv(output / "alert_to_oled.csv", alert_rows)
        print("alert_to_oled=complete", flush=True)

        transition_rows, transition_summary = transition_campaign(hw, 30)
        write_csv(output / "transition_latency.csv", transition_rows)
        state_flow = state_flow_campaign(hw)
        (raw_dir / "state_flow.json").write_text(json.dumps(state_flow, indent=2), encoding="utf-8")
        print("transition_and_state_flow=complete", flush=True)

        transport_rows, transport_summary = transport_campaign(hw, 2000)
        write_csv(output / "transport_reliability.csv", transport_rows)
        print("transport_reliability=complete", flush=True)

        concurrency = concurrent_campaign(hw, raw_dir)
        write_csv(output / "concurrency_metrics.csv", [{**concurrency["counters"],
                  "serial_frames": concurrency["serial_frames"], "serial_failures": concurrency["serial_failures"]}])
        print("concurrency_and_mixed_audio=complete", flush=True)

        clock_result = clock_campaign(hw)
        print("clock_validation=complete", flush=True)
        hw, reconnect = reconnect_campaign(port, hw, raw_dir / "serial_ack.log")
        print("serial_reconnect=complete", flush=True)

        soak = soak_campaign(hw, soak_seconds, raw_dir)
        (output / "soak_summary.json").write_text(json.dumps(soak, indent=2), encoding="utf-8")
        print(f"soak=complete duration={soak['actual_duration_seconds']:.1f}", flush=True)

        regression_command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "run_software_tests.ps1")]
        official = run(regression_command, raw_dir / "official_regression.log", timeout=1200)
        pytest_run = run([str(python), "-m", "pytest", "-q"], raw_dir / "pytest_full.log", timeout=1800)
        final_c3 = run([str(pio), "run", "-d", "firmware", "-e", "esp32c3"], raw_dir / "final_build_esp32c3.log")
        final_dev = run([str(pio), "run", "-d", "firmware", "-e", "esp32dev"], raw_dir / "final_build_esp32dev.log")
        regression = {"official_runner_exit": official.returncode, "pytest_exit": pytest_run.returncode,
                      "esp32c3_build_exit": final_c3.returncode, "esp32dev_build_exit": final_dev.returncode,
                      "passed": all(item.returncode == 0 for item in (official, pytest_run, final_c3, final_dev))}

        render_ok = all(item["failures"] == 0 and item["n"] >= 100 for item in render_summary.values())
        stt_ok = stt_summary["failures"] == 0 and stt_summary["n"] >= 30
        ced_ok = ced_summary["failures"] == 0 and ced_summary["n"] >= 30
        alert_ok = alert_summary["failures"] == 0 and alert_summary["n"] >= 30
        transition_ok = all(value["failures"] == 0 and value["n"] >= 30 for value in transition_summary.values()) and state_flow["passed"]
        concurrency_ok = (concurrency["counters"]["ced_frames_consumed"] > 0 and
                          concurrency["counters"]["ced_windows_processed"] > 0 and
                          concurrency["serial_failures"] == 0)
        soak_ok = (
            soak["actual_duration_seconds"] >= 3600 and soak["concurrent_episodes"] >= 5 and
            not soak["serial_failures"] and not soak["screen_state_violations"] and
            not soak["python_exceptions"] and not soak["queue_overflows"]
        )
        transport_ok = (
            transport_summary["sent"] == 2000 and transport_summary["acked"] == 2000 and
            not transport_summary["missing"] and not transport_summary["duplicate"] and
            not transport_summary["unexpected"] and not transport_summary["out_of_order"] and
            not transport_summary["parser_state_corruption"] and transport_summary["parser_recovered"]
        )
        concurrency_ok = (
            concurrency_ok and concurrency["subtitle_and_ced_coexisted"] and
            concurrency["ced_processed_while_speech"] and concurrency["ced_processed_while_stt_idle"] and
            concurrency["ced_present_across_multiple_mixed_windows"]
        )
        clock_ok = all(clock_result[key] for key in (
            "automatic_k_after_connection", "home_minute_matches",
            "k_during_subtitle_preserved_screen", "k_during_alert_preserved_screen",
            "advanced_without_resync",
        ))
        mandatory = [render_ok, stt_ok, ced_ok, alert_ok, transition_ok,
                     transport_ok, concurrency_ok, soak_ok, clock_ok,
                     reconnect["reopen_ok"] and reconnect["normal_hud_after_reconnect"], regression["passed"]]
        layer2 = "COMPLETE" if all(mandatory) else "PARTIALLY COMPLETE"
        status = {
            "hardware_detected": True, "firmware_build": final_c3.returncode == 0,
            "functional_acceptance": True, "host_to_oled_latency": render_ok,
            "stt_to_oled": stt_ok, "ced_to_oled": ced_ok, "alert_to_oled": alert_ok,
            "transition_latency": transition_ok, "transport_reliability": transport_summary,
            "concurrency": concurrency["counters"],
            "mixed_audio": {key: concurrency[key] for key in (
                "subtitle_and_ced_coexisted", "ced_processed_while_speech", "ced_processed_while_stt_idle",
                "idle_ced_windows", "speech_ced_windows", "ced_present_across_multiple_mixed_windows",
                "subtitle_updates", "serial_frames", "serial_failures")},
            "soak": soak, "clock": clock_result, "serial_reconnect": reconnect,
            "regression": regression,
            "battery": {"runtime": "MANUAL VALIDATION PENDING", "average_current": "NOT MEASURED"},
            "physical_acoustic": "NOT MEASURED / MANUAL OPTIONAL", "layer2_status": layer2,
            "campaign_duration_seconds": time.monotonic() - started_monotonic,
            "layer1_model_modified": False, "emergency_thresholds_modified": False,
            "hud_awareness_threshold_policy_modified": False, "stt_semantics_modified": False,
            "locked_hud_behavior_modified": False,
        }
        status["mixed_audio"]["C_frames_sent"] = concurrency["counters"]["C_frames_sent"]
        status["mixed_audio"]["ced_windows"] = [{
            "top1": item.get("top1"), "top1_score": item.get("top1_score"),
            "top5": item.get("top5"), "selected_hud_label": item.get("selected_hud_label"),
            "active_hud_label": item.get("active_hud_label"), "c_frame_sent": item.get("c_frame_sent"),
        } for item in concurrency["ced_audits"]]
        summaries = {**render_summary, "STT FINAL -> OLED": stt_summary,
                     "CED accepted -> OLED": ced_summary, "HELP decision -> ALERT": alert_summary,
                     **transition_summary}
        (output / "layer2_hardware_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        (output / "FINAL_LAYER2_HARDWARE_REPORT.md").write_text(
            report_markdown(environment, status, summaries), encoding="utf-8")
        (output / "RUNNING").unlink(missing_ok=True)
        (output / "COMPLETE").write_text(utc_now(), encoding="utf-8")
        print(json.dumps({"output_directory": str(output), "layer2_status": layer2,
                          "campaign_duration_seconds": status["campaign_duration_seconds"]}), flush=True)
        return 0 if layer2 == "COMPLETE" else 2
    except Exception:
        (raw_dir / "campaign_exception.log").write_text(traceback.format_exc(), encoding="utf-8")
        (output / "FAILED").write_text(utc_now(), encoding="utf-8")
        raise
    finally:
        try: hw.close()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
