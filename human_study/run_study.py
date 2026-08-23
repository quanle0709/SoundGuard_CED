"""One-click runner for the protocol-locked SoundGuard Layer 3 study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from display_transport import HUDTransport, get_alert_display_state
from emergency_system import EmergencySystem
from hud_awareness import evaluate_ced_for_hud
from sound_classifier import classify_audio_file
from speech_recognizer import transcribe_audio_file
from human_study.study_core import (
    BLOCKS,
    MANIFEST_PATH,
    QUESTIONNAIRE_FIELDS,
    ROOT,
    STUDY_DIR,
    STUDY_VERSION,
    SUS_ITEMS,
    LIKERT_ITEMS,
    TRIAL_FIELDS,
    build_trial_plan,
    calculate_rt_ms,
    condition_order,
    condition_set_map,
    derive_seed,
    load_config,
    load_manifest,
    protocol_lock,
    score_sus,
    session_directory,
    stimulus_asset_bundle_sha256,
    stimulus_asset_paths,
    tracked_worktree_clean,
    utc_now,
    validate_participant_id,
)


PREFLIGHT_DIR = STUDY_DIR / "preflight_logs"
ESP32_VID = 0x303A
ESP32_PID = 0x1001


def yes_no(prompt: str) -> bool:
    while True:
        value = input(f"{prompt} [YES/NO]: ").strip().upper()
        if value in {"Y", "YES"}:
            return True
        if value in {"N", "NO"}:
            return False


def detect_esp32_port(requested: str) -> str:
    if requested.lower() != "auto":
        return requested
    from serial.tools import list_ports

    matches = [
        port for port in list_ports.comports()
        if port.vid == ESP32_VID and port.pid == ESP32_PID
    ]
    if len(matches) != 1:
        found = ", ".join(port.device for port in matches) or "none"
        raise RuntimeError(
            "Expected exactly one ESP32-C3 VID:PID=303A:1001; "
            f"found {len(matches)} ({found})."
        )
    return matches[0].device


def required_stimulus_files(manifest: list[dict[str, str]]) -> list[Path]:
    return stimulus_asset_paths(manifest)


def record_researcher_test_wav(input_device: int | None = None) -> Path:
    import sounddevice as sd
    import soundfile as sf

    sample_rate = 16000
    seconds = 4.0
    input(
        "Press Enter, then the researcher should speak a short non-sensitive "
        "Vietnamese test phrase. This is not participant data."
    )
    audio = sd.rec(
        int(sample_rate * seconds), samplerate=sample_rate, channels=1,
        dtype="float32", device=input_device,
    )
    sd.wait()
    handle = tempfile.NamedTemporaryFile(
        prefix="soundguard_researcher_preflight_", suffix=".wav", delete=False
    )
    handle.close()
    path = Path(handle.name)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path


def select_preflight_ced(result: dict) -> str:
    candidates = [{
        "label": result.get("label", ""),
        "score": result.get("confidence", 0.0),
    }] + list(result.get("top_predictions") or [])
    accepted: list[tuple[float, str]] = []
    for candidate in candidates:
        try:
            score = float(candidate.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        evaluation = evaluate_ced_for_hud(str(candidate.get("label", "")), score)
        if evaluation.get("accepted"):
            accepted.append((score, str(evaluation["display_label"])))
    return max(accepted)[1] if accepted else ""


def write_preflight_report(report: dict) -> Path:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = PREFLIGHT_DIR / f"preflight_{stamp}.json"
    suffix = 1
    while path.exists():
        path = PREFLIGHT_DIR / f"preflight_{stamp}_{suffix}.json"
        suffix += 1
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def confirm_preflight_subtitle(
    hud: HUDTransport,
    confirm: Callable[[str], bool],
) -> bool:
    """Hold the visual-check subtitle until the researcher answers."""
    hud.set_partial_subtitle("PREFLIGHT")
    try:
        return confirm("Researcher: is PREFLIGHT visible as a subtitle on the OLED?")
    finally:
        hud.set_subtitle("")


def run_preflight(
    hud_port: str,
    researcher_test_wav: str | None,
    *,
    input_device: int | None = None,
    confirm: Callable[[str], bool] = yes_no,
) -> tuple[bool, dict, Path]:
    """Run mandatory real-service/hardware preflight without participant data."""
    config = load_config()
    report: dict = {
        "study_version": STUDY_VERSION,
        "started_at": utc_now(),
        "participant_data": False,
        "researcher_test_audio_copied_or_saved_by_platform": False,
        "researcher_test_transcript_retained": False,
        "input_device_index": input_device,
        "checks": {},
        "result": "PREFLIGHT FAIL",
    }
    temporary_audio = False
    hud: HUDTransport | None = None
    stage = "stimuli"
    try:
        manifest = load_manifest()
        missing = [str(path) for path in required_stimulus_files(manifest) if not path.is_file()]
        report["checks"]["stimuli_and_provenance"] = {
            "passed": not missing, "missing": missing,
        }
        if missing:
            raise RuntimeError("Required stimulus preparation is incomplete.")
        report["checks"]["stimuli_and_provenance"]["asset_bundle_sha256"] = (
            stimulus_asset_bundle_sha256(manifest)
        )
        report["checks"]["stimuli_and_provenance"]["file_count"] = len(
            required_stimulus_files(manifest)
        )
        stage = "version_lock"
        clean = tracked_worktree_clean()
        report["checks"]["tracked_worktree_clean"] = {"passed": clean}
        if not clean:
            raise RuntimeError(
                "Tracked git changes are present. Commit the locked study/product "
                "version before preflight."
            )

        stage = "internet"
        started = time.perf_counter()
        with socket.create_connection(
            (config["network_probe_host"], int(config["network_probe_port"])),
            timeout=float(config["network_probe_timeout_seconds"]),
        ):
            pass
        report["checks"]["internet"] = {
            "passed": True,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "host": config["network_probe_host"],
        }

        stage = "google_stt"
        test_path = (
            Path(researcher_test_wav)
            if researcher_test_wav
            else record_researcher_test_wav(input_device)
        )
        temporary_audio = researcher_test_wav is None
        if not test_path.is_file():
            raise FileNotFoundError(f"Researcher test WAV not found: {test_path}")
        started = time.perf_counter()
        transcript = transcribe_audio_file(test_path)
        stt_latency = (time.perf_counter() - started) * 1000.0
        if not transcript:
            raise RuntimeError("Google STT returned no recognized researcher test text.")
        report["checks"]["google_stt"] = {
            "passed": True, "latency_ms": round(stt_latency, 3),
            "provider": "Google Speech Recognition", "recognized_nonempty": True,
        }

        stage = "esp32_com"
        port = detect_esp32_port(hud_port)
        hud = HUDTransport(port)
        if not hud.enabled:
            raise RuntimeError(f"HUD transport did not open on {port}.")
        report["checks"]["esp32_com"] = {"passed": True, "port": port}

        stage = "subtitle_hud"
        subtitle_ok = confirm_preflight_subtitle(hud, confirm)
        report["checks"]["subtitle_hud"] = {
            "passed": subtitle_ok,
            "frame_type": "P",
            "payload": "PREFLIGHT",
        }
        if not subtitle_ok:
            raise RuntimeError("Subtitle HUD visual confirmation failed.")

        stage = "ced_path"
        ced_row = next(row for row in manifest if row["stimulus_id"] == "EN_A01")
        ced_result = classify_audio_file(ROOT / ced_row["source_path"])
        ced_label = select_preflight_ced(ced_result)
        if not ced_label:
            raise RuntimeError("Frozen CED path produced no accepted display class.")
        hud.set_environmental_sound(ced_label)
        ced_ok = confirm(f"Researcher: is the CED label {ced_label} visible on the OLED?")
        report["checks"]["ced_path"] = {
            "passed": ced_ok, "display_label": ced_label,
            "top1": ced_result.get("label"),
        }
        if not ced_ok:
            raise RuntimeError("CED HUD visual confirmation failed.")

        stage = "alert_path"
        alert_row = next(row for row in manifest if row["stimulus_id"] == "CR_A01")
        alert_result = classify_audio_file(ROOT / alert_row["source_path"])
        alert_system = EmergencySystem(decision_mode="continuous")
        decision = None
        for index in range(2):
            decision = alert_system.process_sound_event(
                str(alert_result.get("label", "unknown")),
                float(alert_result.get("confidence", 0.0)),
                timestamp=f"2026-01-01T00:00:0{index * 5}",
            )
        event_label, help_active = get_alert_display_state(
            alert_system.active_sound, alert_system.help_active
        )
        if not decision or not decision.get("alert") or not event_label:
            raise RuntimeError("Frozen critical CED/emergency path produced no ALERT.")
        hud.set_alert_state(event_label, help_active)
        display_alert = event_label.upper()
        alert_ok = confirm(
            f"Researcher: is the {display_alert} ALERT screen visible on the OLED?"
        )
        report["checks"]["alert_path"] = {
            "passed": alert_ok, "label": display_alert,
            "top1": alert_result.get("label"),
            "confidence": alert_result.get("confidence"),
            "event_state": decision.get("event_state"),
        }
        if not alert_ok:
            raise RuntimeError("ALERT HUD visual confirmation failed.")

        hud.set_subtitle("")
        hud.set_environmental_sound("")
        hud.set_alert_state()
        report["result"] = "PREFLIGHT PASS"
    except Exception as exc:
        report["failed_stage"] = stage
        report["failure_type"] = (
            "network_STT_failure"
            if stage in {"internet", "google_stt"}
            else "system_error"
        )
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if hud is not None:
            try:
                hud.set_subtitle("")
                hud.set_environmental_sound("")
                hud.set_alert_state()
            finally:
                hud.close()
        if temporary_audio and "test_path" in locals():
            try:
                test_path.unlink()
            except OSError:
                pass
        report["finished_at"] = utc_now()
        path = write_preflight_report(report)
    return report["result"] == "PREFLIGHT PASS", report, path


class ChoiceUI:
    """Large local Tk UI; each question is a small blocking window."""

    def choose(self, title: str, prompt: str, choices: list[str]) -> tuple[str, float]:
        import tkinter as tk

        result: dict[str, object] = {}
        root = tk.Tk()
        root.title(title)
        root.geometry("900x600")
        root.configure(bg="white")
        label = tk.Label(
            root, text=prompt, font=("Arial", 24, "bold"), wraplength=820,
            justify="center", bg="white", fg="black", pady=30,
        )
        label.pack(fill="x")

        def select(value: str) -> None:
            result["value"] = value
            result["time"] = time.perf_counter()
            root.destroy()

        for index, value in enumerate(choices, start=1):
            button = tk.Button(
                root, text=f"{index}. {value}", font=("Arial", 20),
                command=lambda selected=value: select(selected), pady=12,
            )
            button.pack(fill="x", padx=60, pady=6)
            root.bind(str(index), lambda _event, selected=value: select(selected))
        root.protocol("WM_DELETE_WINDOW", lambda: select("[STOPPED]"))
        root.mainloop()
        return str(result["value"]), float(result["time"])

    def message(self, title: str, text: str) -> None:
        from tkinter import messagebox
        messagebox.showinfo(title, text)


class ProductMonitor:
    """Runs the unchanged product and timestamps its existing stdout evidence."""

    def __init__(self, command: list[str], log_path: Path, start_timeout: float) -> None:
        self.command = command
        self.log_path = log_path
        self.start_timeout = start_timeout
        self.process: subprocess.Popen[str] | None = None
        self.lines: list[dict] = []
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._log_handle = None

    def start(self) -> None:
        if self.process is not None:
            return
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            self.command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env=env,
        )

        def reader() -> None:
            assert self.process is not None and self.process.stdout is not None
            for raw in self.process.stdout:
                self._queue.put(raw.rstrip("\r\n"))

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            self.drain()
            if any("[WAITING FOR SPEECH]" in item["line"] for item in self.lines):
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"SoundGuard exited during startup with {self.process.returncode}.")
            time.sleep(0.05)
        raise TimeoutError("SoundGuard did not report ready before the locked timeout.")

    def drain(self) -> None:
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                return
            event = {"wall": time.time(), "monotonic": time.perf_counter(), "line": line}
            self.lines.append(event)
            if self._log_handle is not None:
                self._log_handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def mark(self) -> int:
        self.drain()
        return len(self.lines)

    def evidence_since(self, mark: int) -> dict:
        self.drain()
        rows = self.lines[mark:]
        errors = [row for row in rows if "STT " in row["line"] and "error:" in row["line"]]
        timing: list[tuple[dict, dict]] = []
        ced_events: list[dict] = []
        for row in rows:
            for prefix in ("STT PARTIAL timing: ", "STT FINAL timing: "):
                if prefix in row["line"]:
                    try:
                        timing.append((row, json.loads(row["line"].split(prefix, 1)[1])))
                    except json.JSONDecodeError:
                        pass
            if "CED window audit: " in row["line"]:
                try:
                    ced_events.append(json.loads(row["line"].split("CED window audit: ", 1)[1]))
                except json.JSONDecodeError:
                    pass
        network_error = any(
            token in row["line"].lower()
            for row in errors
            for token in ("recognition request failed", "network", "connection", "timed out")
        )
        no_result = any("FINAL: [empty]" in row["line"] for row in rows)
        return {
            "stt_timing": timing,
            "stt_errors": errors,
            "network_error": network_error,
            "stt_no_result": no_result,
            "ced_events": ced_events,
            "lines": rows,
        }

    def await_evidence(
        self, mark: int, *, needs_stt: bool, needs_ced: bool, timeout: float
    ) -> dict:
        deadline = time.monotonic() + timeout
        evidence = self.evidence_since(mark)
        while time.monotonic() < deadline:
            stt_ready = (
                not needs_stt
                or bool(evidence["stt_timing"])
                or bool(evidence["stt_errors"])
                or bool(evidence["stt_no_result"])
            )
            ced_ready = not needs_ced or bool(evidence["ced_events"])
            if stt_ready and ced_ready:
                return evidence
            if not self.alive:
                return evidence
            time.sleep(0.05)
            evidence = self.evidence_since(mark)
        return evidence

    def stop(self, timeout: float) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.drain()
        if self._log_handle is not None:
            self._log_handle.close()
        self.process = None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sample_rate)


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    from scipy.signal import resample_poly
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


def prepare_trial_audio(row: dict[str, str], config: dict) -> tuple[np.ndarray, int]:
    primary, sample_rate = load_audio(ROOT / row["source_path"])
    intended_samples = round(float(row["duration_seconds"]) * sample_rate)
    if intended_samples > 0:
        primary = primary[:intended_samples]
    secondary_path = row.get("secondary_source_path", "").strip()
    if not secondary_path:
        return primary, sample_rate
    secondary, secondary_rate = load_audio(ROOT / secondary_path)
    intended_secondary_samples = round(
        float(row["duration_seconds"]) * secondary_rate
    )
    if intended_secondary_samples > 0:
        secondary = secondary[:intended_secondary_samples]
    secondary = resample(secondary, secondary_rate, sample_rate)
    if row["block"] == "speech":
        target_length = len(primary)
    else:
        target_length = max(len(primary), len(secondary))
    if len(primary) < target_length:
        primary = np.pad(primary, (0, target_length - len(primary)))
    if len(secondary) < target_length:
        repeats = math.ceil(target_length / max(1, len(secondary)))
        secondary = np.tile(secondary, repeats)
    secondary = secondary[:target_length]
    gain = 10.0 ** (float(config["mixed_secondary_gain_db"]) / 20.0)
    mixed = primary[:target_length] + secondary * gain
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    limit = float(config["playback_peak_limit"])
    if peak > limit:
        mixed = mixed * (limit / peak)
    return mixed.astype(np.float32), sample_rate


def play_trial(row: dict[str, str], config: dict) -> tuple[float, float]:
    import sounddevice as sd
    audio, sample_rate = prepare_trial_audio(row, config)
    wall = time.time()
    monotonic = time.perf_counter()
    sd.play(
        audio, sample_rate, blocking=False,
        device=config.get("runtime_output_device"),
    )
    sd.wait()
    return wall, monotonic


def reserve_session_dir(path: Path, *, resume: bool, force: bool) -> None:
    if path.exists() and force:
        backup = path.with_name(path.name + ".replaced-" + datetime.now().strftime("%Y%m%dT%H%M%S"))
        shutil.move(str(path), str(backup))
    elif path.exists() and not resume:
        raise FileExistsError(
            f"Session folder already exists: {path}. Use --resume for an incomplete "
            "session or --force to preserve it under a timestamped backup."
        )
    path.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, fieldnames: tuple[str, ...], row: dict) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def load_completed_trial_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["trial_id"] for row in csv.DictReader(handle)}


def parse_hud_observation(ui: ChoiceUI, block: str) -> dict[str, str]:
    subtitle, observed_at = ui.choose(
        "Researcher HUD check", "Researcher only: was subtitle text visible?",
        ["YES", "NO"],
    )
    ced, _ = ui.choose(
        "Researcher HUD check", "Researcher only: which noncritical CED label was visible?",
        ["NONE", "DOG", "ANIMAL", "HORN", "DOOR", "BABY", "OTHER"],
    )
    alert, _ = ui.choose(
        "Researcher HUD check", "Researcher only: which ALERT was visible?",
        ["NONE", "SIREN", "GLASS", "FIRE", "SCREAM", "SOS", "OTHER"],
    )
    return {
        "subtitle_state": subtitle,
        "displayed_ced": ced,
        "displayed_alert": alert,
        "hud_observation_monotonic": observed_at,
        "coexistence_state": (
            "SUBTITLE_AND_CED" if subtitle == "YES" and ced != "NONE"
            else "SUBTITLE_ONLY" if subtitle == "YES"
            else "CED_ONLY" if ced != "NONE"
            else "ALERT" if alert != "NONE"
            else "NONE"
        ),
    }


def clear_hud(port: str) -> None:
    """Put the display in an unassisted idle state between study conditions."""
    with HUDTransport(port) as hud:
        if not hud.enabled:
            raise RuntimeError(f"Unable to clear HUD on {port}")
        hud.set_subtitle("")
        hud.set_environmental_sound("")
        hud.set_alert_state()
        hud.set_status("Ready")


def execute_trial(
    planned,
    participant_id: str,
    pilot: bool,
    seed: int,
    config: dict,
    ui: ChoiceUI,
    monitor: ProductMonitor | None,
) -> dict:
    row = planned.row
    mark = monitor.mark() if monitor else 0
    playback_success = True
    playback_error = ""
    try:
        stimulus_wall, stimulus_monotonic = play_trial(row, config)
    except Exception as exc:
        stimulus_wall, stimulus_monotonic = time.time(), time.perf_counter()
        playback_success = False
        playback_error = f"{type(exc).__name__}: {exc}"

    response, response_monotonic = ui.choose(
        f"{planned.block.upper()} — {planned.condition}",
        row["question"], row["choices"].split("|"),
    )
    environment_response = ""
    environment_correct = ""
    primary_response_monotonic = response_monotonic
    environment_response_monotonic: float | None = None
    if planned.block == "mixed":
        environment_response, environment_time = ui.choose(
            "MIXED — environmental event", "Âm thanh môi trường nào vừa phát?",
            row["environment_choices"].split("|"),
        )
        environment_response_monotonic = environment_time
        response_monotonic = max(response_monotonic, environment_time)
        environment_correct = str(
            environment_response == row["environment_ground_truth"]
        ).lower()

    observation = {
        "subtitle_state": "NOT_APPLICABLE", "displayed_ced": "NOT_APPLICABLE",
        "displayed_alert": "NOT_APPLICABLE", "coexistence_state": "NOT_APPLICABLE",
    }
    evidence = {
        "stt_timing": [], "stt_errors": [], "network_error": False,
        "stt_no_result": False, "ced_events": [],
    }
    if monitor:
        needs_stt = (
            planned.block in {"speech", "mixed"}
            or row.get("type") == "scripted_help"
        )
        needs_ced = planned.block in {"environmental", "mixed"} or row.get("type") == "critical_ced"
        evidence = monitor.await_evidence(
            mark, needs_stt=needs_stt, needs_ced=needs_ced,
            timeout=float(config["stt_evidence_window_seconds"]),
        )
        observation = parse_hud_observation(ui, planned.block)

    correct = response == row["correct_response"]
    speech_correct = str(correct).lower() if planned.block in {"speech", "mixed"} else ""
    if planned.block == "mixed":
        both_correct = str(correct and environment_correct == "true").lower()
    else:
        both_correct = ""

    stt_attempted = bool(
        monitor and (
            planned.block in {"speech", "mixed"}
            or row.get("type") == "scripted_help"
        )
    )
    stt_timing = evidence["stt_timing"]
    stt_success = bool(stt_timing) if stt_attempted else None
    stt_error_type = ""
    network_failure = bool(evidence["network_error"])
    if stt_attempted and not stt_success:
        stt_error_type = (
            "network_STT_failure" if network_failure
            else "recognition_no_result" if evidence.get("stt_no_result")
            else "no_STT_evidence"
        )

    hud_timestamp = ""
    hud_source = ""
    stt_latency_ms = ""
    if stt_timing:
        log_row, parsed = stt_timing[0]
        hud_timestamp = parsed.get("hud_sent_at", log_row["wall"])
        hud_source = "production_STT_hud_sent_at"
        stt_latency_ms = f"{(float(hud_timestamp) - stimulus_wall) * 1000.0:.3f}"
    elif evidence["ced_events"]:
        hud_timestamp = next(
            (event.get("inference_end") for event in evidence["ced_events"] if event.get("c_frame_sent")),
            "",
        )
        hud_source = "production_CED_audit" if hud_timestamp != "" else ""

    hardware_connected = bool(monitor is None or monitor.alive)
    invalid_reasons = []
    if not playback_success:
        invalid_reasons.append("playback_failure")
    if not hardware_connected:
        invalid_reasons.append("hardware_disconnect")
    if stt_attempted and not stt_success:
        invalid_reasons.append(stt_error_type)
    trial_valid = not invalid_reasons
    if not trial_valid:
        error_class = "network_STT_failure" if network_failure else "system_error"
    elif not correct or (planned.block == "mixed" and environment_correct != "true"):
        error_class = "participant_error"
    else:
        error_class = ""

    rt_hud = ""
    if hud_timestamp != "":
        try:
            # Production timestamp is wall time; response monotonic cannot be mixed
            # with it. Use the matched wall response derived from the monotonic delta.
            response_wall = stimulus_wall + (response_monotonic - stimulus_monotonic)
            rt_hud = f"{max(0.0, response_wall - float(hud_timestamp)) * 1000.0:.3f}"
        except (TypeError, ValueError):
            pass
    else:
        response_wall = stimulus_wall + (response_monotonic - stimulus_monotonic)

    return {
        "participant_id": participant_id,
        "study_version": STUDY_VERSION,
        "pilot": str(pilot).lower(),
        "block": planned.block,
        "condition": planned.condition,
        "condition_position": planned.condition_position,
        "trial_id": planned.trial_id,
        "trial_index": planned.trial_index,
        "stimulus_id": row["stimulus_id"],
        "stimulus_set": row["stimulus_set"],
        "ground_truth": row["ground_truth"],
        "environment_ground_truth": row.get("environment_ground_truth", ""),
        "response": response,
        "speech_response": response if planned.block in {"speech", "mixed"} else "",
        "environment_response": environment_response or (
            response if planned.block == "environmental" else ""
        ),
        "correct": str(correct).lower(),
        "speech_correct": speech_correct,
        "environment_correct": environment_correct or (
            str(correct).lower() if planned.block == "environmental" else ""
        ),
        "both_correct": both_correct,
        "missed": str(
            (
                environment_response == "NONE"
                and row.get("environment_ground_truth") not in {"", "NONE"}
            ) if planned.block == "mixed" else (
                response in {"NONE", "[STOPPED]"} and row["ground_truth"] != "NONE"
            )
        ).lower(),
        "false_perceived_event": str(row["ground_truth"] == "NONE" and response != "NONE").lower(),
        "stimulus_start_wall": datetime.fromtimestamp(stimulus_wall).astimezone().isoformat(timespec="milliseconds"),
        "stimulus_start_monotonic": f"{stimulus_monotonic:.9f}",
        "hud_timestamp": hud_timestamp,
        "hud_timestamp_source": hud_source,
        "response_timestamp_wall": datetime.fromtimestamp(response_wall).astimezone().isoformat(timespec="milliseconds"),
        "response_timestamp_monotonic": f"{response_monotonic:.9f}",
        "rt_from_stimulus_ms": calculate_rt_ms(stimulus_monotonic, response_monotonic),
        "rt_from_hud_ms": rt_hud,
        "hud_observation_timestamp_monotonic": observation.get(
            "hud_observation_monotonic", ""
        ),
        "speech_response_timestamp_monotonic": (
            f"{primary_response_monotonic:.9f}"
            if planned.block in {"speech", "mixed"} else ""
        ),
        "environment_response_timestamp_monotonic": (
            f"{(environment_response_monotonic or primary_response_monotonic):.9f}"
            if planned.block in {"environmental", "mixed"} else ""
        ),
        "speech_rt_from_stimulus_ms": (
            calculate_rt_ms(stimulus_monotonic, primary_response_monotonic)
            if planned.block in {"speech", "mixed"} else ""
        ),
        "environment_rt_from_stimulus_ms": (
            calculate_rt_ms(
                stimulus_monotonic,
                environment_response_monotonic or primary_response_monotonic,
            ) if planned.block in {"environmental", "mixed"} else ""
        ),
        "displayed_ced": observation["displayed_ced"],
        "displayed_alert": observation["displayed_alert"],
        "subtitle_state": observation["subtitle_state"],
        "coexistence_state": observation["coexistence_state"],
        "stt_attempted": str(stt_attempted).lower(),
        "stt_success": "" if stt_success is None else str(stt_success).lower(),
        "stt_error_type": stt_error_type,
        "stt_latency_ms": stt_latency_ms,
        "network_failure": str(network_failure).lower(),
        "trial_valid": str(trial_valid).lower(),
        "invalid_reason": "|".join(invalid_reasons) or playback_error,
        "playback_success": str(playback_success).lower(),
        "hardware_connected": str(hardware_connected).lower(),
        "error_class": error_class,
        "randomization_seed": seed,
    }


def collect_questionnaire(ui: ChoiceUI, output: Path, participant_id: str, pilot: bool) -> float:
    sus_values: list[int] = []
    choices = ["1 Strongly disagree", "2", "3", "4", "5 Strongly agree"]
    for index, prompt in enumerate(SUS_ITEMS, start=1):
        response, _ = ui.choose("STANDARD SUS", f"{index}. {prompt}", choices)
        value = int(response[0])
        sus_values.append(value)
        scored = value - 1 if index % 2 else 5 - value
        append_csv(output, QUESTIONNAIRE_FIELDS, {
            "participant_id": participant_id, "study_version": STUDY_VERSION,
            "pilot": str(pilot).lower(), "instrument": "SUS", "item_id": f"SUS{index:02d}",
            "prompt": prompt, "response": value, "scored_value": scored,
            "recorded_at": utc_now(),
        })
    for index, prompt in enumerate(LIKERT_ITEMS, start=1):
        response, _ = ui.choose("SOUNDGUARD-SPECIFIC LIKERT", f"{index}. {prompt}", choices)
        value = int(response[0])
        append_csv(output, QUESTIONNAIRE_FIELDS, {
            "participant_id": participant_id, "study_version": STUDY_VERSION,
            "pilot": str(pilot).lower(), "instrument": "SOUNDGUARD_LIKERT",
            "item_id": f"SG{index:02d}", "prompt": prompt, "response": value,
            "scored_value": value, "recorded_at": utc_now(),
        })
    total = score_sus(sus_values)
    append_csv(output, QUESTIONNAIRE_FIELDS, {
        "participant_id": participant_id, "study_version": STUDY_VERSION,
        "pilot": str(pilot).lower(), "instrument": "SUS_TOTAL", "item_id": "SUS_TOTAL",
        "prompt": "Standard SUS total (0-100)", "response": total,
        "scored_value": total, "recorded_at": utc_now(),
    })
    return total


def run_session(args: argparse.Namespace) -> int:
    config = load_config()
    participant_id = args.participant or input("Participant ID: ")
    pilot = args.pilot
    if args.participant is None:
        pilot = yes_no("Pilot?")
    participant_id = validate_participant_id(participant_id, pilot=pilot)
    if not yes_no("Confirm consent complete and physical hardware/setup checklist ready"):
        print("Session not started.")
        return 2

    passed, report, report_path = run_preflight(
        args.hud_port, args.researcher_test_wav, input_device=args.input_device
    )
    print(report["result"])
    print(f"Preflight report: {report_path}")
    if not passed:
        print("Participant session aborted/postponed; no participant folder was created.")
        return 2

    port = report["checks"]["esp32_com"]["port"]
    base_seed = int(args.seed if args.seed is not None else config["default_base_seed"])
    config = dict(config)
    config["runtime_output_device"] = args.output_device
    participant_seed = derive_seed(participant_id, base_seed)
    path = session_directory(participant_id, pilot)
    reserve_session_dir(path, resume=args.resume, force=args.force)
    trials_path = path / "trials.csv"
    questionnaire_path = path / "questionnaire.csv"
    runtime_path = path / "runtime.log"
    lock = protocol_lock(base_seed)
    plan = build_trial_plan(participant_id, base_seed)
    session_path = path / "session.json"
    if args.resume and session_path.exists():
        session = json.loads(session_path.read_text(encoding="utf-8"))
        for key, value in lock.items():
            if session.get("protocol_lock", {}).get(key) != value:
                raise RuntimeError(f"Resume blocked by protocol lock mismatch: {key}")
        if session.get("status") == "complete":
            raise RuntimeError("Completed sessions are immutable; --resume is not permitted.")
        if session.get("input_device_index") != args.input_device:
            raise RuntimeError("Resume blocked: input device differs from the original session.")
        if session.get("output_device_index") != args.output_device:
            raise RuntimeError("Resume blocked: output device differs from the original session.")
    else:
        session = {
            "participant_id": participant_id,
            "pilot": pilot,
            "status": "in_progress",
            "started_at": utc_now(),
            "study_version": STUDY_VERSION,
            "protocol_lock": lock,
            "participant_randomization_seed": participant_seed,
            "condition_order": list(condition_order(participant_id)),
            "condition_stimulus_sets": condition_set_map(participant_id),
            "preflight_report": str(report_path.relative_to(STUDY_DIR)),
            "stimulus_asset_bundle_sha256": report["checks"]["stimuli_and_provenance"]["asset_bundle_sha256"],
            "frozen_product_command": (
                config["frozen_product_command"]
                + (["--device", str(args.input_device)] if args.input_device is not None else [])
                + ["--hud-port", port]
            ),
            "input_device_index": args.input_device,
            "output_device_index": args.output_device,
            "optional_metadata": {
                "age_band": input("Optional age band (blank to omit): ").strip(),
                "hearing_difficulty_category": input("Optional self-reported hearing difficulty category: ").strip(),
                "usual_assistive_device_usage": input("Optional usual hearing-assistive device: ").strip(),
                "usual_device_used_during_session": input("Usual device used during session (YES/NO/N/A): ").strip().upper(),
            },
            "setup_deviations": input("Setup deviations (blank if none): ").strip(),
            "expected_trial_count": len(plan),
        }
        session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")

    completed = load_completed_trial_ids(trials_path)
    ui = ChoiceUI()
    monitor: ProductMonitor | None = None
    active_condition = None
    questionnaire_collected = False
    if questionnaire_path.exists():
        with questionnaire_path.open(encoding="utf-8-sig", newline="") as handle:
            questionnaire_collected = any(
                row.get("item_id") == "SUS_TOTAL" for row in csv.DictReader(handle)
            )
    try:
        ui.message("SoundGuard study", "The study will begin. Ask the researcher whenever you need a pause.")
        for planned in plan:
            if planned.trial_id in completed:
                continue
            if planned.condition != active_condition:
                if monitor:
                    monitor.stop(float(config["product_stop_timeout_seconds"]))
                    monitor = None
                    clear_hud(port)
                    if not questionnaire_collected:
                        collect_questionnaire(
                            ui, questionnaire_path, participant_id, pilot
                        )
                        questionnaire_collected = True
                active_condition = planned.condition
                ui.message(
                    "Condition change",
                    f"Researcher: prepare condition {active_condition}. Confirm the participant's usual hearing assistance is unchanged.",
                )
                if active_condition == "WITH":
                    command = (
                        [sys.executable] + config["frozen_product_command"]
                        + (["--device", str(args.input_device)] if args.input_device is not None else [])
                        + ["--hud-port", port]
                    )
                    monitor = ProductMonitor(command, runtime_path, float(config["product_start_timeout_seconds"]))
                    monitor.start()
            ui.message(
                f"Block {planned.block}",
                "Press OK when the participant is ready. The stimulus will play once.",
            )
            trial_row = execute_trial(
                planned, participant_id, pilot, participant_seed, config, ui, monitor
            )
            append_csv(trials_path, TRIAL_FIELDS, trial_row)
            if (
                trial_row["playback_success"] != "true"
                or trial_row["hardware_connected"] != "true"
            ):
                raise RuntimeError(
                    "system_error: playback or hardware failed; the invalid trial was "
                    "retained and the session was stopped."
                )
            if (
                trial_row["stt_attempted"] == "true"
                and trial_row["stt_success"] != "true"
            ):
                raise RuntimeError(
                    f"{trial_row['stt_error_type']}: affected STT-dependent block must be "
                    "postponed; the invalid trial was retained and was not scored as "
                    "participant error."
                )
            if trial_row["response"] == "[STOPPED]":
                raise KeyboardInterrupt
        if monitor:
            monitor.stop(float(config["product_stop_timeout_seconds"]))
            monitor = None
            clear_hud(port)
        if not questionnaire_collected:
            sus_total = collect_questionnaire(
                ui, questionnaire_path, participant_id, pilot
            )
            questionnaire_collected = True
        else:
            with questionnaire_path.open(encoding="utf-8-sig", newline="") as handle:
                sus_total = float(next(
                    row["scored_value"] for row in csv.DictReader(handle)
                    if row.get("item_id") == "SUS_TOTAL"
                ))
        session.update({
            "status": "complete", "completed_at": utc_now(),
            "completed_trial_count": len(load_completed_trial_ids(trials_path)),
            "sus_total": sus_total,
        })
        session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
        ui.message("Session complete", "Thank you. The researcher will now validate the anonymous session data.")
        return 0
    except (KeyboardInterrupt, Exception) as exc:
        if monitor:
            monitor.stop(float(config["product_stop_timeout_seconds"]))
        session.update({
            "status": "incomplete", "stopped_at": utc_now(),
            "stop_reason": f"{type(exc).__name__}: {exc}",
            "completed_trial_count": len(load_completed_trial_ids(trials_path)),
        })
        session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Session incomplete: {exc}")
        return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", help="P01-P10 or PILOTnn with --pilot")
    parser.add_argument("--pilot", action="store_true", help="Store separately as pilot data")
    parser.add_argument("--seed", type=int, help="Locked base randomization seed")
    parser.add_argument("--hud-port", default="auto", help="ESP32 COM port or auto")
    parser.add_argument("--input-device", type=int, help="Locked microphone device index")
    parser.add_argument("--output-device", type=int, help="Locked speaker/output device index")
    parser.add_argument("--researcher-test-wav", help="Non-sensitive researcher WAV for real Google STT preflight")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume only an incomplete lock-matching session")
    parser.add_argument("--force", action="store_true", help="Archive an existing folder before a new run")
    args = parser.parse_args()
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    if args.preflight_only:
        passed, report, path = run_preflight(
            args.hud_port, args.researcher_test_wav, input_device=args.input_device
        )
        print(report["result"])
        print(path)
        return 0 if passed else 2
    return run_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
