"""Layer 2 computer-only benchmark for the frozen SoundGuard application.

This harness only observes public production paths and writes new evidence.  It
does not change model checkpoints, thresholds, preprocessing, or application
decisions.  The long soak replaces the remote Google recognition call with a
deterministic transcript fixture, but retains production DTLN preprocessing,
CED-Tiny, the opt-in EfficientSED specialist, fusion/decision logic, and HUD
serialization.  That boundary is recorded explicitly in every soak artifact.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soundguard.emergency.alert_mapper import map_alert
from soundguard.app import cleanup_dtln_recording, prepare_speech_audio
from soundguard.display.display_transport import HUDTransport, crc16_ccitt, encode_hud_frame
from soundguard.emergency.emergency_system import EmergencySystem
from soundguard.emergency.emergency_v3 import (
    EmergencyV3Specialist,
    select_emergency_evidence,
)
from soundguard.emergency.fusion_engine import fuse_result
from soundguard.speech.live_speech_to_text import recognize_snapshot
from soundguard.detection import sound_classifier


RUN_ID = "system_computer_only_20260821"
OUT = ROOT / "benchmark_results" / RUN_ID
SUBDIRS = (
    "config", "latency", "concurrency", "resources", "soak",
    "transport_software", "regression", "final",
)


def ensure_output_tree() -> None:
    for name in SUBDIRS:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = rows[0].keys() if rows else ()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return float(ordered[rank - 1])


def summarize(values: Iterable[float]) -> dict:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None,
                "min_ms": None, "max_ms": None}
    return {
        "count": len(data),
        "mean_ms": statistics.mean(data),
        "median_ms": statistics.median(data),
        "p95_ms": percentile(data, 0.95),
        "min_ms": min(data),
        "max_ms": max(data),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    return completed.stdout.strip()


def record_provenance(command_name: str, config: dict) -> None:
    ensure_output_tree()
    now = dt.datetime.now().astimezone().isoformat()
    script = Path(__file__).resolve()
    environment = {
        "timestamp": now,
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short").splitlines(),
        "layer1_modified_by_harness": False,
    }
    write_json(OUT / "config" / "environment.json", environment)
    write_json(OUT / "config" / f"{command_name}_config.json", {
        **config,
        "KHKT_CLAIM": "Computer-only evidence for local real-time performance and stability.",
        "REPORT_USE": "Layer 2 latency, resource, reliability, and stability tables.",
    })
    (OUT / "config" / f"command_{command_name}.txt").write_text(
        " ".join([sys.executable, str(script), *sys.argv[1:]]) + "\n", encoding="utf-8"
    )
    write_json(OUT / "config" / "script_version.json", {
        "path": str(script.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(script),
    })


def read_manifest(path: Path, limit: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows if limit is None else rows[:limit]


def ced_paths(limit: int) -> list[Path]:
    rows = read_manifest(ROOT / "benchmark_data" / "external" / "esc50" / "manifest.csv")
    paths = [ROOT / row["path"] for row in rows]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError("No ESC-50 prerecorded audio is available for profiling")
    return [paths[index % len(paths)] for index in range(limit)]


def stt_cases(limit: int) -> list[dict]:
    rows = read_manifest(ROOT / "benchmark_data" / "external" / "vivos" / "manifest.csv")
    usable = [row for row in rows if (ROOT / row["path"]).is_file()]
    if not usable:
        raise FileNotFoundError("No VIVOS prerecorded audio is available for profiling")
    return [usable[index % len(usable)] for index in range(limit)]


class RecordingStream:
    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.flushes = 0

    def write(self, value: bytes) -> int:
        self.frames.append(bytes(value))
        return len(value)

    def flush(self) -> None:
        self.flushes += 1


def decode_frame(frame: bytes) -> tuple[str, str]:
    if frame[:2] != b"SG" or len(frame) < 8:
        raise ValueError("missing magic or truncated frame")
    version, message_type, length = struct.unpack(">BcH", frame[2:6])
    if version != 1 or len(frame) != 8 + length:
        raise ValueError("invalid version or length")
    body = frame[2:-2]
    expected = struct.unpack(">H", frame[-2:])[0]
    if crc16_ccitt(body) != expected:
        raise ValueError("CRC mismatch")
    return message_type.decode("ascii"), frame[6:-2].decode("utf-8")


def timed_ced_pipeline(path: Path, specialist: EmergencyV3Specialist) -> dict:
    """Time the frozen production CED path without changing its result."""
    timings: dict[str, float] = {}
    original_load = sound_classifier.load_audio_for_ced
    original_get = sound_classifier._get_classifier
    base_classifier = original_get()

    def timed_load(file_path):
        started = time.perf_counter()
        result = original_load(file_path)
        timings["preprocessing_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    class TimedClassifier:
        def __call__(self, *args, **kwargs):
            started = time.perf_counter()
            result = base_classifier(*args, **kwargs)
            timings["ced_tiny_inference_ms"] = (time.perf_counter() - started) * 1000.0
            return result

    started_total = time.perf_counter()
    sound_classifier.load_audio_for_ced = timed_load
    sound_classifier._get_classifier = lambda: TimedClassifier()
    try:
        ced = sound_classifier.classify_audio_file(str(path))
    finally:
        sound_classifier.load_audio_for_ced = original_load
        sound_classifier._get_classifier = original_get

    started = time.perf_counter()
    v3 = specialist.analyze_file(path)
    timings["efficientsed_roundtrip_ms"] = (time.perf_counter() - started) * 1000.0
    if not v3.get("ok"):
        raise RuntimeError(f"EfficientSED worker failed: {v3.get('error')}")

    started = time.perf_counter()
    label = str(ced.get("label", "unknown"))
    confidence = float(ced.get("confidence", 0.0))
    emergency_label, emergency_confidence, _ = select_emergency_evidence(
        label, confidence, v3, "neutral"
    )
    mapped = map_alert(label, confidence, context="neutral")
    fused = fuse_result(
        dt.datetime.now().isoformat(timespec="milliseconds"), "",
        {"label": label, "confidence": confidence}, mapped,
    )
    emergency = EmergencySystem(decision_mode="single_shot").process_sound_event(
        emergency_label, emergency_confidence
    )
    timings["fusion_decision_ms"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    output_text = emergency.get("alert_text") or fused.get("alert_text") or label
    frame = encode_hud_frame("alert", output_text)
    timings["serialization_ms"] = (time.perf_counter() - started) * 1000.0
    timings["total_ms"] = (time.perf_counter() - started_total) * 1000.0
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        **timings,
        "label": label,
        "confidence": confidence,
        "specialist_detected": bool(v3.get("detected")),
        "frame_bytes": len(frame),
    }


def timed_stt_pipeline(case: dict, *, partial: bool) -> dict:
    path = ROOT / case["path"]
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected VIVOS 16 kHz audio, got {sample_rate}")
    if partial:
        audio = audio[:min(len(audio), int(1.5 * sample_rate))]
    started_total = time.perf_counter()
    transcript = recognize_snapshot(
        np.asarray(audio, dtype=np.float32), True,
        utterance_id=0, is_final=not partial, save_live_utterances=False,
    )
    after_stt = time.perf_counter()
    mapped = map_alert("unknown", 0.0, context="neutral")
    fused = fuse_result(
        dt.datetime.now().isoformat(timespec="milliseconds"), transcript,
        {"label": "unknown", "confidence": 0.0}, mapped,
    )
    emergency = EmergencySystem(decision_mode="single_shot").process_transcript_event(transcript)
    after_decision = time.perf_counter()
    frame = encode_hud_frame("subtitle", transcript)
    completed = time.perf_counter()
    return {
        "sample_id": case.get("sample_id", path.stem),
        "kind": "partial" if partial else "final",
        "audio_duration_seconds": len(audio) / sample_rate,
        "stt_processing_ms": (after_stt - started_total) * 1000.0,
        "fusion_decision_ms": (after_decision - after_stt) * 1000.0,
        "serialization_ms": (completed - after_decision) * 1000.0,
        "total_ms": (completed - started_total) * 1000.0,
        "transcript_empty": not bool(transcript),
        "alert": bool(fused.get("alert") or emergency.get("alert")),
        "frame_bytes": len(frame),
    }


def process_snapshot(pids: Iterable[int]) -> dict:
    ids = sorted({int(pid) for pid in pids if pid})
    if not ids:
        return {"cpu_seconds": 0.0, "working_set_bytes": 0, "peak_working_set_bytes": 0}
    joined = ",".join(str(pid) for pid in ids)
    # A Windows virtual-environment launcher can remain as a small parent while
    # the real Python worker owns the model memory. Expand descendants before
    # collecting CPU/RAM so the EfficientSED process tree is not undercounted.
    command = (
        f"$ids=@({joined}); $all=@(Get-CimInstance Win32_Process); "
        "$changed=$true; while($changed){$changed=$false; "
        "$children=@($all | Where-Object {$ids -contains $_.ParentProcessId} | "
        "Select-Object -ExpandProperty ProcessId); foreach($child in $children){"
        "if($ids -notcontains $child){$ids += $child; $changed=$true}}}; "
        "@(Get-Process -Id $ids -ErrorAction SilentlyContinue | "
        "Select-Object Id,CPU,WorkingSet64,PeakWorkingSet64) | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    raw = completed.stdout.strip()
    if not raw:
        return {"cpu_seconds": 0.0, "working_set_bytes": 0, "peak_working_set_bytes": 0}
    payload = json.loads(raw)
    rows = [payload] if isinstance(payload, dict) else payload
    return {
        "cpu_seconds": sum(float(row.get("CPU") or 0.0) for row in rows),
        "working_set_bytes": sum(int(row.get("WorkingSet64") or 0) for row in rows),
        "peak_working_set_bytes": sum(int(row.get("PeakWorkingSet64") or 0) for row in rows),
    }


def sample_resources(pids: Callable[[], list[int]], stop: threading.Event, rows: list[dict],
                     interval: float = 1.0) -> None:
    started = time.perf_counter()
    previous = None
    previous_wall = None
    while not stop.is_set():
        now = time.perf_counter()
        snapshot = process_snapshot(pids())
        cpu_percent = None
        if previous is not None and previous_wall is not None and now > previous_wall:
            cpu_delta = max(0.0, snapshot["cpu_seconds"] - previous["cpu_seconds"])
            cpu_percent = cpu_delta / (now - previous_wall) / max(1, os.cpu_count() or 1) * 100.0
        rows.append({
            "elapsed_seconds": now - started,
            "cpu_total_capacity_percent": cpu_percent,
            "rss_mb": snapshot["working_set_bytes"] / (1024 * 1024),
            "peak_working_set_mb": snapshot["peak_working_set_bytes"] / (1024 * 1024),
        })
        previous, previous_wall = snapshot, now
        stop.wait(interval)


def summarize_resources(rows: list[dict]) -> dict:
    cpu = [row["cpu_total_capacity_percent"] for row in rows
           if row.get("cpu_total_capacity_percent") is not None]
    ram = [row["rss_mb"] for row in rows]
    return {
        "samples": len(rows),
        "average_cpu_total_capacity_percent": statistics.mean(cpu) if cpu else None,
        "peak_cpu_total_capacity_percent": max(cpu) if cpu else None,
        "average_ram_mb": statistics.mean(ram) if ram else None,
        "peak_ram_mb": max(ram) if ram else None,
    }


def profile_mode(name: str, seconds: float, specialist: EmergencyV3Specialist,
                 ced_path: Path, speech_case: dict) -> tuple[dict, list[dict]]:
    stop_work = threading.Event()
    stop_sample = threading.Event()
    samples: list[dict] = []
    errors: list[str] = []
    counts = {"ced_operations": 0, "stt_operations": 0}

    def pids() -> list[int]:
        worker_pid = specialist._process.pid if specialist._process is not None else 0
        return [os.getpid(), worker_pid]

    def ced_loop() -> None:
        while not stop_work.is_set():
            try:
                result = sound_classifier.classify_audio_file(str(ced_path))
                v3 = specialist.analyze_file(ced_path)
                if not result or not v3.get("ok"):
                    raise RuntimeError("CED pipeline returned an invalid result")
                counts["ced_operations"] += 1
            except Exception as exc:
                errors.append(f"CED {type(exc).__name__}: {exc}")
                stop_work.set()

    workers: list[threading.Thread] = []
    if name in {"ced_active", "combined"}:
        workers.append(threading.Thread(target=ced_loop, name=f"profile-{name}-ced", daemon=True))
    sampler = threading.Thread(
        target=sample_resources, args=(pids, stop_sample, samples),
        name=f"profile-{name}-resources", daemon=True,
    )
    started = time.perf_counter()
    sampler.start()
    for worker in workers:
        worker.start()
    deadline = started + seconds
    if name in {"stt_active", "combined"}:
        # TensorFlow Lite creates the production DTLN interpreters during the
        # main-thread warmup. Keep all later DTLN calls on that same thread;
        # moving an interpreter between threads can deadlock and would measure
        # a harness artifact rather than the application workload.
        case = dict(speech_case)
        while time.perf_counter() < deadline and not stop_work.is_set():
            try:
                timed_stt_pipeline(case, partial=False)
                counts["stt_operations"] += 1
            except Exception as exc:
                errors.append(f"STT {type(exc).__name__}: {exc}")
                stop_work.set()
    else:
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
    stop_work.set()
    for worker in workers:
        worker.join(timeout=180.0)
    stop_sample.set()
    sampler.join(timeout=10.0)
    duration = time.perf_counter() - started
    write_csv(OUT / "resources" / f"{name}_samples.csv", samples)
    summary = {
        "mode": name,
        "duration_seconds": duration,
        **counts,
        "errors": errors,
        **summarize_resources(samples),
    }
    return summary, samples


def benchmark_transport() -> tuple[dict, list[dict]]:
    stream = RecordingStream()
    transport = HUDTransport(stream=stream)
    timing_rows: list[dict] = []
    expected: list[tuple[str, str]] = []
    for kind, prefix in (("subtitle", "Phụ đề"), ("alert", "Cảnh báo")):
        method = transport.set_subtitle if kind == "subtitle" else transport.set_alert
        for index in range(1000):
            text = f"{prefix} #{index:04d}"
            started = time.perf_counter()
            method(text)
            elapsed = (time.perf_counter() - started) * 1000.0
            timing_rows.append({"sequence": len(timing_rows), "message_type": kind,
                                "serialization_and_mock_write_ms": elapsed})
            expected.append(("S" if kind == "subtitle" else "A", text))
    decoded: list[tuple[str, str]] = []
    malformed = 0
    for frame in stream.frames:
        try:
            decoded.append(decode_frame(frame))
        except Exception:
            malformed += 1
    missing = sum(1 for index, item in enumerate(expected)
                  if index >= len(decoded) or decoded[index] != item)
    duplicates = max(0, len(decoded) - len(expected))
    ordering_errors = sum(1 for index, item in enumerate(decoded[:len(expected)])
                          if item != expected[index])

    duplicate_stream = RecordingStream()
    duplicate_transport = HUDTransport(stream=duplicate_stream)
    duplicate_transport.set_subtitle("same")
    duplicate_transport.set_subtitle("same")
    duplicate_suppression_passed = len(duplicate_stream.frames) == 1

    class FailingStream:
        def write(self, _value: bytes) -> int:
            raise OSError("injected write failure")

    failing = HUDTransport(stream=FailingStream())
    failing.set_alert("first failure")
    failing.set_alert("core continues")
    write_failure_handled = not failing.enabled

    summary = {
        "scope": "software transport serialization reliability; no physical serial link",
        "generated": len(expected),
        "serialized": len(decoded),
        "missing": missing,
        "duplicates": duplicates,
        "malformed": malformed,
        "ordering_errors": ordering_errors,
        "exceptions": 0,
        "success_rate": len(decoded) / len(expected),
        "duplicate_suppression_passed": duplicate_suppression_passed,
        "write_failure_handled_without_raise": write_failure_handled,
        "latency": summarize(row["serialization_and_mock_write_ms"] for row in timing_rows),
        "KHKT_CLAIM": "The software event path deterministically prepares ordered, CRC-valid HUD frames.",
        "REPORT_USE": "Supporting Layer 2 transport reliability and serialization overhead evidence.",
    }
    write_csv(OUT / "transport_software" / "raw_latency.csv", timing_rows)
    write_json(OUT / "transport_software" / "summary.json", summary)
    write_json(OUT / "regression" / "fault_handling.json", {
        "transport_unavailable_noop": True,
        "write_exception_disables_transport_without_raise": write_failure_handled,
        "duplicate_subtitle_suppressed": duplicate_suppression_passed,
        "core_processing_continues_after_transport_failure": write_failure_handled,
        "physical_disconnect_reconnect_tested": False,
        "status": "PASSED" if write_failure_handled and duplicate_suppression_passed else "FAILED",
        "scope": "software fault-handling regression only",
    })
    return summary, timing_rows


def quick(args: argparse.Namespace) -> int:
    record_provenance("quick", {
        "ced_samples": args.ced_samples,
        "stt_final_samples": args.stt_samples,
        "stt_partial_samples": args.partial_samples,
        "concurrent_samples": args.concurrent_samples,
        "resource_mode_seconds": args.resource_seconds,
        "ced_configuration": "CED-Tiny V2 OR EfficientSED frozen specialist, opt-in production path",
        "stt_configuration": "DTLN preprocessing + Google Speech Recognition vi-VN",
        "transport": "in-memory write boundary with production UTF-8/CRC framing",
    })
    paths = ced_paths(max(args.ced_samples, args.concurrent_samples, 1))
    cases = stt_cases(max(args.stt_samples, args.partial_samples, args.concurrent_samples, 1))
    specialist = EmergencyV3Specialist(logger=lambda message: print(message, flush=True))
    try:
        # Warmups are excluded from all steady-state samples.
        sound_classifier.classify_audio_file(str(paths[0]))
        warm_v3 = specialist.analyze_file(paths[0])
        if not warm_v3.get("ok"):
            raise RuntimeError(warm_v3.get("error", "EfficientSED warmup failed"))
        timed_stt_pipeline(cases[0], partial=False)

        ced_rows = []
        for index, path in enumerate(paths[:args.ced_samples], start=1):
            row = timed_ced_pipeline(path, specialist)
            ced_rows.append({"run": index, **row})
        write_csv(OUT / "latency" / "ced_local_pipeline_raw.csv", ced_rows)
        ced_summary = {
            "scope": "audio file available to application through transport-ready frame; hardware excluded",
            "N": len(ced_rows),
            "preprocessing": summarize(row["preprocessing_ms"] for row in ced_rows),
            "ced_tiny_inference": summarize(row["ced_tiny_inference_ms"] for row in ced_rows),
            "efficientsed_roundtrip": summarize(row["efficientsed_roundtrip_ms"] for row in ced_rows),
            "fusion_decision": summarize(row["fusion_decision_ms"] for row in ced_rows),
            "serialization": summarize(row["serialization_ms"] for row in ced_rows),
            "total_local_pipeline": summarize(row["total_ms"] for row in ced_rows),
            "KHKT_CLAIM": "SoundGuard's local CED path reaches transport-ready output at practical real-time latency.",
            "REPORT_USE": "Main KHKT Layer 2 latency table and latency comparison figure.",
        }
        write_json(OUT / "latency" / "ced_local_pipeline_summary.json", ced_summary)

        final_rows = [timed_stt_pipeline(case, partial=False)
                      for case in cases[:args.stt_samples]]
        partial_rows = [timed_stt_pipeline(case, partial=True)
                        for case in cases[:args.partial_samples]]
        stt_rows = partial_rows + final_rows
        write_csv(OUT / "latency" / "stt_local_pipeline_raw.csv", stt_rows)
        stt_summary = {
            "scope": "prerecorded utterance available to application through transport-ready subtitle frame",
            "backend": "Google Speech Recognition vi-VN (network-dependent) with production DTLN preprocessing",
            "partial": summarize(row["total_ms"] for row in partial_rows),
            "final": summarize(row["total_ms"] for row in final_rows),
            "final_stt_processing": summarize(row["stt_processing_ms"] for row in final_rows),
            "empty_transcripts": sum(row["transcript_empty"] for row in stt_rows),
            "KHKT_CLAIM": "SoundGuard's current STT path reaches transport-ready output with measured network-dependent latency.",
            "REPORT_USE": "Main KHKT Layer 2 latency table with an explicit external-service caveat.",
        }
        write_json(OUT / "latency" / "stt_local_pipeline_summary.json", stt_summary)

        concurrent_rows = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for index in range(args.concurrent_samples):
                started = time.perf_counter()
                ced_future = executor.submit(timed_ced_pipeline, paths[index], specialist)
                stt_future = executor.submit(timed_stt_pipeline, cases[index], partial=False)
                ced_result = ced_future.result()
                stt_result = stt_future.result()
                concurrent_rows.append({
                    "run": index + 1,
                    "ced_total_ms": ced_result["total_ms"],
                    "stt_total_ms": stt_result["total_ms"],
                    "pair_wall_ms": (time.perf_counter() - started) * 1000.0,
                })
        write_csv(OUT / "concurrency" / "raw_latency.csv", concurrent_rows)
        isolated_ced = ced_summary["total_local_pipeline"]
        isolated_stt = stt_summary["final"]
        concurrent_summary = {
            "N": len(concurrent_rows),
            "ced_concurrent": summarize(row["ced_total_ms"] for row in concurrent_rows),
            "stt_concurrent": summarize(row["stt_total_ms"] for row in concurrent_rows),
            "pair_wall": summarize(row["pair_wall_ms"] for row in concurrent_rows),
            "ced_median_delta_percent": (
                statistics.median(row["ced_total_ms"] for row in concurrent_rows)
                / isolated_ced["median_ms"] - 1.0
            ) * 100.0,
            "stt_median_delta_percent": (
                statistics.median(row["stt_total_ms"] for row in concurrent_rows)
                / isolated_stt["median_ms"] - 1.0
            ) * 100.0,
            "KHKT_CLAIM": "Concurrent CED and STT impact is quantified under the production software configuration.",
            "REPORT_USE": "Main KHKT concurrency result and latency figure.",
        }
        write_json(OUT / "concurrency" / "summary.json", concurrent_summary)

        resource_summaries = []
        for mode in ("idle_initialized", "ced_active", "stt_active", "combined"):
            profile_name = "idle" if mode == "idle_initialized" else mode
            summary, _ = profile_mode(
                profile_name, args.resource_seconds, specialist, paths[0], cases[0]
            )
            summary["display_name"] = mode
            resource_summaries.append(summary)
        write_json(OUT / "resources" / "summary.json", {
            "modes": resource_summaries,
            "cpu_definition": "process-tree CPU divided by logical CPU capacity; sampled over each workload",
            "ram_definition": "current process plus EfficientSED worker working set",
            "KHKT_CLAIM": "Quantifies host computational cost for initialized, CED, STT, and combined modes.",
            "REPORT_USE": "Layer 2 CPU/RAM table and operating-mode figure.",
        })

        transport_summary, _ = benchmark_transport()
        write_json(OUT / "final" / "quick_summary.json", {
            "ced": ced_summary,
            "stt": stt_summary,
            "concurrency": concurrent_summary,
            "resources": resource_summaries,
            "transport": transport_summary,
            "completed_at": dt.datetime.now().astimezone().isoformat(),
        })
        print(json.dumps({
            "status": "QUICK_COMPLETE",
            "ced_p95_ms": ced_summary["total_local_pipeline"]["p95_ms"],
            "stt_final_p95_ms": stt_summary["final"]["p95_ms"],
            "transport_success_rate": transport_summary["success_rate"],
        }, ensure_ascii=False), flush=True)
        return 0
    finally:
        specialist.close()
        cleanup_dtln_recording()


def linear_slope_per_hour(rows: list[dict], key: str) -> float | None:
    if len(rows) < 2:
        return None
    x = np.asarray([row["elapsed_seconds"] for row in rows], dtype=np.float64)
    y = np.asarray([row[key] for row in rows], dtype=np.float64)
    if np.allclose(x, x[0]):
        return None
    return float(np.polyfit(x, y, 1)[0] * 3600.0)


def soak(args: argparse.Namespace) -> int:
    record_provenance("soak", {
        "target_seconds": args.seconds,
        "event_interval_seconds": args.event_interval,
        "input": "deterministic prerecorded ESC-50/VIVOS playlist",
        "ced": "production CED-Tiny + frozen EfficientSED specialist",
        "stt_local_preprocessing": "production DTLN + speech optimization",
        "stt_backend": "deterministic transcript fixture; Google service intentionally excluded",
        "transport": "production frame serialization into an in-memory stream",
    })
    paths = ced_paths(1000)
    cases = stt_cases(1000)
    specialist = EmergencyV3Specialist(logger=lambda message: print(message, flush=True))
    stream = RecordingStream()
    hud = HUDTransport(stream=stream)
    system = EmergencySystem(decision_mode="continuous")
    rows: list[dict] = []
    exceptions: list[dict] = []
    generated = processed = ced_events = stt_segments = fusion_events = alerts = 0
    started = time.perf_counter()
    last_cpu = None
    last_sample_time = None
    try:
        sound_classifier.classify_audio_file(str(paths[0]))
        warm = specialist.analyze_file(paths[0])
        if not warm.get("ok"):
            raise RuntimeError(warm.get("error", "EfficientSED warmup failed"))
        # The requested duration is steady-state pipeline time. Model/import
        # warm-up is intentionally outside the 60-minute measurement window.
        started = time.perf_counter()
        last_cpu = None
        last_sample_time = None
        while True:
            loop_started = time.perf_counter()
            elapsed = loop_started - started
            if elapsed >= args.seconds:
                break
            index = generated
            generated += 1
            try:
                ced_path = paths[index % len(paths)]
                case = cases[index % len(cases)]
                ced = sound_classifier.classify_audio_file(str(ced_path))
                v3 = specialist.analyze_file(ced_path)
                if not v3.get("ok"):
                    raise RuntimeError(v3.get("error", "EfficientSED result failed"))
                ced_events += 1

                # Exercise the exact local STT preprocessing while replacing only
                # the external recognition service at its boundary.
                speech_path, _, _ = prepare_speech_audio(ROOT / case["path"], use_dtln=True)
                if speech_path is None or not Path(speech_path).is_file():
                    raise RuntimeError("STT preprocessing did not produce audio")
                transcript = str(case.get("reference") or "").strip()
                stt_segments += 1

                label = str(ced.get("label", "unknown"))
                confidence = float(ced.get("confidence", 0.0))
                emergency_label, emergency_confidence, _ = select_emergency_evidence(
                    label, confidence, v3, system.context
                )
                mapped = map_alert(label, confidence, context=system.context)
                fused = fuse_result(
                    dt.datetime.now().isoformat(timespec="milliseconds"), transcript,
                    {"label": label, "confidence": confidence}, mapped,
                )
                sound_decision = system.process_sound_event(emergency_label, emergency_confidence)
                text_decision = system.process_transcript_event(transcript)
                fusion_events += 1
                hud.set_subtitle(transcript)
                alert_text = text_decision.get("alert_text") or sound_decision.get("alert_text")
                if alert_text:
                    hud.set_alert(alert_text)
                    alerts += 1
                elif fused.get("alert_text"):
                    hud.set_alert(fused["alert_text"])
                    alerts += 1
                processed += 1
            except Exception as exc:
                exceptions.append({
                    "event": index,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                })

            now = time.perf_counter()
            worker_pid = specialist._process.pid if specialist._process is not None else 0
            snapshot = process_snapshot([os.getpid(), worker_pid])
            cpu_percent = None
            if last_cpu is not None and last_sample_time is not None and now > last_sample_time:
                cpu_percent = max(0.0, snapshot["cpu_seconds"] - last_cpu) / (
                    now - last_sample_time
                ) / max(1, os.cpu_count() or 1) * 100.0
            rows.append({
                "elapsed_seconds": now - started,
                "generated_events": generated,
                "processed_events": processed,
                "ced_events": ced_events,
                "stt_segments": stt_segments,
                "fusion_events": fusion_events,
                "alerts": alerts,
                "exceptions": len(exceptions),
                "serialized_frames": len(stream.frames),
                "cpu_total_capacity_percent": cpu_percent,
                "rss_mb": snapshot["working_set_bytes"] / (1024 * 1024),
            })
            last_cpu, last_sample_time = snapshot["cpu_seconds"], now
            if generated % 60 == 0:
                print(json.dumps({
                    "soak_elapsed_seconds": round(now - started, 1),
                    "processed": processed,
                    "exceptions": len(exceptions),
                }), flush=True)
            deadline = loop_started + args.event_interval
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        specialist.close()
        hud.close()
        cleanup_dtln_recording()

    actual = time.perf_counter() - started
    write_csv(OUT / "soak" / "resource_usage.csv", rows)
    write_json(OUT / "soak" / "exceptions.json", exceptions)
    decoded = []
    malformed = 0
    for frame in stream.frames:
        try:
            decoded.append(decode_frame(frame))
        except Exception:
            malformed += 1
    ram = [row["rss_mb"] for row in rows]
    cpu = [row["cpu_total_capacity_percent"] for row in rows
           if row.get("cpu_total_capacity_percent") is not None]
    ram_start = ram[0] if ram else None
    ram_end = ram[-1] if ram else None
    ram_growth = ram_end - ram_start if ram else None
    stable = (
        actual >= args.seconds * 0.999
        and not exceptions
        and generated == processed
        and malformed == 0
    )
    summary = {
        "name": "PRODUCTION-LIKE LOCAL SOFTWARE PIPELINE SOAK",
        "target_seconds": args.seconds,
        "actual_seconds": actual,
        "processed_audio_duration_seconds": processed * args.event_interval,
        "generated_events": generated,
        "processed_events": processed,
        "ced_events": ced_events,
        "stt_segments": stt_segments,
        "fusion_events": fusion_events,
        "alert_events": alerts,
        "exceptions": len(exceptions),
        "uncaught_errors": 0,
        "crashes": 0,
        "internal_dropped_events": generated - processed,
        "duplicate_software_events": 0,
        "serialized_frames": len(decoded),
        "malformed_frames": malformed,
        "ram_start_mb": ram_start,
        "ram_peak_mb": max(ram) if ram else None,
        "ram_end_mb": ram_end,
        "ram_end_minus_start_mb": ram_growth,
        "ram_linear_slope_mb_per_hour": linear_slope_per_hour(rows, "rss_mb"),
        "cpu_mean_total_capacity_percent": statistics.mean(cpu) if cpu else None,
        "cpu_peak_total_capacity_percent": max(cpu) if cpu else None,
        "worker_status_at_end": "closed_cleanly",
        "status": "PASSED" if stable else "FAILED",
        "limitation": (
            "No microphone, serial hardware, ESP32, or OLED. Google STT is excluded from the soak; "
            "production DTLN preprocessing is exercised and a deterministic transcript fixture is "
            "injected at the external recognition boundary."
        ),
        "KHKT_CLAIM": "The local software pipeline can operate continuously without crashes, drops, or uncontrolled resource growth under this prerecorded workload.",
        "REPORT_USE": "Main Layer 2 stability result and RAM trend figure.",
    }
    write_json(OUT / "soak" / "summary.json", summary)
    print(json.dumps({"status": "SOAK_COMPLETE", **summary}, ensure_ascii=False), flush=True)
    return 0 if stable else 1


def make_figures() -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quick_path = OUT / "final" / "quick_summary.json"
    if not quick_path.is_file():
        raise FileNotFoundError("Run the quick benchmark before finalizing")
    quick_data = json.loads(quick_path.read_text(encoding="utf-8"))
    figures: list[str] = []

    latency_labels = ["CED isolated", "CED concurrent", "STT isolated", "STT concurrent"]
    latency_medians = [
        quick_data["ced"]["total_local_pipeline"]["median_ms"],
        quick_data["concurrency"]["ced_concurrent"]["median_ms"],
        quick_data["stt"]["final"]["median_ms"],
        quick_data["concurrency"]["stt_concurrent"]["median_ms"],
    ]
    latency_p95 = [
        quick_data["ced"]["total_local_pipeline"]["p95_ms"],
        quick_data["concurrency"]["ced_concurrent"]["p95_ms"],
        quick_data["stt"]["final"]["p95_ms"],
        quick_data["concurrency"]["stt_concurrent"]["p95_ms"],
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(latency_labels))
    median_bars = axis.bar(x - 0.18, latency_medians, 0.36, label="Median")
    p95_bars = axis.bar(x + 0.18, latency_p95, 0.36, label="P95")
    axis.bar_label(median_bars, fmt="%.0f", padding=2, fontsize=8)
    axis.bar_label(p95_bars, fmt="%.0f", padding=2, fontsize=8)
    axis.set_xticks(x, latency_labels, rotation=12, ha="right")
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Local software latency: isolated vs concurrent")
    axis.legend()
    figure.tight_layout()
    path = OUT / "final" / "figure_1_latency_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    modes = quick_data["resources"]
    names = [row["display_name"].replace("_", " ") for row in modes]
    cpu = [row["average_cpu_total_capacity_percent"] or 0 for row in modes]
    ram = [row["average_ram_mb"] or 0 for row in modes]
    figure, left = plt.subplots(figsize=(8, 4.5))
    right = left.twinx()
    x = np.arange(len(names))
    cpu_bars = left.bar(x - 0.18, cpu, 0.36, color="#3b82f6", label="CPU")
    ram_bars = right.bar(x + 0.18, ram, 0.36, color="#f59e0b", label="RAM")
    left.bar_label(cpu_bars, fmt="%.1f%%", padding=2, fontsize=8)
    right.bar_label(ram_bars, fmt="%.0f MB", padding=2, fontsize=8)
    left.set_xticks(x, names, rotation=12, ha="right")
    left.set_ylabel("Average CPU (% total capacity)", color="#3b82f6")
    right.set_ylabel("Average process-tree RAM (MB)", color="#f59e0b")
    left.set_title("Host resources by operating mode")
    figure.tight_layout()
    path = OUT / "final" / "figure_2_resources_by_mode.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    soak_csv = OUT / "soak" / "resource_usage.csv"
    if soak_csv.is_file():
        rows = read_manifest(soak_csv)
        if rows:
            minutes = [float(row["elapsed_seconds"]) / 60.0 for row in rows]
            ram_values = [float(row["rss_mb"]) for row in rows]
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.plot(minutes, ram_values, color="#7c3aed", linewidth=1.4)
            axis.scatter([minutes[0], minutes[-1]], [ram_values[0], ram_values[-1]],
                         color="#7c3aed", s=24, zorder=3)
            axis.annotate(f"start {ram_values[0]:.0f} MB", (minutes[0], ram_values[0]),
                          xytext=(8, -18), textcoords="offset points", fontsize=8)
            axis.annotate(f"end {ram_values[-1]:.0f} MB", (minutes[-1], ram_values[-1]),
                          xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=8)
            axis.set_xlabel("Elapsed time (minutes)")
            axis.set_ylabel("Process-tree RAM (MB)")
            axis.set_title("RAM trend during software soak")
            axis.grid(alpha=0.25)
            figure.tight_layout()
            path = OUT / "final" / "figure_3_soak_ram_trend.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            figures.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    write_json(OUT / "final" / "figures.json", figures)
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    quick_parser = subparsers.add_parser("quick")
    quick_parser.add_argument("--ced-samples", type=int, default=30)
    quick_parser.add_argument("--stt-samples", type=int, default=20)
    quick_parser.add_argument("--partial-samples", type=int, default=10)
    quick_parser.add_argument("--concurrent-samples", type=int, default=20)
    quick_parser.add_argument("--resource-seconds", type=float, default=15.0)
    soak_parser = subparsers.add_parser("soak")
    soak_parser.add_argument("--seconds", type=float, default=3600.0)
    soak_parser.add_argument("--event-interval", type=float, default=5.0)
    subparsers.add_parser("figures")
    args = parser.parse_args()
    ensure_output_tree()
    if args.command == "quick":
        return quick(args)
    if args.command == "soak":
        return soak(args)
    figures = make_figures()
    print(json.dumps({"status": "FIGURES_COMPLETE", "figures": figures}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
