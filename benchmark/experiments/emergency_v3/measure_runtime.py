"""Measure the opt-in production adapter after its model is warm."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from emergency_v3 import EmergencyV3Specialist

OUT = ROOT / "benchmark_results" / "emergency_v3" / "latency" / "runtime_resources.json"


def python_processes() -> dict[int, dict]:
    command = (
        "@(Get-Process python* -ErrorAction SilentlyContinue | "
        "Select-Object Id,CPU,WorkingSet64,PeakWorkingSet64,VirtualMemorySize64) | "
        "ConvertTo-Json -Compress"
    )
    output = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", command], text=True
    )
    payload = json.loads(output)
    if isinstance(payload, dict):
        payload = [payload]
    return {int(item["Id"]): item for item in payload}


def process_snapshot(process_ids: set[int]) -> dict:
    processes = python_processes()
    selected = [processes[pid] for pid in process_ids if pid in processes]
    return {
        "Ids": sorted(process_ids),
        "CPU": sum(float(item.get("CPU") or 0.0) for item in selected),
        "WorkingSet64": sum(int(item["WorkingSet64"]) for item in selected),
        "PeakWorkingSet64": sum(int(item["PeakWorkingSet64"]) for item in selected),
        "VirtualMemorySize64": sum(int(item["VirtualMemorySize64"]) for item in selected),
    }


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[ceil(quantile * len(values)) - 1]


def main() -> None:
    with (ROOT / "benchmark_data/external/esc50/manifest.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        paths = [ROOT / row["path"] for row in csv.DictReader(stream)][:30]
    preexisting_process_ids = set(python_processes())
    specialist = EmergencyV3Specialist()
    try:
        warm = specialist.analyze_file(paths[0])
        if not warm.get("ok") or specialist._process is None:
            raise RuntimeError(warm.get("error", "worker failed to load"))
        worker_process_ids = set(python_processes()) - preexisting_process_ids
        before = process_snapshot(worker_process_ids)
        wall_start = time.perf_counter()
        latencies = []
        for path in paths:
            started = time.perf_counter()
            result = specialist.analyze_file(path)
            if not result.get("ok"):
                raise RuntimeError(result.get("error"))
            latencies.append((time.perf_counter() - started) * 1000.0)
        wall_seconds = time.perf_counter() - wall_start
        after = process_snapshot(worker_process_ids)
        cpu_seconds = float(after["CPU"]) - float(before["CPU"])
        payload = {
            "sample_count": len(latencies),
            "scope": "warm production subprocess round trip including file decode and IPC",
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p90_ms": percentile(latencies, 0.90),
            "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99),
            "working_set_bytes": int(after["WorkingSet64"]),
            "peak_working_set_bytes": int(after["PeakWorkingSet64"]),
            "virtual_memory_bytes": int(after["VirtualMemorySize64"]),
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "single_core_utilization_percent": cpu_seconds / wall_seconds * 100.0,
            "total_machine_cpu_capacity_percent": (
                cpu_seconds / wall_seconds / (os.cpu_count() or 1) * 100.0
            ),
            "real_time_factor_for_5_second_clips": statistics.mean(latencies) / 5000.0,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        specialist.close()


if __name__ == "__main__":
    main()
