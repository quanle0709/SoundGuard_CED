"""Targeted real-hardware diagnosis for the Layer 2 USB/HUD transport.

This suite is deliberately separate from the final campaign.  It can exercise
benchmark ACK traffic or ordinary production frames, records the effective
rate/byte load, samples benchmark-only firmware counters, and stops after three
consecutive missing ACKs instead of waiting through the campaign timeout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_layer2_hardware_validation import (  # noqa: E402
    HardwareSession,
    clock_payload,
    discover_port,
    frame,
    utc_now,
)


DIAG_FIELDS = (
    "rx_bytes", "completed_frames", "valid_frames", "crc_failures",
    "parser_recoveries", "malformed_benchmark_frames", "ack_attempts",
    "ack_bytes_accepted", "ack_zero_writes", "tx_available",
    "min_tx_available", "poll_count", "firmware_ms", "free_heap",
    "minimum_free_heap", "largest_free_block", "stack_high_water",
    "reset_reason", "display_commits", "last_display_us", "max_display_us",
    "last_parsed_id", "last_ack_attempt_id",
)

MIX_TYPES = ("S", "P", "C", "E", "K", "A", "T")
MIX_PAYLOADS = {
    "S": "Transport subtitle", "P": "Transport partial", "C": "DOG",
    "E": "1|", "A": "Legacy alert", "T": "READY",
}


class DiagnosticSession(HardwareSession):
    def diagnostics(self, label: str) -> dict[str, Any]:
        trial_id = f"diag-{label}"
        ack = self.send("D", "", trial_id=trial_id, timeout=2.0)
        if ack["failure"]:
            return {"label": label, "failure": ack["failure"]}
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.serial.timeout = max(0.01, deadline - time.monotonic())
            line = self.serial.readline().decode("ascii", errors="replace").strip()
            if not line:
                break
            parts = line.split("|")
            if len(parts) == 2 + len(DIAG_FIELDS) and parts[:2] == ["SGDIAG", trial_id]:
                values: dict[str, Any] = {"label": label, "failure": ""}
                for key, value in zip(DIAG_FIELDS, parts[2:]):
                    values[key] = value if key.startswith("last_") and key.endswith("_id") else int(value)
                self._log(f"DIAG {json.dumps(values, sort_keys=True)}")
                return values
            self._log(f"UNMATCHED_DIAG while_waiting={trial_id} line={line!r}")
        return {"label": label, "failure": "missing_or_malformed_diagnostic"}


def payload_for(message_type: str, index: int) -> str:
    if message_type == "K":
        return clock_payload()
    return MIX_PAYLOADS.get(message_type, f"Value {index & 1}")


def pace(started: float, transactions: int, rate: float | None) -> None:
    if rate is None:
        return
    remaining = started + transactions / rate - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def ack_run(hw: DiagnosticSession, *, count: int, rate: float | None,
            isolated_type: str | None, sample_every: int) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    samples = [hw.diagnostics("start")]
    transactions = sent_bytes = received_bytes = 0
    consecutive_failures = 0
    for index in range(count):
        message_type = isolated_type or MIX_TYPES[index % len(MIX_TYPES)]
        payload = payload_for(message_type, index)
        trial_id = f"ack-{message_type}-{index:05d}"
        encoded = frame("B", f"{trial_id}|{message_type}|{payload}")
        result = hw.send(message_type, payload, trial_id=trial_id, timeout=1.0)
        transactions += 1
        sent_bytes += len(encoded)
        received_bytes += len(result["ack"].encode("ascii", errors="replace"))
        rows.append({"index": index, "type": message_type,
                     "failure": result["failure"],
                     "latency_ms": result["latency_ms"]})
        consecutive_failures = consecutive_failures + 1 if result["failure"] else 0
        pace(started, transactions, rate)
        if consecutive_failures >= 3:
            break
        if isolated_type is None and message_type in {"A", "E"}:
            clear_id = f"ack-{message_type}-clear-{index:05d}"
            encoded = frame("B", f"{clear_id}|{message_type}|")
            clear = hw.send(message_type, "", trial_id=clear_id, timeout=1.0)
            transactions += 1
            sent_bytes += len(encoded)
            received_bytes += len(clear["ack"].encode("ascii", errors="replace"))
            rows.append({"index": index, "type": f"{message_type}-clear",
                         "failure": clear["failure"],
                         "latency_ms": clear["latency_ms"]})
            consecutive_failures = consecutive_failures + 1 if clear["failure"] else 0
            pace(started, transactions, rate)
            if consecutive_failures >= 3:
                break
        if sample_every and (index + 1) % sample_every == 0:
            samples.append(hw.diagnostics(f"{index + 1:05d}"))
    elapsed = time.perf_counter() - started
    post = hw.send("Q", "", trial_id="ack-post-query", timeout=2.0)
    samples.append(hw.diagnostics("end") if not post["failure"] else {
        "label": "end", "failure": "device_unresponsive"})
    return {
        "mode": "ack-type" if isolated_type else "ack-mixed",
        "isolated_type": isolated_type,
        "requested_primary_frames": count,
        "attempted_transactions": transactions,
        "acked": sum(not row["failure"] for row in rows),
        "missing": sum(bool(row["failure"]) for row in rows),
        "first_failures": [row for row in rows if row["failure"]][:10],
        "elapsed_seconds": elapsed,
        "effective_transactions_per_second": transactions / elapsed,
        "target_transactions_per_second": rate,
        "bytes_sent": sent_bytes,
        "bytes_received": received_bytes,
        "outstanding_requests": 1,
        "host_timeout_seconds": 1.0,
        "post_query_ok": not post["failure"],
        "diagnostic_samples": samples,
    }


def no_ack_run(hw: DiagnosticSession, *, count: int, rate: float,
               production_like: bool) -> dict[str, Any]:
    started = time.perf_counter()
    samples = [hw.diagnostics("start")]
    transactions = sent_bytes = 0
    for index in range(count):
        if production_like:
            # A representative live mix: changing partials/finals, CED at a
            # slower cadence, rare alert lifecycle, and periodic clock sync.
            if index % 500 == 0:
                message_type, payload = "K", clock_payload()
            elif index % 250 == 0:
                message_type, payload = "E", "1|SIREN"
            elif index % 250 == 1:
                message_type, payload = "E", ""
            elif index % 25 == 0:
                message_type, payload = "C", "DOG" if (index // 25) & 1 else ""
            elif index % 10 == 0:
                message_type, payload = "S", f"Final {index}"
            else:
                message_type, payload = "P", f"Partial {index}"
        else:
            message_type = MIX_TYPES[index % len(MIX_TYPES)]
            payload = payload_for(message_type, index)
        encoded = frame(message_type, payload)
        hw.serial.write(encoded)
        transactions += 1
        sent_bytes += len(encoded)
        if not production_like and message_type in {"A", "E"}:
            encoded = frame(message_type, "")
            hw.serial.write(encoded)
            transactions += 1
            sent_bytes += len(encoded)
        pace(started, transactions, rate)
    hw.serial.flush()
    time.sleep(1.0)
    elapsed = time.perf_counter() - started
    post = hw.send("Q", "", trial_id="no-ack-post-query", timeout=2.0)
    samples.append(hw.diagnostics("end") if not post["failure"] else {
        "label": "end", "failure": "device_unresponsive"})
    return {
        "mode": "production-like" if production_like else "no-ack-mixed",
        "requested_primary_frames": count,
        "attempted_transactions": transactions,
        "elapsed_seconds": elapsed,
        "effective_transactions_per_second": transactions / elapsed,
        "target_transactions_per_second": rate,
        "bytes_sent": sent_bytes,
        "bytes_received": 0,
        "benchmark_ack_enabled_for_workload": False,
        "post_query_ok": not post["failure"],
        "diagnostic_samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="auto")
    parser.add_argument("--mode", choices=("ack-mixed", "ack-type", "no-ack-mixed", "production-like"), required=True)
    parser.add_argument("--count", type=int, default=2500)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--type", choices=MIX_TYPES + ("Q",))
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or (args.rate is not None and args.rate <= 0):
        parser.error("count and rate must be positive")
    if args.mode == "ack-type" and not args.type:
        parser.error("--type is required for ack-type")
    if args.mode in {"no-ack-mixed", "production-like"} and args.rate is None:
        parser.error("--rate is required for non-ACK workloads")

    port, identity = discover_port(args.port)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_log = args.output.with_suffix(".serial.log")
    hw = DiagnosticSession(port, raw_log)
    try:
        if args.mode.startswith("ack-"):
            result = ack_run(hw, count=args.count, rate=args.rate,
                             isolated_type=args.type if args.mode == "ack-type" else None,
                             sample_every=args.sample_every)
        else:
            result = no_ack_run(hw, count=args.count, rate=float(args.rate),
                                production_like=args.mode == "production-like")
    finally:
        hw.close()
    result.update({"started_utc": utc_now(), "port_identity": identity})
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(result, indent=2))
    return 0 if result["post_query_ok"] and not result.get("missing") else 1


if __name__ == "__main__":
    raise SystemExit(main())
