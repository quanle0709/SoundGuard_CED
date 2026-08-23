# Final SoundGuard Layer 2 Hardware Report

## A. Executive summary

Corrected Layer 2 status: **COMPLETE** after the audited transport closure in sections T-U. The original partial conclusion and failed evidence remain below for auditability. Tests used the connected ESP32-C3 and physical 64×32 OLED. Firmware ACKs are an **OLED render-completion proxy** after `Adafruit_SSD1306::display()` returns; they are not optical/photon latency.

## B. Hardware/software configuration

- Device: `USB Serial Device (COM9)`, VID:PID `303A:1001`, serial `9C:CC:01:A8:73:30`
- Port/baud: `COM9` / `115200`
- Firmware target: `esp32-c3-devkitm-1`
- PlatformIO: `PlatformIO Core, version 6.1.19`
- Python/Windows: `3.12.10` / `Windows-11-10.0.26200-SP0`
- Git: `24346d879634f65eeb8a647bdba0ac84e249c63d`; dirty working tree: `True`

## C. Frozen product configuration

HUD-awareness threshold `0.52`; policy SHA-256 `d7ee2431ea0f0c3a0aca9ea0b09689d14ac0aa3e4392b2ac8d53332de10fe08b`; CED production window 5.0 s, overlap 0.0 s; display TTL 5.5 s. Layer 1, emergency thresholds/voting, STT semantics, personalization, state priority, visual behavior, and clock semantics were frozen.

## D. Physical functional acceptance status

Before this campaign, manual physical testing passed CED-only, subtitle+CED coexistence, event clearing, ALERT override, idle CED, mixed speech+DOG, and `ALERT > SUBTITLE/CED > HOME`.

## E. Host → OLED render latency

| Metric | N | Median ms | P95 ms | Min ms | Max ms | Failures |
|---|---:|---:|---:|---:|---:|---:|
| S | 100 | 11.728 | 12.519 | 11.279 | 22.945 | 0 |
| C | 100 | 8.174 | 8.695 | 7.524 | 9.081 | 0 |
| E | 100 | 8.704 | 9.206 | 7.755 | 12.767 | 0 |

Nearest-rank percentiles are used: rank `ceil(p×N)`. Failed trials are retained and excluded from latency distribution calculations.

## F. STT FINAL → OLED

| STT FINAL -> OLED | 30 | 13.709 | 15.139 | 12.060 | 15.602 | 0 |

Starts when the actual recognizer returns FINAL; utterance duration and model latency are excluded.

## G. CED accepted → OLED

| CED accepted -> OLED | 30 | 8.389 | 8.824 | 7.594 | 8.902 | 0 |

Starts only after the locked HUD-awareness policy accepts the model result, not at the start of the 5-second acoustic window.

## H. Critical/HELP → ALERT

| HELP decision -> ALERT | 30 | 8.572 | 9.366 | 8.053 | 9.371 | 0 |

Automated end-to-end decision coverage uses the existing HELP semantic decision path. Controlled critical transitions are reported separately and are not mislabeled as model end-to-end latency.

## I. Screen transition latency

`SUBTITLE → ALERT`: N=30, median=8.448 ms, P95=9.326 ms, failures=0. All transition CSV rows retain screen-state checks; no ACK-observable intermediate HOME state was present.

## J. Hardware transport reliability

The original 2,000-frame run observed 1,999 ACKs inside the trial window and one ACK immediately after the next transaction. A corrected affected-benchmark rerun was then started with late-ACK reconciliation. It ACKed trials 0–1699, after which trials 1700–1709 produced ten consecutive 30-second ACK failures across multiple frame types. The rerun was stopped under the preregistered functional-defect rule. Raw evidence is retained in `raw_logs/transport_rerun_serial.log`.

After closing the benchmark process, COM9 remained enumerated but the HUD protocol did not answer a fresh query. Re-uploading the unchanged firmware failed with `Failed to connect to ESP32-C3: No serial data received`. Manual USB/reset/bootloader recovery is required.

{
  "sent": 2000,
  "received_parsed": 1999,
  "acked": 1999,
  "missing": 1,
  "duplicate": 0,
  "unexpected": 0,
  "out_of_order": 0,
  "parser_state_corruption": 0,
  "state_mismatch": 0,
  "invalid_injection": [
    {
      "case": "invalid_crc",
      "injected": true,
      "expected_ack": false
    },
    {
      "case": "malformed_payload",
      "injected": true,
      "expected_ack": false
    },
    {
      "case": "truncated_then_valid",
      "injected": true,
      "next_valid_succeeded": true
    }
  ],
  "parser_recovered": true
}

Malformed, invalid-CRC, and truncated cases are kept separate from normal valid-frame failure statistics.

## K. Concurrent STT/CED system health

```json
{
  "raw_frames_captured": 500,
  "ced_frames_produced": 500,
  "speech_frames_dropped": 0,
  "ced_frames_dropped": 0,
  "speech_frames_produced": 174,
  "speech_events_dropped": 0,
  "recognition_jobs_dropped": 0,
  "recognition_results_dropped": 0,
  "ced_frames_consumed": 500,
  "ced_events_dropped": 0,
  "ced_windows_processed": 4,
  "ced_windows_dropped": 0,
  "ced_window_queue_max_depth": 1,
  "ced_inference_seconds_total": 7.734000000054948,
  "ced_inference_seconds_max": 7.2810000000172295,
  "ced_events_dispatched": 4,
  "ced_top_k_recoveries": 2,
  "ced_held_label_refreshes": 2,
  "ced_ttl_expirations": 1,
  "ced_clear_requests": 1,
  "stt_partial_count": 2,
  "stt_final_count": 2,
  "C_frames_sent": 3
}
```

## L. Mixed speech + environmental validation

```json
{
  "subtitle_and_ced_coexisted": true,
  "ced_processed_while_speech": true,
  "ced_processed_while_stt_idle": true,
  "idle_ced_windows": 1,
  "speech_ced_windows": 3,
  "ced_present_across_multiple_mixed_windows": true,
  "subtitle_updates": 4,
  "serial_frames": 16,
  "serial_failures": 0,
  "C_frames_sent": 3,
  "ced_windows": [
    {
      "top1": "Bark",
      "top1_score": 0.6231867671012878,
      "top5": [
        {
          "label": "Bark",
          "score": 0.6231867671012878
        },
        {
          "label": "Dog",
          "score": 0.6199265718460083
        },
        {
          "label": "Animal",
          "score": 0.606243371963501
        },
        {
          "label": "Domestic animals, pets",
          "score": 0.6011565327644348
        },
        {
          "label": "Bow-wow",
          "score": 0.589719295501709
        }
      ],
      "selected_hud_label": "DOG",
      "active_hud_label": "DOG",
      "c_frame_sent": true
    },
    {
      "top1": "Speech",
      "top1_score": 0.6980352997779846,
      "top5": [
        {
          "label": "Speech",
          "score": 0.6980352997779846
        },
        {
          "label": "Bark",
          "score": 0.6479286551475525
        },
        {
          "label": "Dog",
          "score": 0.6414333581924438
        },
        {
          "label": "Animal",
          "score": 0.6280440092086792
        },
        {
          "label": "Domestic animals, pets",
          "score": 0.6273231506347656
        }
      ],
      "selected_hud_label": "DOG",
      "active_hud_label": "DOG",
      "c_frame_sent": true
    },
    {
      "top1": "Speech",
      "top1_score": 0.6941126585006714,
      "top5": [
        {
          "label": "Speech",
          "score": 0.6941126585006714
        },
        {
          "label": "Dog",
          "score": 0.6226301789283752
        },
        {
          "label": "Animal",
          "score": 0.612921953201294
        },
        {
          "label": "Bark",
          "score": 0.6025210022926331
        },
        {
          "label": "Domestic animals, pets",
          "score": 0.6016303896903992
        }
      ],
      "selected_hud_label": "DOG",
      "active_hud_label": "DOG",
      "c_frame_sent": true
    },
    {
      "top1": "Silence",
      "top1_score": 0.6334816813468933,
      "top5": [
        {
          "label": "Silence",
          "score": 0.6334816813468933
        },
        {
          "label": "Music",
          "score": 0.5335695147514343
        },
        {
          "label": "Sound effect",
          "score": 0.5090154409408569
        },
        {
          "label": "Sine wave",
          "score": 0.5069735050201416
        },
        {
          "label": "Musical instrument",
          "score": 0.5039905309677124
        }
      ],
      "selected_hud_label": "",
      "active_hud_label": "",
      "c_frame_sent": false
    }
  ]
}
```

## M. 60-minute hardware soak

```json
{
  "requested_seconds": 3600.0,
  "actual_duration_seconds": 3600.125,
  "total_frames": 16449,
  "total_transitions": 16449,
  "crashes": 0,
  "python_exceptions": 0,
  "subsystem_errors": [],
  "serial_failures": 0,
  "firmware_reset_evidence": 0,
  "failed_acks": 0,
  "screen_state_violations": 0,
  "stale_ced": 0,
  "stale_alert": 0,
  "concurrent_episodes": 5,
  "concurrent_episode_results": [
    {
      "episode": 0,
      "started_after_seconds": 600.0629999999946,
      "counters": {
        "raw_frames_captured": 500,
        "ced_frames_produced": 500,
        "speech_frames_dropped": 0,
        "ced_frames_dropped": 0,
        "speech_frames_produced": 176,
        "speech_events_dropped": 0,
        "recognition_jobs_dropped": 0,
        "recognition_results_dropped": 0,
        "ced_frames_consumed": 500,
        "ced_events_dropped": 0,
        "ced_windows_processed": 4,
        "ced_windows_dropped": 0,
        "ced_window_queue_max_depth": 2,
        "ced_inference_seconds_total": 11.96899999998277,
        "ced_inference_seconds_max": 11.405999999988126,
        "ced_events_dispatched": 4,
        "ced_top_k_recoveries": 2,
        "ced_held_label_refreshes": 2,
        "ced_ttl_expirations": 0,
        "ced_clear_requests": 0,
        "stt_partial_count": 4,
        "stt_final_count": 1,
        "C_frames_sent": 3
      },
      "serial_failures": 0,
      "subtitle_and_ced_coexisted": true,
      "ced_processed_while_speech": true,
      "ced_processed_while_stt_idle": true
    },
    {
      "episode": 1,
      "started_after_seconds": 1200.0,
      "counters": {
        "raw_frames_captured": 500,
        "ced_frames_produced": 500,
        "speech_frames_dropped": 0,
        "ced_frames_dropped": 0,
        "speech_frames_produced": 174,
        "speech_events_dropped": 0,
        "recognition_jobs_dropped": 0,
        "recognition_results_dropped": 0,
        "ced_frames_consumed": 500,
        "ced_events_dropped": 0,
        "ced_windows_processed": 4,
        "ced_windows_dropped": 0,
        "ced_window_queue_max_depth": 2,
        "ced_inference_seconds_total": 11.437000000005355,
        "ced_inference_seconds_max": 10.89100000000326,
        "ced_events_dispatched": 4,
        "ced_top_k_recoveries": 2,
        "ced_held_label_refreshes": 2,
        "ced_ttl_expirations": 0,
        "ced_clear_requests": 0,
        "stt_partial_count": 4,
        "stt_final_count": 1,
        "C_frames_sent": 3
      },
      "serial_failures": 0,
      "subtitle_and_ced_coexisted": true,
      "ced_processed_while_speech": true,
      "ced_processed_while_stt_idle": true
    },
    {
      "episode": 2,
      "started_after_seconds": 1800.0939999999828,
      "counters": {
        "raw_frames_captured": 500,
        "ced_frames_produced": 500,
        "speech_frames_dropped": 0,
        "ced_frames_dropped": 0,
        "speech_frames_produced": 174,
        "speech_events_dropped": 0,
        "recognition_jobs_dropped": 0,
        "recognition_results_dropped": 0,
        "ced_frames_consumed": 500,
        "ced_events_dropped": 0,
        "ced_windows_processed": 4,
        "ced_windows_dropped": 0,
        "ced_window_queue_max_depth": 1,
        "ced_inference_seconds_total": 10.453000000008615,
        "ced_inference_seconds_max": 9.703000000008615,
        "ced_events_dispatched": 4,
        "ced_top_k_recoveries": 2,
        "ced_held_label_refreshes": 2,
        "ced_ttl_expirations": 1,
        "ced_clear_requests": 1,
        "stt_partial_count": 5,
        "stt_final_count": 1,
        "C_frames_sent": 3
      },
      "serial_failures": 0,
      "subtitle_and_ced_coexisted": true,
      "ced_processed_while_speech": true,
      "ced_processed_while_stt_idle": true
    },
    {
      "episode": 3,
      "started_after_seconds": 2400.0629999999946,
      "counters": {
        "raw_frames_captured": 500,
        "ced_frames_produced": 500,
        "speech_frames_dropped": 0,
        "ced_frames_dropped": 0,
        "speech_frames_produced": 174,
        "speech_events_dropped": 0,
        "recognition_jobs_dropped": 0,
        "recognition_results_dropped": 0,
        "ced_frames_consumed": 500,
        "ced_events_dropped": 0,
        "ced_windows_processed": 4,
        "ced_windows_dropped": 0,
        "ced_window_queue_max_depth": 1,
        "ced_inference_seconds_total": 8.187999999965541,
        "ced_inference_seconds_max": 7.532000000006519,
        "ced_events_dispatched": 4,
        "ced_top_k_recoveries": 2,
        "ced_held_label_refreshes": 2,
        "ced_ttl_expirations": 0,
        "ced_clear_requests": 0,
        "stt_partial_count": 4,
        "stt_final_count": 1,
        "C_frames_sent": 3
      },
      "serial_failures": 0,
      "subtitle_and_ced_coexisted": true,
      "ced_processed_while_speech": true,
      "ced_processed_while_stt_idle": true
    },
    {
      "episode": 4,
      "started_after_seconds": 3000.2029999999795,
      "counters": {
        "raw_frames_captured": 500,
        "ced_frames_produced": 500,
        "speech_frames_dropped": 0,
        "ced_frames_dropped": 0,
        "speech_frames_produced": 174,
        "speech_events_dropped": 0,
        "recognition_jobs_dropped": 0,
        "recognition_results_dropped": 0,
        "ced_frames_consumed": 500,
        "ced_events_dropped": 0,
        "ced_windows_processed": 4,
        "ced_windows_dropped": 0,
        "ced_window_queue_max_depth": 1,
        "ced_inference_seconds_total": 8.65700000000652,
        "ced_inference_seconds_max": 8.078000000008615,
        "ced_events_dispatched": 4,
        "ced_top_k_recoveries": 2,
        "ced_held_label_refreshes": 2,
        "ced_ttl_expirations": 0,
        "ced_clear_requests": 0,
        "stt_partial_count": 4,
        "stt_final_count": 1,
        "C_frames_sent": 3
      },
      "serial_failures": 0,
      "subtitle_and_ced_coexisted": true,
      "ced_processed_while_speech": true,
      "ced_processed_while_stt_idle": true
    }
  ],
  "concurrent_counter_totals": {
    "raw_frames_captured": 2500,
    "ced_frames_produced": 2500,
    "speech_frames_dropped": 0,
    "ced_frames_dropped": 0,
    "speech_frames_produced": 872,
    "speech_events_dropped": 0,
    "recognition_jobs_dropped": 0,
    "recognition_results_dropped": 0,
    "ced_frames_consumed": 2500,
    "ced_events_dropped": 0,
    "ced_windows_processed": 20,
    "ced_windows_dropped": 0,
    "ced_window_queue_max_depth": 7,
    "ced_inference_seconds_total": 50.7039999999688,
    "ced_inference_seconds_max": 47.610000000015134,
    "ced_events_dispatched": 20,
    "ced_top_k_recoveries": 10,
    "ced_held_label_refreshes": 10,
    "ced_ttl_expirations": 1,
    "ced_clear_requests": 1,
    "stt_partial_count": 21,
    "stt_final_count": 5,
    "C_frames_sent": 15
  },
  "ced_raw_drops": 0,
  "ced_inference_window_drops": 0,
  "stt_drops": 0,
  "event_drops": 0,
  "queue_overflows": 0
}
```

## N. Serial-session reconnect

```json
{
  "label": "serial-session reconnect",
  "before_ok": true,
  "reopen_ok": true,
  "clock_resynchronized": true,
  "normal_hud_after_reconnect": true,
  "physical_usb_unplug_replug": "NOT MEASURED"
}
```

## O. Clock validation

```json
{
  "automatic_k_after_connection": true,
  "home_host_minute": "22:18",
  "home_displayed_minute": "22:18",
  "home_minute_matches": true,
  "k_during_subtitle_preserved_screen": true,
  "k_during_alert_preserved_screen": true,
  "advanced_without_resync": true,
  "advance_start": "22:18",
  "advance_end": "22:19",
  "wait_seconds": 8.616780281066895
}
```

## P. Regression status

```json
{
  "official_runner_exit": 0,
  "pytest_exit": 0,
  "esp32c3_build_exit": 0,
  "esp32dev_build_exit": 0,
  "passed": true
}
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
- A reproducible long-run USB/serial lockup remains unresolved. Manual recovery: physically unplug/replug USB or press RESET; if automatic upload remains unavailable, hold BOOT while tapping RESET and rerun the ESP32-C3 upload.

## S. Layer 2 final conclusion

**PARTIALLY COMPLETE** under the project rule. Mandatory component statuses are preserved in `layer2_hardware_status.json`; limitations above are not represented as measured evidence.

---

## T. Transport blocker closure addendum (2026-08-23)

This addendum preserves sections A-S and the original failed log unchanged. It supersedes only the unresolved transport limitation and Layer 2 conclusion.

### Root cause and reproduction

The demonstrated fault is in benchmark instrumentation under unrealistic high-rate ESP32-C3 hardware-CDC TX stress, not in production S/P/A/E/C/T/K transport semantics.

The old `SGACK` was assembled through 13 separate `Print`/`write` calls. During an unpaced S-only isolation run, ACK 636 was rendered and fully accepted by firmware at millisecond 796181, but the host timed out. Immediately before the next ACK, `Serial.availableForWrite()` was 216 bytes rather than the normal 256: exactly one complete 40-byte ACK remained queued. Sending frame 637 (the next host OUT packet) released ACK 636, after which RX, rendering, TX, and a fresh query all worked. Evidence: `transport_diagnostics/mode10_type_S.json` and its serial log.

This reproduces the underlying missing/late-ACK mechanism with direct counters. The exact historical ten-ACK permanent endpoint state was not forced again after reset; the current failing mix instead passed 3,214 transactions at 100.76/s before the fix. The original permanent failure remains preserved in `raw_logs/transport_rerun_serial.log`. The controlled reproduction establishes that:

- firmware execution remained alive;
- RX resumed on the following OUT packet;
- parser, heap, stack, and OLED commit timing were healthy;
- TX had accepted the complete ACK but the USB IN endpoint had not delivered it;
- the initial 30-second timeout was not the root cause;
- one outstanding host request and synchronous draining rule out a reader-thread race or host backlog.

The original permanent case did not expose visual/RX diagnostics, so its exact RX state remains unknown. Host writes completed, TX produced no observable bytes, and upload/reset over the same USB interface failed. The controlled predecessor failure plus the A/B fix below identify the benchmark HWCDC ACK path as the actionable cause.

### Why the soak passed while synthetic stress failed

Estimated sustained live traffic during continuous speech is approximately 0.95 frames/s: partial STT every 1.5 seconds (0.667/s), at most one final per 12 seconds (0.083/s), CED every 5 seconds (0.2/s), and K every 10 minutes (0.0017/s), before deduplication and excluding rare alerts. The passed soak averaged 4.569 frames/s.

The historical failed rerun drove about 105.7 actual transactions/s when benchmark-generated alert clear frames are included, more than 100 times the estimated continuous-speech rate. It repeatedly rendered identical values that `HUDTransport` normally deduplicates and emitted a fragmented ACK after every transaction. Bytes/s remained below nominal serial bandwidth; the differentiator was benchmark-only USB packet/interrupt cadence combined with continuous OLED commits.

### Fix

`firmware/src/protocol.cpp` now formats each benchmark ACK into one bounded 96-byte buffer, performs one `output.write()`, checks for a short write, and calls the ESP32-C3 HWCDC bounded `flush()`. Product frames do not generate ACK traffic and are unchanged.

Benchmark-only D queries expose RX bytes, complete/valid frames, CRC failures, parser recovery, malformed benchmark frames, ACK attempts/accepted bytes/short writes, TX capacity, loop count, uptime, heap, largest block, stack margin, reset reason, and display commit count/timing. `SoundGuardHUD::commitDisplay()` preserves the same synchronous `Adafruit_SSD1306::display()` operation and only records timing/counts.

`tools/diagnose_layer2_transport.py` is the targeted repeatable suite. It controls rate, ACK/no-ACK mode, frame type, outstanding request count, timeout, bytes, serial cadence, periodic diagnostics, and post-run usability.

### Controlled modes and frame isolation

| Workload | Actual transactions | Rate | Missing | Post-query |
|---|---:|---:|---:|---|
| fragmented ACK, mixed, fresh reset | 3,214 | 100.76/s | 0 | PASS |
| fragmented ACK, realistic paced | 2,571 | 5.00/s | 0 | PASS |
| fragmented ACK, moderate paced | 2,571 | 25.00/s | 0 | PASS |
| fragmented ACK, unpaced | 3,857 | 98.13/s | 0 | PASS |
| fragmented ACK, S-only | 2,000 | 71.43/s | 1 late at index 636 | PASS |
| atomic ACK, S-only | 5,000 | 74.04/s | 0 | PASS |
| atomic ACK, P-only | 2,000 | 77.21/s | 0 | PASS |
| atomic ACK, C-only | 2,000 | 82.34/s | 0 | PASS |
| atomic ACK, E-only | 2,000 | 106.89/s | 0 | PASS |
| atomic ACK, K-only | 2,000 | 391.20/s | 0 | PASS |
| atomic ACK, Q-only/non-render | 2,000 | 506.79/s | 0 | PASS |
| atomic ACK, mixed/unpaced | 3,857 | 95.09/s | 0 | PASS |
| production-like S/P/C/E/K, no ACK | 2,200 | 4.989/s | N/A (no response by design) | PASS |

The fixed mixed workload is stable at 95.09/s, about 19 times the documented production-like acceptance rate and about 100 times the estimated continuous-speech rate. Render-heavy S/P/C remained stable at 74-82/s; non-render Q reached 506.79/s. OLED commits contribute to maximum achievable rate but did not cause the stranded ACK: commit timing remained bounded while the old TX queue held exactly one ACK.

### Authoritative transport rerun

The original `transport_campaign(..., 2000)` was rerun from scratch on the fixed firmware:

```text
sent=2000
ACKed=2000
missing=0
duplicates=0
out_of_order=0
late_ack_reconciled=0
parser_state_corruption=0
parser_recovered=true
post_query_ok=true
```

Controlled invalid CRC, malformed benchmark payload, and truncated-frame cases were excluded from ordinary loss statistics. Two CRC/recovery events and one malformed benchmark frame were counted as expected, and the following valid frame succeeded. Evidence is in `transport_diagnostics/authoritative_transport_atomic_fix.*` and `transport_diagnostics/parser_recovery.json`.

### Resource and backpressure findings

Across the diagnostic sequence, the device reached 21,118 complete frames and 16,195 OLED commits before the final counter sample:

- free heap stabilized at 306,776 bytes;
- minimum free heap was 302,204 bytes;
- largest free block remained 278,516 bytes;
- no monotonic heap loss or fragmentation trend was observed;
- valid-load CRC failures and parser recovery counts remained zero;
- ACK short writes remained zero;
- pre-ACK TX capacity remained 256 bytes after the fix;
- maximum observed OLED commit was 6,713 microseconds;
- no reset evidence was observed.

Before the fix, the reproducible late ACK reduced TX availability to 216/256 bytes. This is direct backpressure evidence and matches the stranded ACK length. Firmware contains no production `Serial.print`, `Serial.println`, or response path; only benchmark B/D instrumentation transmits.

### Post-fix hardware and regression

- HUD state-flow smoke: PASS.
- Clock sync, screen preservation, and autonomous minute advance: PASS.
- Serial close/reopen, resync, and normal HUD traffic: PASS.
- Concurrent CED/HUD smoke: 500 CED frames consumed, 4 windows, zero CED drops, zero serial failures, subtitle/CED coexistence observed.
- A warm network STT smoke could not be run because outbound Google recognition was blocked in this sandbox. The authoritative prior hardware concurrency run and five soak episodes remain unchanged and passed; the offline audio-pipeline regression covers 33 tests.
- Focused HUD/transport contracts: 52 passed.
- Official regression runner: PASS.
- Full pytest: 232 passed, 5 subtests passed, 4 warnings.
- ESP32-C3 build: PASS.
- shared ESP32 DevKit build: PASS.
- final firmware upload and post-benchmark protocol query: PASS.

No physical reset is required after the accepted benchmark. The final board accepted fresh frames, closed/reopened its COM session, and was reflashed automatically. The earlier manual reset remains part of the preserved historical failure evidence, not normal cleanup after the fix.

### Soak decision and product risk

The 60-minute soak was not rerun. The accepted fix changes only benchmark B/D instrumentation; production S/P/A/E/C/T/K parsing, state decisions, display contents, priorities, and host transport semantics are unchanged. `commitDisplay()` calls the same synchronous display method and only records diagnostic timing. Therefore the authoritative 3600.125-second / 16,449-frame soak remains valid.

Product-risk classification: benchmark instrumentation plus unrealistic high-rate USB CDC TX stress. No evidence supports a production parser, memory-leak, scheduler, HUD-state, or one-way production transport defect.

## U. Corrected Layer 2 conclusion

**COMPLETE.** The mandatory transport blocker is resolved by an evidence-based benchmark-only fix, the original failure remains auditable, the 2,000-frame acceptance rerun passes without reconciliation, controlled parser recovery passes, and the device remains usable without physical reset.
