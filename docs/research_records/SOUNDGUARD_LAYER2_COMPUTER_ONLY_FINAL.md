# SoundGuard Layer 2 — Computer-Only Final

Ngày đo: 2026-08-21  
Evidence package: `benchmark_results/system_computer_only_20260821/`  
Phạm vi: từ audio đã sẵn sàng cho ứng dụng đến output sẵn sàng tại boundary `HUDTransport`; không đo microphone, đường serial vật lý, ESP32 hoặc OLED. Layer 1 không bị sửa, tune hoặc chạy lại để tối ưu kết quả.

## 1. Completed Today

| Test | Result | Evidence |
| --- | --- | --- |
| CED local software pipeline | N=30; median 124.72 ms; P95 149.87 ms | `latency/ced_local_pipeline_*` |
| STT local software pipeline | Partial N=10, P95 914.45 ms; final N=20, P95 1,373.30 ms | `latency/stt_local_pipeline_*` |
| Concurrent CED + STT | N=20; median impact +11.07% CED, +14.88% STT | `concurrency/` |
| CPU/RAM operating modes | Initialized, CED, STT và combined; process-tree sampling | `resources/` |
| 60-minute software soak | 3,601.02 s; 719/719 event; 0 crash/exception/drop | `soak/` |
| Software transport reliability | 2,000/2,000 frame; 0 missing/duplicate/malformed/order error | `transport_software/` |
| Fault handling | Write failure isolated; unavailable transport is no-op; duplicate suppression passed | `regression/fault_handling.json` |
| Targeted Layer 2 regression | 43 passed, 0 failed | `regression/pytest_layer2.xml` |

## 2. KHKT Final Table

| Metric | N / Duration | Result | KHKT Claim | Status |
| --- | ---: | ---: | --- | --- |
| CED local pipeline latency | 30 | Median 124.72 ms; P95 149.87 ms | CED software reaches transport-ready output within the 5 s production analysis interval | `DONE` |
| STT partial latency | 10 | Median 487.19 ms; P95 914.45 ms | Partial subtitle output is produced with measured network-dependent latency | `DONE` |
| STT final latency | 20 | Median 1,013.96 ms; P95 1,373.30 ms | Final subtitle output is produced with measured network-dependent latency | `DONE` |
| Concurrent impact | 20 pairs | CED median +11.07%; STT median +14.88% | Concurrent execution causes measurable but bounded latency overhead on this host | `DONE` |
| Software stability | 3,601.02 s | 719/719 processed; 0 crash/exception/drop | Local pipeline operated continuously without uncontrolled resource growth | `DONE` |
| CED active resources | 15.66 s | CPU avg/peak 64.74%/74.15%; RAM avg/peak 1,180.19/1,181.93 MB | Quantifies high-throughput CED cost on the current laptop | `DONE` |
| Combined resources | 17.98 s | CPU avg/peak 33.35%/62.90%; RAM avg/peak 1,172.49/1,183.44 MB | Quantifies concurrent CED+STT host cost | `DONE` |
| Serialization reliability | 2,000 events | 100%; 0 missing/duplicate/malformed/order error | Software event frames are ordered and CRC-valid at the write boundary | `DONE` |
| Serialization + mock-write latency | 2,000 | Median 0.0556 ms; P95 0.1233 ms | Transport preparation overhead is negligible relative to model latency | `SUPPORTING` |
| Fault-handling regression | 4 behaviors + 43 tests | Passed | Transport faults do not raise through the tested application boundary | `DONE` |

## 3. Latency

### CED local pipeline

Boundary: prerecorded audio file available to app → CED-Tiny → EfficientSED frozen specialist → decision/fusion → CRC frame ready.

| Stage | N | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preprocessing | 30 | 1.45 | 1.32 | 2.02 | 1.20 | 2.27 |
| CED-Tiny inference | 30 | 37.30 | 36.37 | 42.62 | 34.47 | 49.62 |
| EfficientSED round-trip | 30 | 87.48 | 86.50 | 98.28 | 78.73 | 113.74 |
| Fusion/decision | 30 | 0.153 | 0.146 | 0.178 | 0.120 | 0.288 |
| Serialization | 30 | 0.040 | 0.037 | 0.061 | 0.025 | 0.064 |
| **Total local CED pipeline** | **30** | **126.75** | **124.72** | **149.87** | **115.22** | **150.03** |

### STT local pipeline

Boundary: prerecorded utterance available to app → production DTLN preprocessing → Google Speech Recognition `vi-VN` → fusion/decision → subtitle frame ready. Vì backend là dịch vụ ngoài và model version không immutable, kết quả có tính network-dependent.

| Output | N | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Partial transcript | 10 | 517.25 | 487.19 | 914.45 | 400.41 | 914.45 |
| Final transcript | 20 | 1,044.45 | 1,013.96 | 1,373.30 | 716.90 | 1,871.92 |

Không có transcript rỗng trong 30 request đo latency.

### Concurrent mode

| Path | N | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | Median delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CED concurrent | 20 | 168.97 | 138.53 | 267.56 | 114.89 | 354.18 | +11.07% |
| STT concurrent | 20 | 1,140.13 | 1,164.88 | 1,550.40 | 775.41 | 1,874.24 | +14.88% |
| Pair wall time | 20 | 1,141.82 | 1,166.04 | 1,552.22 | 776.89 | 1,875.99 | — |

Kết luận: chạy đồng thời tạo overhead có thể đo được, nhưng CED P95 concurrent 267.56 ms vẫn thấp hơn nhiều so với cửa sổ CED 5 giây. Đây không phải mic→OLED latency.

## 4. Resource Usage

CPU là phần trăm tổng năng lực của tất cả logical CPU; RAM là Windows working set của process chính cộng toàn bộ process tree EfficientSED. Mỗi mode được sample trong workload liên tục khoảng 15–18 giây.

| Mode | Duration | Operations | CPU avg | CPU peak | RAM avg | RAM peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initialized idle | 15.04 s | 0 | 0.24% | 1.02% | 1,175.69 MB | 1,175.91 MB |
| CED active | 15.66 s | 108 CED | 64.74% | 74.15% | 1,180.19 MB | 1,181.93 MB |
| STT active | 16.18 s | 12 STT | 1.13% | 2.36% | 1,175.61 MB | 1,175.71 MB |
| CED + STT | 17.98 s | 66 CED + 10 STT | 33.35% | 62.90% | 1,172.49 MB | 1,183.44 MB |

STT CPU thấp vì phần lớn thời gian chờ remote service; không được diễn giải là chi phí compute của một STT model chạy local.

## 5. Software Stability

```text
Duration: 3601.02 s
Processed audio duration: 3595 s
Generated / processed events: 719 / 719
CED events: 719
STT segments: 719
Fusion events: 719
Alert events: 504
Crashes: 0
Exceptions / uncaught errors: 0 / 0
Dropped internal events: 0
Duplicate software events: 0
Serialized frames: 839
Malformed frames: 0
RAM start / peak / end: 698.29 / 699.71 / 120.20 MB
RAM end - start: -578.09 MB
RAM linear slope: -281.47 MB/hour
CPU mean / peak: 1.88% / 3.75% total capacity
Worker status: closed_cleanly
Conclusion: PASSED; no uncontrolled resource growth observed.
```

Soak chạy CED-Tiny, EfficientSED, production DTLN preprocessing, fusion/decision và HUD serialization trên playlist audio thu sẵn. Google STT được thay bằng transcript deterministic đúng tại external-service boundary để tránh biến service/network thành yếu tố ổn định trong một phép đo local. Vì vậy đây là **production-like local software soak**, không phải cloud-provider soak và không phải full hardware soak.

Windows đã trim working set theo bậc trong khoảng đầu và sau đó ổn định; vì vậy slope âm không được diễn giải là “model chỉ cần 120 MB”. Resource theo mode ở mục 4 đại diện tốt hơn cho footprint khi model đang resident và chạy tích cực.

## 6. Software Transport Reliability

2,000 event gồm 1,000 subtitle và 1,000 alert được gửi qua `HUDTransport` với stream in-memory:

```text
Generated: 2000
Serialized: 2000
Missing: 0
Duplicates: 0
Malformed / CRC errors: 0
Ordering errors: 0
Success rate: 100%
Median serialization + mock-write: 0.0556 ms
P95 serialization + mock-write: 0.1233 ms
```

Đây là **software transport serialization reliability**, không phải serial reliability hoặc laptop→ESP32 delivery reliability.

Fault regression xác nhận: transport disabled là no-op; write exception bị cô lập và disable transport; core call tiếp theo không raise; duplicate subtitle bị suppress đúng thiết kế. Không claim disconnect/reconnect vật lý.

## 7. KHKT Main Results

### Result 1 — CED local latency

```text
Metric: Audio-to-transport-ready CED latency
Result: N=30; median 124.72 ms; P95 149.87 ms
KHKT_CLAIM: SoundGuard's frozen CED software path operates within its 5-second analysis interval.
REPORT_USE: Main system-performance table and Figure 1.
```

### Result 2 — STT final latency

```text
Metric: Utterance-to-transport-ready final subtitle latency
Result: N=20; median 1,013.96 ms; P95 1,373.30 ms
KHKT_CLAIM: Final Vietnamese subtitle output is produced with measured network-dependent latency.
REPORT_USE: Main system-performance table and Figure 1; state Google backend caveat.
```

### Result 3 — Concurrent overhead

```text
Metric: CED + STT concurrent median latency impact
Result: CED +11.07%; STT +14.88% (N=20 pairs)
KHKT_CLAIM: Concurrent execution introduces measurable but bounded contention on this host.
REPORT_USE: Concurrency paragraph and Figure 1.
```

### Result 4 — 60-minute stability

```text
Metric: Production-like local software soak
Result: 3,601.02 s; 719/719 events; 0 crash, exception, or drop
KHKT_CLAIM: The local pipeline operated continuously without failures under a deterministic prerecorded workload.
REPORT_USE: Main stability table and Figure 3.
```

### Result 5 — Active resource cost

```text
Metric: CED active process-tree CPU/RAM
Result: CPU 64.74% average / 74.15% peak; RAM 1,180.19 MB average / 1,181.93 MB peak
KHKT_CLAIM: Quantifies computational cost of the frozen CED configuration on the current laptop.
REPORT_USE: Resource table and Figure 2.
```

### Result 6 — Software transport reliability

```text
Metric: CRC frame serialization reliability
Result: 2,000/2,000 correct; 0 missing, duplicate, malformed, or out-of-order
KHKT_CLAIM: The software transport boundary deterministically prepares valid ordered frames.
REPORT_USE: Supporting reliability table; never label as physical serial reliability.
```

### Result 7 — Fault handling

```text
Metric: Software transport fault-handling regression
Result: All injected behaviors passed; targeted regression 43/43 tests passed
KHKT_CLAIM: Tested software transport faults are isolated without crashing core processing.
REPORT_USE: Reliability/supporting-methods paragraph.
```

## 8. Figures

1. `benchmark_results/system_computer_only_20260821/final/figure_1_latency_comparison.png`
2. `benchmark_results/system_computer_only_20260821/final/figure_2_resources_by_mode.png`
3. `benchmark_results/system_computer_only_20260821/final/figure_3_soak_ram_trend.png`

## 9. HARDWARE_REQUIRED

Các mục sau vẫn `PENDING` và không được thay thế bằng mock result:

1. True microphone/event → ESP32 → OLED end-to-end P50/P95 latency.
2. Physical laptop → ESP32 sent/received/ACK delivery rate.
3. Dropped, duplicate, malformed và CRC-rejected packet trên đường vật lý.
4. Real disconnect/reconnect recovery time và hành vi khi COM biến mất/trở lại.
5. OLED render/update confirmation và refresh success rate.
6. Full 60-minute hardware soak gồm microphone, app, serial, ESP32 và OLED.
7. Microphone capture stability/overflow trên thiết bị thật.
8. Battery life, power draw và thermal nếu dùng trong claim wearable feasibility.

## 10. Updated Layer 2 Status

```text
SYSTEM LAYER TOTAL: 60% (project tracking estimate, not a statistical score)

COMPUTER-ONLY PORTION:
COMPLETE for the requested scope, with a declared deterministic STT backend in the soak.

HARDWARE PORTION:
PENDING
```

Mức 60% phản ánh toàn bộ computer-only P0/P1 trong phiên này đã có evidence, nhưng các phép đo hardware endpoint/reliability/power vẫn là phần lớn evidence triển khai còn thiếu.

## 11. Technical Problems Found

1. Google STT là backend network-dependent và không có immutable model version; latency có thể thay đổi theo thời điểm.
2. TFLite interpreter của DTLN không nên được chuyển sang worker thread sau khi warm ở main thread. Harness đã được sửa để profiling STT giữ nguyên main thread; core app không bị thay đổi.
3. Windows virtualenv launcher và Python worker có PID riêng. Resource sampler đã được sửa để cộng toàn bộ descendant process; nếu chỉ đo PID launcher sẽ thiếu khoảng model memory lớn.
4. Soak working set bị Windows trim mạnh theo bậc. Slope âm chỉ chứng minh không có tăng mất kiểm soát trong lần chạy này, không phải model footprint tối thiểu.
5. Soak không kiểm tra độ ổn định của Google service; điều này được tách rõ khỏi local software stability.

## 12. Final Verdict

```text
SOUNDGUARD LAYER 2 — COMPUTER-ONLY STATUS

Layer 1 modified:
NO

CED software latency:
DONE

STT software latency:
DONE

Concurrent pipeline test:
DONE

CPU/RAM profiling:
DONE

Software soak test:
DONE (local production-like; deterministic STT service boundary)

Software transport reliability:
DONE (software serialization only)

Fault-handling regression:
DONE

COMPUTER-ONLY SYSTEM BENCHMARK READY TO LOCK:
YES, within the explicitly stated laptop boundary and limitations

REQUIRES PHYSICAL GLASSES NEXT:
1. Mic/event → OLED latency
2. Physical delivery/ACK/drop/duplicate rate
3. Real disconnect/reconnect recovery
4. OLED update confirmation and hardware soak
5. Battery/power/thermal measurements

NEXT ACTION AFTER GLASSES ARE POWERED:
Freeze the hardware test protocol and timestamps, then run mic→OLED latency,
physical transport reliability/reconnect, and a 60-minute end-to-end hardware soak
without changing the locked Layer 1 model configuration.
```

## 13. Evidence Integrity

- Commands: `config/command_quick.txt`, `config/command_soak.txt`
- Environment/git state: `config/environment.json`
- Benchmark execution hash: `config/script_version.json`
- Post-run visualization-only hash note: `config/post_run_script_state.json`
- Raw samples: `latency/*.csv`, `concurrency/*.csv`, `resources/*.csv`, `soak/resource_usage.csv`, `transport_software/raw_latency.csv`
- Summaries: corresponding `summary.json` files
- Logs: `quick.stdout.log`, `quick.stderr.log`, `soak.stdout.log`, `soak.stderr.log`
- Regression: `regression/pytest_layer2.xml`

Layer 1 outputs under `benchmark_results/external_results_20260820/` were not overwritten.
