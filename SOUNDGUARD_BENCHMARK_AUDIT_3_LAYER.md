# SoundGuard Benchmark Audit — Framework 3 tầng

Ngày audit: 2026-08-20  
Phạm vi: toàn bộ repository tại commit `24346d879634f65eeb8a647bdba0ac84e249c63d`, branch `experiment/emergency-ced-v3`, với working tree đang dirty. Audit này chỉ đọc và kiểm tra; không train, không đổi threshold, không tải dataset lớn, không sửa core app và không ghi đè output cũ.

Quy ước quan trọng:

- `legacy/default`: CED-Tiny và hành vi cũ khi các feature flag tắt.
- `improved V2`: semantic CED mapping và compound-safe personalization, chỉ bật bằng feature flag.
- `Emergency V3`: EfficientSED specialist trên laptop, opt-in, mặc định tắt.
- Số liệu trên ESC-50 là development/internal benchmark có rủi ro overlap với AudioSet; không được gọi là untouched external holdout.
- `software correctness` không đồng nghĩa với real-world accuracy hoặc Human Metric.

## 1. Executive Summary

SoundGuard đã có một Model Layer đáng kể: CED trên 280 clip ESC-50, STT trên 100 câu VIVOS, sweep nhiễu Gaussian/DEMAND, ablation DTLN, emergency evaluation và raw per-sample outputs. Đây là tầng mạnh nhất. Tuy nhiên chưa có external holdout cuối cùng: FSD50K hiện chỉ có metadata/ground truth, DCASE chỉ có license/credits; cả hai đều không có audio và chưa chạy.

System Layer mới có latency phần mềm cục bộ, latency cloud STT, resource của EfficientSED và một soak 1 giờ chỉ chạy synthetic control path. Chưa có mic → OLED latency, packet delivery/ACK, reconnect, OLED success rate, full-pipeline soak, battery, power hoặc thermal. Không được dùng P95 inference để đại diện cho end-to-end latency.

Human Layer không có dataset, protocol, participant log hay kết quả người dùng. Tất cả Human Metrics là `MISSING`.

Blocker lớn nhất để khóa baseline khoa học là thiếu một evaluation set chưa bị dùng để chọn mapping/threshold/candidate. Blocker lớn nhất của prototype là thiếu measurement trên đường thật microphone → laptop → ESP32 → OLED.

Kết quả test hiện tại trong phiên audit: `149 passed, 1 failed, 5 subtests passed`. Test fail là golden legacy do CED loader cố lấy `processor_config.json` từ Hugging Face trong môi trường không có network; cache có weights nhưng không tự chứa đủ processor artifact. Đây là lỗi reproducibility/dependency packaging, chưa phải bằng chứng về regression logic.

Phân loại inventory A–E:

| Suite/artifact | A–E | Kết luận |
|---|---|---|
| `benchmark_v1` fixed-file suite | C + D | Có raw CSV/report; chỉ 5 WAV, 6 rows, không có emergency positive; claim rất yếu. |
| Preliminary `run_full_benchmark.py` | C + D | Có output thật nhưng CED chỉ 2 clip; soak ban đầu chỉ 60 giây. |
| Expanded benchmark | C, gần E trong phạm vi internal | Có 280 ESC-50, 100 VIVOS, raw predictions, manifests, seed và plots; vẫn có source-overlap risk, cloud mutability và môi trường không khóa phiên bản. |
| Root-cause analysis | C | Có output chi tiết và kiểm chứng lại SNR/alignment; phù hợp diagnostic. |
| Improved V2 | C + D | Có output thật, nhưng cùng 280 development clips đã được xem để chọn candidate; không phải final holdout. |
| Emergency V3 | C + D | Raw frame predictions, checkpoint/hash/threshold/split đầy đủ; threshold chọn trên ESC folds 1–4, fold 5 chỉ internal check, negatives chỉ dog/door. |
| HUD serial/firmware | A | Code, unit tests và manual `--hud-test` tồn tại; không có physical measurement output. |
| FSD50K/DCASE final evaluation | MISSING, chưa đạt A | Có metadata/credits một phần nhưng chưa có evaluator hoàn chỉnh, audio hoặc output. |
| Human study | MISSING | Không tìm thấy script, protocol, consent, participant data hoặc result. |

## 2. Three-Layer Status

### MODEL LAYER

Completion: **62%**  
Status: `PARTIAL`, gần `MOSTLY_COMPLETE` cho CED/STT/noise trong phạm vi development benchmark.  
Evidence: expanded raw predictions/metrics; improved V2; Emergency V3 raw frame predictions; manifests ESC-50/VIVOS/DEMAND; deterministic personalization tests.  
Main missing items: untouched external CED/emergency holdout; real paired fusion corpus; broader hard negatives; confidence intervals cho external safety metrics; remote personalization labeled evaluation nếu muốn claim AI accuracy.

62% phản ánh coverage khá rộng và raw evidence tốt, nhưng bị trừ mạnh vì ESC-50 đã thành development set, AudioSet overlap không loại trừ được, V3 threshold đã calibration trên cùng corpus, fusion N=6 chỉ là logic test, và FSD50K/DCASE chưa chạy.

### SYSTEM LAYER

Completion: **24%**  
Status: `PARTIAL`.  
Evidence: CED/STT/local pipeline latency; EfficientSED CPU/RAM/runtime; synthetic control-path soak 1 giờ; transport encoding/CRC unit tests.  
Main missing items: mic → OLED latency; capture/preprocess/fusion/transport/parser/OLED timestamps; actual full-pipeline soak; sent/received/drop/duplicate/malformed/reconnect metrics; full-mode CPU/RAM; battery/power/temperature.

24% vì các số hiện có chủ yếu dừng ở software timing. Soak 1 giờ không chạy mic, model, serial hay HUD; transport tests không sử dụng ESP32/OLED thật.

### HUMAN LAYER

Completion: **0%**  
Status: `MISSING`.  
Evidence: không có.  
Main missing items: readability, subtitle/alert comprehension, reaction time, missed-alert rate, usability, comfort, cognitive load, acceptance và test với người khiếm thính.

## 3. 3-Layer × Feature Matrix

| Feature | Model metrics | System metrics | Human metrics |
|---|---|---|---|
| CED | `MOSTLY_COMPLETE` — accuracy/F1/per-class/confusion/SNR/latency trên ESC-50; chưa có untouched holdout | `PARTIAL` — inference latency có; capture/queue/live latency và device path thiếu | `MISSING` — không có event-recognition study |
| STT | `MOSTLY_COMPLETE` — WER/CER clean và noisy; cloud model mutable, noisy effective N=30 | `PARTIAL` — final-call latency có; partial/final transcript latency và mic-to-HUD thiếu | `MISSING` — không có comprehension/readability study |
| Emergency/Fusion | `PARTIAL` — emergency development evidence mạnh; fusion chỉ N=6 hand-authored cases | `PARTIAL` — logic latency và lifecycle tests có; alert start/end trên hardware và false alarms theo giờ thiếu | `MISSING` — không có response/reaction/missed-alert study |
| Noise processing | `MOSTLY_COMPLETE` — paired before/after cho CED/STT/SNR; kết quả cho thấy DTLN gây hại downstream trong protocol hiện tại | `PARTIAL` — chưa có latency overhead/resource theo full route và live mic | `MISSING` — không có intelligibility/preference test |
| Personalization | `PARTIAL` — software correctness/invariants có; không phải personalization AI accuracy | `PARTIAL` — deterministic latency cũ có; remote provider skipped, runtime stability chưa đo | `MISSING` — không có usefulness/preference study |
| HUD/transport | `NOT_APPLICABLE` | `PARTIAL` — UTF-8/CRC/dedup/control-line unit tests và firmware compile artifact có; không có physical reliability/latency | `MISSING` — readability/comfort/comprehension chưa đo |

## 4. Existing Metrics

Chỉ liệt kê số có evidence. `N` là số independent source samples khi có thể; các repeated SNR observations được ghi rõ.

| Layer | Module | Metric | Dataset/Test | N | Result | Evidence file | Confidence |
|---|---|---|---|---:|---:|---|---|
| Model | CED legacy | Clean accuracy / macro P / macro R / Macro-F1 | ESC-50, 7 balanced classes | 280 | .4500 / .8571 / .4500 / .5525 | `benchmark_results/expanded/ced/clean_metrics.csv` | MEDIUM |
| Model | CED legacy | Clean Macro-F1 (95% bootstrap CI) | ESC-50 | 280 | 0.5525 (0.5072–0.5941) | `benchmark_results/expanded/ced/confidence_intervals.json` | MEDIUM |
| Model | CED legacy | Per-class recall | ESC-50, 40/class | 280 | baby .65; dog .05; door .90; fire .50; glass .00; siren .60; horn .45 | `benchmark_results/expanded/ced/per_class_metrics.csv` | MEDIUM |
| Model | CED legacy | Real-noise Macro-F1, 20/15/10/5/0 dB | ESC-50 + 4 DEMAND environments | 280 sources | .4834/.4278/.4193/.3696/.3487 | `benchmark_results/expanded/ced/real_noise_metrics_by_snr.csv` | MEDIUM |
| Model | CED legacy | Gaussian-noise Macro-F1, 20/15/10/5/0 dB | ESC-50 + seeded Gaussian | 280 sources | .5419/.5160/.4864/.4757/.4405 | `benchmark_results/expanded/summary.csv` | MEDIUM |
| Model | CED improved V2 | Accuracy / Macro-F1 | Same ESC-50 development set | 280 | .5679 / .6773 | `benchmark_results/improved/ced/metrics.json` | MEDIUM |
| Model | CED improved V2 | Per-class recall | Same ESC-50 development set | 40/class | baby .65; dog .275; door .90; fire .55; glass .125; siren .975; horn .50 | `benchmark_results/improved/ced/per_class_metrics.csv` | MEDIUM |
| Model | STT | Clean WER / CER | VIVOS official test selection, 19 speakers | 100 | .10745 / .05943 | `benchmark_results/expanded/stt/clean_summary.json` | MEDIUM |
| Model | STT | Real-noise WER, 20/15/10/5/0 dB | 30 fixed VIVOS utterances + DEMAND | 30 sources | .1021/.1092/.1056/.1197/.1338 | `benchmark_results/expanded/stt/noise_summary.csv` | MEDIUM |
| Model | Noise/DTLN | Mean intrusive ΔSNR on non-speech events | ESC-50 + DEMAND paired mixtures | 700 pairs | -7.9700 dB | `benchmark_results/expanded/noise_filter/signal_summary.json` | HIGH |
| Model | Noise/DTLN | ΔCED Macro-F1 across SNR | 140 pairs/SNR, 700 total | 700 pairs | -0.2977 weighted; every SNR negative | `benchmark_results/expanded/noise_filter/ced_before_after.csv` | HIGH |
| Model | Noise/DTLN | ΔSTT WER across real-noise SNR | 30 sources × 5 SNR | 150 pairs | +0.01338 (worse) | `benchmark_results/expanded/noise_filter/stt_before_after.csv` | MEDIUM |
| Model | Emergency V2 legacy | Precision / recall / F1 / FNR | ESC-50: 200 positive, 80 negative | 280 | 1.000 / .510 / .6755 / .490 | `benchmark_results/expanded/emergency/metrics.json` | MEDIUM |
| Model | Emergency improved V2 | Precision / recall / F1 / FNR | Same ESC-50 development set | 280 | 1.000 / .575 / .7302 / .425 | `benchmark_results/improved/emergency/metrics.json` | MEDIUM |
| Model | EfficientSED-only | Recall / FPR, folds 1–4 | ESC-50 development selection split | 224 | .96875 / 0 | `benchmark_results/emergency_v3/efficientsed_only/metrics.json` | MEDIUM |
| Model | EfficientSED-only | Recall / FPR, fold 5 internal check | ESC-50 fold 5 | 56 | .950 / 0 | `benchmark_results/emergency_v3/efficientsed_only/metrics.json` | MEDIUM |
| Model | Emergency V3 combined | Recall / FNR / FPR, folds 1–4 | V2 OR EfficientSED | 224 | .975 / .025 / 0 | `benchmark_results/emergency_v3/development_gate.json` | MEDIUM |
| Model | Emergency V3 combined | Recall / FNR / FPR, fold 5 internal check | V2 OR EfficientSED | 56 | .975 / .025 / 0 | `benchmark_results/emergency_v3/fusion_policy_comparison.csv` | MEDIUM |
| Model | EfficientSED-only | Per-class recall, all ESC-50 | 40/class | 200 positive | glass 1.0; horn 1.0; fire .85; siren 1.0; baby .975 | `benchmark_results/emergency_v3/efficientsed_only/per_class_metrics.csv` | MEDIUM |
| Model | Emergency V3 negatives | Binary false positives | dog + door only | 80 | 0 | `benchmark_results/emergency_v3/fusion_policy_comparison.csv` | LOW |
| Model | Fusion | Raw F1 → fused F1 | Hand-authored deterministic cases | 6 | .6667 → 1.000 | `benchmark_results/expanded/fusion/summary.json` | LOW |
| Model | Personalization | Exact priority accuracy | Hand-authored policy assertions | 128 | 1.000 | `benchmark_results/expanded/personalization/metrics.json` | HIGH |
| Model | Personalization | Safety / substring pass rate | Hand-authored invariants | 50 / 5 | .96 / .60 | `benchmark_results/expanded/personalization/metrics.json` | HIGH |
| Model | Personalization improved | Fixed safety / adversarial | Deterministic cases | 50 / 12 | 1.000 / 12 of 12 | `benchmark_results/improved/personalization/metrics.json` | HIGH |
| System | CED legacy | Warm inference P95 | ESC-50 fixed files | 280 | 42.35 ms | `benchmark_results/expanded/ced/latency_summary.json` | HIGH |
| System | STT cloud | Request latency P95 / mean RTF | VIVOS, Google STT | 100 | 1804.09 ms / .310 | `benchmark_results/expanded/stt/latency_summary.json` | MEDIUM |
| System | Local PC pipeline | Warm fixed-file P95 | CED → emergency → fusion; excludes capture/STT/HUD | 100 | 24.87 ms | `benchmark_results/expanded/pipeline/latency_summary.json` | HIGH |
| System | Emergency rules | Logic P95 | Fixed software cases | 8 | .0563 ms | `benchmark_results/emergency/latency_summary.json` | LOW |
| System | Emergency V3 | Sequential local P95 | V2 + EfficientSED; excludes capture/STT/HUD/startup | 280 | 106.82 ms | `benchmark_results/emergency_v3/latency/combined.json` | HIGH |
| System | Emergency V3 | Warm subprocess round-trip P95 | Decode + IPC | 30 | 91.07 ms | `benchmark_results/emergency_v3/latency/runtime_resources.json` | MEDIUM |
| System | Emergency V3 | Worker RAM / CPU | Warm worker, short run | 30 | 385.7 MB working set; 72.6% total 8-thread capacity | `benchmark_results/emergency_v3/latency/runtime_resources.json` | MEDIUM |
| System | Emergency V3 | Model footprint / initialization | EfficientSED CPU worker | 1 load | 15.54 MB; 3,830,798 parameters; 4.223 s load | `benchmark_results/emergency_v3/model_metadata.json` | HIGH |
| System | Personalization | Deterministic P95 latency | Fixed software cases | 8 | 3.418 ms | `benchmark_results/personalization/latency_summary.json` | LOW |
| System | Personalization remote | API benchmark status | OpenAI/OpenRouter | 0 | `SKIPPED` — no credential | `benchmark_results/expanded/personalization/remote_SKIPPED.json` | HIGH |
| System | Stability | 1-hour crash count | Synthetic control path only | 3600 s | 0 crashes; 48,704,800 iterations | `benchmark_results/expanded/stability/summary.json` | HIGH |
| System | Stability | Mean CPU / traced peak allocation | Same synthetic soak | 3586 samples | 12.06% total CPU; 1.998 MB traced peak; RSS unavailable | `benchmark_results/expanded/stability/resource_usage.csv` | MEDIUM |
| System | HUD transport | Software protocol tests | In-memory serial framing | 6 tests | 6 pass within full run | `test_display_transport.py` | HIGH |
| System | Whole software suite | Current audit test result | pytest, network restricted | 150 tests | 149 pass, 1 dependency/cache failure; 5 subtests pass | N/A — audit console only, not persisted | LOW |
| System | Legacy benchmark_v1 | Total latency / CED steady-state / STT steady-state | 5 WAV, 6 rows | 6 | 3.0678 s / .0232 s / .8359 s mean | `benchmark_v1/outputs/benchmark_report.md` | LOW |

Không có dòng Human Metric vì không có evidence.

## 5. Dataset & Leakage Audit

### Dataset status và sample size

| Dataset/test | Hiện trạng | Total N | Phân bố | Validity / sample-size warning |
|---|---|---:|---|---|
| ESC-50 mapped | Audio + official metadata + 280-row manifest có hash | 280 | 7 class × 40; emergency positive 200, negatives 80 | Balanced, nhưng cả CED-Tiny và EfficientSED đều AudioSet-derived; source overlap không loại trừ được. Toàn bộ set đã được xem và dùng development. |
| VIVOS | Audio + 100-row deterministic test manifest có transcript/hash | 100 | 19 speakers | Hợp lý cho baseline cloud STT nhỏ; Google training membership và model version không biết. |
| VIVOS noisy subset | Generated mixtures/output có | 30 source utterances | 5 SNR × 2 noise protocols × 2 filter routes | Effective independent N vẫn là 30, không phải 600. Cần CI theo speaker/utterance. |
| DEMAND | 4 selected recordings + manifest/hash | 4 | traffic, cafeteria, office, home | Chỉ 1 recording/environment; mixtures từ cùng source noise có tương quan cao. |
| Fusion cases | Hand-authored JSON | 6 | 3 positive, 3 negative sau fusion | Quá nhỏ cho accuracy claim; chỉ software logic validation. Real paired CED+STT corpus hiện N=0. |
| Personalization | Hand-authored cases/invariants | 20 profiles; 128 assertions; 50 safety tests | Không phải user-labeled dataset | Chỉ chứng minh deterministic software correctness. |
| Stability | Synthetic loop | 3600 s | 48.7M dependent iterations | Duration hợp lệ; event count không đại diện cho real audio events. |
| benchmark_v1 | Raw CSV còn | 5 WAV, 6 rows | 3 clean speech, 1 noisy speech, 1 environmental | Quá nhỏ; không có positive emergency. |
| FSD50K eval | Chỉ doc/ground-truth/metadata ZIP | 10,231 metadata rows; 0 audio | Target-label metadata có: siren 55, horn 68, fire 124, crackle 107, glass 267, shatter 96, crying/sobbing 42 | Chưa có class mapping frozen, dedup, audio hay prediction. Metadata count không phải evaluated N. |
| DCASE 2017 source events | Chỉ license + credits | 83 credit lines; 0 audio/YAML | Credits có baby/glass sources nhưng chưa có evaluation manifest | Không thể xác nhận sample count/class truth từ credits; chưa benchmark. |

### Split validity

- ESC-50 official fold metadata được giữ. Emergency V3 dùng folds 1–4 để chọn threshold `0.05`, rồi đọc fold 5. Fold 5 là internal check tốt hơn all-data score nhưng không phải untouched external holdout, vì ontology/candidate và corpus ESC-50 đã được sử dụng trong quá trình phát triển.
- VIVOS dùng official test split và deterministic round-robin selection. Không có local training.
- Generated noise variants giữ chung source identity; không thấy exact duplicate trong 280 ESC-50 hoặc 100 VIVOS manifests. Tuy nhiên repeated mixtures không được coi là independent samples.
- Không tìm thấy local CED training/fine-tuning data, train list hoặc checkpoint do dự án train. Vì vậy không có local train-test leakage được chứng minh; upstream AudioSet membership vẫn không tái dựng được.

### Hard negatives và imbalance

- Emergency V3 có 80 negative clip, chỉ từ `dog` và `door_wood_knock`. Đây là negative set cân bằng theo hai family, nhưng chưa đủ để gọi là broad hard-negative benchmark.
- Category-level EfficientSED tạo 6 horn và 2 fire cross-category activations trên các safety-positive clips. Chúng không thành binary false alarms nhưng cho thấy subtype precision không hoàn hảo.
- Chưa có negatives như speech/shout, music, alarms không nguy hiểm, traffic, dishes, construction, TV, trẻ em không khóc, hoặc device-specific noise. False alarm `0/80` vì vậy có confidence thấp cho field claim.

### External evaluation verdict

FSD50K untouched evaluation và DCASE 2017 baby/glass evaluation **chưa chạy**. Hiện chỉ có metadata/credits. Không có output path, predictions, confusion matrix hoặc metric cho hai nguồn này. Đây là P0.

## 6. Reproducibility Audit

| Benchmark | Dataset | Split | Model/config | Command/script | Raw output | Metric output | Verdict |
|---|---|---|---|---|---|---|---|
| Expanded V2 | Có manifests/hash | Có | Seed 42, model ID, SNR/config có | Có | Có | Có | `PARTIALLY_REPRODUCIBLE` |
| Improved V2 | Reuses frozen expanded data | Development set, không holdout | Feature flags và baseline snapshot có | Có | Có | Có | `PARTIALLY_REPRODUCIBLE`; valid cho candidate comparison, không final claim |
| Emergency V3 | ESC manifest/audio có | folds 1–4 + fold 5 | Checkpoint SHA, threshold, taxonomy, source hashes có | Có | Raw frame JSONL + predictions có | Có | `PARTIALLY_REPRODUCIBLE`; chain mạnh nhưng external validity thiếu |
| STT expanded | VIVOS manifest/audio có | Test selection có | Backend/language có | Có | Transcript rows có | Có | `PARTIALLY_REPRODUCIBLE`; Google backend/version/network mutable |
| DTLN paired | Source/generated paths + model hashes có | Fixed sources/SNR | Seed, alignment, two TFLite hashes có | Có | Paired CSV có | Có | `REPRODUCIBLE` trong workspace hiện tại |
| Personalization deterministic | JSON cases có | Fixed | Rules/schema có | Có | Case/assertion CSV có | Có | `REPRODUCIBLE` cho software correctness |
| benchmark_v1 | Manifest/audio paths có | Hand-selected | Run IDs có | Có | `results.csv`, `events.csv` có | Report có | `PARTIALLY_REPRODUCIBLE`; N rất nhỏ và historical runs mixed |
| Physical HUD/system | Không có captured dataset | Không | Firmware/config có | Manual command có | Không | Không | `NON_REPRODUCIBLE` vì chưa đo |
| Human study | Không | Không | Không | Không | Không | Không | `NON_REPRODUCIBLE` / `MISSING` |

Các điểm làm benchmark chưa đạt `COMPLETE`:

- `requirements.txt` và `requirements-dev.txt` không pin version; environment freeze hiện tại không nằm trong manifest chính.
- CED model được ghi bằng model ID, không bundle đầy đủ processor artifact. Golden test hiện fail offline dù weights cache tồn tại.
- Working tree dirty; frozen V3 hashes khớp 9/9 file được liệt kê, nhưng nhiều file benchmark/data/result đang untracked.
- Expanded outputs có commit/seed/config nhưng model revision/checkpoint hash CED không được ghi đầy đủ trong methodology.
- Google STT là dịch vụ ngoài, không có immutable model version và yêu cầu network.
- FSD50K/DCASE không có audio/manifest/evaluator output.

## 7. Missing Metrics

| Layer | Priority | Missing metric/test | Why needed | Automated? | Hardware required? | Human required? |
|---|---|---|---|---|---|---|
| Model | P0 | FSD50K untouched CED evaluation | Khóa Macro-F1/per-class claim ngoài ESC development | Yes | No | Human audit mapping only |
| Model | P0 | DCASE baby cry/glass break source-event evaluation | Hai class transient quan trọng và yếu ở V2 | Yes | No | Human label audit only |
| Model | P0 | Emergency V3 external precision/recall/FNR/FPR | Threshold 0.05 hiện chỉ development-calibrated | Yes | No | No |
| Model | P0 | Broad hard-negative false-alarm benchmark | `0/80` dog/door không bảo vệ field false-alarm claim | Yes | No | Label audit useful |
| Model | P0 | Real paired CED+STT fusion corpus | N=6 logic cases không đo real fusion accuracy | Yes after collection | Recording may be needed | Ground-truth annotation |
| System | P0 | Mic/event → OLED end-to-end latency | Metric hệ thống quan trọng nhất hiện thiếu | Partly | Yes | No |
| System | P0 | 1-hour full pipeline + ESP32/OLED soak | Synthetic control-path soak không đo prototype | Partly | Yes | No |
| System | P0 | Sent/received/drop/duplicate/malformed/ACK rate | Chứng minh transport reliability | Yes | Yes | No |
| System | P0 | Reconnect behavior và recovery time | Cần cho wearable reliability | Yes | Yes | No |
| Human | P0 | Alert/subtitle comprehension + missed-alert rate | Claim benefit cho người khiếm thính cần evidence trực tiếp | No | Yes | Yes |
| Human | P0 | Reaction time to emergency alert | Chứng minh alert hữu ích kịp thời | No | Yes | Yes |
| Model | P1 | External per-class CI/confusion/error publication | Tránh aggregate che class yếu | Yes | No | No |
| Model | P1 | STT noisy CI với nhiều utterance/noise recordings | Effective N=30 và 1 recording/environment còn nhỏ | Yes | No | No |
| Model | P1 | DTLN latency overhead + SI-SNR/intelligibility metric | Hiểu trade-off thay vì chỉ intrusive SNR | Yes | No | Optional |
| Model | P1 | Personalization labeled user/profile evaluation | Chỉ cần nếu muốn claim AI accuracy | Partly | No | Yes |
| System | P1 | Partial transcript and final transcript latency | Cloud call latency không phải subtitle latency | Yes | Optional mic | No |
| System | P1 | Full-mode CPU/RAM/model footprint | Chọn cấu hình laptop khả thi | Yes | No | No |
| System | P1 | Battery/power/temperature/OLED refresh success | Prototype wearable feasibility | Partly | Yes | No |
| Human | P1 | HUD readability, usability, cognitive load | Thiết kế 64×32 cần validation thật | No | Yes | Yes |
| Human | P2 | Comfort, wearable acceptance, preference | Hoàn thiện product evidence | No | Yes | Yes |

## 8. Computer vs Hardware vs Human Tests

### GROUP A — COMPUTER-ONLY

1. Freeze mapping, hashes, candidate config và evaluator trước khi xem prediction.
2. Hoàn tất FSD50K/DCASE audio acquisition theo plan riêng; deduplicate hash + acoustic fingerprint; chạy đúng một lần.
3. Xuất CED/emergency confusion, per-class P/R/F1, FPR/FNR, bootstrap CI và toàn bộ FP/FN.
4. Mở rộng hard negatives và SNR/noise evaluation với independent source identity.
5. Tính STT confidence intervals theo utterance/speaker; đo clean/noisy WER/CER.
6. Đo DTLN overhead, SI-SNR/gain-corrected quality và downstream deltas.
7. Xây real paired fusion evaluation sau khi corpus đã được thu/label.
8. Pin environment/model revision và làm offline golden reproduction pass.
9. Đo CPU/RAM cho legacy, improved V2, V3, STT on/off và DTLN on/off.

### GROUP B — HARDWARE / PROTOTYPE TEST

1. Gắn timestamp tại capture, preprocessing, decision, serial write, ESP32 parse và OLED refresh/ACK.
2. Đo mic/event → OLED latency trên nhiều loại sự kiện và transcript.
3. Chạy 1-hour full pipeline soak có microphone, models, serial, ESP32 và OLED.
4. Ghi sent/received/ACK, CRC reject, malformed, duplicate, dropped, reconnect count và recovery time.
5. Đo microphone stability, OLED update success/refresh behavior, battery/power và temperature.

### GROUP C — HUMAN TEST

1. HUD subtitle readability và comprehension.
2. Alert comprehension, reaction time và missed-alert rate.
3. Usability, cognitive load, comfort và wearable acceptance.
4. So sánh preference giữa các cách hiển thị/cảnh báo đã preregister.
5. Tuyển và báo cáo riêng participant khiếm thính; không thay thế bằng automated test.

## 9. Minimum Defensible Benchmark Set

Bộ 15 figure/metric tối thiểu cho báo cáo KHKT:

1. External CED Macro-F1 + 95% CI trên FSD50K/DCASE frozen holdout.
2. External per-class recall cho 5 safety categories.
3. External emergency precision/recall/F1/FNR.
4. Hard-negative FPR hoặc false alarms/hour với negative coverage được công bố.
5. CED Macro-F1-vs-SNR curve trên real noise.
6. Clean Vietnamese STT WER/CER.
7. Noisy Vietnamese STT WER-vs-SNR với CI.
8. DTLN before/after ΔWER và ΔCED Macro-F1; báo cáo cả kết quả âm.
9. Real paired fusion F1 hoặc ghi `MISSING` nếu chưa có corpus hợp lệ.
10. Local processing P50/P95, tách model load khỏi warm inference.
11. Mic/event → OLED P50/P95 end-to-end latency.
12. 1-hour full-pipeline crash/drop/duplicate counts.
13. Transport delivery/ACK success rate + reconnect recovery.
14. Full-mode CPU/RAM và ESP32 power/battery/temperature.
15. Human alert/subtitle comprehension, reaction time và missed-alert rate.

Hiện có evidence tương đối dùng được cho mục 5–8 và 10; mục 1–4 chưa đạt external holdout, mục 9 underpowered, mục 11–15 thiếu.

## 10. Exact Next Plan

### STEP 1: Finish Layer 1

Chốt trước bằng hash: source files, CED/EfficientSED checkpoints, taxonomy/mapping, threshold `0.05`, code commit, dependency lock, split và evaluator. Hoàn thiện FSD50K eval audio và DCASE source-event audio/manifest mà không xem prediction trước; deduplicate với ESC-50/AudioSet-identifiable sources; chạy một lần và công bố raw predictions + all FP/FN + CI. Không tune lại trên holdout.

### STEP 2: Lock baseline

Chọn rõ baseline reportable: legacy/default, improved V2 hay V3 opt-in. Giữ cả ba như separate configurations, không trộn số. Sau external run, tạo immutable manifest gồm command, environment lock, dataset hashes, model hashes, thresholds, raw outputs và calculated metrics. Golden reproduction phải pass offline hoặc ghi rõ dependency network.

### STEP 3: Finish critical Layer 2 metrics

Thêm measurement hooks/counters không đổi decision logic. Chạy mic → OLED latency, full-pipeline 1-hour soak, transport reliability/reconnect và CPU/RAM/power/thermal. Mỗi latency phải ghi start/end boundary; không gọi inference latency là system latency.

### STEP 4: Prepare Layer 3 protocol

Soạn protocol/ethics/consent, participant criteria và preregister primary outcomes: comprehension, reaction time, missed alerts, readability. Pilot kỹ thuật chỉ kiểm tra protocol; không dùng pilot để tạo final efficacy claim. Formal study phải có người khiếm thính nếu báo cáo nhắm đến nhóm này.

## 11. Files That Need Changes

Chỉ đề xuất, chưa sửa:

| File | Proposed change | Why | Risk to core logic |
|---|---|---|---|
| `requirements.txt` + new benchmark lockfile | Pin exact versions và model/tool revisions | Reproduction hiện phụ thuộc cache/network | Low |
| `benchmark/config.py` hoặc new frozen-holdout config | Khai báo FSD50K/DCASE mapping, split, hashes, exclusions | Tránh result-driven mapping | None nếu benchmark-only |
| `benchmark/download_datasets.py` | Chỉ thêm resumable/verified bounded acquisition theo protocol đã duyệt | Metadata có nhưng audio thiếu | None nếu benchmark-only |
| New `benchmark/run_final_holdout.py` | One-shot evaluator, no threshold search, raw outputs + CI + FP/FN | P0 external baseline | None nếu benchmark-only |
| `benchmark/experiments/emergency_v3/run_evaluation.py` | Tách calibration command khỏi frozen external evaluation command | Ngăn accidental test tuning | None nếu benchmark-only |
| `benchmark/run_expanded_benchmark.py` | Ghi exact CED revision/hash, dependency snapshot và effective independent N | Methodology/reproducibility | None nếu output mới |
| `audio_pipeline.py` | Thêm timestamp/counter instrumentation sau khi design được duyệt | Đo stage/system latency và drops | Low–Medium; hot path |
| `display_transport.py` + `firmware/src/protocol.*` | Optional sequence ID/ACK/CRC/reconnect counters | Đo delivery reliability | Medium; protocol compatibility |
| `firmware/src/hud.*` | Optional render-complete timestamp/counter | Đo OLED endpoint | Low–Medium |
| New `benchmark/hardware_protocol.md` | Chuẩn hóa mic→OLED và soak procedure | Reproducible Layer 2 | None |
| New `benchmark/human_study_protocol.md` | Preregister outcomes, recruitment, consent, analysis | Layer 3 scientific validity | None |

## Final Verdict

```text
SOUNDGUARD BENCHMARK AUDIT

MODEL LAYER: 62%
SYSTEM LAYER: 24%
HUMAN LAYER: 0%

MODEL BASELINE READY TO LOCK:
PARTIALLY

SYSTEM BASELINE READY TO LOCK:
NO

READY FOR HUMAN TESTING:
NO

TOP 5 MISSING BENCHMARKS:
1. Untouched FSD50K/DCASE external CED + emergency evaluation.
2. Broad hard-negative false-alarm/FPR benchmark.
3. Real paired CED+STT fusion benchmark.
4. Mic/event → ESP32 → OLED end-to-end latency.
5. One-hour full-pipeline hardware soak with delivery/reconnect/resource metrics.

TOP 3 DATA/VALIDITY RISKS:
1. ESC-50 is already development data and may overlap AudioSet-derived model training sources.
2. Emergency V3 threshold 0.05 was selected on ESC-50 folds 1–4; fold 5 is not a truly independent external holdout.
3. Negative coverage is only 80 dog/door clips; fusion N=6 and noisy STT effective N=30 are underpowered for broad claims.

NEXT ACTION:
Freeze evaluator/config/dependencies, acquire only the planned FSD50K/DCASE audio,
deduplicate and run the preregistered Layer-1 holdout exactly once without tuning.

SAFE TO START SPEAKER RECOGNITION AFTER:
The Layer-1 external holdout is complete and immutable, the baseline configuration
is explicitly selected and locked, and critical Layer-2 mic→OLED/reliability
measurements have at least a reproducible protocol and initial baseline.
```

## Post-Audit Update — External Validation Completed

Updated: 2026-08-20 23:52+07:00. This section supersedes earlier statements in this historical audit that FSD50K/DCASE audio, external predictions, or broad hard negatives were missing. It does not change the System or Human conclusions.

- Official FSD50K eval audio acquired: 10,231/10,231 WAVs; both archive MD5 values match Zenodo and every extracted entry passed size/CRC-32 verification. One earlier complete-size `.z01` download failed MD5 and is preserved as quarantined, unused evidence.
- Official DCASE source-event audio acquired: 83/83 WAVs; archive MD5 verified.
- Exact provenance overlap found in the full official populations: FSD50K 152 rows and DCASE 6 events.
- Frozen benchmark-eligible population before decontamination: 3,988. Exact-ID exclusions from this population: 49; exact raw/PCM hash duplicates: 0; high-confidence acoustic-fingerprint exclusions: 1; final N: 3,938.
- Final included set: FSD50K fire 121, siren 49, vehicle horn 64, hard negatives 3,657; DCASE glass breaking 30 and baby crying 17.
- Final freeze timestamp: `2026-08-20T23:38:59.807894+07:00`. First inference marker followed at `2026-08-20T23:39:32.924662+07:00`.
- Frozen V2 OR EfficientSED baseline ran once on all 3,938 rows. Post-hoc threshold tuning and selective rerun: `NO`.
- External target Macro-F1: 0.5749. Emergency recall/FNR: 0.5943/0.4057. Hard-negative FPR: 0.0566. DCASE emergency recall: 0.9574. Fire recall: 0.2479 (critical weakness).
- Internal-to-external target Macro-F1 delta: -0.3909. This generalization gap is retained as the scientific result, not tuned away.
- Current software/harness suite after the benchmark-only acquisition, decontamination, and extraction checks: 158 passed, 0 failed, 5 subtests passed.

Updated three-layer status:

```text
MODEL LAYER: 80%
SYSTEM LAYER: 24%
HUMAN LAYER: 0%

MODEL BASELINE READY TO LOCK:
YES

SYSTEM BASELINE READY TO LOCK:
NO

READY FOR HUMAN TESTING:
NO
```

The Model Layer increase reflects completion of the frozen decontaminated external holdout and broad hard-negative evaluation. It does not imply deployment readiness. Remaining Model Layer gaps include real paired CED+STT fusion data, uncertainty/independent replication work, and any labeled personalization claim. System/Human percentages remain unchanged because external model evidence does not measure hardware or users.

Canonical evidence: `benchmark_results/external_results_20260820/FINAL_KHKT_EXTERNAL_VALIDATION.md`.
