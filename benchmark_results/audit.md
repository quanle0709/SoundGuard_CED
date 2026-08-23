# SoundGuard benchmark audit

Baseline commit: `24346d879634f65eeb8a647bdba0ac84e249c63d`

Dirty working tree was preserved:

```text
M .gitignore
 M README.md
 M app.py
 M audio_pipeline.py
 M emergency_system.py
 M requirements.txt
 M test_audio_pipeline.py
?? display_transport.py
?? firmware/
?? personalization/
?? test_display_transport.py
?? test_personalization.py
```

| Feature | Exists? | Implementation | Model/Algorithm | Can benchmark on PC? | Ground truth available? | Planned metrics |
|---|---|---|---|---|---|---|
| Environmental Sound Detection / CED | YES | sound_classifier.py | mispeech/ced-tiny | YES; 2 clips only | 2 labeled clips | Accuracy, per-class P/R/F1, confusion |
| CED noise robustness | YES | benchmark wrapper around CED | Seeded white-noise mixing | YES; preliminary | Derived from 2 clean clips | Accuracy, Macro F1 by SNR |
| CED latency | YES | classify_audio_file | Warm repeated inference | YES | N/A | mean/median/P90/P95/P99 |
| Speech-to-Text | YES | speech_recognizer.py | Google Speech Recognition vi-VN | NETWORK-DEPENDENT | 1 transcript | WER, CER |
| STT noise robustness | YES | same STT interface | Seeded noise | NETWORK-DEPENDENT | 1 transcript | WER/CER by SNR |
| STT latency | YES | transcribe_audio_file | Google remote API | NETWORK-DEPENDENT | N/A | latency, RTF |
| Noise filtering quality | YES | speech_enhancer.py | DTLN two-stage TFLite | YES | Paired clean/noisy | input/output/delta SNR |
| Noise filtering effect on CED | YES | DTLN -> CED experiment | Paired | YES | 2 clips | Macro F1 delta |
| Noise filtering effect on STT | YES | DTLN -> Google STT | Paired | NETWORK-DEPENDENT | 1 transcript | WER/CER delta |
| Emergency detection quality | YES | emergency_system.py | Deterministic thresholds/rules | YES | Fixed policy cases | TP/FP/TN/FN/P/R/F1/FNR |
| Emergency false alarms | YES | EmergencySystem | Deterministic | YES | Fixed negatives | false alerts/clip |
| Emergency latency | YES | EmergencySystem single-shot | Software timing | YES | Fixed cases | mean/median/P95/P99 |
| Fusion CED + STT | YES | fusion_engine.py | Deterministic rules | YES | Fixed cases | before/after accuracy/P/R/F1 |
| Fusion error analysis | YES | fusion_engine.py | A/B/C/D categories | YES | Fixed cases | counts and percentages |
| Speaker/family voice recognition | NO | NOT IMPLEMENTED | N/A | NO | NO | NOT IMPLEMENTED |
| Speaker recognition noise robustness | NO | NOT IMPLEMENTED | N/A | NO | NO | NOT IMPLEMENTED |
| AI personalization | YES | personalization/ | Rules + optional OpenAI/OpenRouter | Rules YES; API credential-dependent | Fixed policy cases | priority/role/context accuracy, invariants, latency |
| AI assistant | NO | NOT IMPLEMENTED as separate feature | N/A | NO | NO | NOT IMPLEMENTED |
| End-to-end software pipeline latency | YES | CED -> emergency -> fusion | PC fixed-file path | YES | 2 clips | PC software latency |
| Continuous software stability | YES | Repeated control-path workload | Bounded synthetic soak | YES | Deterministic events | crashes/exceptions/CPU/RAM |
| Left/right direction or vibration | NO | No localization/decision implementation | N/A | NO | No multichannel truth/hardware | Hardware validation required |
| HUD transport | YES | display_transport.py + ESP32 firmware | CRC framed serial | Logic only | Unit tests only | Physical latency/readability not benchmarkable |

No production/core file is modified by the benchmark runner.
