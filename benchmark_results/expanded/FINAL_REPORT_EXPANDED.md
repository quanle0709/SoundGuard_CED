# SoundGuard Expanded PC Benchmark Report

## 1. System Under Test

Git commit `24346d879634f65eeb8a647bdba0ac84e249c63d`; Python 3.12.10 on Windows-11-10.0.26200-SP0. Production CED, STT, DTLN, emergency, fusion, and personalization behavior was not altered by this benchmark.

## 2. Dataset Expansion

ESC-50: 280 clips (40 each: siren, car horn, dog, crying baby, glass breaking, crackling fire, and door knock), CC BY-NC 3.0, `POSSIBLE SOURCE OVERLAP`. VIVOS: 100 official test utterances selected round-robin across all 19 test speakers, CC BY-NC-SA 4.0, proprietary-provider training membership unknown. DEMAND: one 16 kHz channel from traffic, cafeteria, office, and home/living environments, CC BY-SA 3.0. Full provenance is in `benchmark_data/DATASETS.md` and `dataset_leakage_audit.md`.

## 3. Experimental Method

Seed 42; fixed class-complete ESC-50 selection; fixed 30-utterance VIVOS robustness subset; identical paired signals before/after DTLN; Gaussian and real DEMAND noise at 20/15/10/5/0 dB; anti-clipping common gain; DTLN fixed delay removed only for intrusive SNR measurement. Vietnamese text uses Unicode NFC, lowercase, punctuation removal, and whitespace collapse while preserving diacritics.

## 4. Environmental Sound Detection

Clean accuracy: 0.45 ratio (N=280). Clean Macro F1: 0.5525488037155136 ratio (N=280). Macro F1 averages the seven declared target classes; predictions outside those classes are bucketed as `other`, remain errors, and are visible in the confusion matrix. See per-class metrics and predictions; no threshold was tuned and no hard sample was removed.

## 5. Noise Robustness

Synthetic Gaussian uses all 280 clips at every SNR. Real DEMAND uses a fixed balanced 70-clip subset (10 per class) in all four environments, producing N=280 paired observations per SNR. Results are reported separately for all five SNRs. Additive mixing uses measured RMS, a common anti-clipping gain, and recorded achieved SNR.

## 6. Vietnamese Speech-to-Text

Clean WER: 0.1074540174249758 ratio (N=100); clean CER: 0.05942857142857143 ratio (N=100). All 100 clean and all 600 noisy/filtered requests succeeded. Robustness uses the same 30 utterances at every SNR and filter condition. Google STT is network-dependent; failed requests would remain recorded and denominators would reflect successful responses.

## 7. DTLN Signal Quality

Mean Delta SNR: -7.969985152930216 dB (N=700). This is N=700 (35 balanced clips x 4 environments x 5 SNRs). The intrusive metric accounts for the measured 384-sample algorithmic delay and treats filter distortion/attenuation as error; it is not itself task accuracy.

## 8. DTLN Downstream Effect

CED Delta Macro F1: -0.2976841986694203 ratio (N=700). STT Delta WER: 0.013380281690140846 ratio (N=150). Here DTLN worsened both the mean intrusive SNR and downstream CED. It also increased real-noise STT WER at 20, 15, 5, and 0 dB and tied at 10 dB. Signal quality and task performance are distinct and both are reported without suppressing negative effects.

## 9. Emergency Detection

Recall: 0.51 ratio (N=280); False Negative Rate: 0.49 ratio (N=280); false alerts: 0 count (N=80). Ground truth follows the existing SoundGuard category levels and was fixed before predictions.

## 10. Fusion

Raw F1: 0.6666666666666666 ratio (N=6); fused F1: 1.0 ratio (N=6). This remains deterministic logic validation, not expanded real paired audio/transcript evidence.

## 11. Personalization

Priority accuracy: 1.0 ratio (N=128); safety pass rate: 0.96 ratio (N=50); Universal Critical pass rate: 1.0 ratio (N=20). Ground truth is hand-specified from documented policy. Remote AI personalization was skipped because neither provider credential was visible to the benchmark process.

## 12. Latency

PC software P95: 24.87394498020876 ms (N=100). This excludes network STT, microphone capture, serial transport, ESP32, OLED, and vibration.

## 13. Stability

Crash count: 0 count (N=48704800). Actual runtime: 3600.0079113999964 seconds; processed events: 48704800; exceptions: 0; Python traced allocation growth: 1.77626132965088 MiB. Windows RSS: unavailable. Named `SYNTHETIC CONTROL-PATH STABILITY SOAK`; it is not microphone/model/hardware stability.

## 14. Comparison With Preliminary Benchmark

See `preliminary_vs_expanded.md`. The original preliminary result tree is preserved.

## 15. Weaknesses and Failure Cases

Clean CED missed 55% of clips, with especially severe failures for glass breaking and dog barking; exact per-class values are retained. Emergency recall was 0.51 with a 0.49 false-negative rate, though no non-emergency clip falsely alerted. DTLN reduced mean intrusive SNR and CED performance on the expanded real-noise protocol, while also worsening aggregate real-noise STT WER. Personalization failed two deliberately misleading hyphenated-substring checks (`factory-method` and `baby-blue`), producing a 48/50 safety result; this benchmark finding is not patched in production code. Exact error rows are available in predictions/results CSVs.

## 16. Limitations

This PC benchmark is not a smart-glasses hardware benchmark. Google STT depends on a cloud network service. AudioSet/ESC-50 source overlap cannot be ruled out. Gaussian noise is artificial; four DEMAND recordings do not cover all real acoustic scenes, and additive mixing does not reproduce reverberation or device microphones. There is no hearing-impaired user study and no physical battery, HUD, vibration, temperature, directional, or end-to-end device validation.

## 17. Measurements Still Required on Real Glasses

Battery life/power, thermal behavior, microphone and enclosure effects, real-stream drop rate, serial/ESP32 latency, OLED readability/latency, vibration perception, direction accuracy with ground truth, and trials with hearing-impaired participants.
