# SoundGuard PC Benchmark Report

## 1. System Under Test

Commit `24346d879634f65eeb8a647bdba0ac84e249c63d` on branch `main` with the pre-existing dirty working tree recorded in `methodology.json`. Benchmark code imports production interfaces without changing core behavior.

## 2. Hardware and Software Environment

See `methodology.json`. This is a Windows PC software benchmark using Python 3.12.10.

## 3. Dataset and Ground Truth

Two source WAV files were available: one human-labeled bark clip and one Vietnamese speech clip with a manifest reference transcript. Seeded Gaussian noise generated paired 20/15/10/5/0 dB variants. Emergency, fusion, and personalization use fixed hand-authored policy cases in `benchmark_data/`. Results are preliminary because the audio dataset is extremely small.

## 4. Methodology

Fixed seed 42; identical noisy WAVs for filtered/unfiltered comparisons; warm-up excluded from steady-state CED latency; Unicode NFC/case/punctuation/whitespace normalization for Vietnamese WER/CER; production thresholds unchanged.

## 5. Results

### Environmental Sound Detection

- Clean accuracy: 0.5 ratio (n=2)
- Clean Macro F1: 0.3333333333333333 ratio (n=2)
- Warm P95 latency: 18.28365000837948 ms (n=8)

### Speech-to-Text

- Clean WER: 0.0 ratio (n=1)
- Clean CER: 0.0 ratio (n=1)

### Noise Filtering

- Mean Delta SNR: 6.770421367897822 dB (n=10)
- CED Mean Delta F1: -0.39999999999999997 ratio (n=10)

### Emergency Detection

- Recall: 1.0 ratio (n=8)
- False Negative Rate: 0.0 ratio (n=8)

### Fusion

- Delta F1: 0.33333333333333337 ratio (n=6)

### AI Personalization

- Exact asserted-priority accuracy: 1.0 ratio (n=22)
- Universal Critical pass rate: 1.0 ratio (n=8)
- Remote provider latency/consistency is skipped when no credential is present.

### Sound Direction / Vibration

NOT IMPLEMENTED — NO BENCHMARK RESULT. No localization algorithm or multichannel ground truth exists.

### End-to-End PC Software Pipeline

- PC SOFTWARE END-TO-END P95 latency: 24.981680023483932 ms (n=2)

### Stability

- Crash count: 0 count (n=820400)

## 6. Key Findings

- Emergency policy cases achieved 100% recall with no false negatives or false alarms (8 total cases; 4 negatives).
- Fusion improved binary F1 from 0.667 to 1.000 on six constructed rule cases by repairing one help-request miss and suppressing one benign bark false alarm.
- Deterministic personalization passed all 22 priority assertions and all Universal Critical checks; exact role and context inference were both 1.0 ratio (n=8) and 1.0 ratio (n=8).
- CED clean accuracy was only 0.500 (1/2 clips), and SNR behavior was non-monotonic; the denominator is too small for a general accuracy claim.
- After correcting for DTLN's measured 384-sample algorithmic delay, DTLN improved mean intrusive SNR. Despite that signal-level improvement, it reduced CED Macro F1 and worsened Google STT WER at 10, 5, and 0 dB on this tiny dataset; unfiltered STT remained perfect on the single utterance. These negative downstream results are retained.
- No thresholds were tuned and no hard cases were removed after inspection.

## 7. Limitations

This is a **PC SOFTWARE BENCHMARK**, not real smart-glasses hardware validation. Two audio clips cannot establish general CED/STT accuracy or class balance. Synthetic white noise does not represent all real environments. Google STT is network-dependent. The bounded stability run is shorter than the one-hour target. No claim is made about ESP32/OLED/vibration latency, battery life, physical direction accuracy, or real-world user benefit.

## 8. Missing Hardware/User Validation

- Battery life, power consumption, and device temperature
- OLED readability and physical display latency
- Vibration perception and actuator latency
- End-to-end serial/ESP32 latency
- Two-microphone directional accuracy with recorded ground truth
- Long-duration microphone stability in real environments
- Testing with hearing-impaired participants
