# SoundGuard Emergency V3 — EfficientSED Evaluation

## 1. Objective

Evaluate exactly one external repository, EfficientSED, as an additional framewise emergency specialist without replacing CED-Tiny or changing STT, DTLN, fusion logic, personalization, HUD transport, firmware, direction, or vibration.

## 2. Stable SoundGuard V2 Baseline

The work began from commit 24346d879634f65eeb8a647bdba0ac84e249c63d on improve-ced-safety-v2. The V2 snapshot/hash evidence was verified and all 136 pre-V3 tests passed before the experiment. Frozen V2 metrics on 280 ESC-50 development clips were emergency recall 0.575, FNR 0.425, precision 1.0, FP 0/80, and CED macro F1 0.677323.

## 3. EfficientSED Repository Audit

Repository https://github.com/theMoro/EfficientSED was audited at commit e3449bfd1f4922b81d82410ed842653f462bfca8. It is MIT licensed. Its framewise models are trained with AudioSet weak/strong data and expose an ordered 447-label strong ontology. Details are in EFFICIENTSED_AUDIT.md.

## 4. Selected Model

Only fmn10_strong from upstream release v0.0.1 was selected. It is a 3,830,798-parameter Frame-MobileNet with linear strong heads, a 15,538,778-byte checkpoint, 16 kHz mono input, 10-second padded windows, and 250 output frames (40 ms each). SHA-256 is 92a5c1dc5aa21620e262420c4cc657d2f699bdfa20a748de506c354a2fe2770a.

## 5. Label Ontology Mapping

The taxonomy was frozen before scoring and permits only glass_breaking, vehicle_horn, fire, siren, and baby_crying. Exact labels and narrowly justified strong aliases are included. Generic Animal, Vehicle, Breaking, Crying/sobbing, Crackle, and Honk are rejected as too broad; fireworks and firecrackers are excluded. The complete classification and justification are in the experimental taxonomy JSON.

## 6. Dataset and Leakage Limitations

The benchmark uses all 280 existing ESC-50 clips: 40 each for five safety classes and 40 each for dog/door negatives. EfficientSED and CED-Tiny are AudioSet-derived, so source/training overlap cannot be ruled out. Folds 1–4 selected the threshold; fold 5 was read only afterward, but ESC-50 as a whole is already SoundGuard development data and is not a truly independent deployment holdout.

## 7. EfficientSED Standalone Results

At threshold 0.05, on all 280 clips, EfficientSED alone produced TP 193, FN 7, FP 0, TN 80: recall 0.965, FNR 0.035, precision 1.0, F1 0.982188, FPR 0.0.

Development folds 1–4: TP 155, FN 5, FP 0, TN 64, recall 0.96875. Frozen-threshold fold 5: TP 38, FN 2, FP 0, TN 16, recall 0.95.

Standalone correct-category recall was glass 1.0, horn 1.0, fire 0.85, siren 1.0, baby crying 0.975.

## 8. Frame-Wise Analysis

All 40 glass and 40 horn clips contained correct framewise evidence. Three glass and four horn clips would have been missed by thresholding the temporal mean, demonstrating that transient frame maxima mattered. Fire remained the weakest class at 34/40. Per-clip timings/durations and compressed raw frame evidence are retained.

## 9. Threshold Selection

Thresholds 0.05 through 0.95 were swept only on folds 1–4. Threshold 0.05 maximized recall while retaining 0/64 negative false positives; every higher threshold reduced recall without improving FPR. The maximum safety score among all 80 dog/door negatives was 0.007601, so 0.05 did not behave as a detect-all threshold. CED-Tiny thresholds were unchanged.

## 10. V2 vs EfficientSED vs Combined V3

The selected policy is V2 OR EfficientSED approved-safety evidence. On all 280 clips:

| Metric | V2 | EfficientSED only | Combined V3 |
| --- | ---: | ---: | ---: |
| TP | 115 | 193 | 195 |
| FN | 85 | 7 | 5 |
| Recall | 0.575 | 0.965 | 0.975 |
| FNR | 0.425 | 0.035 | 0.025 |
| Precision | 1.0 | 1.0 | 1.0 |
| FP / negatives | 0 / 80 | 0 / 80 | 0 / 80 |
| FPR | 0.0 | 0.0 | 0.0 |

Combined development folds 1–4 reached recall 0.975 (156/160), while frozen-threshold fold 5 reached 0.975 (39/40).

## 11. Safety-Class Performance

Combined all-data recalls were glass 1.0, vehicle horn 1.0, fire 0.90, siren 1.0, and baby crying 0.975. All requested class directions were met. The remaining five combined misses were fire clips.

## 12. False Positives

Binary emergency false positives were exactly 0/80 dog/door negatives for V2, EfficientSED alone, and combined V3. Category-level scoring produced six horn and two fire cross-category activations on other safety-positive clips; these did not create non-emergency alerts but show that the specialist is more reliable as emergency evidence than as a perfect subtype classifier.

## 13. Latency and Resources

EfficientSED mel plus inference over 280 clips: mean 54.07 ms, median 48.87 ms, P90 70.31 ms, P95 82.98 ms, P99 143.97 ms. Sequential V2 plus specialist estimate: P95 106.82 ms. Warm production subprocess round trip including decode/IPC over 30 clips: P95 91.07 ms. Initialization was 4.22 seconds and is excluded from normal inference latency.

The worker used about 385.7 MB working RAM (398.2 MB peak), the checkpoint is 15.54 MB, and five-second real-time factor was about 0.0122. The short benchmark consumed about 5.81 logical cores (72.6% of the eight-logical-core machine capacity). CUDA was unavailable, so GPU latency/VRAM are not reported. This is a PC-side specialist, not an ESP32 model.

## 14. Integration Architecture

Emergency V3 is explicit opt-in through --emergency-v3 or SOUNDGUARD_ENABLE_EMERGENCY_V3=1. CED-Tiny continues to classify/display all normal sounds and remains the input to unchanged fusion logic. If V2 already supplies dangerous sound evidence it is preserved; otherwise a thresholded specialist category is supplied to the existing emergency state machine as additional categorical evidence. The five approved categories are the specialist's entire authority.

## 15. Regression Safety

With all V3 flags off, the frozen golden output matched exactly. The complete suite passed: 150 tests, 0 failed, plus 5 unittest subtests. This includes 14 new V3 tests. Compilation and import checks passed. CED macro F1 remains the frozen V2 value 0.677323 because EfficientSED never replaces general CED output. Main-environment before/after dependency inventories are identical.

## 16. Failure/Fallback Behavior

The isolated model is loaded lazily once and reused. Missing Python, missing checkout/checkpoint, initialization failure, malformed worker output, inference exceptions, or worker exit logs one concise warning and supplies no specialist evidence. The existing V2 path continues and the application does not require EfficientSED when V3 is disabled.

## 17. Holdout Status

Fold 5 was kept out of threshold selection and yielded 39/40 combined recall with 0/16 negative FP. It is a useful internal check, not a truly independent holdout. No additional external dataset was introduced because the task allowed only the named external repository and existing audio, and repeatedly sourcing/tuning another set would violate the requested discipline.

## 18. Limitations

ESC-50 is small, coarse, and possibly overlaps AudioSet training. Dog and door are only two negative families. The threshold may shift with microphones, gain, background mixtures, cultures, or continuous streaming. CPU/RAM cost is inappropriate for ESP32-class execution. GPU performance and physical device battery/thermal behavior are unmeasured. A preregistered independent field-audio holdout is required before deployment claims.

## 19. Final Verdict

Emergency Recall >= 0.70: **PASS**.

Emergency Recall >= 0.75: **PASS**.

Emergency Recall >= 0.80: **PASS**.

**KEEP Emergency V3** as an opt-in PC/software specialist. It provides a +0.400 absolute recall gain over V2 on the current development audio with no observed binary false-positive increase and exact legacy behavior when disabled. Retention does not imply field validation; an independent holdout remains mandatory before default-on deployment.
