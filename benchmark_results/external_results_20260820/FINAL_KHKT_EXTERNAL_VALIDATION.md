# SoundGuard External Validation — KHKT Evidence Package

Evaluation date: 2026-08-20 (Asia/Saigon)  
Required terminology: **Decontaminated external holdout derived from the official FSD50K/DCASE evaluation data**

This evaluation is not described as an untouched official evaluation split. The source data were filtered using a preregistered, provenance-based decontamination protocol before any external prediction.

## 1. Decontamination Summary

| Dataset | Official population | Mapping-eligible before decontamination | Exact-ID overlap in full official population | Exact-ID excluded from eligible set | Exact audio hash excluded | Fingerprint excluded | Final included |
|---|---:|---:|---:|---:|---:|---:|---:|
| FSD50K eval | 10,231 | 3,934 | 152 | 43 | 0 | 0 | 3,891 |
| DCASE 2017 source events | 83 | 54 | 6 | 6 | 0 | 1 | 47 |
| **Total** | **10,314** | **3,988** | **158** | **49** | **0** | **1** | **3,938** |

The remaining 6,326 official rows were outside the preregistered target/hard-negative scope, not removed because of model behavior. The fingerprint exclusion was DCASE `glassbreak/250709.wav`, matched to development clip ESC-50 `2-250710-A-39.wav`: 190 dominant landmark hits, offset consistency 0.9948, minimum coverage 0.3592, aligned overlap 2.1496 s, and aligned waveform NCC 0.8903.

Archive integrity was checked before extraction. The first complete-size FSD50K `.z01` download failed official MD5 and was quarantined; it was never used. A full clean redownload matched official MD5, and all 10,231 extracted WAV entries passed ZIP uncompressed-size and CRC-32 checks.

## 2. Frozen External Dataset

| Dataset / class | Initial eligible N | Exact-ID excluded | Fingerprint excluded | Final N |
|---|---:|---:|---:|---:|
| FSD50K fire | 124 | 3 | 0 | 121 |
| FSD50K siren | 55 | 6 | 0 | 49 |
| FSD50K vehicle horn | 68 | 4 | 0 | 64 |
| FSD50K hard negatives | 3,687 | 30 | 0 | 3,657 |
| DCASE baby crying | 19 | 2 | 0 | 17 |
| DCASE glass breaking | 35 | 4 | 1 | 30 |
| **Total** | **3,988** | **49** | **1** | **3,938** |

Mapping exclusions fixed before prediction included FSD50K `Crackle` (107 membership rows; 4 exact ESC-50 overlaps; 103 remaining) because generic crackle is not sufficient evidence of fire, broad `Alarm`, generic `Glass`, and `Crying_and_sobbing`. DCASE gunshot (29 clips) is a valid emergency class but was outside the requested DCASE baby/glass scope and was not used as a negative. Hard negatives were not sampled or balanced: all qualifying FSD50K clips were retained.

## 3. External Results

### Overall frozen combined baseline

| Metric | Result | Counts |
|---|---:|---:|
| Target Macro Precision | 0.5368 | 5 target classes |
| Target Macro Recall | 0.7322 | 5 target classes |
| Target Macro F1 | 0.5749 | 5 target classes |
| Emergency Precision | 0.4465 | 167 TP, 207 FP |
| Emergency Recall | 0.5943 | 167 / 281 events |
| Emergency F1 | 0.5099 | 281 target + 3,657 negative clips |
| False Negative Rate | 0.4057 | 114 / 281 missed |
| Hard-negative FPR | 0.0566 | 207 / 3,657 false alerts |

### Per-class one-vs-rest results

| Class | N | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Glass breaking | 30 | 29 | 32 | 1 | 0.4754 | 0.9667 | 0.6374 |
| Vehicle horn | 64 | 48 | 135 | 16 | 0.2623 | 0.7500 | 0.3887 |
| Fire | 121 | 30 | 22 | 91 | 0.5769 | 0.2479 | 0.3468 |
| Siren | 49 | 37 | 40 | 12 | 0.4805 | 0.7551 | 0.5873 |
| Baby crying | 17 | 16 | 2 | 1 | 0.8889 | 0.9412 | 0.9143 |

The per-class FP column uses all other frozen external samples as negatives. DCASE alone contains no hard-negative clips, so its separately reported class precision is target-set discrimination rather than an operational false-alert estimate.

### Dataset-specific safety results

| Dataset | N | Target N | Hard-negative N | Emergency Recall | FNR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| FSD50K-derived | 3,891 | 234 | 3,657 | 0.5214 | 0.4786 | 0.0566 |
| DCASE-derived | 47 | 47 | 0 | 0.9574 | 0.0426 | N/A |

## 4. KHKT Main Results

### Metric 1

Metric: External target Macro-F1  
Result: **0.5749** across five safety-relevant classes.  
KHKT claim: The frozen baseline has measurable, non-perfect generalization beyond the ESC-50 development corpus.  
Report use: Primary external-results table and abstract-level quantitative result.

### Metric 2

Metric: DCASE emergency recall  
Result: **0.9574 (45/47)**.  
KHKT claim: The frozen system detects most decontaminated DCASE baby-cry/glass-break source events.  
Report use: Safety-event validation subsection.

### Metric 3

Metric: Baby-crying F1  
Result: **0.9143**, recall 0.9412, N=17.  
KHKT claim: Baby crying is the strongest independently evaluated category in this holdout.  
Report use: Per-class result chart and discussion.

### Metric 4

Metric: Glass-breaking recall  
Result: **0.9667 (29/30)**.  
KHKT claim: Transient glass-breaking events have high sensitivity after decontamination.  
Report use: Safety recall table; precision must be shown alongside it.

### Metric 5

Metric: Siren recall  
Result: **0.7551 (37/49)**.  
KHKT claim: Siren generalization is materially lower than internal development performance but remains above 75%.  
Report use: Per-class generalization discussion.

### Metric 6

Metric: Hard-negative false-positive rate  
Result: **0.0566 (207/3,657)**.  
KHKT claim: The system rejects 94.34% of a much broader preregistered negative population, while the non-zero false-alert rate defines a real limitation.  
Report use: False-alert table/figure and limitations.

### Metric 7

Metric: Fire recall  
Result: **0.2479 (30/121)**; class FNR 0.7521.  
KHKT claim: Fire is a critical external-generalization weakness and does not support a strong reliability claim.  
Report use: Limitations, future work, and safety-risk discussion.

### Metric 8

Metric: Internal-to-external target Macro-F1 delta  
Result: **−0.3909** (0.9658 internal development to 0.5749 external).  
KHKT claim: Development-only results substantially overestimate external performance.  
Report use: Generalization-gap table and methodology justification.

## 5. Internal vs External

| Metric | Internal / Development | External Decontaminated | Delta |
|---|---:|---:|---:|
| Target Macro Precision | 0.9585 | 0.5368 | −0.4217 |
| Target Macro Recall | 0.9750 | 0.7322 | −0.2428 |
| Target Macro F1 | 0.9658 | 0.5749 | −0.3909 |
| Emergency Recall | 0.9750 | 0.5943 | −0.3807 |
| False Negative Rate | 0.0250 | 0.4057 | +0.3807 |
| Hard-negative FPR | 0.0000 | 0.0566 | +0.0566 |

Internal values reconstruct the same frozen combined class-decision rule from saved ESC-50 development predictions. The internal negative population contains only 80 dog/door clips, whereas the external negative population contains 3,657 clips across preregistered dog/bark, door/knock, speech, music/instrument, engine, and bell families; therefore the FPR delta reflects both generalization and much broader negative coverage.

No threshold, mapping, preprocessing, model, or sample rule was changed after observing this gap.

## 6. Leakage / Validity Statement

Contamination auditing was completed before external prediction. All exact Freesound-ID overlaps with the 2,000-row official ESC-50 metadata were identified from provenance; benchmark-eligible overlaps were excluded without consulting model output. Remaining candidates were compared with available SoundGuard development audio using raw SHA256, canonical decoded-PCM SHA256, and conservative non-embedding acoustic landmarks with time-offset consistency and aligned waveform NCC verification. One high-confidence same-recording match was excluded and no ambiguous match remained. Model/checkpoints, preprocessing, threshold 0.05, mappings, hard-negative rules, included/excluded manifests, code hashes, and dependencies were frozen before the first inference. No post-hoc tuning or confidence-based sample selection was performed.

## 7. Reproducibility

- Frozen CED model: `mispeech/ced-tiny` revision `ace276d29dd0bb3f3517b0fa8cf300738c409019`.
- Frozen EfficientSED checkpoint SHA256: `92a5c1dc5aa21620e262420c4cc657d2f699bdfa20a748de506c354a2fe2770a`.
- Frozen EfficientSED threshold: `0.05`.
- Final external manifest was frozen at `2026-08-20T23:38:59.807894+07:00`.
- First inference marker: `2026-08-20T23:39:32.924662+07:00`.
- Inference completed: `2026-08-20T23:51:54.180693+07:00`, 3,938/3,938 rows.
- Post-hoc threshold tuning: **NO**.
- Selective rerun: **NO**.
- External harness/full suite after additions: **158 passed, 0 failed, 5 subtests passed**.
- Raw predictions, manifests, mappings, checksums, figures, and comparison data are retained under the dated evidence directories.

## 8. Updated Benchmark Status

```text
MODEL LAYER: 80%
SYSTEM LAYER: 24%
HUMAN LAYER: 0%

MODEL BASELINE READY TO LOCK:
YES
```

`YES` means the computer-only model baseline and its limitations can now be locked and reported reproducibly. It does not mean deployment safety is established: external fire recall is weak, the generalization gap is large, System Layer hardware evidence remains incomplete, and Human Layer evidence is absent.

## Final Verdict

```text
EXTERNAL VALIDATION STATUS

FSD50K official audio acquired:
YES

DCASE official audio acquired:
YES

Exact-ID contamination removed:
YES

Hash duplicate check:
DONE

Acoustic fingerprint check:
DONE

Final external manifests frozen before prediction:
YES

External prediction performed only after freeze:
YES

Post-hoc threshold tuning:
NO

FSD50K external evaluation:
DONE

DCASE external evaluation:
DONE

Internal-vs-external comparison:
DONE

MODEL LAYER READY TO LOCK:
YES

TOP KHKT RESULTS:
1. External target Macro-F1 = 0.5749 across five classes.
2. DCASE emergency recall = 0.9574 (45/47).
3. Baby-crying F1 = 0.9143; glass-breaking recall = 0.9667.
4. Hard-negative FPR = 0.0566 on 3,657 broad negatives.
5. Generalization gap: target Macro-F1 delta = -0.3909; fire recall = 0.2479 is the critical limitation.

REMAINING COMPUTER-ONLY BLOCKERS:
1. A real paired CED+STT fusion corpus; current fusion N=6 remains logic validation only.
2. Source/uploader-aware uncertainty intervals and independent replication for future model changes.
3. Full-mode computer resource/long-soak evidence and labeled remote-personalization evaluation if those claims are retained.

NEXT RECOMMENDED ACTION:
Lock this baseline and evidence package. Do not optimize against this holdout. Move to critical System Layer mic-to-OLED latency, transport/reconnect, and full-pipeline soak tests. Any future fire improvement must use separate development data and a newly frozen independent holdout.
```

