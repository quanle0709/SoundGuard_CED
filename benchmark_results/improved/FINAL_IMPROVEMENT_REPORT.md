# SoundGuard Improvement Report

## 1. Stable Baseline

Dirty worktree preserved by binary patches, production hashes, manifest, and a recoverable source archive. Development occurs on `improve-ced-safety-v2`; no baseline tag was made because that would bundle unrelated user work.

## 2. Root Causes

Fine/coarse ontology mismatch, genuine safety-event model misses, non-speech damage from DTLN, and punctuation-normalized personalization compounds.

## 3. Changes Evaluated

Formal semantic taxonomy; max/top-two/p90 one-second pooling; three DTLN routes; compound-aware evidence.

## 4. Changes Rejected

All temporal pooling variants; DTLN before CED; DTLN before STT on this fixed real-noise set; unsafe broad aliases (`Animal`, `Vehicle`, `Breaking`, generic crying). Fusion was not tuned.

## 5. Selected Architecture

Default-off semantic CED adapter; default-off compound-safe personalization; raw CED; raw STT in improved mode via `--no-dtln`; DTLN standalone.

## 6. Baseline vs Improved

CED Macro F1 0.5525 -> 0.6773. Emergency recall 0.5100 -> 0.5750; FP 0 -> 0.

| Target | Result | Measured |
|---|---|---:|
| CED Macro F1 >= 0.65 | **PASS** | 0.677323 |
| CED stretch Macro F1 >= 0.70 | **FAIL** | 0.677323 |
| Mean safety-class recall >= 0.70 | **FAIL** | 0.560000 |
| Emergency recall >= 0.70 | **FAIL** | 0.575000 |
| STT clean WER <= 0.12 | **PASS** | 0.107454 |
| Personalization priority >= 0.95 | **PASS** | 1.000000 |
| Universal Critical = 1.00 | **PASS** | 1.000000 |
| Personalization safety >= 0.98 | **PASS** | 1.000000 |
| Regression failures = 0 | **PASS** | 0 |
| Frozen stability crashes = 0 | **PASS** | 0 |

## 7. Safety-Critical Performance

Emergency target is **FAIL**: 0.5750 < 0.70. Broad aliases were rejected rather than used to manufacture success. Model-level work is required.

## 8. STT Performance

Clean WER 0.1075 remains under 0.12. Raw audio wins real-noise A/B; cloud latency remains separately reported.

## 9. Noise Filtering / Routing Findings

DTLN improves speech intrusive SNR but worsens fixed downstream STT and severely harms non-speech CED. It remains available independently, not forced into improved downstream routes.

## 10. Personalization Safety

Priority 1.0000; Universal Critical 1.0000; fixed safety 1.0000; remaining adversarial compound failures 0.

## 11. Latency

Alternating same-process paired P95 33.5375 -> 31.1201 ms; delta -2.4174 ms (-7.21%). Frozen expanded P95 was 24.8739 ms. The negative paired delta is timing noise, not a claimed speedup; no measurable mapping penalty was observed.

## 12. Regression Safety

Golden legacy comparison with all new flags OFF passed. Complete validation: 136 tests passed, 0 failed, 5 subtests passed; compilation PASS; imports PASS. Frozen stability evidence remains 0 crashes; this is not real-device stability.

## 13. Limitations

Development-set tuning, possible AudioSet/ESC-50 overlap, seven coarse classes, no hardware/user study, network STT variability, and no final holdout result.

## 14. Holdout Validation Status

Candidate frozen; holdout not yet acquired. See `FINAL_HOLDOUT_PLAN.md`.
