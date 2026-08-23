# Baseline versus improved

The same frozen development clips are used; this is not final untouched-holdout evidence.

| Metric | Baseline | Improved | Delta |
|---|---:|---:|---:|
| CED accuracy | 0.450000 | 0.567857 | +0.117857 |
| CED Macro F1 | 0.552549 | 0.677323 | +0.124774 |
| Glass recall | 0.000000 | 0.125000 | +0.125000 |
| Vehicle-horn recall | 0.450000 | 0.500000 | +0.050000 |
| Fire recall | 0.500000 | 0.550000 | +0.050000 |
| Siren recall | 0.600000 | 0.975000 | +0.375000 |
| Baby-crying recall | 0.650000 | 0.650000 | +0.000000 |
| Emergency recall | 0.510000 | 0.575000 | +0.065000 |
| Emergency FNR | 0.490000 | 0.425000 | -0.065000 |
| Emergency precision | 1.000000 | 1.000000 | +0.000000 |
| Emergency false positives | 0.000000 | 0.000000 | +0.000000 |
| STT clean WER | 0.107454 | 0.107454 | +0.000000 |
| STT clean CER | 0.059429 | 0.059429 | +0.000000 |
| Personalization priority accuracy | 1.000000 | 1.000000 | +0.000000 |
| Personalization Universal Critical | 1.000000 | 1.000000 | +0.000000 |
| Personalization fixed safety | 0.960000 | 1.000000 | +0.040000 |
| Local PC P95 ms | 33.537500 | 31.120125 | -2.417375 |

## DTLN routing

Selected: raw audio to CED and STT; DTLN remains standalone. The fixed routing table is preserved at `benchmark_results/improvement/dtln_routing_comparison.csv`. Raw real-noise STT WER is 0.114085 versus 0.127465 with DTLN; raw paired CED Macro F1 is 0.399122 versus 0.099647 with DTLN.
