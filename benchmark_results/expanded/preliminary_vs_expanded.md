# Preliminary versus expanded benchmark

| Area | Preliminary | Expanded |
| --- | ---: | ---: |
| CED clean | 2 clips / 2 classes | 280 clips / 7 mapped ESC-50 classes |
| Vietnamese STT | 1 utterance | 100 VIVOS test utterances (or recorded successful denominator) |
| Real environmental noise | none | 4 DEMAND environments |
| Emergency | 8 fixed cases | 280 real-audio CED-to-policy cases |
| Personalization | 22 priority assertions | 100+ hand-specified priority assertions and 50+ safety checks |
| Stability | 60 seconds | configured 3,600-second synthetic control-path soak |

The expanded estimates should replace the preliminary numbers in reports because their denominators, class balance, real-noise coverage, and fixed public ground truth are substantially stronger. Original files under `benchmark_results/` remain preserved. ESC-50 results still carry `POSSIBLE SOURCE OVERLAP` and must not be described as guaranteed AudioSet-independent validation.
