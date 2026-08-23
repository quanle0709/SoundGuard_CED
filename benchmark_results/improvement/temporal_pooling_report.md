# Temporal pooling experiment

Controlled alternatives use fixed non-overlapping 1-second windows and only three preregistered aggregations: max, top-2 mean, and 90th percentile. The current whole-clip output remains the baseline. No production pooling was changed.

Best pooling-only candidate by the safety-first gate: **Pooling: max_1s**, Macro F1 0.3050, safety recall 0.2150, emergency recall 0.2600, FP 0.

Best combined candidate: **Combined: max_1s**, Macro F1 0.3550, safety recall 0.2700, emergency recall 0.2900, FP 0.

This development set has already been inspected. Results support candidate selection only; they are not final holdout evidence.
