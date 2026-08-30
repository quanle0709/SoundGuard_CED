# Selected EfficientSED threshold

The frozen candidate threshold is **0.05** on the maximum frame probability over approved safety labels.

Selection used only ESC-50 folds 1–4 (N=224: 160 safety-positive, 64 dog/door negatives). Priorities were recall, FNR, false positives, precision, then F1, subject to development FPR <= 0.05. At 0.05 the standalone specialist achieved 155 TP, 5 FN, 0 FP, 64 TN: recall 0.96875, FNR 0.03125, precision 1.0, FPR 0.0. Higher thresholds reduced recall without reducing false positives.

Although 0.05 is numerically low, it is not a detect-all setting for this strong framewise checkpoint: the largest approved-safety score among all 80 dog/door negatives was 0.007601. The threshold was frozen before fold 5 was read for final metrics. CED-Tiny's threshold was not changed.

This remains development calibration, not a universal deployment threshold. An independent non-AudioSet-overlap holdout is still required before a field claim.
