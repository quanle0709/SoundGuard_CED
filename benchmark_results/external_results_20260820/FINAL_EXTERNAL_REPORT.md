# SoundGuard External Validation — 2026-08-20

## 1. Decontamination Summary

See `../external_decontamination_20260820/decontamination_summary.json`. Exact-ID, raw/decoded hash, and conservative acoustic-landmark checks were completed before prediction. The dataset is a **decontaminated external holdout derived from official FSD50K/DCASE evaluation data**, not an untouched official evaluation split.

## 2. Frozen External Dataset

| Dataset | Final N | Target N | Hard-negative N |
|---|---:|---:|---:|
| FSD50K-derived | 3891 | 234 | 3657 |
| DCASE-derived | 47 | 47 | 0 |
| Total | 3938 | 281 | 3657 |

## 3. External Results

| Metric | Result |
|---|---:|
| Target Macro Precision | 0.5368 |
| Target Macro Recall | 0.7322 |
| Target Macro F1 | 0.5749 |
| Emergency Recall | 0.5943 |
| False Negative Rate | 0.4057 |
| Hard-negative FPR | 0.0566 |

## 4. KHKT Main Results

1. **External target Macro-F1:** 0.5749. KHKT claim: category performance generalizes beyond the development corpus. Report use: primary external-results table.
2. **Emergency Recall:** 0.5943 (167/281). KHKT claim: proportion of relevant safety events detected. Report use: safety-results table.
3. **False Negative Rate:** 0.4057 (114 missed events). KHKT claim: empirical missed-event risk. Report use: limitations and safety table.
4. **Hard-negative FPR:** 0.0566 (207/3657). KHKT claim: the system does not alert indiscriminately on preregistered unrelated sound families. Report use: false-alert figure/table.
5. **Strongest class F1:** baby_crying = 0.9143, N=17. KHKT claim: strongest independently evaluated capability. Report use: per-class chart.

## 5. Internal vs External

The comparison is in `internal_vs_external.csv`; deltas are external minus internal and are reported as a generalization gap, without threshold changes.

## 6. Leakage / Validity Statement

Contamination auditing occurred before prediction. Exact ESC-50 Freesound-ID overlaps were excluded using provenance, then source-file/decoded-PCM hashes and conservative time-offset-consistent acoustic landmarks were checked against available SoundGuard development audio. Model, mappings, thresholds, sample population, and hard-negative rules were frozen before the first external inference. No sample was selected from model output and no post-hoc tuning was performed.

## 7. Reproducibility

The model/checkpoint hashes, CED revision, preprocessing, threshold 0.05, mappings, manifests, code hashes, dependency snapshot, and inference start/completion markers are preserved. Raw predictions are retained for every included sample.
