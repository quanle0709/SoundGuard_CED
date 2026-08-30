# SoundGuard candidate decision

The frozen ESC-50 set is a development/improvement evaluation set because its failures were inspected during root-cause analysis. It is not an untouched holdout.

| Candidate | Macro F1 | Safety recall | Emergency recall | Emergency FNR | FP |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.5525 | 0.4400 | 0.5100 | 0.4900 | 0 |
| Semantic mapping | 0.6773 | 0.5600 | 0.5750 | 0.4250 | 0 |
| Best pooling only: 1 s max | 0.3050 | 0.2150 | 0.2600 | 0.7400 | 0 |
| Best combined: mapping + 1 s max | 0.3550 | 0.2700 | 0.2900 | 0.7100 | 0 |

## Gate decision

- **Accept for opt-in integration:** the formal semantic mapper. It exceeds the 0.65 Macro-F1 target, improves safety/emergency recall, creates no observed emergency false positive, and excludes broad score-driven aliases. Its emergency recall remains below target, so this is an incremental candidate, not a complete safety solution.
- **Reject:** all tested temporal pooling variants. They substantially harm every primary metric.
- **Accept for opt-in integration:** compound-aware personalization evidence. It produces 50/50 fixed safety, 12/12 adversarial passes, 128/128 priority assertions, and 20/20 Universal Critical checks.
- **Select architecture C:** raw CED, raw STT, DTLN standalone. Fixed real-noise STT raw WER/CER are 0.1141/0.0669 versus 0.1275/0.0782 through DTLN. The application already sends raw audio to CED; `--no-dtln` selects raw STT.
- **No fusion tuning:** N=6 remains functional validation only.

Emergency recall 0.575 does not meet the 0.70 minimum. Additional saved-prediction recovery would require questionable labels such as generic `Breaking`, `Vehicle`, `Animal`, or `Crying, sobbing`; those mappings are rejected. A model-level safety-event improvement and untouched validation are required.

`Glass -> glass_breaking` is material-specific but coarser than literal breakage. It remains an explicit, auditable, default-off alias whose false-alert cost must be tested on the final holdout.
