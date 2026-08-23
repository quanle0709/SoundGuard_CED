# Semantic mapper experiment

Frozen ESC-50 is now a development/improvement set, not an untouched holdout. The taxonomy was defined from ontology semantics before scoring transitions. Broad labels (`Animal`, `Vehicle`, generic `Breaking`, `Crying, sobbing`) are explicitly rejected.

| Metric | Baseline | Candidate |
|---|---:|---:|
| CED Macro F1 | 0.5525 | 0.6773 |
| Emergency recall | 0.5100 | 0.5750 |
| Emergency FNR | 0.4900 | 0.4250 |
| Emergency precision | 1.0000 | 1.0000 |
| Emergency false positives | 0 | 0 |

Recovered strict-category TPs by alias: `{'Dog': 8, 'Toot': 2, 'Emergency vehicle': 10, 'Police car (siren)': 3, 'Glass': 5, 'Bow-wow': 1, 'Crackle': 2, 'Civil defense siren': 1, 'Ambulance (siren)': 1}`. Newly created classification errors by alias: `{}`. Every transition is retained in `semantic_mapper_transitions.csv`; no observed transition is hidden. The experiment cannot estimate false-positive behavior outside these seven source classes, so semantic validity remains a required gate even when observed FP is zero.
