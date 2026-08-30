# SoundGuard Improvement Plan

Safety for hearing-impaired users determines ordering; no expected final numbers are invented.

| Priority | Change | Target metric | Baseline | Expected direction | Risk | Effort |
|---|---|---|---|---|---|---|
| P0 | Separate ontology scoring from true CED failure; preregister any runtime aliases | Safety-class recall / emergency FN | See frozen manifest | Fewer mapping-caused FN | Alias false positives | Medium |
| P0 | Compare safety-event model/transient-window candidates on the frozen set plus independent validation | Glass, siren, horn, fire recall | Frozen per-class CSV | Up | Model cost/false alarms | High |
| P0 | Test CED bypass or event-aware routing around speech DTLN | DTLN delta CED Macro F1 | Frozen expanded summary | Toward zero/up | Noise robustness | Medium |
| P1 | Validate threshold choices on a separate preregistered set | Recall, FNR, FPR, F1 | Frozen emergency metrics | Recall up with bounded FPR | Test leakage | Medium |
| P1 | Preserve compound boundaries in role evidence | Adversarial safety pass | 48/50 plus diagnostic CSV | Up | False negatives for terse answers | Low |
| P1 | Add gain-corrected/SI-SNR reporting | Signal-quality interpretability | -7.97 dB gain-sensitive | Less ambiguous | Historical comparison | Low |
| P2 | Collect paired real fusion corpus | Fusion F1 and branch coverage | N=6 logic-only, 0 paired | Evidence quality up | Collection bias | High |
| P2 | Address STT substitutions/diacritics after safety items | WER/CER | 0.1075/0.0594 | Down | Provider drift | Medium |

Every future production change must reuse `baseline_manifest.json`, unchanged sample hashes, and the exact frozen test set. Candidate selection must use separate development data.
