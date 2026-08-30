# SoundGuard Benchmark Root-Cause Analysis

## 1. Executive Summary

1. Strict CED weakness combines true top-1 errors with a measurable ontology/canonicalization gap: accuracy 0.4500 / Macro F1 0.5525 versus conservative ontology-aware **diagnostic only** accuracy 0.5321 / Macro F1 0.6580.
2. Emergency false negatives are concentrated by class (glass_breaking=40, vehicle_horn=22, fire=20, siren=11, baby_crying=5) and stage (A=87, B=11); recognition dominates before emergency rules can act.
3. The SNR implementation passes known-signal unit diagnostics, and independent lag estimates center at 384.0 samples. The frozen non-speech CED delta is -7.9700 dB; an independent recomputation is -7.7334 dB and remains -7.5718 dB after optimal gain correction. In contrast, the 150 speech pairs improve by 6.7506 dB: signal domain explains the preliminary/expanded sign reversal.
4. DTLN turns 162 correct CED decisions into errors versus 6 repairs; mean output/input RMS ratio is 0.2172.
5. Personalization hyphen failures arise because punctuation normalization turns compounds into standalone evidence tokens; 10/12 diagnostic compounds infer a role.

## 2. CED

Worst strict recalls: glass breaking 0.000, dog barking 0.050. Full class confusions and confidence summaries are in `ced/per_class_failure_analysis.csv`. The model ontology has Glass/Shatter rather than a literal Glass breaking label, and Dog/Bark/Bow-wow rather than the single SoundGuard dog_barking category. The strict-to-aware gap quantifies evaluation/canonicalization mismatch; residual errors remain genuine model/domain failures. Wrong confidence distributions are class-specific and retained rather than reduced to one aggregate. No direct evidence identifies peak normalization or input shape as the principal cause; the model's configured temporal pooling can plausibly dilute impulses, but this benchmark cannot isolate it without a controlled preprocessing experiment.

## 3. Emergency Detection

All 98 false negatives are enumerated. FN by class: glass_breaking=40, vehicle_horn=22, fire=20, siren=11, baby_crying=5. FN by stage: A=87, B=11. Fusion was not part of this emergency benchmark, so it suppressed none. The saved-prediction threshold sweep shows the maximum attainable trade-off without changing top-1 recognition; it is explicitly offline diagnostic and cannot recover wrong/unmapped labels.

## 4. DTLN

### Is the -7.97 dB Delta SNR measurement valid?

**YES within the stated intrusive-SNR methodology.** The SNR function recovers 20/10/0 dB synthetic cases, 289/290 independent alignment estimates equal 384 samples (one low-correlation outlier is 325), and no sign or delay-correction bug was found. Plain SNR is gain-sensitive, but optimal gain correction still leaves a strongly negative CED-pair mean delta (-7.5718 dB), while mean clean correlation falls from 0.8920 to 0.3333. Thus scaling does not explain away the degradation. It remains an intrusive waveform metric, not a perceptual-quality or intelligibility verdict. The same energy/gain fields are reported for all 150 speech pairs.

The apparent sign reversal is chiefly a domain change: DTLN improves the 150 speech-pair intrusive SNR by 6.7506 dB on average, while it degrades the 700 non-speech CED pairs. This is consistent with a speech denoiser preserving speech more effectively than environmental safety transients.

### Does DTLN harm CED?

**YES on this paired protocol.** Correct→wrong=162; wrong→correct=6. Class/environment/SNR breakdowns are in the transition matrix. The mechanism is downstream representation change after strong gain/spectral processing, especially loss/distortion of discriminative events—not residual alignment alone.

### Does DTLN harm STT?

**MIXED, net harmful.** Improved=5, unchanged=126, worsened=19; aggregate expanded WER delta is positive even though speech intrusive SNR improves. The evidence shows that waveform SNR is not a sufficient proxy for recognizer accuracy; it is consistent with phonetic detail changes/provider sensitivity, but this cache cannot isolate a single acoustic mechanism.

## 5. Personalization

`factory-method`→`factory method` and `baby-blue`→`baby blue` during punctuation normalization. Whole-word matching then treats the pieces as explicit role evidence. This generalizes to other compounds and is a parser/evidence issue, not Universal Critical validation.

## 6. Fusion

Existing CED and STT real audio are unpaired, so legitimate paired fusion cases available now: 0. N>=30 requires a new independently labeled paired corpus; manufacturing combinations is scientifically invalid.

## 7. STT

Clean corpus edit counts are substitutions=88, deletions=21, insertions=2. Mean per-utterance CER change after stripping diacritics is 0.0093; detailed speaker/length and highest-WER rows are retained.

## 8. Ranked Root Causes

| Rank | Problem | Evidence | Impact | Confidence |
|---:|---|---|---|---|
| 1 | CED recognition/ontology failures on safety events | Full 280 top-k; class recall and strict-aware gap | Emergency FN | High |
| 2 | DTLN changes discriminative audio | 700 paired transitions and signal diagnostics | CED and STT degradation | High |
| 3 | Emergency threshold rejects some recognized candidates | Stage-B FN plus frozen-prediction sweep | Additional FN | High |
| 4 | Hyphen normalization creates false personalization evidence | deterministic minimal repros | Profile relevance/safety | High |
| 5 | Fusion evidence is underpowered | 0 paired real cases; N=6 logic cases | Unsupported improvement claim | High |

## 9. Recommended Interventions

| Priority | Component | Proposed modification | Expected benefit | Risk | Production behavior? | Rerun |
|---|---|---|---|---|---|---|
| P0 | CED mapping/evaluation | Preregister ontology-aware mapping, then separately test any production alias change | Separate scoring artifact from real misses | Over-broad aliases | Yes if production | Frozen CED + emergency |
| P0 | Safety-event CED | Evaluate a safety-focused model/temporal transient strategy on frozen clips | Reduce glass/horn/siren FN | False alarms/domain overfit | Yes | Full frozen CED/noise/emergency |
| P0 | DTLN routing | Gate/bypass speech denoiser for transient CED after a frozen ablation | Prevent correct→wrong transitions | More noise reaches CED | Yes | 700 pairs + latency |
| P1 | Emergency thresholds | Consider only after top-1 repair; choose from preregistered validation, not test sweep | Recover threshold-only FN | False alerts | Yes | Frozen positives + negatives/new validation |
| P1 | Personalization parser | Preserve compound boundaries or require contextual role phrases | Eliminate compound false roles | Miss terse true roles | Yes | all personalization + adversarial |
| P1 | DTLN metric | Add SI-SNR/gain-corrected metric beside intrusive SNR | Sounder interpretation | Metric comparability | No | signal diagnostics |
| P2 | Fusion dataset | Collect 30–50+ real paired cases | Valid fusion estimate | Annotation/cost | No initially | preregistered fusion benchmark |
| P2 | STT | Analyze/provider-tune only after safety work | Better utility | Network/provider variability | Yes | fixed VIVOS/noise set |

## 10. Scientific Interpretation

Model limitation and dataset/domain mismatch remain after the conservative ontology correction. Benchmark ontology mismatch is measured, not assumed. Threshold effects are isolated with saved predictions. No preprocessing defect was proven. DTLN has both gain-sensitive metric ambiguity and independently observed downstream harm. Personalization is a deterministic rule/parser issue.

**Production/core behavior modified during this analysis: NO.**
