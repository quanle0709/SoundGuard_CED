# HearVis personalized-recognition dataset benchmark

Run: `20260913T191404+0700`. Fixed seed: `20260913`.

## 1. Software/unit verification

- Focused command: `.\.venv\Scripts\python.exe -m pytest -q tests\test_personalized_recognition.py tests\test_personalized_recognition_dataset.py` — 20 passed, 0 failed in 1.79 s.
- Full command: `.\.venv\Scripts\python.exe -m pytest -q` — 275 passed, 0 failed, 4 warnings, and 7 subtests passed in 59.02 s.
- These software tests are separate from the public-dataset functional results below.

## 2. Public-dataset functional evaluation

### Familiar Sounds — ESC-50 few-shot custom-category behavior

- Enrollment: 15 clips; calibration: 30 known + 24 unknown; holdout: 30 known + 24 unknown.
- Correct known identification: 19/30 (63.3%).
- Known acceptance: 19/30 (63.3%); false rejection: 11/30 (36.7%).
- Unknown rejection: 20/24 (83.3%); false acceptance: 4/24 (16.7%).
- Overall correct open-set decisions: 39/54 (72.2%). This tiny fixed split is a functional result, not a general accuracy claim.
- Known top-1 before rejection: 30/30 (100.0%); CLOCK ALARM: 10/10 top-1, 6/10 accepted, DOOR KNOCK: 10/10 top-1, 9/10 accepted, WASHING MACHINE: 10/10 top-1, 4/10 accepted.
- Calibration-only operating point: cosine threshold `0.77`, runner-up margin `0.10`.
- Confusion matrix, score/margin distributions, and every rejected example with reason are in `sound_metrics.json`; trial rows are in `sound_predictions.csv`.

ESC-50 demonstrates few-shot custom-category behavior. It does **not** demonstrate personalized-instance recognition of a particular household device.

### Familiar Voices — LibriSpeech SLR12 test-clean

- Enrollment: 10 utterances; calibration: 10 known + 15 impostor; holdout: 20 known + 25 impostor.
- Correct enrolled-speaker identification: 20/20 (100.0%).
- False rejection: 0/20 (0.0%); impostor false acceptance: 0/25 (0.0%).
- Impostor UNKNOWN rejection: 25/25 (100.0%); ambiguous decisions: 0/45 (0.0%).
- Overall correct open-set decisions: 45/45 (100.0%). This tiny fixed split is a functional result, not a population estimate.
- Known top-1 before rejection: 20/20 (100.0%).
- Calibration-only operating point: cosine threshold `0.68`, runner-up margin `0.30`.
- Confusion matrix, score/margin distributions, and every rejected example with reason are in `voice_metrics.json`; trial rows are in `voice_predictions.csv`.

This is a public-dataset functional evaluation. It does not establish family-member performance, Vietnamese speech performance, replay-attack resistance, population-level accuracy, or guaranteed independence from every pretraining identity.

### Measured model cost

- EfficientAT `mn10_as` (`a425fdce92572e602a1d5634799bd9f1f2efa806`, SHA-256 `0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e`): cold P50 `16.174` s, cold P95 `24.294` s, cold max `25.196` s (n=3); warm P50 `0.0727` s, warm P95 `0.1474` s, warm max `0.7022` s; peak worker RSS `405.9` MiB.
- WeSpeaker ECAPA-TDNN512-LM (`a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b`, SHA-256 `d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0`): cold P50 `14.873` s, cold P95 `17.660` s, cold max `17.970` s (n=3); warm P50 `0.1524` s, warm P95 `0.4687` s, warm max `0.5408` s; peak worker RSS `448.8` MiB.
- Prototype-build distributions and per-request worker CPU time are in the metrics JSON. Both retained worker processes were measured concurrently; combined current/peak RSS is recorded in `system_metrics.json`.

## 3. Synthetic queue/failure tests

Both real model workers were started in the same isolated runtime. The deterministic sound queue submitted 10, processed 3, dropped 7, timed out 0, and failed 0; the voice queue submitted 10, processed 3, dropped 7, timed out 0, and failed 0. Both processed payloads `[0, 8, 9]`, proving drop-oldest behavior. The injected isolation check submitted 2, processed 2, timed out 1, failed 1, dropped 0, and returned fail-open `UNKNOWN` for both.

## 4. Measurements not performed

- Physical microphone capture, OLED/HUD hardware rendering, end-to-end live CED/STT/model contention, energy draw, and thermal behavior.
- Real household instances, consenting familiar people, Vietnamese voices, overlapping speakers, replay/spoof attacks, and noisy/reverberant field conditions.
- Statistical confidence intervals or population-level validation.

## 5. Claims that are not supported

- No claim of personalized-instance recognition, real family-member accuracy, Vietnamese speaker accuracy, authentication/security suitability, replay resistance, certified-alarm behavior, or population-level accuracy.
- No claim that a small-sample perfect result, if observed, generalizes beyond the exact checksummed split.

## Isolation

Real user profile storage was fingerprinted before and after the run and was unchanged: `True`. Benchmark profiles remained under Git-ignored benchmark storage.
