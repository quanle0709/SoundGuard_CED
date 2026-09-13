# HearVis Personalized Recognition

## Scope and safety boundary

Familiar Sounds and Familiar Voices are integrated into the existing local personalization site at `http://127.0.0.1:8765`. They are host-side, opt-in, and default-off. The original questionnaire, Apply/Sync behavior, `user_profile.json` schema, CED, STT, HELP, emergency rules, and HUD priority remain independent.

This is a research feature, not identity authentication or a certified alarm. A rejected, ambiguous, late, overloaded, or failed recognition becomes `UNKNOWN`; it never suppresses core CED/STT and never creates an emergency alert. The firmware continues to enforce `ALERT > SUBTITLE/CED > HOME`.

## Setup

The optional environment and model cache live under ignored `benchmark_data/external/` directories. The preparation script pins source/model revisions, keeps temporary and Numba cache data on D: beside that environment, and does not modify the primary `.venv`.

```powershell
python tools/prepare_personalized_recognition.py
python -m personalization.web_server
# Open http://127.0.0.1:8765
```

The first sound or voice build lazily loads its model. No optional model loads merely by opening the site.
Finish enrollment before starting live mode. Live mode snapshots enabled prototypes at startup so its real-time microphone callback never performs profile disk I/O; restart live mode after changing profiles.

## Enrollment and testing

### Familiar Sounds

1. Open **Familiar sounds** and create a named profile.
2. Record or upload 2–5 clean WAV examples, ideally from different distances and ordinary background conditions.
3. Remove bad samples. Silent, too-quiet, severely clipped, duplicate, too-short, too-long, and undecodable audio is rejected.
4. Build the prototype. EfficientAT embeds every valid sample and stores an L2-normalized centroid locally.
5. Use **Test WAV** with both a known example and a different sound. Review the score, margin, and `UNKNOWN` reason.

Live sound decisions require cosine score `>= 0.72`, lead over the runner-up `>= 0.08`, two matching decisions within the last three 3-second windows, and a 5-second per-label cooldown. Thresholds remain profile fields for controlled calibration.

### Familiar Voices

1. Obtain the person's informed permission and acknowledge consent in the UI.
2. Record at least three natural utterances. Vary wording; do not enroll a single repeated phrase.
3. Build the profile, then test with held-out known speech and an unenrolled speaker.

Speaker identification runs once on the raw final utterance supplied by the existing VAD/state machine. Score `>= 0.72` and runner-up margin `>= 0.06` are required. Accepted final captions are rendered as `[Display name] transcript`. HELP receives the original unprefixed transcript so speaker metadata cannot change emergency phrase matching.

## Live operation

```powershell
# Either feature may be enabled alone.
python app.py --live-stt --no-dtln --familiar-sounds
python app.py --live-stt --no-dtln --familiar-voices
python app.py --live-stt --no-dtln --familiar-sounds --familiar-voices
```

Use `--recognition-root PATH` for a non-default enrollment directory and `--personalized-python PATH` for a prepared compatible worker interpreter. Without the feature flags, no recognition worker starts and microphone callback behavior stays on the baseline speech/CED paths.

The one physical `sounddevice.InputStream` copies frames into bounded STT and CED queues as before. When Familiar Sounds is both enabled and has a built profile, the same callback makes one additional bounded copy. A lightweight window builder feeds a drop-oldest queue of size two. Voice jobs reuse finalized utterance audio. Model processes run below normal OS priority and return errors as `UNKNOWN`.

## Local storage, consent, and deletion

Enrollment files are stored under:

```text
personalization/enrollments/
  familiar_sounds/profiles/<uuid>/
  familiar_voices/profiles/<uuid>/
```

Each directory contains metadata, PCM WAV samples, and (after build) `prototype.npy`. Raw audio and embeddings are intentionally separate from `user_profile.json` and ignored by Git. Disabling preserves the local data; removing samples invalidates and disables the prototype; deleting a profile removes its directory. Delete a voice profile immediately if consent is withdrawn. Backups of this directory remain the operator's responsibility.

## Models and provenance

| Purpose | Exact implementation | Revision / artifact hash | License and data notes |
| --- | --- | --- | --- |
| Sound embedding | EfficientAT `mn10_as`, official 32 kHz / 128-mel / 800-window / 320-hop preprocessing, 960-D pooled feature | source `a425fdce92572e602a1d5634799bd9f1f2efa806`; weight SHA-256 `0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e` | EfficientAT code is MIT; the checkpoint is AudioSet-trained. Confirm downstream dataset/model terms before redistribution. |
| Speaker embedding | WeSpeaker `ECAPA-TDNN512-LM`, ONNX Runtime CPU, official 16 kHz Kaldi fbank80 + CMN, 192-D output | HF revision `a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b`; ONNX SHA-256 `d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0` | WeSpeaker code is Apache-2.0; the model card identifies CC-BY-4.0 and VoxCeleb2 training data. Voice enrollment itself is biometric/personal data. |

The sound checkpoint's complete SHA-256 is recorded by the worker and in the evaluation artifact. Optional Python versions are pinned in `requirements-personalized-recognition.txt`.

## Verification and measured host cost

The deterministic protocol is in `tools/evaluate_personalized_recognition.py`; its output is under `benchmark_results/personalized_recognition_mvp/`. ESC-50 enrollment uses folds 1–3 and tests folds 4–5. VIVOS uses three enrollment utterances and separate held-out utterances per enrolled speaker plus an unenrolled speaker. These small checks validate plumbing and open-set behavior; they are not population-level accuracy claims.

On the development Windows CPU after cache warm-up:

- EfficientAT: 6/6 trial decisions correct, including 2/2 unknown rejections and zero false accepts; median embedding 0.027 s, maximum 1.935 s in the measured run; worker RSS about 472 MB.
- WeSpeaker: 6/6 trial decisions correct, including 2/2 unknown rejections and zero false accepts; median embedding 0.052 s, maximum 1.953 s; worker RSS about 440 MB.
- Both workers retained concurrently can therefore approach 912 MB RSS. Fresh Numba/model caches can make the first request materially slower; the UI explicitly reports a lazy first build.
- Bounded queue drops and worker failures are exposed in runtime counters. The deterministic software tests exercise drop-oldest behavior and failure isolation without loading either model.

## Remaining manual checks

Physical microphone, real household custom sounds, real consenting familiar/unfamiliar speakers, simultaneous CED/STT/model CPU contention, and OLED readability/priority require hardware/user testing. Thresholds were not calibrated on a diverse target-user cohort. Overlapping speakers, very short speech, playback/replay, reverberation, noise, illness, and microphone changes can cause rejection or misidentification.
