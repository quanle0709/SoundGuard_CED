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

The first sound or voice build lazily loads its model. When live mode or the existing
personalization server starts, each enabled feature with at least one valid built profile begins
one background preload; disabled features and features without a valid profile do not load a
model. Recognition remains fail-open while a model is loading or if preload fails. The
`/api/recognition` response reports each model as `idle`, `loading`, `ready`, or `failed`, including
load timing or the failure text where available. Repeated preload requests do not create duplicate
workers.
Finish enrollment before starting live mode. Live mode snapshots enabled prototypes at startup so its real-time microphone callback never performs profile disk I/O; restart live mode after changing profiles.

## Enrollment and testing

### Familiar Sounds

1. Open **Familiar sounds** and create a named profile.
2. Record or upload at least two valid WAV examples. For reliable use, 5–10 varied samples from different distances and ordinary background conditions are recommended.
3. Remove bad samples. Silent, too-quiet, severely clipped, duplicate, too-short, too-long, and undecodable audio is rejected.
4. Build the prototype. EfficientAT embeds every valid sample and stores an L2-normalized centroid locally.
5. Use **Test WAV** with both a known example and a different sound. Review the score, margin, and `UNKNOWN` reason.

Live sound decisions require cosine score `>= 0.72`, lead over the runner-up `>= 0.08`, two matching decisions within the last three 3-second windows, and a 5-second per-label cooldown. Thresholds remain profile fields for controlled calibration.

### Familiar Voices

1. Obtain the person's informed permission and acknowledge consent in the UI.
2. Record at least three valid natural utterances. For reliable use, 6–10 utterances with varied wording are recommended; do not enroll a single repeated phrase.
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

## Reproducible public-dataset verification

The full functional evaluator is `tools/evaluate_personalized_recognition_dataset.py`. After the official datasets are cached, one command creates a new timestamped isolated run:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_personalized_recognition_dataset.py
```

The cache must contain official ESC-50 metadata and selected official WAV files under `benchmark_data/external/esc50/`, plus the checksummed official LibriSpeech SLR12 `test-clean.tar.gz` and extracted `LibriSpeech/test-clean/` tree under `benchmark_data/external/personalized_recognition_datasets/`. All of that material and all generated benchmark enrollment profiles remain Git-ignored. The script refuses a metadata/archive checksum mismatch and never uses `personalization/enrollments/`.

The recorded run is `benchmark_results/personalized_recognition_dataset/20260913T191404+0700/`. It uses seed `20260913`, exact checksummed file manifests, ESC-50 fold 1 for five enrollment clips per class, folds 2–3 for calibration, and folds 4–5 for holdout. LibriSpeech uses two enrolled speakers with five enrollment utterances each, separate calibration and holdout utterances, three calibration-only impostor speakers, and five different holdout impostor speakers.

Results on this deliberately small fixed split:

- Familiar Sounds: known top-1 was 30/30, but the calibration-derived open-set threshold/margin accepted the correct class for 19/30 known clips and rejected 20/24 unknown clips. There were 11/30 false rejections and 4/24 false accepts; all four false accepts were vacuum-cleaner clips accepted as washing machine. Overall open-set decisions were correct for 39/54 clips.
- Familiar Voices: 20/20 enrolled-speaker holdouts were correctly accepted and 25/25 impostor holdouts were rejected, with 0/25 false accepts and 0/20 false rejects. This 45/45 result is only a public-dataset functional result on two English read-speech identities and five holdout impostor identities, not a population accuracy claim.
- The run also exercised actual HTTP create/list/upload/build/test/enable/disable/sample-delete/profile-delete routes for both feature types, both real workers together, deterministic drop-oldest queues, injected failure/timeout isolation, sound smoothing/cooldown, feature-off behavior, final-caption-only speaker prefixes, raw HELP transcripts, and locked alert priority.

Cold/warm latency, per-request CPU time, current/peak worker RSS, prototype times, score and margin distributions, confusion matrices, every rejected example and reason, exact source/model revisions, API results, queue counters, and unsupported claims are in the run artifacts. Fresh caches and concurrent host activity can materially increase cold-load time.

ESC-50 tests few-shot custom-category behavior, not personalized recognition of a particular household instance. LibriSpeech is a public English read-speech functional check; it does not establish performance for family members, Vietnamese speech, noisy/reverberant use, replay attacks, or identities outside this split. The older `tools/evaluate_personalized_recognition.py` output remains only a small plumbing smoke check.

## Remaining manual checks

Physical microphone, real household custom sounds, real consenting familiar/unfamiliar speakers, simultaneous CED/STT/model CPU contention, and OLED readability/priority require hardware/user testing. Thresholds were not calibrated on a diverse target-user cohort. Overlapping speakers, very short speech, playback/replay, reverberation, noise, illness, and microphone changes can cause rejection or misidentification.
