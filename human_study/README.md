# SoundGuard Layer 3 human-study platform

Current status: **PLATFORM READY — HUMAN DATA PENDING**

This directory contains local research tooling only. It does not modify the
frozen SoundGuard model, CED/STT semantics, emergency/HELP behavior, transport,
or OLED HUD. Do not run a real participant until local approval requirements,
the pilot, HELP recording preparation, standardized setup, and mandatory
preflight have passed.

The evaluated prototype uses its existing network-dependent Google Speech
Recognition implementation for live speech-to-text. Stable internet is a
controlled requirement for speech, mixed, and scripted HELP processing. No
alternate recognizer, known transcript, or mock is used by the study runner.

## Preparation and commands

Install the project requirements and prepare the existing VIVOS and ESC-50
assets. Create the two HELP WAVs and provenance record described in
`stimuli/README.md`. Complete the physical checklist and consent process.
Commit the locked platform and frozen product first: preflight refuses tracked
git changes. It also hashes the actual stimulus files into the participant lock.

Mandatory researcher-only preflight:

```powershell
python human_study/run_study.py --preflight-only --hud-port auto --researcher-test-wav path\to\researcher_test.wav
```

Pilot (stored under `participant_data/pilot`, never included automatically):

```powershell
python human_study/run_study.py --pilot --participant PILOT01 --hud-port auto --researcher-test-wav path\to\researcher_test.wav
```

Main participant:

```powershell
python human_study/run_study.py --participant P01 --hud-port auto --researcher-test-wav path\to\researcher_test.wav
```

If the study computer has multiple audio devices, lock them explicitly for
preflight, pilot, and every main session with `--input-device N` and
`--output-device N`; the selected indices are stored in `session.json`.

The completely prompted one-click form is also available:

```powershell
python human_study/run_study.py
```

Validate one session without changing it:

```powershell
python human_study/validate_session.py human_study/participant_data/main/P01
```

After real P01–P10 sessions are collected and validated, analyze them:

```powershell
python human_study/analyze_results.py
```

With no complete real data, analysis exits without manufacturing results.

## Experimental inventory

Every participant completes both WITHOUT and WITH conditions using distinct
matched stimulus sets. There are 38 scored trials:

- speech comprehension: 12 (quiet and mild-noise VIVOS utterances);
- environmental awareness: 12 (supported DOG/HORN/DOOR/BABY plus benign
  no-event controls from ESC-50);
- mixed speech + environment: 8 (two separate responses); and
- critical/HELP: 6 (safe SIREN and GLASS_BREAKING recordings plus independent
  scripted HELP recordings).

Odd participant IDs start WITHOUT; even IDs start WITH. A four-sequence scheme
cross-balances Set A/B independently of condition order. Order inside each
condition and block is deterministically randomized by protocol version,
participant ID, and seed.

## Evidence and failures

The runner uses monotonic high-resolution timing. It stores host playback onset,
participant response, production HUD-send timestamps where the existing log
provides them, separate RT-from-stimulus and RT-from-HUD values, researcher
observations of the physical OLED, CED/alert/subtitle state, product runtime
evidence, and STT/network fields.

A playback or hardware failure is `system_error`. A parsed Google/network
failure is `network_STT_failure`. Either makes the retained trial invalid. An
STT-dependent trial with no production recognition evidence by the predefined
bounded wait is also invalid and stops/postpones that block. Only an incorrect
response on a technically valid trial is `participant_error`. Analysis excludes
invalid trials and never silently counts them as participant mistakes.

Preflight reports are outside participant data and contain neither the
researcher test transcript nor an audio copy. Raw participant folders,
preflight logs, HELP recordings/provenance, and generated results are gitignored.
`--resume` accepts only an incomplete session with an identical protocol lock;
`--force` moves an existing folder to a timestamped backup before starting.

## Questionnaire

The runner collects the standard ten System Usability Scale items and computes
the conventional 0–100 SUS score. It separately collects seven 1–5 project
items covering subtitle readability, label clarity, alert noticeability,
environmental awareness, distraction, usefulness, and intended use. It never
combines these project items into a fabricated standardized score.

Run the tooling tests with:

```powershell
python -m unittest -v human_study.test_study_tooling
```
