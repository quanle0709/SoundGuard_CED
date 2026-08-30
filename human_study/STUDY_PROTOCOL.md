# SoundGuard Layer 3 study protocol

Protocol lock: **L3_PROTOCOL_V1**  
Status: **PLATFORM READY — HUMAN DATA PENDING**

Do not collect main-study data until the pilot, local ethics/safeguarding
requirements, stimulus preparation, physical setup, and mandatory preflight
have all passed. The evaluated prototype uses its existing network-dependent
Google Speech Recognition implementation for live speech-to-text. It is not a
fully offline prototype.

## Design

This is a within-subject study. Every participant completes WITHOUT and WITH
SoundGuard while keeping their usual hearing assistance constant. P01–P10 are
anonymous main IDs. Odd IDs receive WITHOUT then WITH; even IDs receive WITH
then WITHOUT. A four-sequence schedule independently assigns matched stimulus
sets:

| IDs modulo four | First condition/set | Second condition/set |
|---|---|---|
| P01 pattern | WITHOUT/A | WITH/B |
| P02 pattern | WITH/A | WITHOUT/B |
| P03 pattern | WITHOUT/B | WITH/A |
| P04 pattern | WITH/B | WITHOUT/A |

The pattern repeats for later IDs. Trial order is reproducibly randomized
inside every condition/block from participant ID + base seed + study version.

Each participant completes 38 scored trials: speech 12 (6 per condition),
environmental 12, mixed 8, and critical/HELP 6. Set A and Set B are matched but
non-identical. The manifest is frozen and hashed into each session.

## Before recruitment

- Obtain required ethics, school/competition, privacy, and safeguarding review.
- If minors may participate, use the guardian form and required assent process.
- Record the two HELP stimuli and complete their provenance file.
- Prepare the existing licensed VIVOS and ESC-50 assets.
- Pilot instructions, controls, safe volume, timing, saved data, and recovery.
- Lock the same git commit, configuration, manifest, product command, and setup
  for P01–P10. A material change requires a new protocol version.

## Physical setup record

Complete before every session; use the same values whenever possible.

- Laptop asset ID/model: ____________________
- Speaker asset ID/model: ____________________
- Microphone asset ID/model/input index: ____________________
- SoundGuard ESP32 asset ID/COM port: ____________________
- OLED hardware: ____________________
- Software git commit: ____________________
- OS volume setting: ____________________
- Application volume setting: ____________________
- Measured SPL and method (if available): ____________________
- Speaker-to-participant distance: ____________________
- Speaker position/angle: ____________________
- Microphone position/distance/orientation: ____________________
- OLED position/distance/orientation: ____________________
- Participant seated position: ____________________
- Room and ambient condition: ____________________
- Participant’s usual hearing device in use: YES / NO / N/A
- Setup deviations and reason: ____________________

Do not ask a participant to remove normal hearing assistance. Keep it constant
across WITHOUT and WITH.

## Mandatory preflight

Run preflight immediately before a participant session. It is researcher-only
and is saved outside participant data. It checks:

1. required stimulus files and protocol hashes;
2. network connectivity;
3. real Google STT recognition of a short researcher test recording;
4. ESP32 COM connection;
5. visual confirmation of a subtitle on the HUD;
6. visual confirmation of a noncritical CED label;
7. visual confirmation of an ALERT; and
8. clearing the test state.

The researcher test must contain no participant or sensitive speech. Preflight
does not save its audio or transcript. A failed check produces PREFLIGHT FAIL.
Do not begin the session. If Google STT becomes unavailable during speech or
mixed trials, do not substitute an engine or retry indefinitely; retain the
trial as `network_STT_failure`, set `trial_valid=false`, and postpone/abort the
affected block according to the session record.

## Trial operation

Condition WITHOUT runs with the SoundGuard process stopped. Condition WITH runs
the frozen command recorded in `session.json`. The researcher confirms the
correct state at each transition. The runner plays the locked stimulus and uses
`time.perf_counter()` for onset and response timing. RT from stimulus is the
participant response time minus host playback-start time. RT from HUD is only
calculated from an actual production HUD-send timestamp where parsable. It is
software/event timing, not visual perception latency.

The participant UI uses large multiple-choice controls. Mixed trials collect
separate speech and environmental responses. For WITH trials, the researcher
records what physically appeared on the OLED; this observation is not silently
inferred. Production log evidence is retained in `runtime.log`.

Technical failure policy: playback, hardware, or Google/network STT failure
makes the affected trial invalid. The row remains in raw data with the failure
type and must not be scored as participant error. Participant mistakes remain
valid and use `participant_error` only when the response is incorrect.

## Questionnaire and ending

After WITH SoundGuard, collect the standard ten SUS items (1 strongly disagree
to 5 strongly agree) and seven separate project-specific Likert items. Compute
SUS using the standard odd/even contribution rule and multiply by 2.5. Never
combine project items into a purported standardized score.

Stop immediately if requested. Mark the session incomplete without deleting or
repairing observations. Validate the resulting folder. Completed folders are
not overwritten without explicit `--force`; preserve the original folder before
any forced rerun.

