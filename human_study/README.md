# HearSafe Layer 3 human-facing protocol

Current public status: **planned/preliminary protocol; quantitative outcomes excluded**.

This directory contains research tooling and protocol materials for evaluating the laptop-assisted HearSafe prototype. It does not modify the frozen CED/STT behavior, HELP/emergency policy, transport, or OLED firmware.

Earlier N=15 aggregate percentages, SUS values, and reaction-time claims are not publishable as verified results from the current repository state. The local evidence does not include a sufficiently auditable participant-level archive, denominators, eligibility/consent provenance, and reproducible calculation trail. Do not reconstruct missing records or treat synthetic fixtures as participant evidence.

## Before any participant session

- obtain and document the required ethics, safeguarding, eligibility, and consent approvals;
- confirm the final protocol, recruitment target, inclusion/exclusion rules, and amendment history;
- prepare only redistributable stimuli with complete provenance;
- validate the laptop microphone, audio output, network-dependent Google `vi-VN` STT, ESP32-C3 link, and physical OLED;
- keep participant identifiers, raw records, preflight logs, recordings, and consent materials outside Git.

## Researcher-only commands

Preflight:

```powershell
python human_study/run_study.py --preflight-only --hud-port auto --researcher-test-wav path\to\researcher_test.wav
```

Pilot:

```powershell
python human_study/run_study.py --pilot --participant PILOT01 --hud-port auto --researcher-test-wav path\to\researcher_test.wav
```

Validate a local session:

```powershell
python human_study/validate_session.py human_study/participant_data/main/P01
```

Run study-tooling tests without representing fixtures as evidence:

```powershell
python -m unittest human_study.test_study_tooling
```

## Privacy boundary

Google STT sends speech audio to an external provider. A microphone session can capture private speech. The study operator is responsible for informed consent, data minimization, retention, access control, and provider/privacy compliance.

The `.gitignore` excludes participant data, generated human-result outputs, preflight logs, local HELP recordings, and researcher test audio. Preserve those exclusions. See [`../docs/PRIVACY_AND_LIMITATIONS.md`](../docs/PRIVACY_AND_LIMITATIONS.md).
