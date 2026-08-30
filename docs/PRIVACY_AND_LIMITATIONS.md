# Privacy, Safety, and Publication Boundaries

## Audio and external services

Microphone modes can capture private conversations. Google Vietnamese speech recognition requires Internet access and transfers speech audio to an external provider. Operators must understand the provider terms, obtain appropriate consent, minimize capture, and avoid retaining recordings unless a documented protocol requires it.

Do not commit raw recordings, local utterance dumps, participant identifiers, consent records, credentials, API tokens, machine-specific paths, model caches, or weights without verified redistribution rights.

## Human-study data

`human_study/participant_data/`, generated results, preflight logs, HELP recordings, and consent-bearing records are local-only. Public documentation may describe the protocol, but it must not report earlier N=15 aggregate percentages, SUS scores, or reaction-time claims as verified without the participant-level archive, denominators, eligibility/consent provenance, and reproducible calculations.

Synthetic fixtures test analysis software. They are not observations from people.

## Safety claims

HearVis/HearSafe is a research prototype. It is not a medical device, certified alarm, accessibility guarantee, or deployment-ready safety system. Alerts may be missed or misclassified. The optional V3 external holdout has particularly weak fire recall. Users must not rely on the system as their sole means of detecting hazards or requesting help.

## Hardware and validation gaps

The evaluated system is laptop-assisted and uses a laptop microphone and network STT. Battery capacity/runtime, exact controller/display SKUs, acoustic-onset-to-visible latency, optical readability, long-term comfort, DHH-user effectiveness, sound localization, haptic output, and standalone operation are not established by the frozen evidence.

## Public-release checklist

Before committing or publishing, check tracked and untracked files for:

- `.env` files, tokens, credentials, keys, and service-account files;
- raw/private audio and participant-identifying data;
- consent forms containing names or signatures;
- local absolute paths and diagnostic logs;
- caches, virtual environments, checkpoints, and unredistributable model weights;
- temporary or synthetic outputs that could be mistaken for measured human evidence;
- draft manuscripts not yet frozen for submission.
