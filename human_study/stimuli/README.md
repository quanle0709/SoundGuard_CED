# Layer 3 stimulus preparation

The manifest references the project’s existing local ESC-50 and VIVOS files;
it does not copy or redistribute them. Run the project dataset preparation
before the pilot if those gitignored files are absent.

Two local HELP recordings must be prepared before the pilot:

- `help_A.wav`: a natural Vietnamese request containing the frozen HELP phrase.
- `help_B.wav`: a different natural recording of the same semantic request.

Use one researcher (not a participant), a quiet room, the study microphone,
mono PCM WAV, and no dangerous enactment. Keep each clip approximately two
seconds. Do not tune wording after main collection starts. Record the exact
speaker role, scripted wording, date, equipment, sample rate, and license/owner
permission in `RECORDING_PROVENANCE.md`. These files are deliberately absent;
the runner’s preflight will fail until they and their provenance record exist.

All stimuli are played at the same locked OS/application volume established in
the pilot. “Standard” is a protocol category, not a calibrated dB SPL claim.
Measure and record actual SPL if suitable equipment is available; otherwise
record the exact OS and speaker settings and do not report an acoustic dB value.
