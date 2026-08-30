# Data leakage check

Exact duplicate audio groups: **0**.

- No exact duplicates among evaluated source clips.

No training lists are present in this repository. The bark filename follows ESC-50 naming, while CED uses an AudioSet-derived model; overlap with upstream model training cannot be independently ruled out.
The local speech transcript comes from the human-authored manifest, not Google STT output.
Personalization expectations are hand-authored from documented deterministic policy; the system under test did not generate its own truth.
Generated noise variants remain grouped with their source and are never treated as independent train/test samples.
