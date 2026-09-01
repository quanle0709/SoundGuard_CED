# EfficientSED third-party notice

SoundGuard Emergency V3 optionally uses the separately downloaded EfficientSED project and its fmn10_strong checkpoint.

- Upstream: https://github.com/theMoro/EfficientSED
- Audited commit: e3449bfd1f4922b81d82410ed842653f462bfca8
- License: MIT, copyright 2024 Florian Schmid
- Checkpoint: fmn10_strong.pt, release v0.0.1
- Checkpoint SHA-256: 92a5c1dc5aa21620e262420c4cc657d2f699bdfa20a748de506c354a2fe2770a

No EfficientSED source is vendored into production. The ignored checkout and checkpoint live under benchmark_data/external/efficientsed_repo; the ignored isolated environment lives under benchmark_data/external/efficientsed_venv. SoundGuard's adapter imports the upstream architecture at runtime and uses a minimal compatible strong-head loader to avoid the upstream inference module's unrelated optional Mamba/transformer dependencies.

The upstream MIT license remains in the cloned repository at benchmark_data/external/efficientsed_repo/LICENSE.
