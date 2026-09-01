# Repository Layout

SoundGuard_CED keeps one compatibility launcher at the root while the host implementation lives in the importable `soundguard/` package. A clone can still run `python app.py` without installing the repository itself.

## Current code and entry points

| Path | Ownership |
| --- | --- |
| `app.py` | Thin compatibility launcher and supported runtime command. |
| `soundguard/app.py` | Host CLI implementation and application orchestration. |
| `soundguard/audio/` | Capture, bounded queues, streaming, and pipeline orchestration. |
| `soundguard/speech/` | VAD, optional DTLN enhancement, and online Vietnamese STT. |
| `soundguard/detection/` | CED-Tiny inference, taxonomy, and HUD-awareness policy. |
| `soundguard/emergency/` | Default rules, HELP, fusion, alert mapping, and optional specialist integration. |
| `soundguard/display/` | Host display policy and serial framing. |
| `personalization/` | Optional, post-recognition profile and priority logic. |
| `firmware/` | PlatformIO firmware for the ESP32/ESP32-C3 display controller. |
| `tests/` | Host unit tests and firmware source-contract tests. |
| `tools/` | Operator-invoked hardware and transport diagnostics. |

`run_software_tests.ps1` remains at the root because it is a documented compatibility entry point. Configuration files stay next to the subsystem that owns them: firmware configuration under `firmware/`, study configuration under `human_study/`, benchmark configuration under `benchmark/`, and personalization defaults under `personalization/`.

## Evaluation and research material

| Path | Ownership and handling rule |
| --- | --- |
| `benchmark/` | Current benchmark and evaluation code. Safe smoke tests must use temporary output locations. |
| `benchmark_data/` | Versioned manifests and mappings; large/downloaded or generated data remains local. |
| `benchmark_results/` | Historical research evidence. Do not rewrite values or normalize old paths. |
| `benchmark_v1/` | Legacy fixed-file benchmark implementation and documentation retained for provenance. |
| `human_study/` | Study protocol and tooling. Participant-level data and local derived results are excluded from Git. |
| `docs/` | Durable architecture, evaluation, privacy, attribution, research-record, and public-image documentation. |
| `external/` | Third-party attribution and locally supplied dependencies; model weights are not distributed. |

The historical [`SOUNDGUARD_BENCHMARK_AUDIT_3_LAYER.md`](research_records/SOUNDGUARD_BENCHMARK_AUDIT_3_LAYER.md) and [`SOUNDGUARD_LAYER2_COMPUTER_ONLY_FINAL.md`](research_records/SOUNDGUARD_LAYER2_COMPUTER_ONLY_FINAL.md) reports now live under `docs/research_records/`. They were moved without content changes. Frozen manifests and reports continue to retain their original root paths because those records describe the repository state used for the experiments, not the current layout.

## Local-only material

The ignore rules retain virtual environments, caches, model weights, downloaded/generated benchmark data, private audio, participant records, scratch output, and local recovery snapshots outside the public tree. A file being ignored does not mean it is safe to delete; local research evidence and user work must be reviewed independently.

## Packaging decision

The repository remains dependency-file driven (`requirements.txt` and `requirements-dev.txt`) rather than an installable distribution. The top-level `soundguard/` package is importable directly from a clone, so `python app.py` preserves the existing workflow without `sys.path` manipulation or an editable install.
