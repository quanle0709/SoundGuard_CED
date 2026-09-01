# Repository Layout

SoundGuard_CED intentionally keeps its compatibility-sensitive host runtime as a small set of root-level Python modules. The evaluated commands, research manifests, and historical evidence refer to those module names. Moving them into an install-only package would add risk without changing the laptop-assisted architecture.

## Current code and entry points

| Path | Ownership |
| --- | --- |
| `app.py` | Primary host CLI and supported runtime entry point. |
| `audio_*.py`, `streaming_audio.py` | Capture and pipeline orchestration. |
| `sound_classifier.py`, `sound_taxonomy.py` | Default CED-Tiny inference and label handling. |
| `speech_*.py`, `live_speech_to_text.py`, `voice_activity_detector.py` | VAD, optional DTLN enhancement, and online Vietnamese STT. |
| `emergency_system.py`, `emergency_v3.py`, `fusion_engine.py`, `alert_mapper.py` | Default rules and optional specialist integration. |
| `display_transport.py`, `hud_awareness.py` | Host display policy and serial framing. |
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
| `benchmark_v1/` | Legacy fixed-file benchmark documentation and workflow retained for provenance. |
| `human_study/` | Study protocol and tooling. Participant-level data and local derived results are excluded from Git. |
| `docs/` | Durable architecture, evaluation, privacy, attribution, and public-image documentation. |
| `external/` | Third-party attribution and locally supplied dependencies; model weights are not distributed. |

The root `SOUNDGUARD_BENCHMARK_AUDIT_3_LAYER.md` and `SOUNDGUARD_LAYER2_COMPUTER_ONLY_FINAL.md` reports remain at their historical paths because they are research records rather than general project documentation. Paths embedded in frozen manifests and reports describe the repository state used for those experiments and are not claims about the reorganized tree.

## Local-only material

The ignore rules retain virtual environments, caches, model weights, downloaded/generated benchmark data, private audio, participant records, scratch output, and local recovery snapshots outside the public tree. A file being ignored does not mean it is safe to delete; local research evidence and user work must be reviewed independently.

## Packaging decision

The repository is intentionally dependency-file driven (`requirements.txt` and `requirements-dev.txt`) rather than installable as a Python distribution. Its flat runtime modules, local model paths, firmware, and evidence workflows are designed to run from a clone with `python app.py`. A package migration can be reconsidered after the research configuration is frozen, but it should preserve the existing CLI and historical provenance.
