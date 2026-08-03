# SoundGuard fixed-file benchmark v1

This directory is extended in place. Audio files are never rewritten. Describe every evaluated file in `manifest.csv`; the runner does not infer or invent ground truth. Relative audio paths are resolved from the SoundGuard project root (the parent of `benchmark_v1`), so `samples/test.wav` means `<project>/samples/test.wav`, not `<project>/benchmark_v1/samples/test.wav`. Paths that escape the project root are rejected unless `--allow-external-paths` is explicitly supplied. `--project-root` can override the default when the benchmark directory is stored elsewhere.

## Run

Run software tests first:

```powershell
.\run_software_tests.ps1
```

Run the offline benchmark without external STT:

```powershell
python benchmark_runner.py --run-id baseline_01
```

Google STT is called only with explicit consent:

```powershell
python benchmark_runner.py --run-id baseline_stt_01 --enable-stt
```

Then generate the report:

```powershell
python benchmark_report.py --run-id baseline_stt_01
```

Supplying `--run-id` produces an official single-run report. Omitting it produces an explicitly labeled aggregate historical report and lists every included run ID.

The runner never opens a microphone. CED receives the original raw WAV path. The independent speech branch uses the source directly only when it is already mono 16 kHz; otherwise it creates a temporary mono 16 kHz WAV for VAD, DTLN, and STT. DTLN writes a second temporary speech-only file and neither temporary path can reach CED. Temporary files are cleaned after the row, and the source is never rewritten. `noisy_speech` rows automatically run paired `dtln_enabled` and `dtln_disabled` conditions with the same `sample_id` and `run_id`.

## Manifest schema

`sample_id` and `relative_path` are required. `scenario=noisy_speech` enables paired DTLN conditions. `sequence_id` groups deterministic emergency cycles, `cycle_index` orders them, and `reset_emergency_before=true` resets the condition-specific `EmergencySystem`. Blank `sequence_id` makes a sample independent. Context defaults to `neutral`.

Ground-truth and expectation columns may be blank. Blank truth is excluded from its metric denominator. `ground_truth_category` should use the canonical EmergencySystem category. Boolean expectations accept true/false, yes/no, or 1/0.

## Primary results schema

`results.csv` has exactly one row per `sample_id × condition × run_id`. It contains copied manifest metadata, source hash and duration, speech normalization details, per-stage status, VAD output, raw-CED output, optional final STT transcript, explicit STT/DTLN enablement, DTLN real-time factor, independent sound and transcript emergency results, stage/total latencies, status, error text, and completion time. Completed keys are skipped on resume. Metrics use each stage's status independently: a failed VAD or STT branch does not discard a successful CED result.

Each model-backed stage records `*_cold_start`. Its first successful call is stored in `*_model_load_or_cold_start_latency_seconds`; later successful calls are stored in `*_inference_latency_seconds`. The original stage and total end-to-end latency fields remain for startup-cost analysis. DTLN steady-state RTF is reported separately as `dtln_inference_realtime_factor`. Historical rows created before these fields remain present with blank values rather than being reinterpreted.

`events.csv` is an optional diagnostic log. Its rows are not samples and are never used as metric denominators. It records evidence type/source, sequence metadata, completion time, and emitted emergency state. Fixed files do not have microphone capture timestamps, so those fields remain blank.

Before the original header-only `results.csv` schema was changed, it was copied to a timestamped `results.pre_schema_*.csv` file. Future schema migrations also back up before atomically replacing the CSV and preserve populated rows.

## Limitations

This fixed-file suite measures repeatable model/pipeline quality, not microphone devices, real-time queues, or environmental drift. Those live-stream mechanics are covered separately by `test_audio_pipeline.py`. A small hand-selected dataset does not prove real-world accuracy.
