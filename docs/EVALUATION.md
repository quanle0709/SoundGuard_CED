# Evaluation and Evidence Boundaries

HearVis reports results by configuration, dataset, and measurement endpoint. Values from different layers must not be collapsed into one accuracy, latency, or reliability claim.

## Configuration matrix

| Configuration | CED-Tiny | Google vi-VN STT | EfficientSED | Wearable role |
| --- | --- | --- | --- | --- |
| Default/deployed V2 | On | On for speech | Off | Validate frames and render OLED |
| Optional combined V3 | On | On for speech | Explicit opt-in specialist | Validate frames and render OLED |

## Retained evidence

- `benchmark_results/system_computer_only_20260821/final/quick_summary.json`: host CED and final-STT preparation endpoints.
- `benchmark_results/system_computer_only_20260821/`: bounded local software transport, resource, and soak campaigns.
- `benchmark_results/emergency_v3/` and `benchmark_results/emergency_v3_final/`: optional-specialist configuration, regression, and provenance records.
- `firmware/` and HUD contract tests: display protocol and source-level rendering behavior.
- `docs/images/hearvis-optional-v3-model-results.png`: derived external-holdout summary supplied for the public evidence packet.
- `docs/images/hearvis-endpoint-timing-results.png`: endpoint-separated host and physical display timing summary.

## Interpretation rules

- CED metrics describe a named window-level audio-tagging configuration, not event onset/offset or localization accuracy.
- Optional V3 results do not describe default V2 or ESP32 performance.
- Host CED timing starts with prerecorded audio available to the application.
- Final STT timing starts after the utterance is finalized and is network-dependent.
- Physical timing begins after a host decision or frame is available and ends at firmware `display()` completion.
- None of these measurements is acoustic-onset-to-visible latency.
- A finite soak with zero tracked failures is not proof of universal reliability.
- Weak fire recall on the optional V3 external holdout remains a visible limitation.

## Human-facing evidence

The repository retains a study protocol and research tooling, but public quantitative human outcomes are excluded. The local evidence does not contain a sufficiently auditable participant-level archive, denominators, eligibility/consent provenance, and calculation trail for the earlier N=15 aggregate claims. Treat Layer 3 as planned/preliminary until those records can be reviewed under an appropriate privacy and governance process.

Synthetic study fixtures are software-test data only and must never be presented as participant evidence.

## Safe verification commands

```powershell
.\run_software_tests.ps1
python -m pytest -q
pio run -d firmware -e esp32c3
pio run -d firmware -e esp32dev
```

These commands do not authorize Google STT calls, microphone capture, hardware flashing, or downloading private/external datasets. Run those checks only with explicit consent, suitable hardware, and data redistribution/privacy review.
