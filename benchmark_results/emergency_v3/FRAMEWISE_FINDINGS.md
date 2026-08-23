# Framewise findings

EfficientSED's 40 ms outputs materially help transient events. At the frozen 0.05 threshold, correct-class evidence was found in all 40 glass-breaking clips and all 40 vehicle-horn clips. Three glass clips and four horn clips had correct transient frames while their temporal mean remained below threshold; clip averaging would have suppressed those detections.

| Category | N | Correct class detected | Mean maximum probability | Mean longest duration above threshold |
| --- | ---: | ---: | ---: | ---: |
| glass_breaking | 40 | 40 | 0.6647 | 1.761 s |
| vehicle_horn | 40 | 40 | 0.6523 | 2.416 s |
| fire | 40 | 34 | 0.2498 | 4.018 s |
| siren | 40 | 40 | 0.5490 | 4.828 s |
| baby_crying | 40 | 39 | 0.5473 | 3.761 s |

The detailed per-clip maximum, maximum timestamp, temporal mean, frame count, and longest run are in framewise_analysis.csv. Compressed raw approved-label probabilities and top-five predictions for every frame are in efficientsed_only/raw_frame_predictions.jsonl.gz.
