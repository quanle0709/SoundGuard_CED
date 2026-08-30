# Expanded benchmark datasets

All acquisition is reproducible with `python benchmark/download_datasets.py`. Downloaded binaries live under the gitignored `benchmark_data/external/` directory.

| Dataset | Source | License | Downloaded portion | Selected classes / environments | N | Evaluation role | Leakage status |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| ESC-50 | https://github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 | All official clips from seven mapped classes | siren, car_horn, dog, crying_baby, glass_breaking, crackling_fire, door_wood_knock | 280 | CED, emergency, robustness, DTLN/CED | POSSIBLE SOURCE OVERLAP with AudioSet; no local fine-tuning found |
| VIVOS | https://zenodo.org/records/7068130 | CC BY-NC-SA 4.0 | 100 official test utterances selected deterministically round-robin by speaker | Vietnamese read speech, 19 test speakers | 100 | clean/noisy Google Vietnamese STT | Provider-training membership undisclosed |
| DEMAND | https://zenodo.org/records/1227120 | CC BY-SA 3.0 | One 16 kHz channel from four environments | traffic, cafeteria, office, home/living | 4 | real environmental noise mixing | CED: no known overlap; DTLN upstream DNS membership cannot be reconstructed |

ESC-50 selection is class-complete (40 clips per mapped class), fixed before prediction, and uses official metadata. VIVOS selection is deterministic and transcript-grounded. No generated transcript or model-derived ground truth is used.
