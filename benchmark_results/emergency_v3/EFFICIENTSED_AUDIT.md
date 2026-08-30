# EfficientSED repository audit

Audited repository state: https://github.com/theMoro/EfficientSED at commit e3449bfd1f4922b81d82410ed842653f462bfca8 (2025-06-12). The checkout was made only under the ignored experimental data directory.

| Item | Finding |
| --- | --- |
| Repository | theMoro/EfficientSED, a framewise sound-event-detection research repository derived from PretrainedSED |
| License | MIT; copyright 2024 Florian Schmid |
| Model architecture | Frame-MobileNet (MobileNetV3-derived CNN) plus linear strong/weak heads; the selected model has no sequence model |
| Pretrained dataset | AudioSet weak pretraining followed by AudioSet Strong training, according to the upstream model table/checkpoint naming |
| Label ontology | Ordered 447-class as_strong_train_classes AudioSet-style ontology |
| Input sample rate | 16 kHz mono waveform |
| Input duration | Upstream inference splits/pads to 10-second segments (160,000 samples) |
| Preprocessing | Pre-emphasis then 128-bin log-mel STFT; 400-sample window, 160-sample hop, 512 FFT |
| Output tensor | Strong logits shaped (batch, 447, 250) for a 10-second segment; sigmoid is applied for probabilities |
| Frame resolution | 40 ms (250 frames per 10 seconds); padded frames beyond each clip's true duration are discarded |
| Selected checkpoint | fmn10_strong.pt from upstream release v0.0.1 |
| Checkpoint integrity | SHA-256 92a5c1dc5aa21620e262420c4cc657d2f699bdfa20a748de506c354a2fe2770a |
| Parameters | 3,830,798 |
| Model size | 15,538,778 bytes on disk |
| Upstream dependencies | PyTorch plus a broad research stack; Mamba is optional in the README but imported unconditionally by the generic prediction wrapper |
| Isolated dependencies used | Python 3.12.10; torch 2.13.0+cpu; torchvision 0.28.0+cpu; torchaudio 2.11.0+cpu; librosa 1.0.0; numpy 2.5.2; scipy 1.18.0 |
| CPU support | Confirmed on Windows CPU |
| GPU support | Upstream inference supports CUDA; no CUDA device was available locally, so GPU latency/VRAM were not measured |
| Integration risks | Research dependency breadth, about 386 MB warm worker RAM, about 4.2 s initialization, AudioSet/ESC-50 overlap risk, and an experimental threshold calibrated on ESC-50 |

## Selection decision

fmn10_strong was selected before scoring because it is the requested/preferred plain framewise checkpoint, exists in the repository's exact checkpoint table, avoids optional sequence-model dependencies, and is substantially smaller than the wider variants. No other checkpoint or competing repository was benchmarked.

## Isolation decision

The main SoundGuard environment was recorded before installation and remained byte-for-byte identical in the before/after dependency inventories. EfficientSED runs in benchmark_data/external/efficientsed_venv through a persistent subprocess. The adapter imports the upstream MobileNet and mel implementation, recreates only the two linear checkpoint heads, and applies the upstream checkpoint key-remapping rules. It does not copy the repository into production.
