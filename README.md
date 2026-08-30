# HearVis

HearVis is a research prototype that turns Vietnamese speech and selected environmental sounds into a compact visual heads-up display. The evaluated ISIF 2026 configuration is laptop-assisted: the host performs capture, recognition, audio tagging, and priority policy; an ESP32-C3 controller validates serial frames and renders a 64 x 32 OLED.

> **Naming note:** **HearVis** is the current public-facing project name. **HearSafe Glasses** (or **HearSafe**) names the prototype evaluated for the ISIF 2026 manuscript. **SoundGuard** and `SoundGuard_CED` are retained only where they record development history, repository provenance, compatibility-sensitive paths, configuration variables, or protocol literals. The repository slug remains `quanle0709/SoundGuard_CED` until the ISIF submission is frozen.

![Historical HearSafe prototype assembly](docs/images/hearvis-prototype-overview.jpg)

*Historical HearSafe prototype assembly. The photograph documents the physical build but does not establish current operation.*

## Why We Built HearVis

Speech captions alone do not describe everything happening around a listener. HearVis explores whether a small visual interface can combine Vietnamese captions, selected environmental labels, and clearly preemptive alerts without pretending that a prototype is a certified safety or medical device.

The project emphasizes explicit system boundaries and reproducible, configuration-specific measurements. It does not publish a single generic "HearVis accuracy" score.

## What HearVis Does

| Capability | Current implementation | Important boundary |
| --- | --- | --- |
| Live Vietnamese captions | Silero VAD, utterance segmentation, optional DTLN enhancement, Google `vi-VN` recognition, partial/final handling | Google STT requires Internet and sends speech audio to an external service. |
| Environmental awareness | CED-Tiny window-level audio tagging, candidate ranking, filtering, and display policy | This is window-level classification, not verified event onset/offset detection. |
| Priority alerts | Deterministic priority and display preemption | Not certified emergency detection and not a safety guarantee. |
| HELP | Triggered only from an accepted final Vietnamese transcript | Separate from environmental classification. |
| Personalization | Opt-in, rule-based priority adjustment after recognition | No personalized model training, familiar-voice recognition, or acoustic retraining. |
| HUD | HOME, SUBTITLE, CED, and ALERT states over CRC-protected framed USB serial | The wearable performs frame validation and rendering, not AI inference. |

## Current ISIF 2026 Prototype

The frozen evaluation boundary is **HearSafe, default V2**:

- integrated laptop microphone array for audio input;
- CED-Tiny for environmental audio tagging;
- Silero VAD and optional DTLN on the speech branch;
- Google Speech Recognition configured for Vietnamese (`vi-VN`);
- host-side HELP, emergency, priority, personalization, and display policy;
- USB serial framing with CRC-16-CCITT;
- ESP32-C3 SuperMini development board and a 0.49-inch, 64 x 32, SSD1306-compatible I2C OLED.

EfficientSED is an optional combined V3 specialist. It is disabled by default, runs on the host, and was evaluated separately. No AI inference runs on the ESP32-C3 in the evaluated system.

## System Architecture

![Detailed laptop-assisted HearVis architecture](docs/images/hearvis-system-architecture-detailed.png)

*Detailed ISIF architecture: the laptop host performs audio processing, AI inference, and policy; the ESP32-C3 validates frames and renders the OLED.*

The microphone stream is copied into independent bounded queues. Raw audio windows feed CED-Tiny; the speech branch uses VAD, utterance boundaries, optional DTLN, and network-dependent Google STT. Accepted results enter a deterministic host policy before serialization. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module and protocol details.

## Prototype

![Historical prototype, controller, OLED, and wiring views](docs/images/hearvis-prototype-three-view-reference.jpg)

*Historical assembly, controller/OLED close-up, and wiring detail. These photographs document construction and do not prove current operation.*

The recorded mass measurements were 40.0 g, 41.5 g, and 40.5 g (N=3; mean approximately 40.7 g). Boot-to-HOME measurements from USB power were 10.0, 10.5, 10.2, 9.8, and 10.0 seconds (N=5; mean 10.1 seconds). Boot timing ends at the first HOME display; it is not application or recognition startup latency.

## Hardware

![ESP32-C3 controller and OLED close-up](docs/images/hearvis-controller-oled-closeup.jpg)

*ESP32-C3 SuperMini and 0.49-inch OLED close-up from the historical prototype.*

| Item | Evaluated/configured fact |
| --- | --- |
| Controller | ESP32-C3 SuperMini development board; PlatformIO target `esp32-c3-devkitm-1` does not establish an exact board SKU |
| Display | 0.49-inch, 64 x 32 monochrome, SSD1306-compatible OLED |
| Display bus | I2C address `0x3C`, SDA GPIO4, SCL GPIO5 in the ESP32-C3 build configuration |
| Host link | USB serial at 115200 bit/s with length and CRC validation |
| Audio input | Integrated host-laptop microphone array, not an eyewear-mounted microphone |
| Battery record | Reported protected 3.7 V Li-Po, approximately 800-1000 mAh; capacity and runtime were not independently verified |

The frozen evidence contains no implemented sound localization or haptic output.

## Software Pipeline

1. `streaming_audio.py` captures one 16 kHz mono stream and feeds bounded speech and environmental queues.
2. `sound_classifier.py` applies CED-Tiny to rolling audio windows and returns ranked tags.
3. `voice_activity_detector.py`, `live_speech_to_text.py`, and `speech_recognizer.py` form utterances and request Vietnamese recognition. `speech_enhancer.py` provides optional two-stage DTLN enhancement.
4. `emergency_system.py`, `fusion_engine.py`, and `alert_mapper.py` apply deterministic evidence and priority rules. Accepted final transcript text can trigger HELP.
5. `display_transport.py` normalizes text and sends framed, CRC-protected serial messages to the firmware.
6. `firmware/` validates frames, maintains display state, and renders the OLED.

The default/deployed V2 path uses CED-Tiny and leaves EfficientSED off. Optional V3 adds a narrowly scoped host-side specialist without replacing normal CED output.

## Display and Priority Logic

![HearVis HUD state reconstructions](docs/images/hearvis-hud-states-composite.png)

*Deterministic framebuffer reconstructions, not photographs of the physical OLED.*

![HearVis display-state lifecycle](docs/images/hearvis-display-state-lifecycle.png)

*Verified display-state lifecycle and alert preemption logic. No timing or safety guarantee is implied.*

ALERT has visual priority over SUBTITLE and CED; HOME is the idle state. A HELP event is visually dominant and originates from accepted final transcript content. The policy retains recent subtitle and environmental state so normal presentation can resume after an alert.

| HOME | SUBTITLE | CED | ALERT |
| --- | --- | --- | --- |
| ![HOME reconstruction](docs/images/hearvis-hud-home-reconstruction.png) | ![SUBTITLE reconstruction](docs/images/hearvis-hud-subtitle-reconstruction.png) | ![HORN reconstruction](docs/images/hearvis-hud-horn-reconstruction.png) | ![ALERT reconstruction](docs/images/hearvis-hud-alert-reconstruction.png) |

*Software framebuffer reconstructions: the HearSafe HOME image is a name-only publication reconstruction and was not flashed or physically retested; the other images demonstrate deterministic rendering, not recognition or emergency reliability.*

## Evaluation Strategy

Evidence is separated by layer and endpoint:

- software unit and contract tests check recognition handling, policy, transport, personalization, and HUD behavior;
- offline benchmarks report a named model/configuration and a fixed dataset denominator;
- host timing begins only after prerecorded input or a finalized utterance is available;
- firmware/display-call timing begins after the host decision or frame is available;
- physical checks document the controller/display path, boot-to-HOME, prototype mass, and bounded transport behavior;
- Layer 3 remains a planned/preliminary human-facing protocol for public reporting because the participant-level records, denominators, eligibility/consent provenance, and calculations are not locally auditable.

Details and artifact pointers are in [docs/EVALUATION.md](docs/EVALUATION.md).

## Measured Results

### Optional combined V3 external holdout

![Optional V3 model results](docs/images/hearvis-optional-v3-model-results.png)

*Optional/default-off combined V3 results on a derived external holdout. These are not default V2 or wearable performance.*

| Dataset/configuration | N | Target positive | Hard negative | Macro P / R / F1 | Emergency recall | Hard-negative clip FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Derived external holdout, optional combined V3 | 3,938 | 281 | 3,657 | 0.5368 / 0.7322 / 0.5749 | 0.5943 | 0.0566 |

Fire was the weakest reported class: 30/121 recall (0.2479). This result is a limitation, not evidence of deployment readiness. EfficientSED remains off by default.

### Timing and bounded operation

![Endpoint-specific HearVis timing results](docs/images/hearvis-endpoint-timing-results.png)

*Host and post-decision/display-call measurements use different endpoints. Neither is acoustic-onset-to-visible latency.*

| Measurement | N | Median | P95 | Boundary |
| --- | ---: | ---: | ---: | --- |
| Host CED preparation | 30 | 124.721 ms | 149.869 ms | Prerecorded input available to transport-ready frame |
| Final STT preparation | 20 | 1013.958 ms | 1373.304 ms | Finalized utterance available to transport-ready frame; network-dependent |
| Physical FINAL display-call completion | - | - | 15.139 ms | Accepted FINAL to firmware `display()` completion |
| Physical CED display-call completion | - | - | 8.824 ms | Accepted CED to firmware `display()` completion |
| Physical HELP display-call completion | - | - | 9.366 ms | HELP decision to ALERT `display()` completion |
| Physical alert preemption | - | - | 9.326 ms | Accepted alert preemption endpoint |

A finite soak campaign ran for 3,600.125 seconds with 16,449 transitions and zero failures in the tracked categories. This does not establish universal reliability or field performance.

## What We Physically Validated

- the historical laptop-assisted host-to-ESP32-C3-to-OLED path;
- CRC-protected frame receipt, state updates, and 64 x 32 rendering;
- HOME, subtitle, CED-label, ALERT, and HELP presentation behavior;
- post-decision/display-call timing for the named endpoints;
- five USB-power-to-first-HOME trials;
- three prototype mass measurements and worn placement documentation.

Physical checks do **not** establish acoustic-onset-to-visible latency, field detection accuracy, battery endurance, long-term comfort, optical readability for DHH users, standalone operation, or certified safety performance.

## Current Limitations

- The evaluated system needs a laptop; it is not standalone eyewear.
- Google STT requires Internet and transfers speech audio to an external provider.
- CED-Tiny produces window-level tags rather than verified event boundaries or localization.
- Optional V3 has weak fire recall on the reported external holdout and is disabled by default.
- Priority alerts are deterministic prototype behavior, not certified emergency detection.
- Battery capacity/runtime and exact component SKUs are not independently verified.
- No haptic output or sound-direction feature exists in the frozen evidence.
- No verified DHH-user effectiveness result is published under the current Layer 3 provenance state.
- The repository has no project-wide open-source license; third-party components retain their own licenses.

See [docs/PRIVACY_AND_LIMITATIONS.md](docs/PRIVACY_AND_LIMITATIONS.md) before recording audio or conducting a study.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `app.py` | Main file, microphone, continuous, live-STT, and HUD CLI |
| `audio_*.py`, `streaming_audio.py` | Capture, queues, preprocessing, and orchestration |
| `sound_classifier.py`, `sound_taxonomy.py` | CED-Tiny inference and label policy |
| `speech_*.py`, `live_speech_to_text.py`, `voice_activity_detector.py` | VAD, optional enhancement, and Vietnamese recognition |
| `emergency_system.py`, `emergency_v3.py`, `fusion_engine.py` | Default policy and optional V3 specialist integration |
| `display_transport.py`, `hud_awareness.py` | Host-side serial framing and display policy |
| `firmware/` | PlatformIO ESP32/ESP32-C3 OLED firmware |
| `personalization/` | Opt-in post-recognition rule/profile adapter |
| `benchmark/`, `benchmark_results/` | Reproducible benchmark code and retained evidence |
| `human_study/` | Protocol and local study tooling; private/raw results excluded |
| `docs/` | Architecture, evaluation, privacy boundaries, and public images |

## Installation

Windows PowerShell is the documented host environment. Python package and model compatibility can vary; use an isolated environment.

```powershell
git clone https://github.com/quanle0709/SoundGuard_CED.git
cd SoundGuard_CED
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CED-Tiny and Silero VAD may download on first use. DTLN weights are intentionally not distributed in this repository; place authorized upstream `model_1.tflite` and `model_2.tflite` files under `external/DTLN-master/pretrained_model/` before enabling DTLN. Do not commit model weights.

For the wearable firmware, install PlatformIO and build the evaluated ESP32-C3 configuration:

```powershell
pio run -d firmware -e esp32c3
```

Confirm the actual wiring and serial port before upload. The `esp32c3` build target is a toolchain target, not proof of the exact physical board SKU.

## Running HearVis / HearSafe

List audio devices and inspect all options:

```powershell
python app.py --list-devices
python app.py --help
```

Run a local WAV without DTLN, or use the microphone paths:

```powershell
python app.py .\path\to\audio.wav --no-dtln
python app.py --mic --duration 8 --no-dtln
python app.py --continuous --duration 5 --no-dtln
python app.py --live-stt --end-silence-ms 700 --post-roll-ms 500 --no-dtln
```

Send deterministic HUD test frames, then run live captions with the display:

```powershell
python app.py --hud-port COM5 --hud-test
python app.py --live-stt --hud-port COM5 --no-dtln
```

Only enable optional V3 when its isolated dependencies and audited checkpoint are available:

```powershell
$env:SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING = '1'
python app.py --mic --duration 5 --no-dtln --emergency-v3
```

Environment-variable and protocol names containing `SOUNDGUARD` are preserved for compatibility and provenance.

## Testing and Benchmarks

Run the lightweight offline software checks:

```powershell
.\run_software_tests.ps1
```

Run the broader local suite without deliberately invoking microphone capture or Google STT:

```powershell
python -m pytest -q
```

Build both maintained firmware environments:

```powershell
pio run -d firmware -e esp32c3
pio run -d firmware -e esp32dev
```

Network STT, private/local audio, external dataset downloads, hardware upload, and live microphone tests require explicit operator consent and suitable hardware/data. Benchmark commands and evidence boundaries are documented in [docs/EVALUATION.md](docs/EVALUATION.md).

## Research Paper

The ISIF 2026 manuscript documents the laptop-assisted **HearSafe** configuration. The current draft is intentionally not published. After submission is frozen, link the final frozen/submitted manuscript at:

`docs/paper/HearSafe_ISIF_2026_Full_Paper.pdf`

The manuscript must not be described as peer-reviewed, published, or accepted unless that status later becomes true.

## Future Development

Future HearVis work may explore a standalone compute path, on-device or privacy-preserving speech recognition, improved fire-sound evidence, optical/ergonomic iteration, battery characterization, and auditable studies with DHH participants. These are development directions, not current capabilities.

![Team-member worn fit check](docs/images/hearvis-worn-fit-check.jpg)

*Team-member fit check with the historical prototype. The wearer is a project team member, not a DHH study participant; this image documents worn placement only.*

## Third-Party Components and Attribution

- CED-Tiny model ID: `mispeech/ced-tiny`; model terms apply separately.
- Silero VAD: loaded through the `silero-vad` package under its upstream terms.
- DTLN: third-party work by Nils L. Westhausen; the retained upstream license is at `external/DTLN-master/LICENSE`.
- EfficientSED: optional/default-off experimental specialist; see `THIRD_PARTY_EFFICIENTSED.md` and its upstream license/terms.
- Google Speech Recognition: external, network-dependent service accessed through the Python SpeechRecognition package; provider terms and privacy requirements apply.
- Adafruit GFX, Adafruit SSD1306, and U8g2 for Adafruit GFX: firmware libraries installed by PlatformIO under their respective licenses.

No repository-wide license is currently granted for the original HearVis/HearSafe/SoundGuard code. Contact the project team before reuse.

## Contributors

HearVis / HearSafe project team. Individual development contributions and historical SoundGuard provenance are preserved in the Git history.

## Project Status

**Research prototype - active development.** The laptop-assisted HearSafe evaluation configuration is frozen for ISIF 2026 evidence reporting. This repository is not a medical device, is not certified for safety use, and is not deployment-ready.
