# HearVis Architecture

## Evaluated boundary

The ISIF 2026 HearSafe prototype is laptop-assisted. The laptop captures audio, performs all model inference and policy decisions, and serializes display frames. The wearable controller validates those frames and renders the OLED. It does not run CED, VAD, DTLN, STT, or emergency inference.

```text
Integrated laptop microphone (16 kHz mono)
  -> capture hub and bounded queues
     -> raw windows -> CED-Tiny -> environmental candidates --+
     -> speech frames -> Silero VAD -> optional DTLN            |
        -> Google vi-VN STT -> partial/final transcript --------+
  -> deterministic host HELP/emergency/priority/display policy
  -> framed USB serial (length + CRC-16-CCITT, 115200 bit/s)
  -> ESP32-C3 frame validation and display state
  -> SSD1306-compatible 64 x 32 OLED
```

## Default V2 and optional V3

Default V2 uses CED-Tiny for environmental tagging. Optional combined V3 adds an EfficientSED specialist for a limited safety taxonomy. V3 is disabled by default, runs on the laptop, preserves ordinary CED output, and falls back to V2 if its isolated worker is unavailable.

## Display messages

The compatibility-sensitive serial protocol uses distinct frame types for subtitle, partial subtitle, CED, alert/emergency, clock, and benchmark traffic. Text is NFC-normalized on the host. Firmware accepts complete CRC-valid frames, updates retained HOME/SUBTITLE/CED/ALERT state, and commits the framebuffer.

Priority is deterministic: ALERT preempts SUBTITLE and CED, while HOME is the idle state. HELP is produced from an accepted final Vietnamese transcript, not from environmental classification.

## Key implementation paths

| Concern | Files |
| --- | --- |
| Capture and queues | `audio_capture.py`, `streaming_audio.py`, `audio_pipeline.py` |
| Environmental tagging | `sound_classifier.py`, `sound_taxonomy.py` |
| Speech branch | `voice_activity_detector.py`, `speech_enhancer.py`, `live_speech_to_text.py`, `speech_recognizer.py` |
| Policy | `emergency_system.py`, `fusion_engine.py`, `alert_mapper.py`, `hud_awareness.py` |
| Optional specialist | `emergency_v3.py`, `benchmark/experiments/emergency_v3/` |
| Host transport | `display_transport.py` |
| Wearable firmware | `firmware/src/`, `firmware/platformio.ini` |

## Non-capabilities

The frozen system has no implemented localization, haptic output, eyewear microphone, standalone AI compute, offline Vietnamese STT, or verified battery runtime. Future architecture proposals must not be described as evaluated features.
