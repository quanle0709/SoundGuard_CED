import argparse
import datetime
import tempfile
import time
from pathlib import Path

from soundguard.emergency.alert_mapper import map_alert
from soundguard.audio.audio_capture import (
    cleanup_temp_recordings,
    create_speech_optimized_wav,
    get_peak_amplitude,
    list_input_devices,
    record_audio,
)
from soundguard.emergency.fusion_engine import fuse_result
from soundguard.display.display_transport import (
    HUDTransport,
    get_alert_display_state,
    get_screen2_ced_label,
)
from soundguard.emergency.emergency_system import CONTEXTS, EmergencySystem
from soundguard.emergency.emergency_v3 import (
    create_emergency_v3_specialist,
    select_emergency_evidence,
)
from personalization import PriorityAdapter, ProfileManager
from soundguard.detection.sound_classifier import classify_audio_file
from soundguard.speech.speech_enhancer import enhance_audio_file
from soundguard.speech.speech_recognizer import transcribe_audio_file
from soundguard.speech.voice_activity_detector import detect_speech


DTLN_TEMP_PATH = (
    Path(tempfile.gettempdir()) / "soundguard_dtln_recording.wav"
)

HUD_TEST_SUBTITLES = (
    "Xin chào",
    "Tôi đang ở ngoài đường.",
    "Mẹ nói con nhớ mang áo mưa nhé.",
    "Có một chiếc xe đang đến gần.",
    "Cảnh báo: còi xe.",
    "CẢNH BÁO NGUY HIỂM",
    "Tiếng còi phát ra từ bên trái.",
    "ă â ê ô ơ ư Ă Â Ê Ô Ơ Ư đ Đ",
    "á à ả ã ạ ấ ầ ẩ ẫ ậ ế ề ể ễ ệ ố ồ ổ ỗ ộ ớ ờ ở ỡ ợ ứ ừ ử ữ ự",
)

# Temporary HUD debugging toggle. Disable after end-to-end CED validation.
SHOW_ALL_DETECTED_SOUNDS_ON_HUD = True


def cleanup_dtln_recording() -> None:
    if DTLN_TEMP_PATH.exists():
        try:
            DTLN_TEMP_PATH.unlink()
        except OSError:
            pass


def evaluate_voice_activity(
    raw_audio_path: str | Path,
    use_vad: bool,
    threshold: float,
) -> dict:
    if not use_vad:
        return {
            "status": "disabled",
            "has_speech": True,
            "segments": [],
            "audio_duration": 0.0,
            "speech_duration": 0.0,
            "speech_ratio": 0.0,
            "threshold": threshold,
        }

    try:
        result = detect_speech(
            raw_audio_path,
            threshold=threshold,
        )

        return {
            "status": "enabled",
            **result,
        }

    except Exception as exc:
        print(f"VAD warning: {exc}")
        print(
            "VAD fallback: vẫn chạy DTLN và "
            "Speech-to-Text để không mất lời nói."
        )

        return {
            "status": "fallback",
            "has_speech": True,
            "segments": [],
            "audio_duration": 0.0,
            "speech_duration": 0.0,
            "speech_ratio": 0.0,
            "threshold": threshold,
        }


def prepare_speech_audio(
    raw_audio_path: str | Path,
    use_dtln: bool,
) -> tuple[Path, Path | None, bool]:
    raw_path = Path(raw_audio_path)

    if not use_dtln:
        speech_path = create_speech_optimized_wav(raw_path)
        return speech_path, None, False

    try:
        denoised_path = Path(
            enhance_audio_file(
                input_wav=raw_path,
                output_wav=DTLN_TEMP_PATH,
            )
        )

        speech_path = create_speech_optimized_wav(
            denoised_path
        )

        return speech_path, denoised_path, True

    except Exception as exc:
        print(f"DTLN warning: {exc}")
        print(
            "DTLN fallback: sử dụng pipeline "
            "Speech-to-Text cũ."
        )

        speech_path = create_speech_optimized_wav(
            raw_path
        )

        return speech_path, None, False


def run_analysis(
    audio_path: str,
    speech_audio_path: str | None,
    run_speech_to_text: bool,
    emergency_system: EmergencySystem,
    hud: HUDTransport | None = None,
    emergency_v3_specialist=None,
) -> None:
    hud = hud or HUDTransport()
    print(f"Audio source: {audio_path}")

    transcript = ""

    if run_speech_to_text:
        try:
            transcript = transcribe_audio_file(
                speech_audio_path or audio_path
            )
        except Exception as exc:
            transcript = f"[speech error] {exc}"

        if transcript:
            print(f"Transcript: {transcript}")
            hud.set_subtitle(transcript)
        else:
            print("Transcript: [empty]")
    else:
        print(
            "Transcript: "
            "[skipped because VAD detected no speech]"
        )

    try:
        result = classify_audio_file(audio_path)

        detected_sound = result.get(
            "label",
            "unknown",
        )
        confidence = result.get(
            "confidence",
            0.0,
        )
        top_predictions = result.get(
            "top_predictions",
            [],
        )

    except Exception as exc:
        detected_sound = f"[CED error] {exc}"
        confidence = "N/A"
        top_predictions = []

    if isinstance(confidence, (int, float)):
        sound_confidence = float(confidence)
    else:
        sound_confidence = 0.0

    specialist_result = (
        emergency_v3_specialist.analyze_file(audio_path)
        if emergency_v3_specialist is not None
        else None
    )
    emergency_label, emergency_confidence, emergency_source = select_emergency_evidence(
        detected_sound, sound_confidence, specialist_result, emergency_system.context
    )

    alert_result = map_alert(
        detected_sound,
        sound_confidence,
        context=emergency_system.context,
    )

    fusion_result = fuse_result(
        datetime.datetime.now().isoformat(
            timespec="seconds"
        ),
        transcript,
        {
            "label": detected_sound,
            "confidence": sound_confidence,
        },
        alert_result,
    )

    emergency_result = emergency_system.evaluate(
        transcript=transcript,
        sound_label=emergency_label,
        sound_confidence=emergency_confidence,
        timestamp=fusion_result["timestamp"],
    )

    print(f"Detected sound: {detected_sound}")
    print(f"Confidence: {confidence}")
    if specialist_result is not None:
        print(
            "Emergency V3: "
            f"source={emergency_source}, detected={specialist_result.get('detected', False)}, "
            f"category={specialist_result.get('category', 'none')}, "
            f"score={float(specialist_result.get('score', 0.0)):.4f}"
        )

    vietnamese_sound = alert_result.get(
        "message_vi",
        "",
    )

    print(
        "Vietnamese sound: "
        f"{vietnamese_sound or '[none]'}"
    )
    print(
        "Help request: "
        f"{emergency_result['help_request_detected']}"
    )
    print(
        "Alert level: "
        f"{emergency_result['alert_level']}"
    )
    print(
        "Alert message: "
        f"{emergency_result['alert_text'] or '[none]'}"
    )
    active_sound = getattr(emergency_system, "active_sound", None)
    help_active = bool(getattr(emergency_system, "help_active", False))
    if not hasattr(emergency_system, "active_sound") and emergency_result["alert_text"]:
        category = emergency_result.get("category", "alert")
        if category == "help_request":
            help_active = True
        else:
            active_sound = category
    event_label, help_active = get_alert_display_state(
        active_sound, help_active
    )
    screen2_label = get_screen2_ced_label(detected_sound, emergency_result)
    if event_label or help_active:
        hud.set_alert_state(event_label, help_active)
    else:
        hud.set_alert_state()
    hud.set_environmental_sound(
        screen2_label if SHOW_ALL_DETECTED_SOUNDS_ON_HUD else ""
    )
    print(f"Context: {emergency_result['context']}")
    print(f"Event state: {emergency_result['event_state']}")
    print(f"Decision mode: {emergency_result['decision_mode']}")
    print(
        "Temporal confirmation: "
        f"{emergency_result['temporal_confirmation']}"
    )
    print(f"History window: {emergency_result['history_window']}")
    print(f"Missing cycles: {emergency_result['missing_cycles']}")
    print(
        "Emitted this cycle: "
        f"{emergency_result['emitted_this_cycle']}"
    )

    print("Top predictions:")

    if top_predictions:
        for item in top_predictions:
            print(
                f"  - {item['label']}: "
                f"{item['score']:.4f}"
            )
    else:
        print("  - None")


def print_vad_information(vad_result: dict) -> None:
    status = vad_result.get("status", "fallback")

    if status == "disabled":
        print("Silero VAD: DISABLED")
        return

    if status == "fallback":
        print("Silero VAD: FALLBACK")
        return

    print("Silero VAD: ENABLED")
    print(
        f"Speech detected: "
        f"{vad_result.get('has_speech', False)}"
    )
    print(
        "Speech duration: "
        f"{vad_result.get('speech_duration', 0.0):.3f} seconds"
    )
    print(
        "Speech ratio: "
        f"{vad_result.get('speech_ratio', 0.0):.3f}"
    )
    print(
        "Speech segments: "
        f"{len(vad_result.get('segments', []))}"
    )


def print_audio_information(
    raw_path: Path,
    speech_path: Path | None,
    denoised_path: Path | None,
    dtln_applied: bool,
) -> None:
    raw_peak = get_peak_amplitude(raw_path)

    print(f"Raw recording path: {raw_path}")
    print(f"Raw peak amplitude: {raw_peak:.6f}")

    if speech_path is None:
        print(
            "Speech processing: "
            "SKIPPED because no speech was detected"
        )
        print("DTLN denoising: SKIPPED")
        return

    processed_peak = get_peak_amplitude(
        speech_path
    )

    if dtln_applied and denoised_path is not None:
        print(
            "DTLN-denoised file path: "
            f"{denoised_path}"
        )
        print("DTLN denoising: ENABLED")
    else:
        print("DTLN denoising: DISABLED")

    print(
        "Speech-optimized file path: "
        f"{speech_path}"
    )
    print(
        "Processed peak amplitude: "
        f"{processed_peak:.6f}"
    )


def process_audio(
    raw_path: Path,
    use_vad: bool,
    vad_threshold: float,
    use_dtln: bool,
    emergency_system: EmergencySystem,
    hud: HUDTransport | None = None,
    emergency_v3_specialist=None,
) -> None:
    vad_result = evaluate_voice_activity(
        raw_audio_path=raw_path,
        use_vad=use_vad,
        threshold=vad_threshold,
    )

    print_vad_information(vad_result)

    has_speech = bool(
        vad_result.get("has_speech", True)
    )

    speech_path: Path | None = None
    denoised_path: Path | None = None
    dtln_applied = False

    if has_speech:
        (
            speech_path,
            denoised_path,
            dtln_applied,
        ) = prepare_speech_audio(
            raw_audio_path=raw_path,
            use_dtln=use_dtln,
        )

    print_audio_information(
        raw_path=raw_path,
        speech_path=speech_path,
        denoised_path=denoised_path,
        dtln_applied=dtln_applied,
    )

    run_analysis(
        audio_path=str(raw_path),
        speech_audio_path=(
            str(speech_path)
            if speech_path is not None
            else None
        ),
        run_speech_to_text=has_speech,
        emergency_system=emergency_system,
        hud=hud,
        emergency_v3_specialist=emergency_v3_specialist,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an audio file or record "
            "from the microphone"
        )
    )

    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Path to a WAV file",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Record from the default microphone",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep recording and analyzing in a loop",
    )
    parser.add_argument(
        "--live-stt",
        action="store_true",
        help="Continuously transcribe microphone speech near real time",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Recording duration in seconds",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Microphone device index",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Pause between continuous cycles",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices",
    )
    parser.add_argument(
        "--no-dtln",
        action="store_true",
        help="Disable DTLN noise suppression",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Silero VAD",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD threshold from 0.0 to 1.0",
    )
    parser.add_argument(
        "--hud-port",
        help="ESP32 serial port (for example COM5); omitted disables HUD output",
    )
    parser.add_argument(
        "--hud-baud",
        type=int,
        default=115200,
        help="ESP32 HUD serial baud rate (default: 115200)",
    )
    parser.add_argument(
        "--hud-test",
        action="store_true",
        help="Send deterministic Vietnamese HUD test messages and exit",
    )
    parser.add_argument(
        "--context",
        choices=CONTEXTS,
        default="neutral",
        help="Emergency context mode (default: neutral)",
    )
    parser.add_argument(
        "--personalized-alerts",
        action="store_true",
        help="Apply the validated web personalization profile to alert priority",
    )
    parser.add_argument(
        "--profile-path",
        help="Optional personalized profile JSON path",
    )
    parser.add_argument(
        "--partial-interval",
        type=float,
        default=1.5,
        help="Seconds between live partial transcript snapshots",
    )
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=700,
        help="Silence that finalizes a live utterance",
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=12.0,
        help="Maximum live utterance duration",
    )
    parser.add_argument(
        "--pre-roll-ms",
        type=int,
        default=250,
        help="Audio retained before live speech starts",
    )
    parser.add_argument(
        "--post-roll-ms",
        type=int,
        default=500,
        help="Audio retained after live end-silence detection",
    )
    parser.add_argument(
        "--save-live-utterances",
        action="store_true",
        help="Save raw, DTLN, and optimized FINAL utterance WAV files",
    )
    parser.add_argument(
        "--live-frame-ms",
        type=int,
        choices=(32,),
        default=32,
        help="Live VAD frame duration (Silero requires 32 ms)",
    )
    parser.add_argument(
        "--live-queue-seconds",
        type=float,
        default=4.0,
        help="Maximum queued microphone audio for live STT",
    )
    parser.add_argument(
        "--ced-overlap-seconds",
        type=float,
        default=0.0,
        help=("CED rolling-window overlap in seconds (default: 0; overlapping "
              "windows are correlated and can affect 2-of-3 voting)"),
    )
    parser.add_argument(
        "--emergency-v3",
        action="store_true",
        help="Enable the opt-in EfficientSED emergency specialist",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.hud_test and not args.hud_port:
        print("--hud-test requires --hud-port")
        return 1

    if not 0.0 <= args.vad_threshold <= 1.0:
        print(
            "--vad-threshold must be "
            "between 0.0 and 1.0"
        )
        return 1

    selected_input_modes = sum(bool(value) for value in (
        args.live_stt,
        args.mic,
        args.continuous,
        args.audio_path,
    ))
    if selected_input_modes > 1:
        print("Choose only one of audio_path, --mic, --continuous, or --live-stt")
        return 1
    if args.live_stt or args.mic or args.continuous:
        if args.no_vad:
            print("Microphone utterance modes require Silero VAD; remove --no-vad")
            return 1
        if args.partial_interval <= 0:
            print("--partial-interval must be greater than 0")
            return 1
        if args.end_silence_ms <= 0:
            print("--end-silence-ms must be greater than 0")
            return 1
        if args.max_utterance_seconds <= args.partial_interval:
            print("--max-utterance-seconds must exceed --partial-interval")
            return 1
        if args.pre_roll_ms < 0 or args.pre_roll_ms >= args.max_utterance_seconds * 1000:
            print("--pre-roll-ms must be non-negative and shorter than the maximum utterance")
            return 1
        if args.post_roll_ms < 0:
            print("--post-roll-ms must be non-negative")
            return 1
        if args.live_queue_seconds <= 0:
            print("--live-queue-seconds must be greater than 0")
            return 1
        if args.duration <= 0:
            print("--duration must be greater than 0")
            return 1
        if args.ced_overlap_seconds < 0 or args.ced_overlap_seconds >= args.duration:
            print("--ced-overlap-seconds must be non-negative and shorter than --duration")
            return 1

    if args.list_devices:
        devices = list_input_devices()

        if devices:
            print("Available input devices:")

            for device in devices:
                print(
                    f"  {device['index']}: "
                    f"{device['name']} "
                    f"(channels="
                    f"{device['max_input_channels']})"
                )
        else:
            print("No input devices found.")

        return 0

    try:
        hud = HUDTransport(args.hud_port, args.hud_baud)
    except Exception as exc:
        print(f"HUD warning: unable to open {args.hud_port}: {exc}")
        hud = HUDTransport()
    if hud.enabled:
        # Remove the post-flash validation screen before live runtime output.
        hud.set_subtitle("")
        hud.set_alert_state()
        hud.set_environmental_sound("")
        hud.set_status("LIVE")
    if args.hud_test:
        try:
            for subtitle in HUD_TEST_SUBTITLES:
                hud.set_subtitle(subtitle)
                time.sleep(1.0)
            hud.set_alert_state("siren")
            print(f"Sent {len(HUD_TEST_SUBTITLES)} Vietnamese HUD test screens.")
            return 0
        finally:
            hud.close()

    use_dtln = not args.no_dtln
    use_vad = not args.no_vad
    decision_mode = (
        "continuous" if args.continuous or args.live_stt else "single_shot"
    )
    priority_adapter = None
    if args.personalized_alerts:
        priority_adapter = PriorityAdapter(ProfileManager(args.profile_path))
        print(f"Personalized alerts: {priority_adapter.source} profile")
    emergency_system = EmergencySystem(
        context=args.context,
        decision_mode=decision_mode,
        priority_provider=(priority_adapter.get_priority if priority_adapter else None),
    )
    emergency_v3_specialist = create_emergency_v3_specialist(args.emergency_v3)
    if emergency_v3_specialist is not None:
        print("Emergency V3: enabled (EfficientSED fmn10_strong; lazy load)")

    if args.live_stt or args.mic or args.continuous:
        from soundguard.audio.audio_pipeline import MicrophonePipeline

        mode = "live-stt" if args.live_stt else "continuous" if args.continuous else "mic"
        if mode == "continuous":
            print("Continuous mode started. Press Ctrl+C to stop.")
        try:
            pipeline = MicrophonePipeline(
                mode=mode, emergency_system=emergency_system,
                device_index=args.device, duration=args.duration,
                ced_overlap_seconds=args.ced_overlap_seconds,
                vad_threshold=args.vad_threshold, use_dtln=use_dtln,
                partial_interval=args.partial_interval,
                end_silence_ms=args.end_silence_ms,
                max_utterance_seconds=args.max_utterance_seconds,
                pre_roll_ms=args.pre_roll_ms, post_roll_ms=args.post_roll_ms,
                queue_seconds=args.live_queue_seconds,
                save_live_utterances=args.save_live_utterances,
                show_all_detected_sounds_on_hud=(
                    SHOW_ALL_DETECTED_SOUNDS_ON_HUD
                ),
                hud=hud,
                emergency_v3_specialist=emergency_v3_specialist,
            )
            return pipeline.run()
        finally:
            if emergency_v3_specialist is not None:
                emergency_v3_specialist.close()
            hud.close()

    if not args.audio_path:
        print(
            'Usage: python app.py '
            '"path_to_audio.wav"'
        )
        print(
            "       python app.py "
            "--mic --duration 5"
        )
        print(
            "       python app.py "
            "--continuous --duration 5"
        )
        print(
            "       python app.py "
            "--mic --duration 5 --no-dtln"
        )
        print(
            "       python app.py "
            "--mic --duration 5 --no-vad"
        )

        if emergency_v3_specialist is not None:
            emergency_v3_specialist.close()
        hud.close()
        return 1

    input_path = Path(args.audio_path)

    if not input_path.exists():
        print(f"Audio file not found: {input_path}")
        if emergency_v3_specialist is not None:
            emergency_v3_specialist.close()
        hud.close()
        return 1

    try:
        process_audio(
            raw_path=input_path,
            use_vad=use_vad,
            vad_threshold=args.vad_threshold,
            use_dtln=use_dtln,
            emergency_system=emergency_system,
            hud=hud,
            emergency_v3_specialist=emergency_v3_specialist,
        )
    finally:
        if emergency_v3_specialist is not None:
            emergency_v3_specialist.close()
        hud.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
