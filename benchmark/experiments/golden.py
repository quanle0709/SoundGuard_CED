"""Capture deterministic pre-integration outputs for legacy-off regression."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark_results" / "improvement" / "golden_baseline_manifest.json"
FLAGS = ("SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING", "SOUNDGUARD_ENABLE_TEMPORAL_POOLING_V2",
         "SOUNDGUARD_SAFE_COMPOUND_EVIDENCE")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture() -> dict:
    for name in FLAGS:
        os.environ.pop(name, None)
    from emergency_system import EmergencySystem
    from fusion_engine import fuse_result
    from personalization.profile_generator import generate_rule_based
    from sound_classifier import classify_audio_file
    from speech_enhancer import enhance_audio_file
    from speech_recognizer import transcribe_audio_file

    ced_paths = [
        ROOT / "benchmark_data/external/esc50/audio/1-100032-A-0.wav",
        ROOT / "benchmark_data/external/esc50/audio/1-101336-A-30.wav",
        ROOT / "benchmark_data/external/esc50/audio/1-110389-A-0.wav",
    ]
    ced = []
    for path in ced_paths:
        result = classify_audio_file(path)
        ced.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "label": result["label"],
                    "confidence": result["confidence"], "top_predictions": result["top_predictions"]})
    emergency_cases = [("Siren", .85), ("Glass breaking", .80), ("Bark", .85), ("Siren", .20)]
    emergency = [EmergencySystem(decision_mode="single_shot").process_sound_event(label, confidence)
                 for label, confidence in emergency_cases]
    for row in emergency:
        row.pop("timestamp", None)
    fusion = fuse_result("golden", "", {"label": "Siren", "confidence": .85},
                         {"category": "siren", "is_dangerous": True, "message_vi": "golden"})
    personalization = [generate_rule_based([text]) for text in
                       ("I drive in road traffic.", "A factory-method pattern is software.", "The baby-blue color is bright.")]
    speech_path = ROOT / "benchmark_data/external/vivos/selected/VIVOSDEV01/VIVOSDEV01_R002.wav"
    with patch("speech_recognizer.sr.Recognizer.recognize_google", return_value="  xin chào  "):
        stt = transcribe_audio_file(speech_path)
    dtln_input = ROOT / "benchmark_data/expanded_generated/dtln/inputs/traffic/20/1-187207-A-20.wav"
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "golden.wav"
        enhance_audio_file(dtln_input, target, verbose=False)
        dtln = {"input_sha256": sha256(dtln_input), "output_sha256": sha256(target),
                "output_size": target.stat().st_size}
    return {"schema_version": 1, "feature_flags": {name: False for name in FLAGS}, "ced": ced,
            "stt_mocked_interface": stt, "emergency": emergency, "fusion": fusion,
            "noise_filter_interface": dtln, "personalization": personalization}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(capture(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
