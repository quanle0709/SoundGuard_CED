from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmark_data"
RESULTS_DIR = ROOT / "benchmark_results"
SEED = 42
SNR_LEVELS = (None, 20, 15, 10, 5, 0)

AUDIO_CASES = (
    {
        "sample_id": "esc50_bark_100032",
        "path": ROOT / "samples" / "audio_1-100032-A-0.wav",
        "true_label": "dog_barking",
        "provenance": "ESC-50-style filename; manifest human label Bark",
    },
    {
        "sample_id": "local_vi_speech",
        "path": ROOT / "samples" / "test_speech.wav",
        "true_label": "speech",
        "reference": "xin chào đây là thử nghiệm giọng nói tiếng việt",
        "provenance": "project-local recording with manifest reference transcript",
    },
)
