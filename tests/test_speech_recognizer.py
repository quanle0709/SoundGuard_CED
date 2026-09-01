from pathlib import Path

from soundguard.speech import speech_recognizer


class FakeAudioFile:
    received = []

    def __init__(self, filename):
        assert isinstance(filename, str)
        self.filename = filename
        self.received.append(filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeRecognizer:
    def record(self, source):
        return f"audio:{source.filename}"

    def recognize_google(self, audio, language):
        assert audio.startswith("audio:") and language == "vi-VN"
        return "  deterministic transcript  "


def with_mocked_speech_recognition(value):
    original_audio_file = speech_recognizer.sr.AudioFile
    original_recognizer = speech_recognizer.sr.Recognizer
    FakeAudioFile.received = []
    speech_recognizer.sr.AudioFile = FakeAudioFile
    speech_recognizer.sr.Recognizer = FakeRecognizer
    try:
        transcript = speech_recognizer.transcribe_audio_file(value)
        return transcript, list(FakeAudioFile.received)
    finally:
        speech_recognizer.sr.AudioFile = original_audio_file
        speech_recognizer.sr.Recognizer = original_recognizer


def test_transcribe_audio_file_accepts_pathlib_path():
    path = Path("temporary") / "normalized.wav"
    transcript, received = with_mocked_speech_recognition(path)
    assert transcript == "deterministic transcript"
    assert received == [str(path)]


def test_transcribe_audio_file_accepts_string_path():
    path = r"C:\temporary\dtln.wav"
    transcript, received = with_mocked_speech_recognition(path)
    assert transcript == "deterministic transcript"
    assert received == [str(Path(path))]


def run_tests():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test(); print(f"{test.__name__}: passed")
    print(f"speech recognizer tests passed ({len(tests)})")


if __name__ == "__main__": run_tests()
