from pathlib import Path

import speech_recognition as sr


def transcribe_audio_file(file_path):
    """Transcribe a WAV file using Google Speech Recognition.

    This function does not open the microphone. It requires an internet
    connection because it uses the Google recognition service.
    """

    # SpeechRecognition.AudioFile rejects pathlib.Path/WindowsPath even though
    # they are valid PathLike values. Normalize once at this stable boundary.
    audio_path = str(Path(file_path))
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        transcript = recognizer.recognize_google(audio_data, language="vi-VN")
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        raise RuntimeError(f"Speech recognition request failed: {exc}") from exc

    return transcript.strip()
