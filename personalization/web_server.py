"""Dependency-free local web MVP for the SoundGuard personalization interview."""

from __future__ import annotations

import argparse
import json
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .profile_generator import generate_profile
from .profile_manager import ProfileManager
from .recognition import (
    MAX_UPLOAD_BYTES,
    EmbeddingClient,
    ModelUnavailable,
    RecognitionError,
    RecognitionStore,
    decode_audio_base64,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
QUESTIONS = (
    "What do you do most days—work, study, caregiving, or something else?",
    "How do you usually travel, and how often are you around road traffic?",
    "Which places do you spend the most time in?",
    "Who lives with you, and is there anyone you regularly look after?",
    "Are you often around machinery, vehicles, construction, or medical equipment?",
    "Which alarms, warnings, or everyday sounds must you never miss?",
    "Is there anything else about your routine that should change which sounds get your attention first?",
)


class SessionStore:
    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def start(self) -> tuple[str, str]:
        session_id = secrets.token_urlsafe(18)
        self.sessions[session_id] = {"answers": [], "draft": None, "provider": None}
        return session_id, QUESTIONS[0]

    def answer(self, session_id: str, answer: str) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("unknown or expired session")
        cleaned = answer.strip()
        if not cleaned:
            raise ValueError("answer cannot be empty")
        if len(cleaned) > 1000:
            raise ValueError("answer is too long")
        session["answers"].append(cleaned)
        count = len(session["answers"])
        enough_detail = count >= 5 and sum(len(item.split()) for item in session["answers"]) >= 25
        user_finished = count >= 5 and cleaned.lower() in {"no", "nothing else", "done", "that's all", "that is all"}
        if count >= len(QUESTIONS) or enough_detail or user_finished:
            profile, provider = generate_profile(session["answers"])
            session["draft"], session["provider"] = profile, provider
            print(f"[personalization] profile generated via {provider}")
            return {"complete": True, "profile": profile, "provider": provider}
        return {"complete": False, "question": QUESTIONS[count], "progress": count + 1}


SESSIONS = SessionStore()
MANAGER = ProfileManager()
RECOGNITION_STORE = RecognitionStore()
EMBEDDING_CLIENT = EmbeddingClient()
URL_KINDS = {"familiar-sounds": "sound", "familiar-voices": "voice"}


class PersonalizationHandler(BaseHTTPRequestHandler):
    server_version = "SoundGuardPersonalization/1.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[personalization] {format % args}")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self, max_bytes: int = 32_000) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length <= 0 or length > max_bytes:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            body = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/profile":
            profile, source = MANAGER.load()
            self._json(200, {"profile": profile, "source": source})
        elif path == "/api/recognition":
            self._json(200, {
                "familiar_sounds": RECOGNITION_STORE.list("sound"),
                "familiar_voices": RECOGNITION_STORE.list("voice"),
                "model_worker_configured": EMBEDDING_CLIENT.python_path.is_file(),
                "model_worker_loaded": EMBEDDING_CLIENT.loaded,
                "storage_root": str(RECOGNITION_STORE.root),
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            path = urlsplit(self.path).path
            is_audio_upload = path.endswith("/samples") or path.endswith("/test")
            body = self._read_json(MAX_UPLOAD_BYTES * 2 if is_audio_upload else 32_000)
            if path == "/api/start":
                session_id, question = SESSIONS.start()
                self._json(200, {"session_id": session_id, "question": question, "progress": 1})
            elif path == "/api/chat":
                self._json(200, SESSIONS.answer(str(body.get("session_id", "")), str(body.get("answer", ""))))
            elif path == "/api/apply":
                session = SESSIONS.sessions.get(str(body.get("session_id", "")))
                if not session or not session.get("draft"):
                    raise ValueError("generate a profile before applying it")
                profile = MANAGER.save(session["draft"])
                self._json(200, {"applied": True, "profile": profile})
            elif path in ("/api/familiar-sounds", "/api/familiar-voices"):
                kind = URL_KINDS[path.removeprefix("/api/")]
                profile = RECOGNITION_STORE.create(
                    kind, body.get("display_name", ""),
                    relationship=body.get("relationship", ""),
                    contexts=body.get("contexts", []), priority=body.get("priority", 3),
                    consent_acknowledged=body.get("consent_acknowledged", False),
                )
                self._json(201, {"profile": profile})
            elif (match := re.fullmatch(
                    r"/api/(familiar-sounds|familiar-voices)/([^/]+)/samples", path)):
                kind, profile_id = URL_KINDS[match.group(1)], match.group(2)
                sample = RECOGNITION_STORE.add_sample(
                    kind, profile_id, decode_audio_base64(str(body.get("audio_base64", "")))
                )
                self._json(201, {"sample": sample})
            elif (match := re.fullmatch(
                    r"/api/(familiar-sounds|familiar-voices)/([^/]+)/build", path)):
                kind, profile_id = URL_KINDS[match.group(1)], match.group(2)
                profile = RECOGNITION_STORE.build(kind, profile_id, EMBEDDING_CLIENT.embed)
                self._json(200, {"profile": profile})
            elif (match := re.fullmatch(
                    r"/api/(familiar-sounds|familiar-voices)/test", path)):
                kind = URL_KINDS[match.group(1)]
                result = RECOGNITION_STORE.test(
                    kind, decode_audio_base64(str(body.get("audio_base64", ""))),
                    EMBEDDING_CLIENT.embed,
                )
                self._json(200, {"result": result})
            else:
                self._json(404, {"error": "not found"})
        except ModelUnavailable as exc:
            self._json(503, {"error": str(exc), "feature_available": False})
        except (ValueError, KeyError, RecognitionError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except OSError:
            self._json(500, {"error": "profile could not be saved"})

    def do_PATCH(self) -> None:
        try:
            path = urlsplit(self.path).path
            match = re.fullmatch(r"/api/(familiar-sounds|familiar-voices)/([^/]+)", path)
            if not match:
                self._json(404, {"error": "not found"})
                return
            profile = RECOGNITION_STORE.update(
                URL_KINDS[match.group(1)], match.group(2), self._read_json()
            )
            self._json(200, {"profile": profile})
        except (ValueError, RecognitionError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})

    def do_DELETE(self) -> None:
        try:
            path = urlsplit(self.path).path
            sample = re.fullmatch(
                r"/api/(familiar-sounds|familiar-voices)/([^/]+)/samples/([^/]+)", path
            )
            profile = re.fullmatch(
                r"/api/(familiar-sounds|familiar-voices)/([^/]+)", path
            )
            if sample:
                result = RECOGNITION_STORE.delete_sample(
                    URL_KINDS[sample.group(1)], sample.group(2), sample.group(3)
                )
            elif profile:
                result = RECOGNITION_STORE.delete(
                    URL_KINDS[profile.group(1)], profile.group(2)
                )
            else:
                self._json(404, {"error": "not found"})
                return
            self._json(200, result)
        except RecognitionError as exc:
            self._json(400, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="SoundGuard personalization web assistant")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PersonalizationHandler)
    print(f"SoundGuard personalization: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        EMBEDDING_CLIENT.close()


if __name__ == "__main__":
    main()
