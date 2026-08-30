"""Atomic profile persistence with safe fallback behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .role_knowledge import DEFAULT_PRIORITIES
from .profile_validator import validate_profile


PACKAGE_DIR = Path(__file__).resolve().parent


class ProfileManager:
    def __init__(self, profile_path: str | Path | None = None, default_path: str | Path | None = None):
        self.profile_path = Path(profile_path or os.environ.get("SOUNDGUARD_PROFILE_PATH", PACKAGE_DIR / "user_profile.json"))
        self.default_path = Path(default_path or PACKAGE_DIR / "default_profile.json")

    def _read(self, path: Path) -> dict:
        return validate_profile(json.loads(path.read_text(encoding="utf-8"))).to_dict()

    def load(self) -> tuple[dict, str]:
        try:
            return self._read(self.profile_path), "user"
        except (OSError, ValueError, json.JSONDecodeError):
            try:
                return self._read(self.default_path), "default"
            except (OSError, ValueError, json.JSONDecodeError):
                builtin = {
                    "profile_version": "1.0",
                    "roles": [],
                    "contexts": ["home"],
                    "responsibilities": [],
                    "priority_profile": dict(DEFAULT_PRIORITIES),
                    "reasoning_summary": {
                        label: "Built-in safe fallback."
                        for label in DEFAULT_PRIORITIES
                    },
                }
                return validate_profile(builtin).to_dict(), "built_in_default"

    def save(self, payload: dict) -> dict:
        profile = validate_profile(payload).to_dict()
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.profile_path.with_suffix(self.profile_path.suffix + ".tmp")
        temporary.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.profile_path)
        return profile
