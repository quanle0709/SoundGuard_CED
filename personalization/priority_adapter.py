"""Thin, deterministic bridge from a validated profile to EmergencySystem."""

from __future__ import annotations

from .profile_manager import ProfileManager
from .sound_labels import canonicalize_label


class PriorityAdapter:
    def __init__(self, manager: ProfileManager | None = None):
        self.manager = manager or ProfileManager()
        self.profile, self.source = self.manager.load()
        self._user_mtime = self._mtime()

    def _mtime(self) -> int | None:
        try:
            return self.manager.profile_path.stat().st_mtime_ns
        except OSError:
            return None

    def reload(self) -> str:
        self.profile, self.source = self.manager.load()
        self._user_mtime = self._mtime()
        return self.source

    def get_priority(self, label: str, context: str = "neutral") -> int | None:
        del context  # Reserved for a future context switch without changing core API.
        if self._mtime() != self._user_mtime:
            self.reload()
        canonical = canonicalize_label(label)
        if canonical is None:
            return None
        return self.profile["priority_profile"].get(canonical)
