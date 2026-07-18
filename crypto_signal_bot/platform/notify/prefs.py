"""File-backed per-chat language preferences for the Telegram bot.

One small JSON file mapping a Telegram ``chat_id`` to its chosen language code —
the same lightweight, stdlib-only persistence pattern used for risk state. Read
when rendering/handling a message, written when a user picks a language.

Independent by construction: imports only stdlib and the i18n defaults. It knows
nothing about brokers, execution or the transport.
"""

from __future__ import annotations

import json
from pathlib import Path

from crypto_signal_bot.platform.notify.i18n import DEFAULT_LANGUAGE, normalize


class LanguagePrefsStore:
    """Load/save a ``{chat_id: language}`` map as one small JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, str]:
        """Return the raw map, or an empty map if unreadable/absent."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def get(self, chat_id: str | int) -> str:
        """Return the chat's language, or the default if it hasn't chosen."""
        return normalize(self._load().get(str(chat_id)))

    def has(self, chat_id: str | int) -> bool:
        """True if ``chat_id`` has already picked (and stored) a language.

        Used to tell a first-ever onboarding apart from a later /start, so the
        risk disclaimer is only appended to the very first greeting.
        """
        return str(chat_id) in self._load()

    def set(self, chat_id: str | int, language: str) -> None:
        """Persist ``language`` for ``chat_id`` (normalized, dir auto-created)."""
        data = self._load()
        data[str(chat_id)] = normalize(language)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")


__all__ = ["LanguagePrefsStore", "DEFAULT_LANGUAGE"]
