"""File-backed per-chat personal risk level for the VIP /risk command.

One small JSON file mapping a Telegram ``chat_id`` to a risk level
(``low`` / ``medium`` / ``high``) — same lightweight, stdlib-only persistence
pattern as the language prefs and subscription stores. A VIP subscriber sets
their preferred level; it is stored per chat and read back on demand. (Wiring the
level into position-sizing suggestions is a later step — this owns the setting.)

Independent by construction: imports only stdlib.
"""

from __future__ import annotations

import json
from pathlib import Path

RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_RISK = "medium"

# Accepted spellings (case-insensitive), EN + RU, mapped to the canonical level.
_ALIASES: dict[str, str] = {
    "low": "low", "l": "low", "низкий": "low", "консервативный": "low",
    "medium": "medium", "med": "medium", "m": "medium",
    "средний": "medium", "умеренный": "medium",
    "high": "high", "h": "high", "высокий": "high", "агрессивный": "high",
}


def normalize_risk(value: str | None) -> str | None:
    """Map a user-typed level to a canonical one, or ``None`` if unrecognised."""
    if not value:
        return None
    return _ALIASES.get(value.strip().lower())


class RiskPrefsStore:
    """Load/save a ``{chat_id: level}`` map as one small JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def get(self, chat_id: str | int) -> str:
        """Return the chat's risk level, or the default if it hasn't chosen."""
        level = self._load().get(str(chat_id), DEFAULT_RISK)
        return level if level in RISK_LEVELS else DEFAULT_RISK

    def set(self, chat_id: str | int, level: str) -> None:
        """Persist a (normalized) risk level for ``chat_id``; ignores invalid input."""
        canonical = normalize_risk(level)
        if canonical is None:
            return
        data = self._load()
        data[str(chat_id)] = canonical
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")


__all__ = ["RiskPrefsStore", "normalize_risk", "RISK_LEVELS", "DEFAULT_RISK"]
