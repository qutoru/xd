"""File-backed registry of everyone who has interacted with the bot.

The admin grants access by numeric ``chat_id``, but a Telegram bot can't resolve
an @username to an id until that person has messaged it. So we record every user
who sends a message — id, username, first name — and expose the list via the
admin ``/users`` command, giving the admin the ids to grant.

Independent by construction: stdlib only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnownUser:
    """A user the bot has seen at least once."""

    chat_id: str
    username: str | None = None
    first_name: str | None = None


class UsersStore:
    """Load/save ``{chat_id: {username, first_name}}`` as one small JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def record(
        self,
        chat_id: str | int,
        *,
        username: str | None = None,
        first_name: str | None = None,
    ) -> None:
        """Remember (or refresh) a user's id, username and first name."""
        data = self._load()
        data[str(chat_id)] = {"username": username, "first_name": first_name}
        self._save(data)

    def all(self) -> list[KnownUser]:
        """Every known user, in insertion order."""
        return [
            KnownUser(cid, rec.get("username"), rec.get("first_name"))
            for cid, rec in self._load().items()
        ]

    def find_by_username(self, username: str) -> str | None:
        """Return the chat_id for a @username (case-insensitive), or None.

        Only resolves users the bot has already seen — Telegram gives no way to
        look up an arbitrary username until that person has messaged the bot.
        """
        target = username.lstrip("@").lower()
        if not target:
            return None
        for cid, rec in self._load().items():
            uname = rec.get("username")
            if uname and uname.lower() == target:
                return cid
        return None
