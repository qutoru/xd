"""File-backed subscription store for the Telegram bot.

One small JSON file mapping a Telegram ``chat_id`` to its subscription: an expiry
timestamp and an optional plan label. Time-based — a subscription is *active*
while its ``expires_at`` is in the future — so the admin grants access for N days
and it lapses on its own. Same stdlib-only persistence pattern as the other
notify stores.

Independent by construction: imports only stdlib. It knows nothing about the
transport, the formatter or trading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


@dataclass(frozen=True)
class Subscription:
    """One chat's subscription state."""

    chat_id: str
    expires_at: datetime
    plan: str | None = None

    def is_active(self, now: datetime | None = None) -> bool:
        return self.expires_at > (now or _now())


class SubscriptionStore:
    """Load/save ``{chat_id: {expires_at, plan}}`` as one small JSON file."""

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
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, chat_id: str | int) -> Subscription | None:
        """Return the stored subscription for ``chat_id`` (active or lapsed)."""
        rec = self._load().get(str(chat_id))
        if not rec:
            return None
        expires = _parse(rec.get("expires_at", ""))
        if expires is None:
            return None
        return Subscription(str(chat_id), expires, rec.get("plan"))

    def is_active(self, chat_id: str | int, now: datetime | None = None) -> bool:
        """True if ``chat_id`` has a subscription that has not yet expired."""
        sub = self.get(chat_id)
        return sub is not None and sub.is_active(now)

    def grant(
        self,
        chat_id: str | int,
        days: int,
        plan: str | None = None,
        now: datetime | None = None,
    ) -> Subscription:
        """Grant/extend ``days`` of access. Extends from the current expiry if the
        subscription is still active, otherwise from now."""
        now = now or _now()
        existing = self.get(chat_id)
        base = existing.expires_at if (existing and existing.is_active(now)) else now
        expires = base + timedelta(days=days)
        resolved_plan = plan if plan is not None else (existing.plan if existing else None)
        data = self._load()
        data[str(chat_id)] = {"expires_at": expires.isoformat(), "plan": resolved_plan}
        self._save(data)
        return Subscription(str(chat_id), expires, resolved_plan)

    def revoke(self, chat_id: str | int) -> bool:
        """Remove ``chat_id``'s subscription. Returns True if one existed."""
        data = self._load()
        if str(chat_id) in data:
            del data[str(chat_id)]
            self._save(data)
            return True
        return False

    def active(self, now: datetime | None = None) -> list[Subscription]:
        """All currently-active subscriptions, ascending by expiry."""
        now = now or _now()
        subs = [self.get(cid) for cid in self._load()]
        active = [s for s in subs if s is not None and s.is_active(now)]
        return sorted(active, key=lambda s: s.expires_at)
