"""File-backed store of signals awaiting the owner's manual approval.

Semi-automatic mode: the daily pipeline does not place orders itself. Instead it
records each proposed :class:`~crypto_signal_bot.platform.execution.intent.TradeIntent`
here (keyed by a short opaque token) and pushes the owner a Telegram message with
Accept / Ignore buttons. When the owner taps Accept the listener loads the pending
signal by token, runs it through the risk gate and places the order.

The two halves run in separate processes (one-shot ``cmd_trade`` pipeline vs the
long-poll ``cmd_bot`` listener), so the hand-off must be durable — hence a small
JSON file, same stdlib-only pattern as the other notify stores. It knows nothing
about brokers, exchanges or Telegram.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from crypto_signal_bot.platform.execution.intent import TradeIntent

# Lifecycle of a pending signal. Only ``pending`` is actionable; the rest are
# terminal outcomes recorded once the owner (or an expiry) resolves it.
STATUS_PENDING = "pending"
STATUS_ACCEPTED = "accepted"
STATUS_IGNORED = "ignored"
STATUS_BLOCKED = "blocked"    # accepted by the owner but refused by the risk gate
STATUS_EXPIRED = "expired"


@dataclass
class PendingSignal:
    """One proposed trade awaiting the owner's decision."""

    token: str
    intent: dict                       # TradeIntent.to_dict()
    status: str = STATUS_PENDING
    created_at: str | None = None      # ISO-8601 UTC
    chat_id: str | None = None         # owner chat the approval buttons were sent to
    result: str = ""                   # human-readable outcome, filled on resolution

    def is_pending(self) -> bool:
        return self.status == STATUS_PENDING

    def age_seconds(self, now: pd.Timestamp | None = None) -> float | None:
        """Seconds since creation, or None when the timestamp is missing."""
        if not self.created_at:
            return None
        now = now if now is not None else pd.Timestamp.now(tz="UTC")
        return float((now - pd.Timestamp(self.created_at)).total_seconds())


class PendingSignalStore:
    """Load/save ``{token: PendingSignal}`` as one small JSON file."""

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
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add(self, intent: TradeIntent, *, chat_id: str | None = None) -> PendingSignal:
        """Persist a new pending signal for ``intent`` and return it (with token)."""
        signal = PendingSignal(
            token=uuid.uuid4().hex[:12],
            intent=intent.to_dict(),
            status=STATUS_PENDING,
            created_at=pd.Timestamp.now(tz="UTC").isoformat(),
            chat_id=str(chat_id) if chat_id is not None else None,
        )
        data = self._load()
        data[signal.token] = asdict(signal)
        self._save(data)
        return signal

    def get(self, token: str) -> PendingSignal | None:
        rec = self._load().get(token)
        return None if rec is None else PendingSignal(**rec)

    def resolve(self, token: str, status: str, result: str = "") -> PendingSignal | None:
        """Set the terminal ``status``/``result`` for ``token`` (no-op if absent)."""
        data = self._load()
        rec = data.get(token)
        if rec is None:
            return None
        rec["status"] = status
        rec["result"] = result
        self._save(data)
        return PendingSignal(**rec)

    def all_pending(self) -> list[PendingSignal]:
        return [
            PendingSignal(**rec)
            for rec in self._load().values()
            if rec.get("status") == STATUS_PENDING
        ]
