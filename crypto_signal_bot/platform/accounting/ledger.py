"""Realized-PnL ledger and its file-backed store.

The ledger is the durable record of what actually happened on the account: one
:class:`LedgerEntry` per executed fill, carrying the venue's realized PnL, fee and
the fill's attribution (see :mod:`.attribution`). It is append-only and de-dupes by
``exec_id`` so re-polling an overlapping execution page — within a run or across a
restart — never double-counts. :class:`AccountingStore` persists it as one small
JSON file (same pattern as the risk-state store): stdlib only, no database, no
framework, and it knows nothing about brokers, signals or portfolios.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from crypto_signal_bot.platform.accounting.attribution import classify
from crypto_signal_bot.platform.execution.domain import Fill


@dataclass(frozen=True)
class LedgerEntry:
    """One booked fill. Plain JSON-friendly scalars so it round-trips as-is."""

    exec_id: str | None
    timestamp: str | None  # ISO-8601 UTC (stable across JSON)
    symbol: str
    side: str              # Side value ("buy" / "sell")
    quantity: float        # base units
    price: float | None
    fee: float
    realized_pnl: float
    attribution: str       # Attribution value


class AccountingLedger:
    """Append-only realized-PnL ledger, de-duplicated by ``exec_id``."""

    def __init__(self, entries: Iterable[LedgerEntry] | None = None) -> None:
        self._entries: list[LedgerEntry] = list(entries or [])
        self._seen: set[str] = {e.exec_id for e in self._entries if e.exec_id is not None}

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def record(self, fills: Iterable[Fill]) -> list[LedgerEntry]:
        """Book each fill (skipping already-seen ``exec_id``s); return the new rows.

        A fill with no ``exec_id`` cannot be de-duplicated and is always appended,
        so callers polling a real venue should ensure executions carry one.
        """
        added: list[LedgerEntry] = []
        for f in fills:
            if f.exec_id is not None and f.exec_id in self._seen:
                continue
            entry = LedgerEntry(
                exec_id=f.exec_id,
                timestamp=None if f.timestamp is None else pd.Timestamp(f.timestamp).isoformat(),
                symbol=f.symbol,
                side=f.side.value,
                quantity=float(f.quantity),
                price=None if f.price is None else float(f.price),
                fee=float(f.fee),
                realized_pnl=float(f.realized_pnl),
                attribution=classify(f).value,
            )
            self._entries.append(entry)
            if f.exec_id is not None:
                self._seen.add(f.exec_id)
            added.append(entry)
        return added

    @property
    def realized_pnl(self) -> float:
        """Gross realized PnL booked so far (before fees)."""
        return float(sum(e.realized_pnl for e in self._entries))

    @property
    def total_fees(self) -> float:
        return float(sum(e.fee for e in self._entries))

    @property
    def net_pnl(self) -> float:
        """Realized PnL net of fees."""
        return self.realized_pnl - self.total_fees

    def realized_by_symbol(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self._entries:
            out[e.symbol] = out.get(e.symbol, 0.0) + e.realized_pnl
        return out

    def realized_by_attribution(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self._entries:
            out[e.attribution] = out.get(e.attribution, 0.0) + e.realized_pnl
        return out

    def watermark(self) -> pd.Timestamp | None:
        """Latest fill timestamp booked — a lower bound for the next poll."""
        stamps = [pd.Timestamp(e.timestamp) for e in self._entries if e.timestamp]
        return max(stamps) if stamps else None

    def __len__(self) -> int:
        return len(self._entries)


class AccountingStore:
    """Load/save an :class:`AccountingLedger` as one JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> AccountingLedger:
        """Return the persisted ledger, or an empty one if unreadable/absent."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return AccountingLedger()
        entries = [LedgerEntry(**row) for row in data.get("entries", [])]
        return AccountingLedger(entries)

    def save(self, ledger: AccountingLedger) -> None:
        """Persist the whole ledger (append-only in effect; small single file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"entries": [asdict(e) for e in ledger.entries]}),
            encoding="utf-8",
        )
