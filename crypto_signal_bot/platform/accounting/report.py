"""PerformanceReport — a realized-PnL summary derived purely from the ledger.

A read-only projection of :class:`AccountingLedger`: it re-uses the ledger's own
aggregations and adds no bookkeeping of its own (no broker calls, no re-summing of
fills). Figures are account-cumulative — the realized PnL recorded so far — since
that is exactly what the persisted ledger holds.
"""

from __future__ import annotations

from dataclasses import dataclass

from crypto_signal_bot.platform.accounting.attribution import Attribution
from crypto_signal_bot.platform.accounting.ledger import AccountingLedger

# The close reasons surfaced in the report (ENTRY opens nothing to realize).
_CLOSE_REASONS = (
    Attribution.TAKE_PROFIT,
    Attribution.STOP_LOSS,
    Attribution.REBALANCE,
    Attribution.MANUAL,
)


@dataclass(frozen=True)
class PerformanceReport:
    """Account-cumulative realized-PnL summary built from the ledger."""

    realized_pnl: float               # gross realized PnL (before fees)
    fees: float                       # total fees paid
    net_pnl: float                    # realized_pnl - fees
    realized_by_symbol: dict[str, float]
    realized_by_reason: dict[str, float]  # keys: take_profit/stop_loss/rebalance/manual

    @classmethod
    def from_ledger(cls, ledger: AccountingLedger) -> "PerformanceReport":
        """Project a ledger into a report (delegates all math to the ledger)."""
        by_attr = ledger.realized_by_attribution()
        return cls(
            realized_pnl=ledger.realized_pnl,
            fees=ledger.total_fees,
            net_pnl=ledger.net_pnl,
            realized_by_symbol=ledger.realized_by_symbol(),
            realized_by_reason={r.value: by_attr.get(r.value, 0.0) for r in _CLOSE_REASONS},
        )
