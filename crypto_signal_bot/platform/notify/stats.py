"""Trade-statistics projection for the /stats command (PRO/VIP).

A read-only summary derived from the realized-PnL
:class:`~crypto_signal_bot.platform.accounting.ledger.AccountingLedger`: closed-
trade count, win rate, net PnL and the best/worst symbol. The bot surfaces the
*strategy's* realized performance (the same figures for every subscriber) — it
does not track per-user trades, since subscribers only receive signals.

Rendering is localized here (labels via the i18n catalog); the maths delegates to
the ledger. Independent of the transport.
"""

from __future__ import annotations

from dataclasses import dataclass

from crypto_signal_bot.platform.accounting.ledger import AccountingLedger
from crypto_signal_bot.platform.notify.i18n import t


@dataclass(frozen=True)
class TradeStats:
    """Account-cumulative trade statistics distilled from the ledger."""

    n_closed: int                          # closing fills (realized_pnl != 0)
    wins: int
    losses: int
    realized_pnl: float                    # gross, before fees
    fees: float
    net_pnl: float                         # realized - fees
    best_symbol: tuple[str, float] | None  # (symbol, realized) or None
    worst_symbol: tuple[str, float] | None

    @property
    def win_rate(self) -> float:
        """Fraction of closed trades that were profitable (0 if none closed)."""
        return self.wins / self.n_closed if self.n_closed else 0.0

    @classmethod
    def from_ledger(cls, ledger: AccountingLedger) -> "TradeStats":
        """Project a ledger into trade stats (opening fills realize nothing)."""
        closes = [e for e in ledger.entries if e.realized_pnl != 0.0]
        wins = sum(1 for e in closes if e.realized_pnl > 0.0)
        losses = sum(1 for e in closes if e.realized_pnl < 0.0)
        by_symbol = ledger.realized_by_symbol()
        best = max(by_symbol.items(), key=lambda kv: kv[1]) if by_symbol else None
        # Only meaningful once more than one symbol has realized PnL.
        worst = min(by_symbol.items(), key=lambda kv: kv[1]) if len(by_symbol) > 1 else None
        return cls(
            n_closed=len(closes),
            wins=wins,
            losses=losses,
            realized_pnl=ledger.realized_pnl,
            fees=ledger.total_fees,
            net_pnl=ledger.net_pnl,
            best_symbol=best,
            worst_symbol=worst,
        )


def format_stats(stats: TradeStats, lang: str) -> str:
    """Render trade statistics as a localized Telegram message."""
    if stats.n_closed == 0:
        return t(lang, "stats_empty")
    lines = [
        t(lang, "stats_header"),
        f"{t(lang, 'stats_trades')}: {stats.n_closed} ({stats.wins}✅ / {stats.losses}❌)",
        f"{t(lang, 'stats_winrate')}: {stats.win_rate:.0%}",
        f"{t(lang, 'stats_net')}: {stats.net_pnl:+.2f} "
        f"({t(lang, 'stats_fees')}: {stats.fees:.2f})",
    ]
    if stats.best_symbol is not None:
        lines.append(f"{t(lang, 'stats_best')}: "
                     f"{stats.best_symbol[0]} {stats.best_symbol[1]:+.2f}")
    if stats.worst_symbol is not None:
        lines.append(f"{t(lang, 'stats_worst')}: "
                     f"{stats.worst_symbol[0]} {stats.worst_symbol[1]:+.2f}")
    return "\n".join(lines)
