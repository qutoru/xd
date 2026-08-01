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

import datetime as _dt
from collections import defaultdict
from dataclasses import dataclass

from crypto_signal_bot.platform.accounting.ledger import AccountingLedger, LedgerEntry
from crypto_signal_bot.platform.notify.i18n import t

# A single venue close (stop-loss / take-profit) can settle in many partial
# executions booked as separate ledger rows sharing a timestamp. Fills of the same
# symbol + attribution within this gap are one trade episode; a wider gap marks a
# genuinely separate trade. Keeps /stats counting *trades*, not partial fills.
_EPISODE_GAP_S = 120.0


def _parse_ts(ts: str | None) -> _dt.datetime | None:
    try:
        return _dt.datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


def _episodes(closes: list[LedgerEntry]) -> list[tuple[str, float]]:
    """Collapse partial-fill closes into per-trade episodes; return ``(symbol,
    realized_pnl)`` for each trade. Fills of one venue close (same symbol +
    attribution, within ``_EPISODE_GAP_S``) sum into one episode."""
    groups: dict[tuple[str, str], list[LedgerEntry]] = defaultdict(list)
    for e in closes:
        groups[(e.symbol, e.attribution)].append(e)
    episodes: list[tuple[str, float]] = []
    for (symbol, _attr), items in groups.items():
        items = sorted(items, key=lambda e: e.timestamp or "")
        cur: float | None = None
        prev_t: _dt.datetime | None = None
        for e in items:
            ts = _parse_ts(e.timestamp)
            if cur is None:
                cur = 0.0
            elif prev_t is None or ts is None or (ts - prev_t).total_seconds() > _EPISODE_GAP_S:
                episodes.append((symbol, cur))
                cur = 0.0
            cur += e.realized_pnl
            prev_t = ts
        if cur is not None:
            episodes.append((symbol, cur))
    return episodes


@dataclass(frozen=True)
class TradeStats:
    """Account-cumulative trade statistics distilled from the ledger."""

    n_closed: int                          # closed trade episodes (partial fills merged)
    wins: int
    losses: int
    realized_pnl: float                    # gross, before fees
    fees: float
    net_pnl: float                         # realized - fees
    best_trade: tuple[str, float] | None   # (symbol, realized) of the single best trade
    worst_trade: tuple[str, float] | None  # (symbol, realized) of the single worst trade

    @property
    def win_rate(self) -> float:
        """Fraction of closed trades that were profitable (0 if none closed)."""
        return self.wins / self.n_closed if self.n_closed else 0.0

    @classmethod
    def from_ledger(cls, ledger: AccountingLedger) -> "TradeStats":
        """Project a ledger into trade stats (opening fills realize nothing).

        Trades are counted per close *episode*, not per partial fill: a stop-loss
        that settles in N executions is one trade, not N. Aggregate PnL/fees stay
        the ledger's exact sums.
        """
        closes = [e for e in ledger.entries if e.realized_pnl != 0.0]
        episodes = [(sym, p) for sym, p in _episodes(closes) if p != 0.0]
        wins = sum(1 for _sym, p in episodes if p > 0.0)
        losses = sum(1 for _sym, p in episodes if p < 0.0)
        best = max(episodes, key=lambda sp: sp[1]) if episodes else None
        # Only meaningful once more than one trade has closed.
        worst = min(episodes, key=lambda sp: sp[1]) if len(episodes) > 1 else None
        return cls(
            n_closed=len(episodes),
            wins=wins,
            losses=losses,
            realized_pnl=ledger.realized_pnl,
            fees=ledger.total_fees,
            net_pnl=ledger.net_pnl,
            best_trade=best,
            worst_trade=worst,
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
    if stats.best_trade is not None:
        lines.append(f"{t(lang, 'stats_best')}: "
                     f"{stats.best_trade[0]} {stats.best_trade[1]:+.2f}")
    if stats.worst_trade is not None:
        lines.append(f"{t(lang, 'stats_worst')}: "
                     f"{stats.worst_trade[0]} {stats.worst_trade[1]:+.2f}")
    return "\n".join(lines)
