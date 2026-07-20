"""Daily market snapshot — the single cross-sectional input to the signal layer.

Composes the three providers (universe, bars, funding) into one point-in-time
:class:`MarketSnapshot`: aligned close panel, daily returns, and funding, all on
a common index/columns with no rows after ``asof``. Signals consume the snapshot;
they never talk to a data source directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_signal_bot.platform.data.interfaces import (
    DailyBarProvider,
    FundingProvider,
    UniverseProvider,
)


@dataclass(frozen=True)
class MarketSnapshot:
    """Point-in-time daily cross-sectional state as of ``asof``."""

    asof: pd.Timestamp
    symbols: list[str]
    closes: pd.DataFrame  # time x symbol
    returns: pd.DataFrame  # daily simple returns, same shape as closes
    funding: pd.DataFrame  # time x symbol daily realized funding


class DailySnapshotProvider:
    """Assembles a :class:`MarketSnapshot` from injectable providers."""

    def __init__(
        self,
        universe: UniverseProvider,
        bars: DailyBarProvider,
        funding: FundingProvider,
    ) -> None:
        self.universe = universe
        self.bars = bars
        self.funding = funding

    def snapshot(
        self,
        asof: pd.Timestamp,
        lookback_days: int,
        *,
        now: pd.Timestamp | None = None,
    ) -> MarketSnapshot:
        asof = pd.Timestamp(asof)
        start = asof - pd.Timedelta(days=lookback_days)
        symbols = list(self.universe.universe(asof))

        # Completeness guard: the current UTC day is still forming — today's daily
        # close is a not-yet-closed bar and today's realized-funding sum is partial
        # (settlements accrue through the day at 00/08/16 UTC). The backtest only
        # ever uses fully-closed days, so a live snapshot must too; otherwise the
        # signal's trailing mean ingests incomplete data. We therefore keep only
        # bars strictly before the start of the current UTC day. Historical asof
        # (backtest/replay, dated in the past) is unaffected — nothing is dropped.
        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        today = now.tz_convert("UTC").normalize() if now.tzinfo else now.normalize()
        cutoff = min(asof, today - pd.Timedelta(microseconds=1))

        closes = self.bars.close_panel(symbols, start, asof)
        # Point-in-time guard: never expose a bar dated after asof, nor one on the
        # still-forming current UTC day.
        closes = closes[closes.index <= cutoff].sort_index()
        symbols = [s for s in symbols if s in closes.columns]
        closes = closes[symbols]

        returns = closes.pct_change()

        funding = self.funding.funding_panel(symbols, start, asof)
        funding = funding[funding.index <= cutoff]
        funding = funding.reindex(index=returns.index, columns=symbols)

        return MarketSnapshot(
            asof=asof,
            symbols=symbols,
            closes=closes,
            returns=returns,
            funding=funding,
        )
