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

    def snapshot(self, asof: pd.Timestamp, lookback_days: int) -> MarketSnapshot:
        asof = pd.Timestamp(asof)
        start = asof - pd.Timedelta(days=lookback_days)
        symbols = list(self.universe.universe(asof))

        closes = self.bars.close_panel(symbols, start, asof)
        # Point-in-time guard: never expose a bar dated after asof.
        closes = closes[closes.index <= asof].sort_index()
        symbols = [s for s in symbols if s in closes.columns]
        closes = closes[symbols]

        returns = closes.pct_change()

        funding = self.funding.funding_panel(symbols, start, asof)
        funding = funding[funding.index <= asof]
        funding = funding.reindex(index=returns.index, columns=symbols)

        return MarketSnapshot(
            asof=asof,
            symbols=symbols,
            closes=closes,
            returns=returns,
            funding=funding,
        )
