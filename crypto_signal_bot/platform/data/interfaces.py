"""Data-source contracts for the platform.

Every source is an injectable interface so signals depend on *shapes*, not on
Bybit. A source may be the live exchange, a cached panel, a historical replay or
a test fake — all interchangeable. All panels are (time x symbol) with a
tz-aware daily UTC DatetimeIndex; every value is point-in-time (index <= asof).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class UniverseProvider(Protocol):
    """Resolves the tradable symbol set as of a date."""

    def universe(self, asof: pd.Timestamp) -> list[str]:
        ...


@runtime_checkable
class DailyBarProvider(Protocol):
    """Supplies a (time x symbol) daily close-price panel."""

    def close_panel(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        ...


@runtime_checkable
class FundingProvider(Protocol):
    """Supplies a (time x symbol) daily realized-funding panel.

    Daily funding = sum of the intraday settlements within each UTC day. Sign
    convention: positive = longs pay shorts (Bybit native).
    """

    def funding_panel(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        ...
