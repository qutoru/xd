"""Bybit-backed data providers (thin wrappers over the production data layer).

These implement the platform data interfaces by delegating to
:mod:`crypto_signal_bot.data` (Production Core) — the same public helpers the ML
bot uses. No research code is imported. Panels are aggregated to a daily UTC grid
here (production-owned logic), not borrowed from research.
"""

from __future__ import annotations

import pandas as pd

from crypto_signal_bot.data.derivatives import fetch_funding
from crypto_signal_bot.data.fetcher import fetch_ohlcv
from crypto_signal_bot.data.universe import get_universe

from crypto_signal_bot.platform.data.cache import ParquetCache

_SECONDS_PER_DAY = 24 * 60 * 60


def _lookback_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max(1, int((end - start).total_seconds() // _SECONDS_PER_DAY) + 2)


class BybitUniverseProvider:
    """Live top-N liquid USDT perps (delegates to the production universe)."""

    def universe(self, asof: pd.Timestamp) -> list[str]:  # asof unused: live set
        return list(get_universe())


class BybitDailyBarProvider:
    """Daily close panel assembled from Bybit klines."""

    def __init__(self, cache: ParquetCache | None = None) -> None:
        self.cache = cache

    def _closes(self, symbol: str, days: int) -> pd.Series:
        key = f"close_{symbol}"
        if self.cache is not None and self.cache.is_fresh(key, _SECONDS_PER_DAY):
            cached = self.cache.load(key)
            if cached is not None:
                return cached.iloc[:, 0]
        df = fetch_ohlcv(symbol=symbol, interval="D", history_days=days)
        ser = pd.Series(
            df["close"].to_numpy(dtype="float64"),
            index=pd.to_datetime(df["datetime"], utc=True),
            name=symbol,
        )
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        if self.cache is not None:
            self.cache.save(key, ser.to_frame())
        return ser

    def close_panel(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        days = _lookback_days(start, end)
        cols = {s: self._closes(s, days) for s in symbols}
        panel = pd.DataFrame(cols).sort_index()
        return panel.loc[(panel.index >= start) & (panel.index <= end)]


class BybitFundingProvider:
    """Daily realized-funding panel (sum of settlements per UTC day)."""

    def __init__(self, cache: ParquetCache | None = None) -> None:
        self.cache = cache

    def _daily_funding(self, symbol: str, days: int) -> pd.Series:
        key = f"funding_{symbol}"
        if self.cache is not None and self.cache.is_fresh(key, _SECONDS_PER_DAY):
            cached = self.cache.load(key)
            if cached is not None:
                return cached.iloc[:, 0]
        f = fetch_funding(symbol, history_days=days)
        if f.empty:
            return pd.Series(dtype="float64", name=symbol)
        ser = pd.Series(
            f["funding_rate"].to_numpy(dtype="float64"),
            index=pd.to_datetime(f["timestamp"], unit="ms", utc=True),
        )
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        daily = ser.resample("1D").sum()
        daily.name = symbol
        if self.cache is not None:
            self.cache.save(key, daily.to_frame())
        return daily

    def funding_panel(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        days = _lookback_days(start, end)
        cols = {s: self._daily_funding(s, days) for s in symbols}
        panel = pd.DataFrame(cols).sort_index()
        return panel.loc[(panel.index >= start) & (panel.index <= end)]
