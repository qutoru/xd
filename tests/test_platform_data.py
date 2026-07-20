"""Stage 1 — platform data layer tests (no network; in-memory fakes)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.data.cache import ParquetCache
from crypto_signal_bot.platform.data.interfaces import (
    DailyBarProvider,
    FundingProvider,
    UniverseProvider,
)
from crypto_signal_bot.platform.data.snapshot import DailySnapshotProvider, MarketSnapshot


def _panels(n=12, syms=("A", "B", "C")):
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.02, (n, len(syms))), axis=0),
        index=idx, columns=list(syms),
    )
    funding = pd.DataFrame(
        rng.normal(0, 1e-4, (n, len(syms))), index=idx, columns=list(syms)
    )
    return closes, funding


class _FakeUniverse:
    def __init__(self, syms): self.syms = list(syms)
    def universe(self, asof): return list(self.syms)


class _FakeBars:
    def __init__(self, closes): self.closes = closes
    def close_panel(self, symbols, start, end):
        p = self.closes.loc[(self.closes.index >= start) & (self.closes.index <= end)]
        return p[symbols]


class _FakeFunding:
    def __init__(self, funding): self.funding = funding
    def funding_panel(self, symbols, start, end):
        p = self.funding.loc[(self.funding.index >= start) & (self.funding.index <= end)]
        return p[symbols]


def test_fakes_satisfy_protocols():
    closes, funding = _panels()
    assert isinstance(_FakeUniverse(["A"]), UniverseProvider)
    assert isinstance(_FakeBars(closes), DailyBarProvider)
    assert isinstance(_FakeFunding(funding), FundingProvider)


def test_snapshot_alignment_and_point_in_time():
    closes, funding = _panels(n=12)
    prov = DailySnapshotProvider(_FakeUniverse(["A", "B", "C"]), _FakeBars(closes), _FakeFunding(funding))
    asof = pd.Timestamp("2023-01-08", tz="UTC")
    snap = prov.snapshot(asof, lookback_days=5)

    assert isinstance(snap, MarketSnapshot)
    # Point-in-time: nothing after asof leaks in.
    assert snap.closes.index.max() <= asof
    assert snap.funding.index.max() <= asof
    # Aligned shapes / columns.
    assert list(snap.returns.columns) == snap.symbols
    assert snap.funding.shape == snap.returns.shape
    assert snap.returns.index.equals(snap.funding.index)
    # Returns are the pct-change of the close panel.
    assert np.allclose(
        snap.returns.iloc[-1].to_numpy(),
        (snap.closes.iloc[-1] / snap.closes.iloc[-2] - 1).to_numpy(),
    )


def test_lookback_window_respected():
    closes, funding = _panels(n=20)
    prov = DailySnapshotProvider(_FakeUniverse(["A", "B", "C"]), _FakeBars(closes), _FakeFunding(funding))
    asof = pd.Timestamp("2023-01-15", tz="UTC")
    snap = prov.snapshot(asof, lookback_days=6)
    assert snap.closes.index.min() >= asof - pd.Timedelta(days=6)


def test_incomplete_current_day_dropped():
    # Panel runs through "today"; the current UTC day is still forming (partial
    # funding sum + not-yet-closed daily close) and must not enter the snapshot.
    now = pd.Timestamp("2023-01-12 03:00", tz="UTC")  # early in the current day
    idx = pd.date_range("2023-01-01", periods=12, freq="D", tz="UTC")  # ...through Jan 12
    rng = np.random.default_rng(1)
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (12, 2)), axis=0),
                          index=idx, columns=["A", "B"])
    funding = pd.DataFrame(rng.normal(0, 1e-4, (12, 2)), index=idx, columns=["A", "B"])
    prov = DailySnapshotProvider(_FakeUniverse(["A", "B"]), _FakeBars(closes), _FakeFunding(funding))

    asof = pd.Timestamp("2023-01-12", tz="UTC")  # live runner passes today's midnight
    snap = prov.snapshot(asof, lookback_days=10, now=now)

    # Jan 12 (today, forming) is excluded; the last usable bar is Jan 11 (closed).
    assert snap.closes.index.max() == pd.Timestamp("2023-01-11", tz="UTC")
    assert snap.funding.index.max() == pd.Timestamp("2023-01-11", tz="UTC")


def test_historical_asof_keeps_its_own_bar():
    # A backtest/replay asof dated in the past is fully complete: the completeness
    # guard must NOT strip the asof-dated bar (only the live current day is dropped).
    closes, funding = _panels(n=12)
    prov = DailySnapshotProvider(_FakeUniverse(["A", "B", "C"]), _FakeBars(closes), _FakeFunding(funding))
    asof = pd.Timestamp("2023-01-08", tz="UTC")
    snap = prov.snapshot(asof, lookback_days=6, now=pd.Timestamp("2026-07-20", tz="UTC"))
    assert snap.closes.index.max() == asof  # past day fully retained


def test_parquet_cache_roundtrip(tmp_path):
    cache = ParquetCache(tmp_path / "c")
    assert cache.load("k") is None
    df = pd.DataFrame({"A": [1.0, 2.0]})
    cache.save("k", df)
    pd.testing.assert_frame_equal(cache.load("k"), df)
    assert cache.is_fresh("k", 3600)
    assert not cache.is_fresh("missing", 3600)
