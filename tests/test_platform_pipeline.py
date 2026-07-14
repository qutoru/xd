"""Stage 6 — integration pipeline (offline, injected fake data source)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
from crypto_signal_bot.platform.pipeline import DailyPipeline, PipelineResult
from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig
from crypto_signal_bot.platform.shadow.runner import ShadowRunner

SYMS = [f"C{i}" for i in range(8)]


def _snapshot(asof, n=15):
    idx = pd.date_range(end=asof, periods=n, freq="D")
    rng = np.random.default_rng(7)
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (n, 8)), axis=0),
                          index=idx, columns=SYMS)
    # per-symbol funding means so the ALX score is non-degenerate
    means = np.linspace(-1e-4, 1e-4, 8)
    funding = pd.DataFrame(means + rng.normal(0, 1e-5, (n, 8)), index=idx, columns=SYMS)
    return MarketSnapshot(asof, SYMS, closes, closes.pct_change(), funding)


class _FakeSnapshotProvider:
    def __init__(self, snap): self.snap = snap
    def snapshot(self, asof, lookback_days): return self.snap


def _pipeline(snap, execute=True):
    return DailyPipeline(
        _FakeSnapshotProvider(snap),
        signal_names=["alx"],
        execution_engine=ExecutionEngine(InMemoryBroker(value=100.0)),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        lookback_days=10,
        execute=execute,
    )


def test_pipeline_runs_full_chain():
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    snap = _snapshot(asof)
    result = _pipeline(snap).run_once(asof)

    assert isinstance(result, PipelineResult)
    # Portfolio: dollar-neutral book
    assert np.isclose(result.book.net, 0.0, atol=1e-9)
    assert np.isclose(result.book.gross, 1.0, atol=1e-9)
    # Execution ran through the (fake) broker
    assert result.execution is not None and result.execution.n_filled > 0
    # Shadow produced a finite virtual PnL
    assert np.isfinite(result.shadow.daily_pnl)
    assert result.shadow.asof == result.book.asof
    # Futures TradeIntents were generated for the traded names
    assert result.intents
    assert all(i.entry is not None and i.confidence is not None for i in result.intents)


def test_pipeline_carries_prev_book_for_turnover():
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    pipe = _pipeline(_snapshot(asof))
    r1 = pipe.run_once(asof)
    r2 = pipe.run_once(asof + pd.Timedelta(days=1))
    # second day's shadow turnover is measured vs the first book (small since
    # the signal is unchanged here) — the pipeline threaded prev_book through.
    assert r2.shadow.turnover <= r1.shadow.turnover + 1e-9


def test_pipeline_execute_false_skips_orders():
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    result = _pipeline(_snapshot(asof), execute=False).run_once(asof)
    assert result.execution is None
    assert np.isfinite(result.shadow.daily_pnl)  # shadow still runs
