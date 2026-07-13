"""Platform cross-sectional business logic — via the NEW architecture.

Replaces the legacy sleeve.py known-answer test. Anchors the same funding-ranked
dollar-neutral book logic and the shadow price+funding PnL leg, but through the
plugin -> portfolio -> shadow path (no legacy module imported).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.portfolio import allocator
from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig, build_target_book
from crypto_signal_bot.platform.shadow.replay import replay_day
from crypto_signal_bot.platform.shadow.report import ShadowParams
from crypto_signal_bot.platform.signals.registry import load_signal

SYMS = ["A", "B", "C", "D", "E", "F"]


def _snapshot(asof, n=12):
    idx = pd.date_range(end=asof, periods=n, freq="D")
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.02, (n, 6)), axis=0), index=idx, columns=SYMS
    )
    # Constant per-symbol funding means A<B<...<F. ALX = -(mean funding), so the
    # most-negative-funding name (A) gets the highest score -> should be long.
    means = np.array([-3, -2, -1, 1, 2, 3]) * 1e-4
    funding = pd.DataFrame(np.tile(means, (n, 1)), index=idx, columns=SYMS)
    return MarketSnapshot(asof, SYMS, closes, closes.pct_change(), funding)


def test_alx_plugin_builds_dollar_neutral_funding_book():
    asof = pd.Timestamp("2023-06-01", tz="UTC")
    scores = load_signal("alx").generate(_snapshot(asof), {"lookback": 7})
    book = build_target_book(
        {"alx": scores}, config=PortfolioConfig(alloc="topk", k_pct=0.34, gross_target=1.0)
    )
    assert np.isclose(book.net, 0.0, atol=1e-12)  # dollar-neutral
    assert np.isclose(book.gross, 1.0)
    # Cheapest-funding name long, richest short (same business logic as sleeve).
    assert book.weights["A"] > 0 and book.weights["F"] < 0


def test_allocator_dollar_neutral_topk_known_answer():
    w = allocator.dollar_neutral_topk(pd.Series([1, 2, 3, 4, 5, 6], index=SYMS, dtype="float64"), 0.34)
    assert np.isclose(w.sum(), 0.0) and np.isclose(w.abs().sum(), 1.0)
    assert (w[["E", "F"]] > 0).all() and (w[["A", "B"]] < 0).all()


def test_shadow_replay_price_plus_funding_leg():
    # Same economics sleeve.shadow_replay encoded: net = price + funding - costs.
    asof = pd.Timestamp("2023-06-01", tz="UTC")
    snap = _snapshot(asof)
    book = TargetBook(asof=asof, weights=pd.Series({"A": 0.5, "F": -0.5}))
    rep = replay_day(book, None, snap, ShadowParams(fee_rate=0.0, slippage_rate=0.0))
    assert np.isclose(rep.daily_pnl, rep.price_pnl + rep.funding_pnl)
    assert rep.funding_pnl != 0.0  # the funding leg is present and non-trivial
