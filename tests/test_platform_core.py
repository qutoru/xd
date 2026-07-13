"""Known-answer tests for the autonomous production platform.

No dependency on research/alpha_library: inputs are tiny hand-built panels whose
dollar-neutral weights and PnL are verifiable by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.core.portfolio import build_book, tranche_weights
from crypto_signal_bot.platform.sleeve import AlxSleeve


def test_tranche_weights_dollar_neutral():
    # 4 names, k_pct 0.30 -> k = max(1, int(4*0.30)) = 1: long highest, short lowest.
    w = tranche_weights(np.array([1.0, 2.0, 3.0, 4.0]), 0.30)
    assert w.sum() == 0.0  # dollar neutral
    assert np.abs(w).sum() == 1.0  # gross 1
    assert w[3] == 0.5 and w[0] == -0.5


def test_build_book_known_pnl():
    idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    cols = ["A", "B", "C", "D"]
    # Constant ranking A<B<C<D -> long D (+0.5), short A (-0.5).
    sig = pd.DataFrame([[1, 2, 3, 4]] * 3, index=idx, columns=cols, dtype=float)
    r = pd.DataFrame(0.0, index=idx, columns=cols)
    r.iloc[1, cols.index("D")] = 0.10
    r.iloc[1, cols.index("A")] = -0.10
    res = build_book(r, sig, hold=1, rebalance=1, k_pct=0.30, exec_lag=1, cost=0.0)
    # exec_lag=1: day1 applies day0 weights; price = 0.5*0.10 + (-0.5)*(-0.10) = 0.10.
    assert np.isclose(res.gross[1], 0.10)
    assert np.isclose(res.net[1], 0.10)
    assert np.isclose(res.turnover[1], 1.0)  # first application: |0.5|+|-0.5|


def test_sign_reversal_flips_pnl():
    idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    cols = ["A", "B", "C", "D"]
    sig = pd.DataFrame([[1, 2, 3, 4]] * 3, index=idx, columns=cols, dtype=float)
    r = pd.DataFrame(0.0, index=idx, columns=cols)
    r.iloc[1, cols.index("D")] = 0.10
    r.iloc[1, cols.index("A")] = -0.10
    up = build_book(r, sig, hold=1, rebalance=1, k_pct=0.30, exec_lag=1, cost=0.0)
    dn = build_book(r, -sig, hold=1, rebalance=1, k_pct=0.30, exec_lag=1, cost=0.0)
    assert np.isclose(up.gross[1], -dn.gross[1])


def test_sleeve_target_book_neutral():
    idx = pd.date_range("2023-01-01", periods=20, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    funding = pd.DataFrame(rng.normal(0, 1e-4, (20, 8)), index=idx, columns=cols)
    book = AlxSleeve().target_book(funding)
    # 8 names, k = int(8*0.30) = 2 -> long 2 short 2, each 0.25.
    assert np.isclose(book.sum(), 0.0, atol=1e-12)
    assert np.isclose(book.abs().sum(), 1.0)


def test_sleeve_shadow_replay_includes_funding_leg():
    idx = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(8)]
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.02, (30, 8)), index=idx, columns=cols)
    funding = pd.DataFrame(rng.normal(0, 1e-4, (30, 8)), index=idx, columns=cols)
    out = AlxSleeve().shadow_replay(returns, funding, cost=0.00075)
    # net = price leg + funding leg, exactly.
    assert np.allclose(out["net"], out["price_net"] + out["funding_pnl"])
    assert np.isfinite(out["net_sharpe"])
