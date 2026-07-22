"""Known-answer unit tests for the ALX5 maker-execution mechanics.

These guard the cost/turnover plumbing (NOT the alpha): the hold=1 book equals
the frozen backtester bit-for-bit, turnover is monotone non-increasing in the
hold, net is monotone decreasing in cost, and the phi-haircut arithmetic is
exactly phi*(price+funding) - turnover*cost. No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx_funding_price_validation.validate import backtest
from alpha_library.alx4_regime_analysis import characterize as ch
from alpha_library.alx5_maker_execution import execution as ex


def _panels(T=300, N=10, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0),
                         index=idx, columns=cols)
    dfund = pd.DataFrame(rng.normal(0, 1e-4, (T, N)), index=idx, columns=cols)
    ret, fund, signal = ch.build_signal(close, dfund)
    return ret, fund, signal


def test_hold1_reproduces_frozen_book():
    """held_applied(hold=1) == the frozen backtester's applied weights, and
    book_from_applied reproduces its price/funding/turnover exactly."""
    ret, fund, signal = _panels()
    res = backtest(ret, signal, funding=fund, lag=ex.LAG, cost=ex.MED, k_pct=ex.K_PCT)
    applied = ex.held_applied(signal, ret, hold=1, lag=ex.LAG)
    assert np.allclose(applied, res["applied"], atol=1e-15)
    bk = ex.book_from_applied(applied, ret.to_numpy(dtype="float64"),
                              fund.reindex_like(ret).to_numpy(dtype="float64"))
    assert np.allclose(bk["price"], res["price"], atol=1e-15)
    assert np.allclose(bk["funding"], res["funding"], atol=1e-15)
    assert np.allclose(bk["turnover"], res["turnover"], atol=1e-15)


def test_turnover_monotone_in_hold():
    """Longer holds trade less: average turnover is non-increasing in the hold."""
    ret, fund, signal = _panels()
    avg = []
    for hold in (1, 2, 3, 5):
        applied = ex.held_applied(signal, ret, hold=hold)
        bk = ex.book_from_applied(applied, ret.to_numpy(dtype="float64"),
                                  fund.reindex_like(ret).to_numpy(dtype="float64"))
        avg.append(float(np.mean(bk["turnover"])))
    assert all(avg[i] >= avg[i + 1] - 1e-12 for i in range(len(avg) - 1)), avg
    assert avg[0] > avg[-1]  # strictly cheaper at hold=5 than daily


def test_cost_model_monotone_in_cost():
    """With phi fixed, mean net PnL is strictly decreasing in per-turnover cost."""
    ret, fund, signal = _panels()
    book = ex.frozen_book(ret, signal, fund)
    means = {c: float(np.mean(ex.apply_cost(book["price"], book["funding"],
                                            book["turnover"], c)))
             for c in (ex.IDEAL_MAKER, ex.REAL_MAKER_COST, ex.MED, ex.TAKER)}
    ordered = [means[c] for c in (ex.IDEAL_MAKER, ex.REAL_MAKER_COST, ex.MED, ex.TAKER)]
    assert all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)), ordered


def test_phi_haircut_arithmetic():
    """apply_cost is exactly phi*(price+funding) - turnover*cost."""
    price = np.array([0.010, -0.004, 0.006], dtype="float64")
    funding = np.array([0.001, 0.002, -0.001], dtype="float64")
    turn = np.array([0.5, 1.0, 0.25], dtype="float64")
    cost, phi = ex.REAL_MAKER_COST, ex.REAL_MAKER_PHI
    got = ex.apply_cost(price, funding, turn, cost, phi)
    want = phi * (price + funding) - turn * cost
    assert np.allclose(got, want, atol=1e-18)
    # phi=1 default leaves gross untouched
    got1 = ex.apply_cost(price, funding, turn, cost)
    assert np.allclose(got1, (price + funding) - turn * cost, atol=1e-18)
