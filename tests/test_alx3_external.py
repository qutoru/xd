"""Known-answer unit tests for the ALX3 external-replication mechanics.

These guard the reconstruction/backtest plumbing (not the alpha): funding sign
convention, point-in-time no-leak, and the statistics helpers. No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx3_external_replication import replicate as rp


def _panels(T=400, N=8, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0),
                         index=idx, columns=cols)
    dfund = pd.DataFrame(rng.normal(0, 1e-4, (T, N)), index=idx, columns=cols)
    return close, dfund, idx, cols


def test_point_in_time_no_leak():
    """Corrupting funding AFTER t0 must not change PnL through t0."""
    close, dfund, idx, cols = _panels()
    res = rp.make_book(close, dfund, cols, idx)
    net = res["net"]
    t0 = len(net) // 2
    dfund_c = dfund.copy()
    dfund_c.loc[res["index"][t0 + 1]:] += 1.0
    net_c = rp.make_book(close, dfund_c, cols, idx)["net"]
    assert np.allclose(net[:t0], net_c[:t0], atol=1e-12)


def test_funding_sign_short_receives():
    """A name with persistently POSITIVE funding is shorted (signal = -mean<0),
    and shorting a positive-funding name RECEIVES funding => funding PnL >= 0."""
    T, N = 200, 6
    idx = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(N)]
    close = pd.DataFrame(100.0, index=idx, columns=cols)   # flat prices, no price PnL
    dfund = pd.DataFrame(0.0, index=idx, columns=cols)
    dfund["C0"] = 0.001                                    # crowded-long name pays funding
    dfund["C1"] = -0.001                                   # discouraged name
    res = rp.make_book(close, dfund, cols, idx)
    assert np.nansum(res["funding"]) > 0                   # net funding income positive
    assert np.allclose(np.nansum(res["price"]), 0.0, atol=1e-9)  # flat prices -> no price PnL


def test_bootstrap_ci_orders():
    close, dfund, idx, cols = _panels()
    net = rp.make_book(close, dfund, cols, idx)["net"]
    lo, hi = rp.block_bootstrap_ci(net, n=300)
    assert lo <= hi


def test_deflated_sharpe_bounds_and_monotone():
    rng = np.random.default_rng(0)
    weak = rng.normal(0.0001, 0.02, 800)
    strong = rng.normal(0.003, 0.02, 800)
    dw = rp.deflated_sharpe(weak)
    dstr = rp.deflated_sharpe(strong)
    assert 0.0 <= dw <= 1.0 and 0.0 <= dstr <= 1.0 and dstr > dw
