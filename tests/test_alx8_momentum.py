"""Known-answer unit tests for ALX8 momentum-sleeve mechanics (NOT the alpha).

Guard the plumbing: the momentum signal is exactly the frozen factor, the 50/50
weight blend nets correctly (blending a book with itself == the book), the
realistic-maker net formula, and random-placebo validity. No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.research.xsection import features as ft
from alpha_library.alx8_momentum_sleeve import momentum as mo


def _panel(T=300, N=8, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0),
                         index=idx, columns=cols)
    dfund = pd.DataFrame(rng.normal(0, 1e-4, (T, N)), index=idx, columns=cols)
    ret = close.pct_change()
    return ret, dfund, close


def test_momentum_is_frozen_factor():
    """momentum_signal == relative_strength(log1p(ret), 24), lookback not tuned."""
    ret, _, _ = _panel()
    got = mo.momentum_signal(ret)
    want = ft.relative_strength(np.log1p(ret), 24)
    assert mo.MOM_LOOKBACK == 24
    pd.testing.assert_frame_equal(got, want)


def test_blend_with_self_equals_single_book():
    """A 50/50 blend of a book with ITSELF nets to the same weights -> same net."""
    ret, dfund, _ = _panel()
    fund = dfund.reindex_like(ret)
    res = mo.sleeve(ret, mo.momentum_signal(ret), fund)
    single = mo.realistic_net(res)
    blended, comb = mo.blend_net(res["applied"], res["applied"],
                                 ret.to_numpy("float64"),
                                 fund.to_numpy("float64"))
    assert np.allclose(comb, res["applied"], atol=1e-15)
    assert np.allclose(blended, single, atol=1e-15)


def test_realistic_net_formula():
    """realistic_net = phi*(price+funding) - turnover*cost."""
    res = {"price": np.array([0.01, -0.004, 0.006]),
           "funding": np.array([0.001, 0.0, -0.001]),
           "turnover": np.array([0.2, 0.5, 0.1])}
    got = mo.realistic_net(res)
    want = mo.PHI * (res["price"] + res["funding"]) - res["turnover"] * mo.COST
    assert np.allclose(got, want, atol=1e-18)


def test_random_applied_dollar_neutral():
    """Random placebo weights are dollar-neutral (~0 sum) and only on valid names."""
    ret, _, _ = _panel()
    ap = mo.random_applied(ret, seed=7)
    sums = np.nansum(ap, axis=1)
    assert np.allclose(sums[30:], 0.0, atol=1e-12)     # net-neutral each traded day
    assert np.allclose(ap[0], 0.0, atol=1e-15)         # lag-1: first day flat
