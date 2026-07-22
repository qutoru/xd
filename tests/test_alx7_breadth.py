"""Known-answer unit tests for ALX7 breadth mechanics (NOT the alpha).

Guard the decoupled-anchor plumbing: book() reduces to the frozen engine under a
uniform cost (so the 40-major reference reproduces ALX5), the new-name screen and
PIT masking behave, majors vs new names get the right cost split, and added names
carry zero weight before their onboard+buffer. No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx_funding_price_validation.validate import backtest
from alpha_library.alx7_breadth import breadth as bd


def _panel(T=400, N=6, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0),
                         index=idx, columns=cols)
    dfund = pd.DataFrame(rng.normal(0, 1e-4, (T, N)), index=idx, columns=cols)
    qvol = pd.DataFrame(1e8, index=idx, columns=cols)
    return close, dfund, qvol, idx, cols


def test_book_reduces_to_frozen_engine():
    """book() with a UNIFORM cost and phi=1 equals validate.backtest's net —
    this is what makes the 40-major reference reproduce the ALX5 book."""
    close, dfund, _, _, cols = _panel()
    c = 0.0007
    cost_map = {col: c for col in cols}
    got = bd.book(close, dfund, cols, cost_map, phi=1.0)["net"]
    ret = close[cols].pct_change()
    fund = dfund[cols].reindex(index=ret.index)
    sig = (-fund.rolling(7).mean())
    want = backtest(ret, sig, funding=fund, lag=bd.LAG, cost=c, k_pct=bd.K_PCT)["net"]
    assert np.allclose(got, want, atol=1e-15)


def test_new_cost_tiers_and_major_split():
    """New names get tiered cost; the major flat rate is distinct from thin tiers."""
    assert bd.new_cost_for_adv(60e6) == bd.TIER_50M
    assert bd.new_cost_for_adv(20e6) == bd.TIER_10M
    assert bd.new_cost_for_adv(6e6) == bd.TIER_5M
    assert bd.MAJOR_COST == bd.TIER_50M < bd.TIER_10M < bd.TIER_5M


def test_screen_new_filters_short_and_illiquid():
    close, dfund, qvol, idx, cols = _panel()
    starts = {c: idx[0] for c in cols}
    starts["C0"] = idx[300]                       # only 100 eligible days (< 252)
    cm, _ = bd.mask_new(close, dfund, starts)
    q = qvol.copy(); q["C1"] = 1e6                # C1 median ADV $1M (< $5M)
    adv = bd.window_adv(q, cm)
    admitted = bd.screen_new(cm, adv)
    assert "C0" not in admitted and "C1" not in admitted and "C2" in admitted


def test_new_name_zero_weight_before_onboard():
    close, dfund, qvol, idx, cols = _panel()
    starts = {c: idx[0] for c in cols}
    starts["C0"] = idx[150]
    cm, fm = bd.mask_new(close, dfund, starts)
    adv = bd.window_adv(qvol, cm)
    cost_map = {c: bd.MAJOR_COST for c in cols}
    bk = bd.book(cm, fm, cols, cost_map)
    j = cols.index("C0")
    assert np.allclose(bk["applied"][:150, j], 0.0, atol=1e-15)
