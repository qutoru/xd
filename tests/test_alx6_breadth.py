"""Known-answer unit tests for ALX6 breadth mechanics (NOT the alpha).

Guard the new plumbing only: illiquidity-tier cost monotonicity, PIT-admission
masking, the liquidity/history screen, and pre-onboard zero-weight (entry has no
look-ahead). No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx6_breadth import breadth as bd


def _panel(T=400, N=6, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0),
                         index=idx, columns=cols)
    dfund = pd.DataFrame(rng.normal(0, 1e-4, (T, N)), index=idx, columns=cols)
    qvol = pd.DataFrame(1e8, index=idx, columns=cols)  # all liquid
    return close, dfund, qvol, idx, cols


def test_cost_tiers_monotone():
    """Cheaper tier only for deeper liquidity; cost is non-increasing in ADV."""
    assert bd.cost_for_adv(60e6) == bd.TIER_50M
    assert bd.cost_for_adv(20e6) == bd.TIER_10M
    assert bd.cost_for_adv(6e6) == bd.TIER_5M
    assert bd.TIER_50M < bd.TIER_10M < bd.TIER_5M


def test_mask_pit_nans_before_start():
    """mask_pit NaNs every close/funding value before a name's eligibility start."""
    close, dfund, _, idx, cols = _panel()
    starts = {c: idx[0] for c in cols}
    starts["C0"] = idx[100]                      # C0 admitted only from day 100
    cm, fm = bd.mask_pit(close, dfund, starts)
    assert cm["C0"].iloc[:100].isna().all()
    assert cm["C0"].iloc[100:].notna().all()
    assert fm["C0"].iloc[:100].isna().all()
    assert cm["C1"].notna().all()                # untouched name intact


def test_screen_filters_short_and_illiquid():
    """Screen drops names with < MIN_DAYS eligible days or median ADV < MIN_ADV."""
    close, dfund, qvol, idx, cols = _panel()
    starts = {c: idx[0] for c in cols}
    starts["C0"] = idx[T_short := 400 - 100]     # C0 has only 100 eligible days (< 252)
    cm, _ = bd.mask_pit(close, dfund, starts)
    q = qvol.copy()
    q["C1"] = 1e6                                 # C1 median ADV = $1M (< $5M)
    adv = bd.window_adv(q, cm)
    admitted = bd.screen(cm, adv)
    assert "C0" not in admitted                   # too short
    assert "C1" not in admitted                   # too illiquid
    assert "C2" in admitted


def test_tiered_book_pre_onboard_zero_weight():
    """A masked name carries zero applied weight before its eligibility start."""
    close, dfund, qvol, idx, cols = _panel()
    starts = {c: idx[0] for c in cols}
    starts["C0"] = idx[150]
    cm, fm = bd.mask_pit(close, dfund, starts)
    adv = bd.window_adv(qvol, cm)
    book = bd.tiered_book(cm, fm, cols, adv)
    j = cols.index("C0")
    assert np.allclose(book["applied"][:150, j], 0.0, atol=1e-15)
