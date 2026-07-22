"""ALX7 breadth (decoupled anchor) — book construction.

Self-contained (does not import the FAILED ALX6 book code). Feeds panels to the
FROZEN backtester (`validate.backtest`) unchanged; the signal is never touched.
New machinery only: PIT masking of NEW names, the locked liquidity screen on NEW
names, and a per-name cost vector (majors @7 bps, new names @illiquidity-tier),
all with the ALX5 non-fill haircut φ = 0.80. Thresholds frozen in
PREREGISTRATION.md §2/§4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx_funding_price_validation.validate import backtest
from alpha_library.alx4_regime_analysis import characterize as ch

# LOCKED params (PREREGISTRATION.md §2, §4)
BUFFER_DAYS = 30      # NEW-name warm-up only
MIN_DAYS = 252
MIN_ADV = 5e6
PHI = 0.80
K_PCT = 0.30
LAG = 1

MAJOR_COST = 0.0007   # reference 40 majors: flat ALX5 maker rate
TIER_50M = 0.0007     # new name, ADV >= $50M
TIER_10M = 0.0012     # new name, $10M <= ADV < $50M
TIER_5M = 0.0020      # new name, $5M  <= ADV < $10M


def new_cost_for_adv(adv: float) -> float:
    """Locked per-new-name round-trip maker cost by median eligible-window ADV."""
    if adv >= 50e6:
        return TIER_50M
    if adv >= 10e6:
        return TIER_10M
    return TIER_5M


def new_start(onboard_ms: dict, base_to_sym: dict, cols) -> dict:
    """First eligible ts per NEW name = onboard + BUFFER_DAYS (PIT entry)."""
    out = {}
    for c in cols:
        ob = pd.to_datetime(onboard_ms[base_to_sym[c]], unit="ms", utc=True)
        out[c] = ob.normalize() + pd.Timedelta(days=BUFFER_DAYS)
    return out


def mask_new(close: pd.DataFrame, dfund: pd.DataFrame, starts: dict):
    """NaN every value before each NEW name's eligibility start."""
    cm, fm = close.copy(), dfund.copy()
    for c in cm.columns:
        s = starts.get(c)
        if s is not None:
            cm.loc[cm.index < s, c] = np.nan
            if c in fm.columns:
                fm.loc[fm.index < s, c] = np.nan
    return cm, fm


def window_adv(qvol: pd.DataFrame, close_m: pd.DataFrame) -> pd.Series:
    """Median daily quote volume over each name's eligible (post-mask) window."""
    q = qvol.reindex_like(close_m).where(close_m.notna())
    return q.median()


def screen_new(close_m: pd.DataFrame, adv: pd.Series):
    """Admitted NEW names = >= MIN_DAYS eligible days AND median ADV >= MIN_ADV."""
    days = close_m.notna().sum()
    return [c for c in close_m.columns
            if days.get(c, 0) >= MIN_DAYS and adv.get(c, 0.0) >= MIN_ADV]


def book(close_p: pd.DataFrame, dfund_p: pd.DataFrame, cols, cost_map: dict,
         *, phi: float = PHI, lag: int = LAG):
    """Frozen book on `cols` with a per-name cost vector `cost_map`.

    net = φ·gross − Σ_i (per-name turnover_i · cost_i). gross/price/funding come
    from the frozen engine (cost=0); per-name cost applied here.
    """
    ret = close_p[cols].pct_change()
    fund = dfund_p[cols].reindex(index=ret.index)
    signal = (-fund.rolling(7).mean())
    res = backtest(ret, signal, funding=fund, lag=lag, cost=0.0, k_pct=K_PCT)
    applied = res["applied"]
    gross = res["price"] + res["funding"]
    dw = np.abs(np.diff(applied, axis=0, prepend=0.0))
    c = np.array([cost_map[col] for col in cols])
    net = phi * gross - (dw * c).sum(axis=1)
    ic = ch.rank_ic_series(signal, ret)
    return {"net": net, "applied": applied, "gross": gross, "ic": ic,
            "cols": list(cols), "index": ret.index}
