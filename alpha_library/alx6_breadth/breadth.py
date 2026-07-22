"""ALX6 breadth — book construction on a PIT-admitted, liquidity-screened universe.

Feeds panels to the FROZEN backtester
(`alpha_library.alx_funding_price_validation.validate.backtest`) unchanged — the
signal (-7d funding mean, top/bottom 30%, lag 1, hold 1) is never touched. The
ONLY new machinery is (a) point-in-time admission masking and (b) the locked
illiquidity-tiered maker cost applied to per-name turnover. All thresholds are
frozen in PREREGISTRATION.md §2/§4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx_funding_price_validation.validate import backtest
from alpha_library.alx4_regime_analysis import characterize as ch

# LOCKED admission / cost parameters (PREREGISTRATION.md §2, §4)
BUFFER_DAYS = 30
MIN_DAYS = 252
MIN_ADV = 5e6
PHI = 0.80
K_PCT = 0.30
LAG = 1

# illiquidity-tiered realistic-maker round-trip cost (fractional) by median ADV
TIER_50M = 0.0007   # ADV >= $50M  (= ALX5 realistic maker)
TIER_10M = 0.0012   # $10M <= ADV < $50M
TIER_5M = 0.0020    # $5M  <= ADV < $10M


def cost_for_adv(adv: float) -> float:
    """Locked per-name round-trip maker cost by median eligible-window ADV."""
    if adv >= 50e6:
        return TIER_50M
    if adv >= 10e6:
        return TIER_10M
    return TIER_5M  # >= $5M guaranteed by the screen


def eligibility_start(onboard_ms: dict, base_to_sym: dict, index: pd.DatetimeIndex,
                      cols) -> dict:
    """First eligible timestamp per base = onboard + BUFFER_DAYS (PIT entry)."""
    out = {}
    for c in cols:
        sym = base_to_sym[c]
        ob = pd.to_datetime(onboard_ms[sym], unit="ms", utc=True)
        out[c] = ob.normalize() + pd.Timedelta(days=BUFFER_DAYS)
    return out


def mask_pit(close: pd.DataFrame, dfund: pd.DataFrame, starts: dict):
    """NaN every value before each name's eligibility start (kills entry look-ahead)."""
    cm, fm = close.copy(), dfund.copy()
    for c in close.columns:
        s = starts.get(c)
        if s is not None:
            cm.loc[cm.index < s, c] = np.nan
            if c in fm.columns:
                fm.loc[fm.index < s, c] = np.nan
    return cm, fm


def window_adv(qvol: pd.DataFrame, close_m: pd.DataFrame) -> pd.Series:
    """Median daily quote volume over each name's eligible (post-mask) window."""
    elig = close_m.notna()
    q = qvol.reindex_like(close_m).where(elig)
    return q.median()


def screen(close_m: pd.DataFrame, adv: pd.Series):
    """Admitted cols = >= MIN_DAYS eligible days AND median ADV >= MIN_ADV."""
    days = close_m.notna().sum()
    ok = [c for c in close_m.columns
          if days.get(c, 0) >= MIN_DAYS and adv.get(c, 0.0) >= MIN_ADV]
    return ok


def tiered_book(close_m: pd.DataFrame, dfund_m: pd.DataFrame, cols, adv: pd.Series,
                *, phi: float = PHI, lag: int = LAG):
    """Frozen book on `cols` with the locked illiquidity-tiered maker cost.

    Returns dict: net (φ·gross − Σ per-name turnover·cost), applied, ic-series,
    gross, and the per-day tiered cost. gross/price/funding come from the frozen
    engine (cost=0); the tiered cost is applied to per-name turnover here.
    """
    idx = close_m.index
    ret = close_m[cols].pct_change()
    fund = dfund_m[cols].reindex(index=idx)
    signal = (-fund.rolling(7).mean())
    res = backtest(ret, signal, funding=fund, lag=lag, cost=0.0, k_pct=K_PCT)
    applied = res["applied"]                       # (T, N), lagged
    gross = res["price"] + res["funding"]          # cost=0 -> net is gross
    dw = np.abs(np.diff(applied, axis=0, prepend=0.0))
    c = np.array([cost_for_adv(adv.get(col, 0.0)) for col in cols])  # (N,)
    day_cost = (dw * c).sum(axis=1)
    net = phi * gross - day_cost
    ic = ch.rank_ic_series(signal, ret)
    return {"net": net, "applied": applied, "gross": gross, "day_cost": day_cost,
            "ic": ic, "cols": list(cols), "index": idx}
