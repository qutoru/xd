"""Rule-based signal generators for the research hypotheses.

Every signal is the target position in {-1, 0, 1} decided at a bar's CLOSE using
only information available up to and including that close. Rolling windows use
``.shift(1)`` where the current bar must be excluded, so there is no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hold(trigger: pd.Series, hold: int) -> pd.Series:
    """Forward-fill discrete triggers for ``hold`` bars (opposite trigger wins)."""
    # NaN between triggers; ffill with a limit keeps a position for `hold` bars.
    filled = trigger.replace(0.0, np.nan).ffill(limit=hold - 1)
    return filled.fillna(0.0)


def sweep_reversion(df: pd.DataFrame, *, lookback: int = 20, hold: int = 8) -> pd.Series:
    """H3 — liquidity-sweep (swing-failure) reversion.

    Bullish: the bar's low pierces the lowest low of the previous ``lookback``
    bars but the close reclaims back above that level (stops grabbed, then
    rejected) -> go long. Bearish is the mirror image -> go short. Positions are
    held for ``hold`` bars unless an opposite sweep fires.
    """
    low, high, close = df["low"], df["high"], df["close"]
    prior_low = low.rolling(lookback).min().shift(1)
    prior_high = high.rolling(lookback).max().shift(1)

    bull = (low < prior_low) & (close > prior_low)
    bear = (high > prior_high) & (close < prior_high)

    trig = pd.Series(0.0, index=df.index)
    trig[bull] = 1.0
    trig[bear] = -1.0
    trig[bull & bear] = 0.0  # ambiguous double-sweep: abstain
    return _hold(trig, hold)


def funding_reversion(df: pd.DataFrame, *, z_window: int = 96, z_thresh: float = 1.5, hold: int = 8) -> pd.Series:
    """H5 — funding-rate imbalance reversion.

    When funding is extremely positive (crowded longs paying shorts) we fade it
    short; extremely negative -> long. "Extreme" is a rolling z-score of the
    funding rate over ``z_window`` bars, thresholded at ``z_thresh``.
    Requires a ``funding_rate`` column.
    """
    if "funding_rate" not in df.columns:
        raise KeyError("funding_reversion needs a 'funding_rate' column (fetch with USE_DERIVATIVES)")
    f = df["funding_rate"]
    mean = f.rolling(z_window).mean().shift(1)
    std = f.rolling(z_window).std().shift(1)
    z = (f - mean) / std

    trig = pd.Series(0.0, index=df.index)
    trig[z > z_thresh] = -1.0  # crowded longs -> fade short
    trig[z < -z_thresh] = 1.0  # crowded shorts -> fade long
    return _hold(trig, hold)


def oi_divergence(df: pd.DataFrame, *, ret_window: int = 4, oi_window: int = 4, hold: int = 8) -> pd.Series:
    """H6 — open-interest divergence.

    A price move on FALLING open interest is position-covering, not new money,
    so it tends to fade: price up + OI down -> short; price down + OI up (new
    shorts piling in near a low) -> long. Requires an ``open_interest`` column.
    """
    if "open_interest" not in df.columns:
        raise KeyError("oi_divergence needs an 'open_interest' column (fetch with USE_DERIVATIVES)")
    price_chg = df["close"].pct_change(ret_window)
    oi_chg = df["open_interest"].pct_change(oi_window)

    trig = pd.Series(0.0, index=df.index)
    trig[(price_chg > 0) & (oi_chg < 0)] = -1.0  # rally on covering -> fade
    trig[(price_chg < 0) & (oi_chg > 0)] = 1.0  # sell-off on new shorts -> fade
    return _hold(trig, hold)
