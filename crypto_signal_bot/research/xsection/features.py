"""Cross-sectional features and the market-neutral forward target.

All quantities are matrices with shape (time x symbol). Every feature is
computable at the close of bar ``t`` (rolling windows exclude the future). The
prediction target is the market-neutral forward return, i.e. each symbol's
forward return with the cross-sectional (equal-weight market) mean removed —
this is the residual a cross-sectional strategy actually trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-bar log returns from a wide close-price panel."""
    return np.log(panel / panel.shift(1))


def residualize(returns: pd.DataFrame) -> pd.DataFrame:
    """Remove the cross-sectional (equal-weight market) mean from each bar.

    This is the minimal market-neutralization: resid[t,i] = r[t,i] - mean_i r[t].
    Beta-weighted neutralization is a refinement we deliberately do not take in
    this smallest-implementation pass.
    """
    return returns.sub(returns.mean(axis=1), axis=0)


def forward_target(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Market-neutral forward return over ``horizon`` bars (the label).

    fwd[t,i] = residualized sum of returns over (t+1 .. t+horizon). Uses future
    bars only for the label (as any supervised target must); features never do.
    """
    fwd = returns.rolling(horizon).sum().shift(-horizon)
    return residualize(fwd)


# --- Predefined features (no additions allowed by the research protocol) -----

def relative_strength(returns: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """Cross-sectional momentum: trailing ``lookback``-bar return, demeaned."""
    trailing = returns.rolling(lookback).sum()
    return trailing.sub(trailing.mean(axis=1), axis=0)


def short_term_reversal(returns: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Cross-sectional reversal: negative of the recent return, demeaned."""
    trailing = returns.rolling(lookback).sum()
    demeaned = trailing.sub(trailing.mean(axis=1), axis=0)
    return -demeaned


def funding_dispersion(funding: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional funding tilt: fade rich funding (demeaned, sign-flipped).

    High relative funding = crowded longs -> expected to underperform, so the
    feature is the negative cross-sectional demeaned funding rate. Requires a
    (time x symbol) funding-rate panel.
    """
    demeaned = funding.sub(funding.mean(axis=1), axis=0)
    return -demeaned
