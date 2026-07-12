"""E9 cross-asset lead-lag features (isolated E9 alpha — see PREREGISTRATION.md).

The tradable claim is *differential* diffusion: a common "leader up -> everything
up" move is removed by market-neutral residualisation, so what remains is the
cross-sectional dispersion in how strongly each follower loads on (and lags) the
leader's most recent move. The signal is therefore the follower's leader-beta
times the leader's recent return, cross-sectionally demeaned.

Every quantity is computable at the close of bar ``t``: the rolling beta window
and the leader return both use only information through ``t``. Demeaning is
cross-sectional (across symbols at the same bar) and introduces no time leakage.

This module deliberately does NOT import or use the E7/E8 reversal signal.
"""

from __future__ import annotations

import pandas as pd

BETA_WINDOW = 168  # 7 days at 1h — a standard market-beta window, NOT tuned.
SUSTAINED_AGG = 4  # LL2 leader-impulse aggregation (bars), NOT tuned.


def rolling_beta(followers: pd.DataFrame, leader: pd.Series, window: int = BETA_WINDOW) -> pd.DataFrame:
    """Trailing OLS beta of each follower's return on the leader's return.

    beta_i(t) = cov(r_i, r_L) / var(r_L) over the trailing ``window`` bars ending
    at (and including) bar ``t``. Population moments are used consistently in the
    numerator and denominator so the ddof convention cancels.

    Args:
        followers: (time x symbol) follower log returns.
        leader: (time,) leader log returns, aligned to ``followers.index``.
        window: trailing window length in bars.

    Returns:
        (time x symbol) rolling beta. The first ``window-1`` rows are NaN.
    """
    lead = leader.reindex(followers.index)
    mean_l = lead.rolling(window).mean()
    mean_i = followers.rolling(window).mean()
    mean_il = followers.mul(lead, axis=0).rolling(window).mean()
    cov = mean_il.sub(mean_i.mul(mean_l, axis=0))
    var_l = lead.pow(2).rolling(window).mean() - mean_l.pow(2)
    return cov.div(var_l, axis=0)


def _demean(sig: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (equal-weight) demean at each bar."""
    return sig.sub(sig.mean(axis=1), axis=0)


def lead_lag_impulse(
    followers: pd.DataFrame, leader: pd.Series, *, window: int = BETA_WINDOW
) -> pd.DataFrame:
    """LL1 (primary): beta_i(t) * r_leader(t), cross-sectionally demeaned.

    High-beta followers, right after a leader up-move, are predicted to have
    positive next-bar residual return (they lag and catch up).
    """
    beta = rolling_beta(followers, leader, window)
    sig = beta.mul(leader.reindex(followers.index), axis=0)
    return _demean(sig)


def lead_lag_sustained(
    followers: pd.DataFrame,
    leader: pd.Series,
    *,
    window: int = BETA_WINDOW,
    agg: int = SUSTAINED_AGG,
) -> pd.DataFrame:
    """LL2 (secondary): beta_i(t) * cumulative leader return over ``agg`` bars.

    Tests whether the diffusion is a multi-bar rather than single-bar effect.
    """
    beta = rolling_beta(followers, leader, window)
    lead_cum = leader.reindex(followers.index).rolling(agg).sum()
    sig = beta.mul(lead_cum, axis=0)
    return _demean(sig)
