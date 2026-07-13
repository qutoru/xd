"""Turn a combined cross-sectional score into raw dollar-neutral weights.

Two schemes, both gross=1 and net=0 by construction. Exposure scaling, per-name
caps and turnover live in their own modules — the allocator only maps score->raw
book shape.
"""

from __future__ import annotations

import pandas as pd


def dollar_neutral_topk(score: pd.Series, k_pct: float) -> pd.Series:
    """Long top-K%, short bottom-K%, equal-weight, gross 1, net 0."""
    s = score.dropna()
    n = len(s)
    w = pd.Series(0.0, index=score.index)
    if n < 2:
        return w
    k = max(1, int(n * k_pct))
    order = s.sort_values()
    shorts, longs = order.index[:k], order.index[-k:]
    w[longs] = 0.5 / len(longs)
    w[shorts] = -0.5 / len(shorts)
    return w


def proportional(score: pd.Series) -> pd.Series:
    """Weights proportional to the demeaned score, gross 1, net 0."""
    d = (score - score.mean()).fillna(0.0)
    gross = d.abs().sum()
    if gross == 0:
        return pd.Series(0.0, index=score.index)
    return d / gross
