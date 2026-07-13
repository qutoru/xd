"""Cross-sectional normalization of a per-symbol series.

Puts heterogeneous signal outputs (scores or weights) on a common scale before
combination. Pure functions on a pandas Series; no strategy knowledge.
"""

from __future__ import annotations

import pandas as pd


def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score (population std). Degenerate input -> all zeros."""
    s = s.astype("float64")
    valid = s.dropna()
    sd = valid.std(ddof=0)
    if valid.empty or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - valid.mean()) / sd).fillna(0.0)


def demean(s: pd.Series) -> pd.Series:
    """Subtract the cross-sectional mean (market-neutralize the series)."""
    s = s.astype("float64")
    return s - s.mean()


def rank(s: pd.Series) -> pd.Series:
    """Cross-sectional rank in [-1, 1] (robust to outliers). NaNs -> 0."""
    r = s.rank(method="average")
    n = r.notna().sum()
    if n <= 1:
        return pd.Series(0.0, index=s.index)
    return ((r - 1) / (n - 1) * 2 - 1).fillna(0.0)
