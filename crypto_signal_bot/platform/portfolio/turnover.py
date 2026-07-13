"""Turnover control against the previously held book.

Scales the whole target-minus-previous delta so that sum|Δw| does not exceed the
cap. Preserves net exposure when both books are dollar-neutral (a scaled
zero-sum delta stays zero-sum). Pure function.
"""

from __future__ import annotations

import pandas as pd


def limit_turnover(
    target: pd.Series,
    prev: pd.Series | None,
    max_turnover: float | None,
) -> pd.Series:
    """Cap one-step turnover sum|target - prev| at ``max_turnover``."""
    if prev is None:
        prev = pd.Series(0.0, index=target.index)
    index = target.index.union(prev.index)
    t = target.reindex(index).fillna(0.0)
    p = prev.reindex(index).fillna(0.0)
    delta = t - p
    tv = float(delta.abs().sum())
    if max_turnover is None or tv <= max_turnover or tv == 0.0:
        return t
    return p + delta * (max_turnover / tv)
