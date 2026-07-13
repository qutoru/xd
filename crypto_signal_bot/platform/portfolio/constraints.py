"""Exposure constraints on a weight vector (net, gross, per-name cap).

Pure functions; each enforces exactly one limit. The builder composes them.
"""

from __future__ import annotations

import pandas as pd


def enforce_net(weights: pd.Series, net_target: float = 0.0) -> pd.Series:
    """Shift weights so their sum equals ``net_target`` (0 = dollar-neutral)."""
    n = len(weights)
    if n == 0:
        return weights
    return weights - (weights.sum() - net_target) / n


def scale_gross(weights: pd.Series, gross_target: float) -> pd.Series:
    """Scale so sum|w| == ``gross_target`` (preserves net if net==0)."""
    gross = weights.abs().sum()
    if gross == 0:
        return weights
    return weights * (gross_target / gross)


def max_position(weights: pd.Series, max_weight: float, iters: int = 100) -> pd.Series:
    """Cap |w_i| <= ``max_weight`` while keeping the book dollar-neutral.

    Clips over-cap names and redistributes the residual across the still-free
    names, iterating until both the cap and net==0 hold (or names run out).
    """
    w = enforce_net(weights)
    for _ in range(iters):
        capped = w.clip(-max_weight, max_weight)
        residual = -capped.sum()
        free = capped.abs() < max_weight - 1e-12
        if abs(residual) < 1e-15 or free.sum() == 0:
            return capped
        w = capped.copy()
        w[free] = w[free] + residual / int(free.sum())
    return w.clip(-max_weight, max_weight)
