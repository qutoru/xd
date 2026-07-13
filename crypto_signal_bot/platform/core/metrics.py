"""Performance metrics for a per-bar return series (vendored, production-owned).

Faithful standalone copy of the neutral annualization/Sharpe machinery. Imports
only numpy. Returns are treated as simple per-bar returns.
"""

from __future__ import annotations

import numpy as np

_MINUTES_PER_YEAR = 365 * 24 * 60


def bars_per_year(interval_minutes: int) -> float:
    """Number of bars in a 365-day year for a given interval in minutes."""
    return _MINUTES_PER_YEAR / interval_minutes


def sharpe(x: np.ndarray, interval_minutes: int) -> float:
    """Annualized Sharpe of a per-bar return array (0.0 if degenerate)."""
    x = np.asarray(x, dtype="float64")
    sd = x.std(ddof=1) if x.size > 1 else 0.0
    if sd <= 0:
        return 0.0
    return float(x.mean() / sd * np.sqrt(bars_per_year(interval_minutes)))


def book_metrics(
    gross: np.ndarray,
    net: np.ndarray,
    turnover: np.ndarray,
    *,
    interval_minutes: int,
) -> dict:
    """Gross/net Sharpe, drawdown, annual turnover and cost drag for a book."""
    net = np.asarray(net, dtype="float64")
    bpy = bars_per_year(interval_minutes)
    equity = np.cumprod(1.0 + net) if net.size else np.array([1.0])
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min()) if equity.size else 0.0
    years = net.size / bpy if bpy else 0.0
    ann_return = (
        float(equity[-1] ** (1.0 / years) - 1.0)
        if years > 0 and equity[-1] > 0
        else -1.0
    )
    return {
        "gross_sharpe": sharpe(gross, interval_minutes),
        "net_sharpe": sharpe(net, interval_minutes),
        "ann_return": ann_return,
        "max_drawdown": max_dd,
        "turnover_annual": float(np.mean(turnover) * bpy) if net.size else 0.0,
        "cost_drag_annual": float((np.mean(gross) - np.mean(net)) * bpy) if net.size else 0.0,
        "n_bars": int(net.size),
    }
