"""Smallest market-neutral long/short portfolio for Stage C (survival test).

Cross-sectional ranking each rebalance; long the top K%, short the bottom K%;
dollar-neutral, equal-weighted, gross exposure = 1.0. Overlapping tranches
(Jegadeesh-Titman): a portfolio formed at a rebalance is held ``hold`` bars, and
the live book is the average of all tranches currently active — this is the only
mechanism that reduces turnover, and it is a construction choice, not a tuned
knob. No leverage, no vol targeting, no sizing tricks, no risk model beyond
dollar neutrality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_signal_bot.research.metrics import bars_per_year


@dataclass(frozen=True)
class PortfolioResult:
    """Per-bar gross/net returns and turnover of the long/short book."""

    gross: np.ndarray
    net: np.ndarray
    turnover: np.ndarray  # per-bar sum_i |Δweight_i|


def _tranche_weights(feat_row: np.ndarray, k_pct: float) -> np.ndarray:
    """Dollar-neutral equal-weight weights: +0.5 across top-K%, -0.5 bottom-K%."""
    w = np.zeros_like(feat_row)
    valid = np.where(~np.isnan(feat_row))[0]
    n = valid.size
    if n < 4:
        return w
    k = max(1, int(n * k_pct))
    order = valid[np.argsort(feat_row[valid])]  # ascending
    shorts, longs = order[:k], order[-k:]
    w[longs] = 0.5 / len(longs)
    w[shorts] = -0.5 / len(shorts)
    return w


def backtest_portfolio(
    returns: pd.DataFrame,
    feature: pd.DataFrame,
    *,
    hold: int,
    rebalance: int,
    k_pct: float,
    exec_lag: int,
    cost: float,
) -> PortfolioResult:
    """Backtest the market-neutral book on a returns panel.

    Args:
        returns: (time x symbol) per-bar simple/log returns.
        feature: (time x symbol) signal, higher = go long. Decided at each bar's
            close (no lookahead).
        hold: bars each tranche is held.
        rebalance: bars between forming new tranches (must be <= hold).
        k_pct: fraction of names on each side (e.g. 0.20 for top/bottom 20%).
        exec_lag: bars between decision and fill (>=1, no lookahead).
        cost: cost per unit of turnover |Δweight| (commission + slippage).

    Returns:
        A :class:`PortfolioResult`.
    """
    f = feature.reindex_like(returns).to_numpy(dtype="float64")
    r = returns.to_numpy(dtype="float64")
    T, N = r.shape

    # Accumulate overlapping tranche weights, normalized per bar so the live
    # book is always dollar-neutral with gross ~1.
    acc = np.zeros((T, N))
    cnt = np.zeros(T)
    for s in range(0, T, rebalance):
        g = _tranche_weights(f[s], k_pct)
        if not g.any():
            continue
        end = min(s + hold, T)
        acc[s:end] += g
        cnt[s:end] += 1
    live = np.divide(acc, cnt[:, None], out=np.zeros_like(acc), where=cnt[:, None] > 0)

    # Apply the execution lag: weights decided at close t fill exec_lag bars later.
    applied = np.zeros_like(live)
    applied[exec_lag:] = live[:-exec_lag]

    gross = np.nansum(applied * r, axis=1)
    dturn = np.abs(np.diff(applied, axis=0, prepend=0.0)).sum(axis=1)
    net = gross - dturn * cost
    return PortfolioResult(gross=gross, net=net, turnover=dturn)


def portfolio_metrics(res: PortfolioResult, interval_minutes: int) -> dict:
    """Gross/net Sharpe, return, drawdown, turnover, cost attribution."""
    bpy = bars_per_year(interval_minutes)

    def _sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0

    net = res.net
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min()) if equity.size else 0.0
    years = net.size / bpy
    ann_return = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else -1.0
    turnover_annual = float(res.turnover.mean() * bpy)
    cost_drag_annual = float((res.gross.mean() - res.net.mean()) * bpy)

    return {
        "gross_sharpe": _sharpe(res.gross),
        "net_sharpe": _sharpe(net),
        "ann_return": ann_return,
        "max_drawdown": max_dd,
        "turnover_annual": turnover_annual,
        "cost_drag_annual": cost_drag_annual,
        "n_bars": int(net.size),
    }
