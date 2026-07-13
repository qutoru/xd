"""Dollar-neutral cross-sectional long/short book (vendored, production-owned).

Cross-sectional ranking each rebalance; long the top K%, short the bottom K%;
dollar-neutral, equal-weight, gross ~1.0. Overlapping tranches held ``hold`` bars
(Jegadeesh-Titman) reduce turnover. This is pure machinery — the signal that
drives the ranking is supplied by the caller. Imports only numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BookResult:
    """Per-bar economics of the long/short book plus the applied weights."""

    gross: np.ndarray  # per-bar pre-cost price return
    net: np.ndarray  # gross - turnover*cost (price leg only; funding added by caller)
    turnover: np.ndarray  # per-bar sum_i |Delta weight_i|
    weights: np.ndarray  # (T x N) weights actually applied (post exec-lag)


def tranche_weights(row: np.ndarray, k_pct: float) -> np.ndarray:
    """Dollar-neutral equal-weight: +0.5 across top-K%, -0.5 across bottom-K%."""
    w = np.zeros_like(row, dtype="float64")
    valid = np.where(~np.isnan(row))[0]
    n = valid.size
    if n < 4:
        return w
    k = max(1, int(n * k_pct))
    order = valid[np.argsort(row[valid])]  # ascending
    shorts, longs = order[:k], order[-k:]
    w[longs] = 0.5 / len(longs)
    w[shorts] = -0.5 / len(shorts)
    return w


def build_book(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    hold: int,
    rebalance: int,
    k_pct: float,
    exec_lag: int,
    cost: float,
) -> BookResult:
    """Backtest / replay the market-neutral book on a returns panel.

    Args:
        returns: (time x symbol) per-bar returns.
        signal: (time x symbol) score, higher = go long. Decided at bar close.
        hold: bars each tranche is held.
        rebalance: bars between forming tranches (<= hold).
        k_pct: fraction of names on each side.
        exec_lag: bars between decision and fill (>= 1; no lookahead).
        cost: cost per unit of turnover |Delta weight|.
    """
    f = signal.reindex_like(returns).to_numpy(dtype="float64")
    r = returns.to_numpy(dtype="float64")
    T, N = r.shape

    acc = np.zeros((T, N))
    cnt = np.zeros(T)
    for s in range(0, T, rebalance):
        g = tranche_weights(f[s], k_pct)
        if not g.any():
            continue
        end = min(s + hold, T)
        acc[s:end] += g
        cnt[s:end] += 1
    live = np.divide(acc, cnt[:, None], out=np.zeros_like(acc), where=cnt[:, None] > 0)

    applied = np.zeros_like(live)
    applied[exec_lag:] = live[:-exec_lag]

    gross = np.nansum(applied * r, axis=1)
    dturn = np.abs(np.diff(applied, axis=0, prepend=0.0)).sum(axis=1)
    net = gross - dturn * cost
    return BookResult(gross=gross, net=net, turnover=dturn, weights=applied)
