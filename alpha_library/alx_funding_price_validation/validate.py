"""ALX independent falsification of the funding->price alpha (see PREREGISTRATION.md).

Deliberately re-derives the funding-ranked dollar-neutral daily book from scratch
(own weight builder and backtester) rather than importing AL-1's carry code, so a
matching result rules out AL-1-specific implementation bugs. Reuses only neutral
shared infrastructure (universe loader, metrics, Newey-West, factor features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.research.metrics import bars_per_year

BPY = bars_per_year(24 * 60)  # daily
NW_LAG = 5


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    sd = x.std(ddof=1) if x.size > 1 else 0.0
    return float(x.mean() / sd * np.sqrt(BPY)) if sd > 0 else 0.0


def weights_from_signal(sig_row: np.ndarray, k_pct: float = 0.30) -> np.ndarray:
    """Independent dollar-neutral equal-weight top/bottom-k% weights (gross 1)."""
    w = np.zeros_like(sig_row, dtype="float64")
    valid = np.where(~np.isnan(sig_row))[0]
    n = valid.size
    if n < 4:
        return w
    k = max(1, int(n * k_pct))
    order = valid[np.argsort(sig_row[valid])]  # ascending
    shorts, longs = order[:k], order[-k:]
    w[longs] = 0.5 / len(longs)
    w[shorts] = -0.5 / len(shorts)
    return w


def backtest(
    ret: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    funding: pd.DataFrame | None = None,
    lag: int = 1,
    cost: float = 0.00075,
    k_pct: float = 0.30,
) -> dict:
    """Independent daily long/short backtest. net = price + funding - turnover cost."""
    f = signal.reindex_like(ret).to_numpy(dtype="float64")
    r = ret.to_numpy(dtype="float64")
    fund = (funding.reindex_like(ret).to_numpy(dtype="float64")
            if funding is not None else np.zeros_like(r))
    T, N = r.shape

    w = np.zeros((T, N))
    for t in range(T):
        w[t] = weights_from_signal(f[t], k_pct)
    applied = w.copy() if lag == 0 else np.zeros_like(w)
    if lag > 0:
        applied[lag:] = w[:-lag]

    price = np.nansum(applied * r, axis=1)
    funding_pnl = -np.nansum(applied * np.nan_to_num(fund), axis=1)
    turn = np.abs(np.diff(applied, axis=0, prepend=0.0)).sum(axis=1)
    net = price + funding_pnl - turn * cost
    return {"price": price, "funding": funding_pnl, "net": net, "turnover": turn,
            "applied": applied}


def ols_hac(y: np.ndarray, X: np.ndarray, lag: int = NW_LAG):
    """OLS with Newey-West (HAC) standard errors. X must include a const column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        wgt = 1.0 - l / (lag + 1)
        G = u[l:].T @ u[:-l]
        S += wgt * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else 0.0
    return {"beta": beta, "t": tstat, "resid": resid, "r2": float(r2)}
