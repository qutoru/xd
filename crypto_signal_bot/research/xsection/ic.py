"""Rank-IC diagnostics for cross-sectional features (Stage A/B statistics).

Everything here is a measurement, not an optimization: rank information
coefficient per bar, its Newey-West t-stat (bar-level ICs autocorrelate, so the
naive t is inflated), IC information ratio, IC autocorrelation, walk-forward
fold stability with a binomial sign test, volatility-regime breakdown, and the
microstructure kill-test (one-bar execution gap). Stage A adds the cost-anchored
cross-sectional dispersion check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _row_rank(m: np.ndarray) -> np.ndarray:
    """Rank each row (ignoring NaN), returning NaN where input is NaN."""
    out = np.full(m.shape, np.nan)
    for i in range(m.shape[0]):
        row = m[i]
        mask = ~np.isnan(row)
        if mask.sum() >= 2:
            out[i, mask] = stats.rankdata(row[mask])
    return out


def rank_ic_series(feature: pd.DataFrame, target: pd.DataFrame, *, min_names: int = 5) -> pd.Series:
    """Per-bar Spearman rank-IC between a feature and a forward target.

    Both are (time x symbol). Bars with fewer than ``min_names`` jointly-valid
    symbols are dropped (IC undefined / too noisy).
    """
    f, t = feature.align(target, join="inner")
    fr = _row_rank(f.to_numpy(dtype="float64"))
    tr = _row_rank(t.to_numpy(dtype="float64"))
    ics = np.full(fr.shape[0], np.nan)
    for i in range(fr.shape[0]):
        mask = ~np.isnan(fr[i]) & ~np.isnan(tr[i])
        if mask.sum() >= min_names:
            a, b = fr[i][mask], tr[i][mask]
            if a.std() > 0 and b.std() > 0:
                ics[i] = np.corrcoef(a, b)[0, 1]
    return pd.Series(ics, index=f.index).dropna()


def newey_west_tstat(x: np.ndarray, lag: int) -> float:
    """t-stat of the mean of ``x`` using a Newey-West (HAC) standard error."""
    x = np.asarray(x, dtype="float64")
    n = x.size
    if n < 2:
        return 0.0
    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / n
    var = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1)
        cov = np.dot(xc[l:], xc[:-l]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 0.0) / n)
    return float(x.mean() / se) if se > 0 else 0.0


def ic_autocorr(ic: pd.Series, lags: int = 5) -> list[float]:
    """Autocorrelation of the IC series at lags 1..``lags``."""
    x = ic.to_numpy()
    x = x - x.mean()
    denom = np.dot(x, x)
    return [float(np.dot(x[l:], x[:-l]) / denom) if denom > 0 else 0.0 for l in range(1, lags + 1)]


def summarize_ic(ic: pd.Series, *, nw_lag: int) -> dict:
    """Core IC statistics for a signal."""
    x = ic.to_numpy()
    mean, std = float(x.mean()), float(x.std(ddof=1))
    return {
        "mean_ic": mean,
        "std_ic": std,
        "ic_ir": mean / std if std > 0 else 0.0,
        "nw_tstat": newey_west_tstat(x, nw_lag),
        "n_obs": int(x.size),
        "hit_rate": float(np.mean(x > 0)),
        "ac1": ic_autocorr(ic, 1)[0],
    }


def walk_forward_folds(ic: pd.Series, n_splits: int = 6) -> tuple[pd.DataFrame, float]:
    """Mean IC per chronological fold + binomial p that the sign is not chance."""
    x = ic.to_numpy()
    parts = np.array_split(x, n_splits)
    rows = [{"fold": k + 1, "mean_ic": float(p.mean()), "n": int(p.size)} for k, p in enumerate(parts)]
    table = pd.DataFrame(rows)
    signs = np.sign(table["mean_ic"].to_numpy())
    dominant = 1 if (signs > 0).sum() >= (signs < 0).sum() else -1
    k = int((signs == dominant).sum())
    # Two-sided binomial test that k of n folds share a sign under p=0.5.
    p_binom = float(stats.binomtest(k, len(signs), 0.5, alternative="greater").pvalue)
    return table, p_binom


def regime_breakdown(ic: pd.Series, returns: pd.DataFrame, *, vol_window: int = 96) -> pd.DataFrame:
    """Mean IC within low/mid/high market-volatility terciles."""
    mkt_vol = returns.mean(axis=1).rolling(vol_window).std().shift(1)
    mkt_vol = mkt_vol.reindex(ic.index)
    q1, q2 = mkt_vol.quantile([1 / 3, 2 / 3])
    label = pd.Series("mid", index=ic.index)
    label[mkt_vol <= q1] = "low_vol"
    label[mkt_vol >= q2] = "high_vol"
    rows = []
    for name in ("low_vol", "mid", "high_vol"):
        seg = ic[label == name]
        rows.append({"regime": name, "mean_ic": float(seg.mean()) if len(seg) else np.nan, "n": int(len(seg))})
    return pd.DataFrame(rows)


def cross_sectional_dispersion(returns: pd.DataFrame) -> pd.Series:
    """Per-bar cross-sectional std of market-neutral returns (Stage A)."""
    resid = returns.sub(returns.mean(axis=1), axis=0)
    return resid.std(axis=1)
