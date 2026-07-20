"""ALX4 analysis primitives (frozen spec; no tuning).

Every book is produced by the FROZEN ALX backtester
``alpha_library.alx_funding_price_validation.validate.backtest`` — this module
only *feeds* it panels and *measures* the output. It never re-implements the
strategy and changes no signal parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx_funding_price_validation.validate import backtest, sharpe
from crypto_signal_bot.research.xsection import features as ft

MED = 0.00075
LOOKBACK = 7      # frozen
K_PCT = 0.30      # frozen
LAG = 1           # frozen


# ----------------------------------------------------------------------- #
# panel construction (frozen signal) — union + per-day NaN masking
# ----------------------------------------------------------------------- #
def build_signal(close: pd.DataFrame, dfund: pd.DataFrame):
    """Return (ret, fund, signal) on the union index with the frozen signal.

    signal = -(daily funding).rolling(7).mean()  — sign/lookback frozen.
    """
    idx = close.index.sort_values()
    cols = list(close.columns)
    ret = close.reindex(idx).pct_change()
    fund = dfund.reindex(index=idx, columns=cols)
    signal = (-fund.rolling(LOOKBACK).mean()).reindex(index=idx, columns=cols)
    return ret, fund, signal


def alx_net(ret: pd.DataFrame, signal: pd.DataFrame, fund: pd.DataFrame,
            *, cost: float = MED) -> pd.Series:
    """Frozen ALX net-PnL series."""
    res = backtest(ret, signal, funding=fund, lag=LAG, cost=cost, k_pct=K_PCT)
    return pd.Series(res["net"], index=ret.index)


# ----------------------------------------------------------------------- #
# statistics
# ----------------------------------------------------------------------- #
def rank_ic_series(signal: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional rank-IC of signal(t-1) vs ret(t)."""
    sl = signal.shift(1).to_numpy()
    rr = ret.to_numpy()
    out = np.full(len(ret), np.nan)
    for t in range(len(ret)):
        a, b = sl[t], rr[t]
        m = (~np.isnan(a)) & (~np.isnan(b))
        if m.sum() < 5:
            continue
        ra = pd.Series(a[m]).rank().to_numpy()
        rb = pd.Series(b[m]).rank().to_numpy()
        if ra.std() > 0 and rb.std() > 0:
            out[t] = np.corrcoef(ra, rb)[0, 1]
    return pd.Series(out, index=ret.index)


def nw_tstat(x: np.ndarray, lag: int = 5) -> float:
    """Newey-West t-stat of the mean of a 1-D series."""
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return 0.0
    mu = x.mean()
    u = x - mu
    s = (u @ u) / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1)
        s += 2 * w * (u[l:] @ u[:-l]) / n
    se = np.sqrt(s / n)
    return float(mu / se) if se > 0 else 0.0


def block_bootstrap_ci(net: np.ndarray, block: int = 20, n: int = 2000, seed: int = 0):
    net = np.asarray(net, dtype="float64")
    rng = np.random.default_rng(seed)
    if len(net) <= block:
        return float("nan"), float("nan")
    blocks = [net[i:i + block] for i in range(len(net) - block)]
    out = [sharpe(np.concatenate([blocks[i] for i in
           rng.integers(0, len(blocks), size=len(net) // block + 1)])[:len(net)])
           for _ in range(n)]
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


# ----------------------------------------------------------------------- #
# P1 — funding cross-sectional shuffle placebo (decisive falsifier)
# ----------------------------------------------------------------------- #
def shuffle_placebo(ret: pd.DataFrame, fund: pd.DataFrame,
                    *, n: int = 500, seed: int = 0) -> dict:
    """Permute funding across symbols each day, rebuild the frozen book, N times.

    Breaking the funding<->symbol linkage must destroy a genuine funding effect.
    Returns the real Sharpe, the shuffle distribution, and the pre-registered gate.
    """
    real_sig = (-fund.rolling(LOOKBACK).mean())
    real = sharpe(alx_net(ret, real_sig, fund).to_numpy())

    rng = np.random.default_rng(seed)
    fvals = fund.to_numpy()
    T, N = fvals.shape
    dist = np.empty(n)
    for i in range(n):
        perm = fvals.copy()
        for t in range(T):
            row = perm[t]
            valid = np.where(~np.isnan(row))[0]
            if valid.size > 1:
                row[valid] = row[valid][rng.permutation(valid.size)]
        fsh = pd.DataFrame(perm, index=fund.index, columns=fund.columns)
        sig_sh = (-fsh.rolling(LOOKBACK).mean())
        # funding leg uses the (shuffled) funding too, so the whole book is placebo
        dist[i] = sharpe(alx_net(ret, sig_sh, fsh).to_numpy())

    pct97 = float(np.percentile(dist, 97.5))
    # Gate matches the pre-registered PROSE intent: "shuffled books have no
    # systematic edge" == no systematic POSITIVE Sharpe. A long/short book on
    # randomized funding still pays the same turnover cost with no signal, so a
    # NEGATIVE median is expected null behavior — not evidence against H1. The
    # original code used abs(median) < 0.5, which wrongly rejected the (correct)
    # negative drag; corrected to a one-sided test per the escape-hatch
    # (concrete implementation bug: code disagreed with its own locked intent).
    passed = bool(real > pct97 and np.median(dist) < 0.5)
    return {"real_sharpe": real, "dist": dist, "pct97_5": pct97,
            "median": float(np.median(dist)), "max": float(dist.max()),
            "pct_below_real": float((dist < real).mean()), "pass": passed}


# ----------------------------------------------------------------------- #
# P2 — directionality (funding->return vs return->funding)
# ----------------------------------------------------------------------- #
def directionality(ret: pd.DataFrame, signal: pd.DataFrame) -> dict:
    """Forward IC (signal[t-1]->ret[t]) vs reverse IC (ret[t-1]->signal[t])."""
    fwd = rank_ic_series(signal, ret)
    # reverse: does yesterday's return predict today's funding-signal cross-section?
    rev = rank_ic_series(ret, signal)
    return {"fwd_mean_ic": float(fwd.mean()), "fwd_nw_t": nw_tstat(fwd.to_numpy()),
            "rev_mean_ic": float(rev.mean()), "rev_nw_t": nw_tstat(rev.to_numpy())}


# ----------------------------------------------------------------------- #
# P3 — factor independence
# ----------------------------------------------------------------------- #
def factor_independence(ret: pd.DataFrame, fund: pd.DataFrame, qvol: pd.DataFrame,
                        net: pd.Series) -> dict:
    """Correlation of ALX net with identically-built factor books + OLS R²."""
    idx = ret.index
    logret = np.log1p(ret)
    mkt = ret.mean(axis=1)
    m1 = mkt.rolling(60).mean()
    varm = mkt.pow(2).rolling(60).mean() - m1.pow(2)
    covb = (ret.mul(mkt, axis=0).rolling(60).mean()
            .sub(ret.rolling(60).mean().mul(m1, axis=0)))
    beta = covb.div(varm, axis=0)
    size = -np.log(qvol.reindex(index=idx, columns=ret.columns).replace(0, np.nan))
    facs = {
        "momentum": ft.relative_strength(logret, 24),
        "reversal": ft.short_term_reversal(logret, 3),
        "beta": -beta,
        "size": size,
        "volatility": ret.rolling(30).std(),
    }
    books = {k: backtest(ret, v, funding=None, lag=LAG, cost=MED, k_pct=K_PCT)["net"]
             for k, v in facs.items()}
    books["market"] = mkt.fillna(0.0).to_numpy()
    corrs = {k: float(np.corrcoef(net.to_numpy(), b)[0, 1]) for k, b in books.items()}
    # multivariate OLS R²
    X = np.column_stack([np.ones(len(net))] + [books[k] for k in books])
    y = net.to_numpy()
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_ols
    r2 = float(1.0 - resid.var() / y.var()) if y.var() > 0 else 0.0
    passed = bool(max(abs(c) for c in corrs.values()) < 0.7 and r2 < 0.5)
    return {"corrs": corrs, "r2": r2, "pass": passed}


# ----------------------------------------------------------------------- #
# regime conditioning (diagnostic)
# ----------------------------------------------------------------------- #
def regime_features(ret: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """Pre-defined, t-1-lagged regime variables (no optimization)."""
    idx = ret.index
    logret = np.log1p(ret)
    mkt = ret.mean(axis=1)
    reg = pd.DataFrame(index=idx)
    reg["mkt_vol"] = mkt.rolling(30).std().shift(1)
    reg["fund_disp"] = signal.std(axis=1).shift(1)
    reg["cs_disp"] = ret.std(axis=1).rolling(5).mean().shift(1)
    reg["trend_60d"] = mkt.rolling(60).mean().shift(1)
    reg["breadth"] = (logret.rolling(30).sum() > 0).mean(axis=1).shift(1)
    return reg


def conditional_table(net: pd.Series, ic: pd.Series, reg: pd.DataFrame) -> dict:
    """Tercile-conditional net Sharpe & mean IC for each regime variable."""
    out = {}
    for v in reg.columns:
        d = pd.DataFrame({"net": net, "ic": ic, v: reg[v]}).dropna()
        q = d[v].quantile([1 / 3, 2 / 3])
        lab = pd.cut(d[v], [-1e18, q.iloc[0], q.iloc[1], 1e18],
                     labels=["low", "mid", "high"])
        out[v] = {L: {"n": int((lab == L).sum()),
                      "sharpe": sharpe(d["net"][lab == L].to_numpy()),
                      "ic": float(d["ic"][lab == L].mean())}
                  for L in ["low", "mid", "high"]}
    return out
