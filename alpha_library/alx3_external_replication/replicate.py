"""ALX3 — build independent per-exchange panels and run the FROZEN ALX book.

Every panel is fed to the frozen backtester
``alpha_library.alx_funding_price_validation.validate.backtest`` unchanged, so the
strategy is identical across exchanges — only the data source differs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from crypto_signal_bot.research.xsection import features as ft
from alpha_library.alx_funding_price_validation.validate import backtest, ols_hac, sharpe
from alpha_library.alx3_external_replication import data_sources as ds

MED = 0.00075
SETTLE_HOURS = (0, 8, 16)


# ----------------------------------------------------------------------- #
# raw -> daily panels (independent reconstruction)
# ----------------------------------------------------------------------- #
def build_daily(exchange: str, bases: list[str]):
    """Return (close, qvol, daily_fund) wide daily panels + audit dict.

    daily realized funding = sum of a day's {0,8,16} UTC settlements (ALX rule).
    """
    close, qvol, dfund = {}, {}, {}
    off_grid = total = 0
    got = []
    for base in bases:
        kl, fu = ds.fetch_symbol(exchange, base)
        if kl is None:
            continue
        close[base] = kl.set_index("date")["close"]
        qvol[base] = kl.set_index("date")["qvol"]
        ts = pd.to_datetime(fu["ts"].to_numpy(), unit="ms", utc=True)
        hrs = ts.hour
        total += len(ts)
        off_grid += int((~pd.Index(hrs).isin(SETTLE_HOURS)).sum())
        mask = pd.Index(hrs).isin(SETTLE_HOURS)
        fser = pd.Series(fu["rate"].to_numpy()[mask], index=ts[mask]).sort_index()
        dfund[base] = fser.resample("1D").sum()
        got.append(base)
    close = pd.DataFrame(close).sort_index()
    qvol = pd.DataFrame(qvol).reindex(close.index)
    dfund = pd.DataFrame(dfund)
    audit = {"n_symbols": len(got), "off_grid_frac": (off_grid / total if total else 0.0),
             "symbols": got}
    logger.info("{}: {} symbols  {}..{}", exchange, len(got),
                close.index.min().date() if len(close) else None,
                close.index.max().date() if len(close) else None)
    return close, qvol, dfund, audit


def faithful_cols(close: pd.DataFrame, index, coverage=0.98):
    """ALX coverage rule: keep symbols with >= coverage valid closes over ``index``."""
    sub = close.reindex(index)
    return [c for c in sub.columns if sub[c].notna().mean() >= coverage]


def intersection_window(close: pd.DataFrame, bases=None):
    """Common window = latest first-valid .. earliest last-valid (ALX construction)."""
    cols = [c for c in (bases or close.columns) if c in close.columns]
    firsts, lasts = [], []
    for c in cols:
        s = close[c].dropna()
        if len(s):
            firsts.append(s.index.min()); lasts.append(s.index.max())
    if not firsts:
        return None, None
    return max(firsts), min(lasts)


def make_book(close, dfund, cols, index, cost=MED):
    """Frozen ALX book on the given cols/index. Returns backtest dict + signal."""
    ret = close[cols].reindex(index).pct_change().dropna(how="all")
    idx = ret.index
    fund = dfund.reindex(index=idx, columns=cols)
    signal = (-fund.rolling(7).mean()).reindex(index=idx, columns=cols)
    res = backtest(ret, signal, funding=fund, lag=1, cost=cost)
    res["index"] = idx
    res["ret"] = ret
    res["fund"] = fund
    res["signal"] = signal
    return res


# ----------------------------------------------------------------------- #
# statistics
# ----------------------------------------------------------------------- #
def block_bootstrap_ci(net, block=20, n=1000, seed=0):
    net = np.asarray(net)
    rng = np.random.default_rng(seed)
    if len(net) <= block:
        return float("nan"), float("nan")
    blocks = [net[i:i + block] for i in range(len(net) - block)]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(blocks), size=len(net) // block + 1)
        out.append(sharpe(np.concatenate([blocks[i] for i in pick])[:len(net)]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def deflated_sharpe(net, n_trials=100):
    import math
    from scipy.stats import norm
    x = np.asarray(net, dtype="float64")
    T = x.size
    mu, sd = x.mean(), x.std(ddof=1)
    sr = mu / sd if sd > 0 else 0.0
    z = (x - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    gamma = 0.5772156649015329
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * math.e))
    sr_star = (1 / math.sqrt(T - 1)) * ((1 - gamma) * z1 + gamma * z2)
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return float(norm.cdf((sr - sr_star) * math.sqrt(T - 1) / denom))


# ----------------------------------------------------------------------- #
# factor exposures (Stage 4) — reuse ALX construction
# ----------------------------------------------------------------------- #
def factor_exposures(res):
    ret = res["ret"]
    logret = np.log1p(ret)
    fund = res["fund"]
    net = res["net"]
    mkt = ret.mean(axis=1)
    books = {
        "reversal": ft.short_term_reversal(logret, 3),
        "momentum": ft.relative_strength(logret, 24),
        "raw_funding": -fund,
    }
    out = {}
    for k, v in books.items():
        b = backtest(ret, v, funding=None, lag=1, cost=MED)["net"]
        out[k] = float(np.corrcoef(net, b)[0, 1])
    return out
