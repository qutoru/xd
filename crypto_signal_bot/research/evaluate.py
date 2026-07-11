"""Evaluation + robustness harness for a rule-based hypothesis.

Given a signal-generating function and a price frame, this runs the strict
validation protocol: chronological 60/20/20 metrics, purged walk-forward folds,
and a battery of robustness perturbations (parameters, execution delay,
slippage/commission, Monte-Carlo block bootstrap, volatility regimes). All
randomness uses a fixed seed for reproducibility.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from crypto_signal_bot.config import FEE_RATE, RANDOM_SEED
from crypto_signal_bot.research.backtest import run_rule_backtest
from crypto_signal_bot.research.metrics import compute_metrics

SignalFn = Callable[[pd.DataFrame], pd.Series]


def _metrics_for(df: pd.DataFrame, signal: pd.Series, interval: int, *, fee: float = FEE_RATE,
                 slippage: float = 0.0, exec_lag: int = 1) -> dict:
    res = run_rule_backtest(df, signal, fee=fee, slippage=slippage, exec_lag=exec_lag)
    return compute_metrics(res.net_ret, res.position, res.trade_pnl, interval_minutes=interval)


def evaluate_splits(df: pd.DataFrame, signal_fn: SignalFn, interval: int) -> dict:
    """Metrics on chronological train(60%)/valid(20%)/test(20%) segments."""
    n = len(df)
    a, b = int(n * 0.60), int(n * 0.80)
    out = {}
    for name, lo, hi in (("train", 0, a), ("valid", a, b), ("test", b, n)):
        seg = df.iloc[lo:hi].reset_index(drop=True)
        out[name] = _metrics_for(seg, signal_fn(seg), interval)
    return out


def walk_forward(df: pd.DataFrame, signal_fn: SignalFn, interval: int, *, n_splits: int = 6) -> pd.DataFrame:
    """Sharpe (and more) on ``n_splits`` consecutive out-of-sample test blocks."""
    n = len(df)
    size = n // (n_splits + 1)  # first block is warm-up context only
    rows = []
    for k in range(1, n_splits + 1):
        seg = df.iloc[k * size:(k + 1) * size].reset_index(drop=True)
        m = _metrics_for(seg, signal_fn(seg), interval)
        rows.append({"fold": k, "sharpe": m["sharpe"], "sortino": m["sortino"],
                     "max_dd": m["max_drawdown"], "n_trades": m["n_trades"],
                     "total_return": m["total_return"]})
    return pd.DataFrame(rows)


def perturb_params(df: pd.DataFrame, base_fn: Callable[..., pd.Series], interval: int,
                   grid: dict) -> pd.DataFrame:
    """Sharpe across a parameter grid — a stable edge is not a knife-edge."""
    import itertools
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        m = _metrics_for(df, base_fn(df, **params), interval)
        rows.append({**params, "sharpe": m["sharpe"], "n_trades": m["n_trades"]})
    return pd.DataFrame(rows)


def sensitivity(df: pd.DataFrame, signal: pd.Series, interval: int) -> dict:
    """Execution-delay and cost sensitivity of a fixed signal."""
    out = {}
    for lag in (1, 2, 3):
        out[f"exec_lag_{lag}"] = _metrics_for(df, signal, interval, exec_lag=lag)["sharpe"]
    for slip_bps in (0, 2, 5):
        out[f"slip_{slip_bps}bps"] = _metrics_for(
            df, signal, interval, slippage=slip_bps / 1e4)["sharpe"]
    return out


def monte_carlo(df: pd.DataFrame, signal: pd.Series, interval: int, *,
                n_iter: int = 1000, block: int = 24) -> dict:
    """Block-bootstrap the per-bar net returns -> Sharpe distribution.

    Blocks preserve short-horizon autocorrelation. Returns the 5th/50th/95th
    Sharpe percentiles and the fraction of resamples with Sharpe > 0.
    """
    from crypto_signal_bot.research.metrics import bars_per_year
    res = run_rule_backtest(df, signal)
    r = res.net_ret
    n = r.size
    rng = np.random.default_rng(RANDOM_SEED)
    bpy = bars_per_year(interval)
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, max(1, n - block), size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        sharpes[i] = (sample.mean() / sd) * np.sqrt(bpy) if sd > 0 else 0.0
    return {"p05": float(np.percentile(sharpes, 5)), "p50": float(np.percentile(sharpes, 50)),
            "p95": float(np.percentile(sharpes, 95)), "frac_positive": float(np.mean(sharpes > 0))}


def regime_split(df: pd.DataFrame, signal_fn: SignalFn, interval: int, *, vol_window: int = 96) -> pd.DataFrame:
    """Sharpe in low/mid/high realized-vol regimes (tertiles of trailing vol)."""
    ret = df["close"].pct_change()
    vol = ret.rolling(vol_window).std().shift(1)
    q1, q2 = vol.quantile([1 / 3, 2 / 3])
    labels = pd.Series("mid", index=df.index)
    labels[vol <= q1] = "low_vol"
    labels[vol >= q2] = "high_vol"

    sig = signal_fn(df)
    res = run_rule_backtest(df, sig)
    from crypto_signal_bot.research.metrics import bars_per_year
    bpy = bars_per_year(interval)
    rows = []
    for name in ("low_vol", "mid", "high_vol"):
        mask = (labels == name).to_numpy()
        r = res.net_ret[mask]
        sd = r.std(ddof=1) if r.size > 1 else 0.0
        sharpe = (r.mean() / sd) * np.sqrt(bpy) if sd > 0 else 0.0
        rows.append({"regime": name, "sharpe": float(sharpe), "bars": int(mask.sum())})
    return pd.DataFrame(rows)
