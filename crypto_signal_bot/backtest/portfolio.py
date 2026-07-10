"""Cross-sectional market-neutral portfolio backtest (RESEARCH P0-1).

Tests whether ranking the universe by predicted signal strength and going long
the top-k / short the bottom-k gives a better (diversified) Sharpe than trading
each symbol alone. Evaluated on each symbol's out-of-sample test segment.

Signal strength per bar/symbol = ``P(long) - P(short)`` from the model. Each bar
we form an equal-weight, dollar-neutral book (gross exposure 1.0: 0.5 long +
0.5 short). We report both **gross** (no fees) and **net** (with per-bar
rebalancing fees) stats — gross answers "is there a cross-sectional edge at
all?", net answers "does it survive costs with naive daily-bar rebalancing?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from crypto_signal_bot.config import BARS_PER_YEAR, FEE_RATE, INTERVAL
from crypto_signal_bot.data.storage import load_processed
from crypto_signal_bot.data.universe import get_universe
from crypto_signal_bot.model.predict import SignalModel
from crypto_signal_bot.model.train import model_path
from crypto_signal_bot.model.splits import time_split


def _sharpe(returns: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> float:
    std = returns.std(ddof=0)
    if std <= 0 or len(returns) == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(bars_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0


def _collect_signals(symbols: list[str], interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build wide (datetime x symbol) strength and next-bar return frames."""
    strengths: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}

    for symbol in symbols:
        if not model_path(symbol, interval).exists():
            continue
        df = load_processed(symbol, interval).sort_values("timestamp").reset_index(drop=True)
        splits = time_split(len(df))
        seg = df.iloc[splits.test].reset_index(drop=True)
        if seg.empty:
            continue

        model = SignalModel.load(symbol, interval)
        proba = model.predict_proba(seg)
        # class_order is [-1, 0, 1] -> P(short)=col0, P(long)=col2.
        strength = proba[:, 2] - proba[:, 0]

        idx = pd.to_datetime(seg["datetime"].to_numpy())
        strengths[symbol] = pd.Series(strength, index=idx)
        returns[symbol] = pd.Series(
            seg["close"].pct_change().shift(-1).to_numpy(), index=idx
        )

    strength_df = pd.DataFrame(strengths).sort_index()
    return_df = pd.DataFrame(returns).reindex_like(strength_df)
    return strength_df, return_df


def run_portfolio_backtest(
    symbols: list[str] | None = None,
    *,
    interval: str = INTERVAL,
    k: int = 5,
    fee_rate: float = FEE_RATE,
) -> dict:
    """Backtest a market-neutral top-k/bottom-k portfolio over the universe.

    Args:
        symbols: Universe (defaults to the cached list).
        k: Number of longs and of shorts per bar.
        fee_rate: Per-unit-turnover fee for the net curve.

    Returns:
        Stats dict with gross/net Sharpe, returns, drawdown and turnover.
    """
    if symbols is None:
        symbols = get_universe()

    strength_df, return_df = _collect_signals(symbols, interval)
    n_bars, n_syms = strength_df.shape
    logger.info("Portfolio universe: {} symbols x {} test bars", n_syms, n_bars)
    if n_syms < 2 * k:
        raise ValueError(f"Need >= {2 * k} symbols, have {n_syms}")

    # Rank across symbols each bar (NaNs are ignored by rank).
    rank_hi = strength_df.rank(axis=1, ascending=False)  # 1 = strongest long
    rank_lo = strength_df.rank(axis=1, ascending=True)  # 1 = strongest short
    available = strength_df.notna().sum(axis=1)

    weights = pd.DataFrame(0.0, index=strength_df.index, columns=strength_df.columns)
    weights[rank_hi <= k] = 0.5 / k
    weights[rank_lo <= k] = -0.5 / k
    # Only take bars where we can fill both legs.
    weights = weights.where(available >= 2 * k, other=0.0)

    gross_ret = (weights * return_df).sum(axis=1, skipna=True)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    net_ret = gross_ret - fee_rate * turnover

    gross_eq = (1.0 + gross_ret).cumprod()
    net_eq = (1.0 + net_ret).cumprod()

    active_bars = int((available >= 2 * k).sum())
    stats = {
        "symbols": n_syms,
        "bars": n_bars,
        "active_bars": active_bars,
        "k": k,
        "gross_total_return": float(gross_eq.iloc[-1] - 1.0) if n_bars else 0.0,
        "gross_sharpe": _sharpe(gross_ret),
        "net_total_return": float(net_eq.iloc[-1] - 1.0) if n_bars else 0.0,
        "net_sharpe": _sharpe(net_ret),
        "net_max_drawdown": _max_drawdown(net_eq),
        "avg_turnover": float(turnover.mean()) if n_bars else 0.0,
        "total_fees": float((fee_rate * turnover).sum()),
    }
    return stats


def log_portfolio(stats: dict) -> None:
    """Pretty-log the portfolio stats."""
    logger.info(
        "[portfolio k={}] {} symbols, {} active bars",
        stats["k"],
        stats["symbols"],
        stats["active_bars"],
    )
    logger.info(
        "  GROSS: total_return={:+.2%}  sharpe={:.2f}",
        stats["gross_total_return"],
        stats["gross_sharpe"],
    )
    logger.info(
        "  NET:   total_return={:+.2%}  sharpe={:.2f}  max_dd={:.2%}",
        stats["net_total_return"],
        stats["net_sharpe"],
        stats["net_max_drawdown"],
    )
    logger.info(
        "  avg_turnover/bar={:.3f}  total_fees={:.2%}",
        stats["avg_turnover"],
        stats["total_fees"],
    )
