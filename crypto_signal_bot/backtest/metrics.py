"""Performance metrics for a backtest result frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from crypto_signal_bot.config import BARS_PER_YEAR


def _max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough drop of an equity curve, as a negative fraction."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def performance(df: pd.DataFrame, *, bars_per_year: int = BARS_PER_YEAR) -> dict:
    """Compute headline performance stats from a :func:`run_backtest` frame.

    Returns:
        Dict of strategy stats plus a buy&hold benchmark over the same window.
    """
    net = df["net"]
    n = len(df)
    equity_final = float(df["equity"].iloc[-1]) if n else 1.0
    total_return = equity_final - 1.0

    # Annualization from per-bar stats.
    if n > 0 and net.std(ddof=0) > 0:
        sharpe = float(net.mean() / net.std(ddof=0) * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0
    ann_return = (equity_final ** (bars_per_year / n) - 1.0) if n > 0 else 0.0

    active = df["position"] != 0
    n_active = int(active.sum())
    wins = int((net[active] > 0).sum())

    # A "trade" is any bar where the position changes.
    trades = int((df["position"].diff().fillna(df["position"]) != 0).sum())

    # Buy & hold over the same window (long 1 unit, single entry fee ignored).
    buy_hold = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) if n else 0.0

    return {
        "bars": n,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(df["equity"]) if n else 0.0,
        "exposure": float(n_active / n) if n else 0.0,
        "trades": trades,
        "win_rate": float(wins / n_active) if n_active else 0.0,
        "total_fees": float(df["cost"].sum()),
        "buy_hold_return": buy_hold,
    }


def log_performance(name: str, stats: dict) -> None:
    """Pretty-log a performance dict from :func:`performance`."""
    logger.info("[{}] backtest over {} bars", name, stats["bars"])
    logger.info(
        "  total_return={:+.2%}  ann_return={:+.2%}  sharpe={:.2f}  max_dd={:.2%}",
        stats["total_return"],
        stats["ann_return"],
        stats["sharpe"],
        stats["max_drawdown"],
    )
    logger.info(
        "  exposure={:.1%}  trades={}  win_rate={:.1%}  total_fees={:.2%}",
        stats["exposure"],
        stats["trades"],
        stats["win_rate"],
        stats["total_fees"],
    )
    logger.info("  benchmark buy&hold={:+.2%}", stats["buy_hold_return"])
