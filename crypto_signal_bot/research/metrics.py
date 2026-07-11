"""Quant performance metrics for a per-bar strategy return series.

All metrics are computed from a net (after-cost) per-bar return array plus the
position array (for exposure/turnover) and a per-trade PnL list (for
win-rate/profit-factor/expectancy). Returns are treated as simple (arithmetic)
per-bar returns; annualization uses the bar frequency.
"""

from __future__ import annotations

import numpy as np

# Minutes per 365-day year, used to derive bars-per-year from the interval.
_MINUTES_PER_YEAR = 365 * 24 * 60


def bars_per_year(interval_minutes: int) -> float:
    """Number of bars in a year for a given kline interval in minutes."""
    return _MINUTES_PER_YEAR / interval_minutes


def _max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough drawdown of an equity curve (<= 0)."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def compute_metrics(
    net_ret: np.ndarray,
    position: np.ndarray,
    trade_pnl: np.ndarray,
    *,
    interval_minutes: int,
) -> dict:
    """Compute the full metric panel for one backtest.

    Args:
        net_ret: After-cost per-bar simple returns of the strategy.
        position: Position held during each bar (for exposure/turnover). Same
            length as ``net_ret``.
        trade_pnl: Realized PnL (simple return) per closed trade.
        interval_minutes: Bar interval in minutes (annualization).

    Returns:
        Dict of metrics (Sharpe, Sortino, Calmar, max_drawdown, win_rate,
        profit_factor, expectancy, exposure, turnover_annual, and context).
    """
    net_ret = np.asarray(net_ret, dtype="float64")
    position = np.asarray(position, dtype="float64")
    trade_pnl = np.asarray(trade_pnl, dtype="float64")
    n = net_ret.size
    bpy = bars_per_year(interval_minutes)

    mean = float(net_ret.mean()) if n else 0.0
    std = float(net_ret.std(ddof=1)) if n > 1 else 0.0
    downside = net_ret[net_ret < 0.0]
    dstd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0

    sharpe = (mean / std) * np.sqrt(bpy) if std > 0 else 0.0
    sortino = (mean / dstd) * np.sqrt(bpy) if dstd > 0 else 0.0

    equity = np.cumprod(1.0 + net_ret) if n else np.array([1.0])
    total_return = float(equity[-1] - 1.0) if n else 0.0
    years = n / bpy if bpy else 0.0
    ann_return = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0
    max_dd = _max_drawdown(equity)
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    wins = trade_pnl[trade_pnl > 0.0]
    losses = trade_pnl[trade_pnl < 0.0]
    n_trades = int(trade_pnl.size)
    win_rate = float(wins.size / n_trades) if n_trades else 0.0
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    expectancy = float(trade_pnl.mean()) if n_trades else 0.0

    exposure = float(np.mean(position != 0.0)) if n else 0.0
    # Annualized turnover: average per-bar position change, scaled to a year.
    turnover_annual = float(np.mean(np.abs(np.diff(position, prepend=0.0)))) * bpy if n else 0.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
        "exposure": exposure,
        "turnover_annual": turnover_annual,
        "total_return": total_return,
        "ann_return": ann_return,
        "n_trades": n_trades,
        "n_bars": n,
    }
