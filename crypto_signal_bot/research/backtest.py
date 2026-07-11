"""Vectorized, no-lookahead backtester for rule-based target-position signals.

Convention (no lookahead): a ``signal`` value is the *target position* decided
at the CLOSE of bar ``t``. It is applied to the return of bar ``t + exec_lag``,
i.e. we act on the next bar's move, never the current one. Costs (commission +
slippage) are charged on position turnover ``|Δposition|``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_signal_bot.config import FEE_RATE


@dataclass(frozen=True)
class BacktestResult:
    """Per-bar arrays and per-trade PnL from a rule backtest."""

    net_ret: np.ndarray  # after-cost per-bar returns
    position: np.ndarray  # position applied to each bar's return
    trade_pnl: np.ndarray  # realized simple return per closed trade


def _extract_trades(position: np.ndarray, net_ret: np.ndarray) -> np.ndarray:
    """Sum net returns over contiguous constant-sign position blocks -> trades."""
    trades: list[float] = []
    n = position.size
    i = 0
    while i < n:
        side = position[i]
        if side == 0.0:
            i += 1
            continue
        j = i
        pnl = 0.0
        while j < n and position[j] == side:
            pnl += net_ret[j]
            j += 1
        trades.append(pnl)
        i = j
    return np.asarray(trades, dtype="float64")


def run_rule_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    fee: float = FEE_RATE,
    slippage: float = 0.0,
    exec_lag: int = 1,
) -> BacktestResult:
    """Backtest a target-position signal on a price frame.

    Args:
        df: Frame with a ``close`` column (ascending by time).
        signal: Target position in {-1, 0, 1} decided at each bar's close.
        fee: Commission per unit turnover (one side).
        slippage: Extra cost per unit turnover (one side).
        exec_lag: Bars between decision and execution (>=1 avoids lookahead;
            higher values model a delayed fill for robustness tests).

    Returns:
        A :class:`BacktestResult`.
    """
    close = df["close"].to_numpy(dtype="float64")
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0

    pos = pd.Series(signal, dtype="float64").shift(exec_lag).fillna(0.0).to_numpy()
    gross = pos * ret
    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = turnover * (fee + slippage)
    net = gross - cost

    trades = _extract_trades(pos, net)
    return BacktestResult(net_ret=net, position=pos, trade_pnl=trades)
