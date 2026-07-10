"""Event-driven backtest with risk management (Phase 5).

Unlike the Phase 4 vectorized engine (which flips a +/-1 position every bar and
bleeds fees), this holds **one position at a time** and manages it explicitly:

* **Entry** at the close of bar ``t`` when flat and the model signals long/short
  with top-class probability >= ``threshold``.
* **Sizing** is risk-based: notional is chosen so that hitting the ATR stop
  loses ``risk_per_trade`` of current equity, capped at ``max_leverage``.
* **Exit** on whichever comes first: stop-loss, take-profit (both ATR-sized,
  checked intrabar via high/low) or the time barrier (``horizon`` bars).

Fees are charged on entry and exit notional. Equity is marked to market each
bar (close-to-close, or to the exit price on the exit bar) so Sharpe and
drawdown are meaningful. No look-ahead: the entry decision at ``t`` uses only
bar-``t`` information and the position starts being at risk from ``t+1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_signal_bot.config import (
    FEE_RATE,
    HORIZON,
    MAX_LEVERAGE,
    RISK_PER_TRADE,
    SL_ATR_MULT,
    TP_ATR_MULT,
)


@dataclass
class _Position:
    direction: int  # +1 long, -1 short
    notional: float  # units of equity (leverage) committed at entry
    entry_price: float
    stop: float
    take: float
    entry_i: int
    bars_held: int = 0


@dataclass(frozen=True)
class BacktestResult:
    """Per-bar equity frame plus the per-trade log."""

    bars: pd.DataFrame
    trades: pd.DataFrame


def run_event_backtest(
    seg: pd.DataFrame,
    signals: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold: float,
    fee_rate: float = FEE_RATE,
    horizon: int = HORIZON,
    sl_mult: float = SL_ATR_MULT,
    tp_mult: float = TP_ATR_MULT,
    risk_per_trade: float = RISK_PER_TRADE,
    max_leverage: float = MAX_LEVERAGE,
) -> BacktestResult:
    """Run the risk-managed backtest over one time-ordered segment.

    Args:
        seg: Segment frame with ``datetime, high, low, close, atr`` columns.
        signals: Per-bar ``-1/0/1`` predictions aligned to ``seg``.
        confidence: Per-bar top-class probability aligned to ``seg``.
        threshold: Minimum confidence to open a trade.

    Returns:
        A :class:`BacktestResult` with a per-bar frame (``datetime, close,
        position, net, cost, equity``) and a per-trade frame.
    """
    dt = seg["datetime"].to_numpy()
    high = seg["high"].to_numpy(dtype="float64")
    low = seg["low"].to_numpy(dtype="float64")
    close = seg["close"].to_numpy(dtype="float64")
    atr = seg["atr"].to_numpy(dtype="float64")
    n = len(seg)

    equity = 1.0
    pos: _Position | None = None

    bar_position = np.zeros(n)  # signed notional held *during* each bar
    bar_net = np.zeros(n)  # equity return contributed by each bar
    bar_cost = np.zeros(n)  # fee fraction charged on each bar
    bar_equity = np.zeros(n)
    trades: list[dict] = []

    for i in range(n):
        net_i = 0.0
        cost_i = 0.0

        # --- Manage an open position on this bar ---------------------------
        if pos is not None:
            bar_position[i] = pos.direction * pos.notional
            pos.bars_held += 1

            exit_price: float | None = None
            reason = ""
            if pos.direction == 1:
                if low[i] <= pos.stop:  # stop checked first (conservative)
                    exit_price, reason = pos.stop, "sl"
                elif high[i] >= pos.take:
                    exit_price, reason = pos.take, "tp"
            else:
                if high[i] >= pos.stop:
                    exit_price, reason = pos.stop, "sl"
                elif low[i] <= pos.take:
                    exit_price, reason = pos.take, "tp"
            if exit_price is None and pos.bars_held >= horizon:
                exit_price, reason = close[i], "time"

            mark = exit_price if exit_price is not None else close[i]
            # Mark-to-market from the previous close (== entry price on the
            # first held bar) to the mark/exit price.
            net_i += pos.direction * pos.notional * (mark / close[i - 1] - 1.0)

            if exit_price is not None:
                cost_i += fee_rate * pos.notional  # exit fee
                net_i -= fee_rate * pos.notional
                trade_ret = pos.direction * (exit_price / pos.entry_price - 1.0)
                trades.append(
                    {
                        "entry_time": dt[pos.entry_i],
                        "exit_time": dt[i],
                        "direction": pos.direction,
                        "notional": pos.notional,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "bars_held": pos.bars_held,
                        "reason": reason,
                        "gross_return": trade_ret * pos.notional,
                        "net_return": trade_ret * pos.notional - 2 * fee_rate * pos.notional,
                    }
                )
                pos = None

        # --- Consider opening a new position at this close -----------------
        # Skip the last bar: a fresh position would have no bar to be marked on.
        if pos is None and i < n - 1 and signals[i] != 0 and confidence[i] >= threshold:
            direction = int(signals[i])
            entry_price = close[i]
            stop_dist = sl_mult * atr[i]
            take_dist = tp_mult * atr[i]
            if stop_dist > 0:
                stop_ret = stop_dist / entry_price
                notional = min(risk_per_trade / stop_ret, max_leverage)
                pos = _Position(
                    direction=direction,
                    notional=notional,
                    entry_price=entry_price,
                    stop=entry_price - direction * stop_dist,
                    take=entry_price + direction * take_dist,
                    entry_i=i,
                )
                cost_i += fee_rate * notional  # entry fee
                net_i -= fee_rate * notional

        equity *= 1.0 + net_i
        bar_net[i] = net_i
        bar_cost[i] = cost_i
        bar_equity[i] = equity

    bars = pd.DataFrame(
        {
            "datetime": dt,
            "close": close,
            "position": bar_position,
            "net": bar_net,
            "cost": bar_cost,
            "equity": bar_equity,
        }
    )
    return BacktestResult(bars=bars, trades=pd.DataFrame(trades))
