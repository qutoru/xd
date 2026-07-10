"""A minimal vectorized backtest engine (no risk management).

Convention (no look-ahead): at the close of bar ``t`` we already have all
features for ``t`` and take the signalled position; that position earns the
close-to-close return of the *next* bar ``t -> t+1``. A fee proportional to the
change in position (turnover) is charged whenever we enter, flip or exit.

This deliberately omits stop-loss / take-profit and position sizing (always
+/-1 unit of notional) — that is Phase 5. The point here is to see the raw
signal's edge net of trading costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.config import FEE_RATE


def run_backtest(
    close: pd.Series,
    signals: np.ndarray,
    *,
    fee_rate: float = FEE_RATE,
) -> pd.DataFrame:
    """Simulate the signal strategy over ``close`` and return a per-bar frame.

    Args:
        close: Close prices for the backtest window (time-ordered).
        signals: Per-bar target position in ``{-1, 0, 1}``, aligned to ``close``.
        fee_rate: Fee charged per unit of turnover (|Δposition|).

    Returns:
        DataFrame indexed like ``close`` (last bar dropped — no forward return)
        with columns ``close, position, ret, gross, cost, net, equity``.
    """
    close = close.reset_index(drop=True)
    position = pd.Series(np.asarray(signals, dtype="float64"), index=close.index)

    # Next-bar close-to-close return realized by the position held at bar t.
    fwd_ret = close.pct_change().shift(-1)

    # Turnover: change in position vs the previous bar (entry from 0 counts).
    prev_position = position.shift(1).fillna(0.0)
    turnover = (position - prev_position).abs()

    gross = position * fwd_ret
    cost = turnover * fee_rate
    net = gross - cost

    df = pd.DataFrame(
        {
            "close": close,
            "position": position,
            "ret": fwd_ret,
            "gross": gross,
            "cost": cost,
            "net": net,
        }
    )
    # Drop the final bar: it has no next-bar return to realize.
    df = df.iloc[:-1].copy()
    df["equity"] = (1.0 + df["net"]).cumprod()
    return df
