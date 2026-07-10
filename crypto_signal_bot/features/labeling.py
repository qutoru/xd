"""Triple-barrier labeling of OHLCV bars (Lopez de Prado).

For every bar ``t`` we anchor at ``close[t]`` and place three barriers:

* an **upper** horizontal barrier at ``close[t] + ATR_MULT * ATR[t]``,
* a **lower** horizontal barrier at ``close[t] - ATR_MULT * ATR[t]``,
* a **vertical** (time) barrier ``HORIZON`` bars ahead.

Scanning bars ``t+1 .. t+HORIZON`` we assign:

* ``LABEL_LONG``  (1)  if the upper barrier is touched first,
* ``LABEL_SHORT`` (-1) if the lower barrier is touched first,
* ``LABEL_FLAT``  (0)  if the time barrier is reached first (neither touched).

Barrier width is volatility-scaled via ATR measured **at the anchor bar**, so
the target uses only ``close[t]``/``ATR[t]`` for its levels and future
highs/lows for resolution — no other look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.config import (
    ATR_MULT,
    HORIZON,
    LABEL_FLAT,
    LABEL_LONG,
    LABEL_SHORT,
)
from crypto_signal_bot.features.indicators import compute_atr

# Sentinel "never touched" offset, larger than any real barrier offset.
_NEVER = np.iinfo(np.int64).max


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon: int = HORIZON,
    atr_mult: float = ATR_MULT,
    atr: pd.Series | None = None,
) -> pd.Series:
    """Compute triple-barrier labels for each bar.

    Args:
        df: OHLCV frame (ascending by time) with ``high``, ``low``, ``close``.
        horizon: Number of future bars to scan (vertical barrier).
        atr_mult: Barrier half-width as a multiple of ATR.
        atr: Optional precomputed ATR series aligned to ``df`` (recomputed if
            not supplied). Reuse avoids double work when features already
            computed it.

    Returns:
        Integer label series (``-1``/``0``/``1``) aligned to ``df.index``.
        The last ``horizon`` bars, and any bar whose anchor ATR is NaN
        (warm-up), are set to NaN because their outcome cannot be resolved;
        callers should drop these rows.
    """
    if atr is None:
        atr = compute_atr(df)

    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    atr_arr = atr.to_numpy(dtype="float64")

    n = len(df)
    upper = close + atr_mult * atr_arr
    lower = close - atr_mult * atr_arr

    # First offset (1..horizon) at which each barrier is touched; _NEVER if not.
    first_up = np.full(n, _NEVER, dtype="int64")
    first_dn = np.full(n, _NEVER, dtype="int64")

    for offset in range(1, horizon + 1):
        # high/low of the bar `offset` steps ahead, aligned back to anchor t.
        fut_high = np.concatenate([high[offset:], np.full(offset, np.nan)])
        fut_low = np.concatenate([low[offset:], np.full(offset, np.nan)])

        up_hit = fut_high >= upper
        dn_hit = fut_low <= lower

        # Record the earliest offset only (np.where keeps the existing smaller).
        first_up = np.where(up_hit & (first_up == _NEVER), offset, first_up)
        first_dn = np.where(dn_hit & (first_dn == _NEVER), offset, first_dn)

    labels = np.full(n, LABEL_FLAT, dtype="int64")
    labels = np.where(first_up < first_dn, LABEL_LONG, labels)
    labels = np.where(first_dn < first_up, LABEL_SHORT, labels)
    # Ties (both barriers first touched on the same future bar) stay FLAT:
    # with only OHLC we cannot know intrabar ordering, so we abstain.

    out = pd.Series(labels, index=df.index, dtype="float64")

    # Invalidate bars that cannot be fully resolved.
    if n > 0:
        out.iloc[n - horizon :] = np.nan  # not enough future bars
    out[atr.isna().to_numpy()] = np.nan  # anchor ATR undefined (warm-up)

    return out
