"""Triple-barrier labeling of OHLCV bars (Lopez de Prado), cost-aware & asymmetric.

For every bar ``t`` we anchor at ``close[t]`` and evaluate each trade side with
its own take-profit / stop distances (in ATR units, measured at the anchor):

* **long** wins if the up TP ``close + tp_mult*ATR`` is touched before the down
  stop ``close - sl_mult*ATR``,
* **short** wins if the down TP ``close - tp_mult*ATR`` is touched before the up
  stop ``close + sl_mult*ATR``,
* a **vertical** (time) barrier ``HORIZON`` bars ahead caps the scan.

Scanning bars ``t+1 .. t+HORIZON`` we assign:

* ``LABEL_LONG``  (1)  if only the long side resolves in its favour,
* ``LABEL_SHORT`` (-1) if only the short side resolves in its favour,
* ``LABEL_FLAT``  (0)  otherwise (neither side, or the rare conflict).

With ``tp_mult > sl_mult`` (E4: 2:1) a directional class is only assigned to
sizeable favourable moves with positive expectancy after fees. Setting
``tp_mult == sl_mult`` recovers the classic symmetric labeling: long = up first,
short = down first.

Barrier widths are volatility-scaled via ATR at the anchor bar, so the target
uses only ``close[t]``/``ATR[t]`` for its levels and future highs/lows for
resolution — no other look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.config import (
    HORIZON,
    LABEL_FLAT,
    LABEL_LONG,
    LABEL_SHORT,
    LABEL_SL_MULT,
    LABEL_TP_MULT,
)
from crypto_signal_bot.features.indicators import compute_atr

# Sentinel "never touched" offset, larger than any real barrier offset.
_NEVER = np.iinfo(np.int64).max


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon: int = HORIZON,
    tp_mult: float = LABEL_TP_MULT,
    sl_mult: float = LABEL_SL_MULT,
    atr: pd.Series | None = None,
) -> pd.Series:
    """Compute cost-aware asymmetric triple-barrier labels for each bar.

    Args:
        df: OHLCV frame (ascending by time) with ``high``, ``low``, ``close``.
        horizon: Number of future bars to scan (vertical barrier).
        tp_mult: Take-profit distance as a multiple of ATR (per side).
        sl_mult: Stop distance as a multiple of ATR (per side). ``tp_mult ==
            sl_mult`` reproduces the classic symmetric labeling.
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
    # Per-side barrier levels. TP is the favourable target, SL the adverse stop.
    up_tp = close + tp_mult * atr_arr  # long take-profit
    dn_sl = close - sl_mult * atr_arr  # long stop
    dn_tp = close - tp_mult * atr_arr  # short take-profit
    up_sl = close + sl_mult * atr_arr  # short stop

    # First offset (1..horizon) at which each of the four levels is touched.
    first_up_tp = np.full(n, _NEVER, dtype="int64")
    first_dn_sl = np.full(n, _NEVER, dtype="int64")
    first_dn_tp = np.full(n, _NEVER, dtype="int64")
    first_up_sl = np.full(n, _NEVER, dtype="int64")

    for offset in range(1, horizon + 1):
        # high/low of the bar `offset` steps ahead, aligned back to anchor t.
        fut_high = np.concatenate([high[offset:], np.full(offset, np.nan)])
        fut_low = np.concatenate([low[offset:], np.full(offset, np.nan)])

        up_hit = fut_high >= up_tp  # touches an upside level
        up_stop_hit = fut_high >= up_sl
        dn_hit = fut_low <= dn_tp  # touches a downside level
        dn_stop_hit = fut_low <= dn_sl

        # Record the earliest offset only (np.where keeps the existing smaller).
        first_up_tp = np.where(up_hit & (first_up_tp == _NEVER), offset, first_up_tp)
        first_up_sl = np.where(up_stop_hit & (first_up_sl == _NEVER), offset, first_up_sl)
        first_dn_tp = np.where(dn_hit & (first_dn_tp == _NEVER), offset, first_dn_tp)
        first_dn_sl = np.where(dn_stop_hit & (first_dn_sl == _NEVER), offset, first_dn_sl)

    # A side "wins" if its TP is reached strictly before its stop.
    long_win = first_up_tp < first_dn_sl
    short_win = first_dn_tp < first_up_sl

    labels = np.full(n, LABEL_FLAT, dtype="int64")
    labels = np.where(long_win & ~short_win, LABEL_LONG, labels)
    labels = np.where(short_win & ~long_win, LABEL_SHORT, labels)
    # Conflicts (both sides "win") are near-impossible with tp>=sl and stay
    # FLAT: with only OHLC we cannot know intrabar ordering, so we abstain.

    out = pd.Series(labels, index=df.index, dtype="float64")

    # Invalidate bars that cannot be fully resolved.
    if n > 0:
        out.iloc[n - horizon :] = np.nan  # not enough future bars
    out[atr.isna().to_numpy()] = np.nan  # anchor ATR undefined (warm-up)

    return out
