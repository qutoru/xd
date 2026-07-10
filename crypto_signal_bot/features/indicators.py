"""Technical-indicator feature engineering from OHLCV data.

All features are computed using only information available up to and including
the current bar (no look-ahead): every transform is either a point-wise
function of the current row or a *trailing* rolling/EWM window. The forward-
looking target is produced separately in :mod:`crypto_signal_bot.features.labeling`.

Indicators are implemented in plain pandas/numpy to avoid a heavyweight TA
dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.config import ATR_PERIOD

# Lookback windows (in bars) reused across several feature families.
_RETURN_WINDOWS = (1, 4, 8, 16, 24)
_VOL_WINDOWS = (8, 24, 48)
_EMA_WINDOWS = (9, 21, 50, 200)
_RSI_PERIOD = 14
_BB_PERIOD = 20
_BB_STD = 2.0
_VOLUME_Z_WINDOW = 48

# Bars per day at 15m resolution, used for cyclical time-of-day features.
_BARS_PER_DAY = 96


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range (Wilder's smoothing) over ``period`` bars.

    True range uses the previous close, so the value at bar ``t`` depends only
    on bars ``<= t``. The first ``period`` values are NaN (warm-up).

    Args:
        df: OHLCV frame with ``high``, ``low``, ``close`` columns.
        period: Smoothing lookback.

    Returns:
        ATR series aligned to ``df.index``.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing == EWM with alpha = 1/period.
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    """Relative Strength Index (Wilder) in the 0..100 range."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the model feature matrix from a raw OHLCV frame.

    Args:
        df: Raw OHLCV frame (ascending by time) with ``open, high, low, close,
            volume, turnover`` columns, as produced by the fetcher.

    Returns:
        A new DataFrame indexed like ``df`` containing only feature columns.
        Early rows contain NaNs from indicator warm-up and are meant to be
        dropped by the caller.
    """
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    feats: dict[str, pd.Series] = {}

    # --- Momentum / returns -------------------------------------------------
    log_close = np.log(close)
    log_ret = log_close.diff()
    for w in _RETURN_WINDOWS:
        feats[f"ret_{w}"] = log_close.diff(w)

    # --- Realized volatility ------------------------------------------------
    for w in _VOL_WINDOWS:
        feats[f"vol_{w}"] = log_ret.rolling(w).std()

    # --- ATR (absolute and normalized by price) ----------------------------
    atr = compute_atr(df)
    feats["atr"] = atr
    feats["atr_pct"] = atr / close

    # --- Trend: price vs EMAs ----------------------------------------------
    for w in _EMA_WINDOWS:
        ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
        feats[f"ema_ratio_{w}"] = close / ema - 1.0

    # --- MACD ---------------------------------------------------------------
    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    # Normalize by price so the feature is comparable across regimes.
    feats["macd"] = macd / close
    feats["macd_hist"] = (macd - macd_signal) / close

    # --- RSI ----------------------------------------------------------------
    feats["rsi"] = _rsi(close)

    # --- Bollinger position -------------------------------------------------
    bb_ma = close.rolling(_BB_PERIOD).mean()
    bb_std = close.rolling(_BB_PERIOD).std()
    feats["bb_pos"] = (close - bb_ma) / (_BB_STD * bb_std)

    # --- Candle shape -------------------------------------------------------
    rng = (high - low).replace(0.0, np.nan)
    feats["candle_range"] = rng / close
    feats["candle_body"] = (close - open_) / rng
    feats["upper_wick"] = (high - close.combine(open_, max)) / rng
    feats["lower_wick"] = (close.combine(open_, min) - low) / rng

    # --- Volume -------------------------------------------------------------
    vol_mean = volume.rolling(_VOLUME_Z_WINDOW).mean()
    vol_std = volume.rolling(_VOLUME_Z_WINDOW).std()
    feats["volume_z"] = (volume - vol_mean) / vol_std
    feats["volume_chg"] = volume.pct_change()

    # --- Cyclical time-of-day (intraday seasonality) ------------------------
    bar_of_day = (df["timestamp"] // (15 * 60 * 1000)) % _BARS_PER_DAY
    angle = 2.0 * np.pi * bar_of_day / _BARS_PER_DAY
    feats["tod_sin"] = np.sin(angle)
    feats["tod_cos"] = np.cos(angle)

    out = pd.DataFrame(feats, index=df.index)
    # Guard against inf produced by divisions on degenerate bars.
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(df_features: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature column names."""
    return list(df_features.columns)
