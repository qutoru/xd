"""Orchestrate a backtest on a held-out split of the processed dataset."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from crypto_signal_bot.config import BACKTEST_DIR, INTERVAL, SYMBOL
from crypto_signal_bot.data.storage import load_processed
from crypto_signal_bot.backtest.engine import run_backtest
from crypto_signal_bot.backtest.metrics import log_performance, performance
from crypto_signal_bot.model.predict import SignalModel
from crypto_signal_bot.model.splits import time_split

# Which chronological split to evaluate on. Test is the honest out-of-sample set.
_SEGMENTS = ("train", "valid", "test")


def run(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    segment: str = "test",
) -> dict:
    """Backtest the trained model on one chronological segment.

    Args:
        symbol: Trading symbol.
        interval: Kline interval in minutes as a string.
        segment: One of ``train``/``valid``/``test`` (default out-of-sample test).

    Returns:
        The performance stats dict (the per-bar frame is also saved to disk).
    """
    if segment not in _SEGMENTS:
        raise ValueError(f"segment must be one of {_SEGMENTS}, got {segment!r}")

    df = load_processed(symbol, interval)
    df = df.sort_values("timestamp").reset_index(drop=True)

    model = SignalModel.load(symbol, interval)
    splits = time_split(len(df))
    seg_idx = getattr(splits, segment)
    seg = df.iloc[seg_idx].reset_index(drop=True)
    logger.info(
        "Backtesting '{}' segment: {} bars ({} .. {})",
        segment,
        len(seg),
        seg["datetime"].iloc[0],
        seg["datetime"].iloc[-1],
    )

    signals = model.predict_signals(seg)
    result = run_backtest(seg["close"], signals)
    result.insert(0, "datetime", seg["datetime"].iloc[: len(result)].to_numpy())

    stats = performance(result)
    log_performance(segment, stats)

    _save(result, symbol, interval, segment)
    return stats


def _save(result: pd.DataFrame, symbol: str, interval: str, segment: str) -> None:
    """Persist the per-bar backtest frame for later inspection/plotting."""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKTEST_DIR / f"{symbol}_{interval}_{segment}.parquet"
    result.to_parquet(path, index=False)
    logger.info("Saved backtest curve ({} rows) -> {}", len(result), path)
