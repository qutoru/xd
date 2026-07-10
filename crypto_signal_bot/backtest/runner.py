"""Orchestrate a backtest on a held-out split of the processed dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from crypto_signal_bot.config import (
    BACKTEST_DIR,
    INTERVAL,
    PROB_THRESHOLD_GRID,
    SYMBOL,
)
from crypto_signal_bot.data.storage import load_processed
from crypto_signal_bot.backtest.engine import run_backtest
from crypto_signal_bot.backtest.event_engine import run_event_backtest
from crypto_signal_bot.backtest.metrics import (
    event_performance,
    log_event_performance,
    log_performance,
    performance,
)
from crypto_signal_bot.model.predict import SignalModel
from crypto_signal_bot.model.splits import time_split

# A tuned threshold must produce at least this many valid-set trades to be
# trusted; otherwise Sharpe is too noisy to rank on.
_MIN_TRADES_FOR_TUNING = 5

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


def _tune_threshold(seg: pd.DataFrame, signals, confidence) -> float:
    """Pick the confidence threshold maximizing valid-set Sharpe.

    Only thresholds yielding at least ``_MIN_TRADES_FOR_TUNING`` trades are
    considered; falls back to the lowest grid value if none qualify.
    """
    best_thr = PROB_THRESHOLD_GRID[0]
    best_sharpe = -np.inf
    logger.info("Tuning confidence threshold on valid ({} bars):", len(seg))
    for thr in PROB_THRESHOLD_GRID:
        res = run_event_backtest(seg, signals, confidence, threshold=thr)
        stats = event_performance(res.bars, res.trades)
        logger.info(
            "  thr={:.2f}: trades={:>3} return={:+.2%} sharpe={:.2f}",
            thr,
            stats["n_trades"],
            stats["total_return"],
            stats["sharpe"],
        )
        if stats["n_trades"] >= _MIN_TRADES_FOR_TUNING and stats["sharpe"] > best_sharpe:
            best_sharpe = stats["sharpe"]
            best_thr = thr
    logger.info("Selected threshold={:.2f} (valid sharpe={:.2f})", best_thr, best_sharpe)
    return best_thr


def run_risk_managed(symbol: str = SYMBOL, interval: str = INTERVAL) -> dict:
    """Backtest v2: tune the confidence threshold on valid, evaluate on test.

    Returns:
        The test-segment performance stats dict (curves/trades saved to disk).
    """
    df = load_processed(symbol, interval)
    df = df.sort_values("timestamp").reset_index(drop=True)

    model = SignalModel.load(symbol, interval)
    splits = time_split(len(df))

    valid = df.iloc[splits.valid].reset_index(drop=True)
    test = df.iloc[splits.test].reset_index(drop=True)
    valid_sig, valid_conf = model.predict_with_conf(valid)
    test_sig, test_conf = model.predict_with_conf(test)

    threshold = _tune_threshold(valid, valid_sig, valid_conf)

    logger.info(
        "Risk-managed backtest on test: {} bars ({} .. {})",
        len(test),
        test["datetime"].iloc[0],
        test["datetime"].iloc[-1],
    )
    result = run_event_backtest(test, test_sig, test_conf, threshold=threshold)
    stats = event_performance(result.bars, result.trades)
    stats["threshold"] = threshold
    log_event_performance("test", stats)

    _save(result.bars, symbol, interval, "test_rm")
    if not result.trades.empty:
        tpath = BACKTEST_DIR / f"{symbol}_{interval}_test_rm_trades.parquet"
        result.trades.to_parquet(tpath, index=False)
        logger.info("Saved {} trades -> {}", len(result.trades), tpath)
    return stats
