"""Command-line entry point for crypto_signal_bot.

Usage:
    py main.py fetch
    py main.py build
    py main.py train
    py main.py backtest [--segment {train,valid,test}]
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from crypto_signal_bot.config import HISTORY_DAYS, INTERVAL, SYMBOL
from crypto_signal_bot.data.fetcher import fetch_ohlcv
from crypto_signal_bot.data.storage import save_parquet
from crypto_signal_bot.backtest.runner import run as run_backtest
from crypto_signal_bot.features.dataset import build_and_save
from crypto_signal_bot.model.train import train as train_model


def cmd_fetch(_args: argparse.Namespace) -> int:
    """Download OHLCV history from Bybit and store it as parquet.

    Returns:
        Process exit code (0 on success, 1 if nothing was fetched).
    """
    logger.info("Starting fetch for {} {}m ({} days)", SYMBOL, INTERVAL, HISTORY_DAYS)
    df = fetch_ohlcv(symbol=SYMBOL, interval=INTERVAL, history_days=HISTORY_DAYS)

    if df.empty:
        logger.error("No candles were fetched — nothing to save")
        return 1

    path = save_parquet(df, SYMBOL, INTERVAL)
    logger.success(
        "Fetched {} candles from {} to {} -> {}",
        len(df),
        df["datetime"].min(),
        df["datetime"].max(),
        path,
    )
    return 0


def cmd_build(_args: argparse.Namespace) -> int:
    """Build the features+labels dataset from raw OHLCV and store it.

    Returns:
        Process exit code (0 on success, 1 if the dataset ended up empty).
    """
    logger.info("Building dataset for {} {}m", SYMBOL, INTERVAL)
    dataset = build_and_save(SYMBOL, INTERVAL)

    if dataset.empty:
        logger.error("Dataset is empty after processing — nothing to save")
        return 1

    logger.success("Built dataset with {} rows", len(dataset))
    return 0


def cmd_train(_args: argparse.Namespace) -> int:
    """Train the LightGBM signal model and persist it with its metrics.

    Returns:
        Process exit code (always 0 on a successful run).
    """
    logger.info("Training model for {} {}m", SYMBOL, INTERVAL)
    meta = train_model(SYMBOL, INTERVAL)
    test = meta["test_metrics"]
    logger.success(
        "Trained model — test accuracy={:.3f} macro_f1={:.3f}",
        test["accuracy"],
        test["macro_f1"],
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Backtest the trained model on a held-out split (no risk management).

    Returns:
        Process exit code (always 0 on a successful run).
    """
    logger.info("Backtesting {} {}m on '{}' segment", SYMBOL, INTERVAL, args.segment)
    stats = run_backtest(SYMBOL, INTERVAL, segment=args.segment)
    logger.success(
        "Backtest done — total_return={:+.2%} sharpe={:.2f} (buy&hold {:+.2%})",
        stats["total_return"],
        stats["sharpe"],
        stats["buy_hold_return"],
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="crypto_signal_bot",
        description="MVP crypto signal bot for Bybit (no order execution).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Download OHLCV history from Bybit and save to parquet."
    )
    fetch_parser.set_defaults(func=cmd_fetch)

    build_parser = subparsers.add_parser(
        "build", help="Build features + triple-barrier labels from raw OHLCV."
    )
    build_parser.set_defaults(func=cmd_build)

    train_parser = subparsers.add_parser(
        "train", help="Train the LightGBM signal model and report metrics."
    )
    train_parser.set_defaults(func=cmd_train)

    backtest_parser = subparsers.add_parser(
        "backtest", help="Backtest the trained model (no risk management)."
    )
    backtest_parser.add_argument(
        "--segment",
        choices=("train", "valid", "test"),
        default="test",
        help="Which chronological split to evaluate (default: test).",
    )
    backtest_parser.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
