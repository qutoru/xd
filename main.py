"""Command-line entry point for crypto_signal_bot.

Usage:
    py main.py fetch
    py main.py build
    py main.py train
    py main.py backtest [--segment {train,valid,test}]
    py main.py backtest-rm
    py main.py signal [--dry-run]
    py main.py live [--once] [--notify-no-trade] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from crypto_signal_bot.config import HISTORY_DAYS, INTERVAL, SYMBOL
from crypto_signal_bot.data.fetcher import fetch_ohlcv
from crypto_signal_bot.data.storage import save_parquet
from crypto_signal_bot.backtest.runner import run as run_backtest
from crypto_signal_bot.backtest.runner import run_risk_managed
from crypto_signal_bot.features.dataset import build_and_save
from crypto_signal_bot.live.scheduler import run_live
from crypto_signal_bot.live.signal import generate_signal
from crypto_signal_bot.live.telegram import send_telegram
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


def cmd_backtest_rm(_args: argparse.Namespace) -> int:
    """Risk-managed backtest v2 (tune threshold on valid, evaluate on test).

    Returns:
        Process exit code (always 0 on a successful run).
    """
    logger.info("Risk-managed backtest for {} {}m", SYMBOL, INTERVAL)
    stats = run_risk_managed(SYMBOL, INTERVAL)
    logger.success(
        "Backtest v2 done — thr={:.2f} total_return={:+.2%} sharpe={:.2f} "
        "trades={} (buy&hold {:+.2%})",
        stats["threshold"],
        stats["total_return"],
        stats["sharpe"],
        stats["n_trades"],
        stats["buy_hold_return"],
    )
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    """Generate a signal for the latest closed bar and notify via Telegram.

    Returns:
        Process exit code (always 0 on a successful run).
    """
    logger.info("Generating signal for {} {}m", SYMBOL, INTERVAL)
    sig = generate_signal(SYMBOL, INTERVAL)
    text = sig.format()
    logger.info("Signal message:\n{}", text)
    send_telegram(text, dry_run=args.dry_run)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Run the live signalling loop (or a single iteration with --once).

    Returns:
        Process exit code (always 0 on a clean run/stop).
    """
    logger.info("Starting live mode for {} {}m", SYMBOL, INTERVAL)
    run_live(
        SYMBOL,
        INTERVAL,
        notify_no_trade=args.notify_no_trade,
        dry_run=args.dry_run,
        once=args.once,
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

    backtest_rm_parser = subparsers.add_parser(
        "backtest-rm",
        help="Risk-managed backtest v2 (SL/TP, sizing, tuned threshold).",
    )
    backtest_rm_parser.set_defaults(func=cmd_backtest_rm)

    signal_parser = subparsers.add_parser(
        "signal", help="Generate a signal for the latest closed bar (Telegram)."
    )
    signal_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message instead of sending it to Telegram.",
    )
    signal_parser.set_defaults(func=cmd_signal)

    live_parser = subparsers.add_parser(
        "live", help="Run the live loop: emit a signal after each bar closes."
    )
    live_parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single iteration and exit (e.g. for cron).",
    )
    live_parser.add_argument(
        "--notify-no-trade",
        action="store_true",
        help="Also push 'no-trade' bars to Telegram (default: log only).",
    )
    live_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages instead of sending them to Telegram.",
    )
    live_parser.set_defaults(func=cmd_live)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
