"""Command-line entry point for crypto_signal_bot.

Usage:
    python main.py fetch
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from crypto_signal_bot.config import HISTORY_DAYS, INTERVAL, SYMBOL
from crypto_signal_bot.data.fetcher import fetch_ohlcv
from crypto_signal_bot.data.storage import save_parquet


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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
