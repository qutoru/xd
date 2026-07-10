"""Command-line entry point for crypto_signal_bot.

Multi-symbol: fetch/build/train/signal/live operate over the whole universe
(top-N liquid Bybit perps) by default; pass --symbol to target a single pair.

Usage:
    py main.py symbols                 # discover & cache the top-N universe
    py main.py fetch   [--symbol S]
    py main.py build   [--symbol S]
    py main.py train   [--symbol S]
    py main.py backtest    [--symbol S] [--segment {train,valid,test}]
    py main.py backtest-rm [--symbol S]
    py main.py signal  [--symbol S] [--dry-run]
    py main.py live    [--symbol S] [--once] [--notify-no-trade] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from crypto_signal_bot.config import INTERVAL, SYMBOL
from crypto_signal_bot.data.universe import get_universe, refresh_universe
from crypto_signal_bot.backtest.runner import run as run_backtest
from crypto_signal_bot.backtest.runner import run_risk_managed
from crypto_signal_bot.live.scheduler import run_live
from crypto_signal_bot.live.signal import generate_signal
from crypto_signal_bot.live.telegram import send_telegram
from crypto_signal_bot.model.predict import SignalModel
from crypto_signal_bot.pipeline import build_all, fetch_all, train_all


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    """Return [--symbol] if given, else the cached universe."""
    if getattr(args, "symbol", None):
        return [args.symbol]
    symbols = get_universe()
    logger.info("Operating over universe of {} symbols", len(symbols))
    return symbols


def cmd_symbols(_args: argparse.Namespace) -> int:
    """Discover and cache the top-N liquid universe."""
    symbols = refresh_universe()
    logger.success("Universe refreshed: {} symbols", len(symbols))
    logger.info("Top 10: {}", ", ".join(symbols[:10]))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download OHLCV history for the resolved symbol(s)."""
    fetch_all(_resolve_symbols(args), interval=INTERVAL)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build features+labels datasets for the resolved symbol(s)."""
    build_all(_resolve_symbols(args), interval=INTERVAL)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model per resolved symbol."""
    train_all(_resolve_symbols(args), interval=INTERVAL)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Backtest one symbol on a held-out split (no risk management)."""
    symbol = args.symbol or SYMBOL
    logger.info("Backtesting {} {}m on '{}' segment", symbol, INTERVAL, args.segment)
    stats = run_backtest(symbol, INTERVAL, segment=args.segment)
    logger.success(
        "Backtest done — total_return={:+.2%} sharpe={:.2f} (buy&hold {:+.2%})",
        stats["total_return"],
        stats["sharpe"],
        stats["buy_hold_return"],
    )
    return 0


def cmd_backtest_rm(args: argparse.Namespace) -> int:
    """Risk-managed backtest v2 for one symbol (tune on valid, eval on test)."""
    symbol = args.symbol or SYMBOL
    logger.info("Risk-managed backtest for {} {}m", symbol, INTERVAL)
    stats = run_risk_managed(symbol, INTERVAL)
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
    """Generate a one-shot signal for the resolved symbol(s) and notify."""
    symbols = _resolve_symbols(args)
    for symbol in symbols:
        try:
            model = SignalModel.load(symbol, INTERVAL)
            sig = generate_signal(symbol, INTERVAL, model=model)
            logger.info("{} signal:\n{}", symbol, sig.format())
            if sig.is_trade or args.notify_no_trade:
                send_telegram(sig.format(), dry_run=args.dry_run)
        except Exception as exc:  # isolate a bad symbol
            logger.exception("{}: signal failed, continuing: {}", symbol, exc)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Run the live signalling loop over the universe (or a single symbol)."""
    symbols = [args.symbol] if args.symbol else None
    run_live(
        symbols,
        interval=INTERVAL,
        notify_no_trade=args.notify_no_trade,
        dry_run=args.dry_run,
        once=args.once,
    )
    return 0


def _add_symbol_arg(sub: argparse.ArgumentParser) -> None:
    """Attach the shared optional --symbol override."""
    sub.add_argument(
        "--symbol",
        default=None,
        help="Target a single symbol instead of the whole universe.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="crypto_signal_bot",
        description="MVP crypto signal bot for Bybit (no order execution).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    symbols_parser = subparsers.add_parser(
        "symbols", help="Discover and cache the top-N liquid Bybit universe."
    )
    symbols_parser.set_defaults(func=cmd_symbols)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Download OHLCV history from Bybit and save to parquet."
    )
    _add_symbol_arg(fetch_parser)
    fetch_parser.set_defaults(func=cmd_fetch)

    build_parser_ = subparsers.add_parser(
        "build", help="Build features + triple-barrier labels from raw OHLCV."
    )
    _add_symbol_arg(build_parser_)
    build_parser_.set_defaults(func=cmd_build)

    train_parser = subparsers.add_parser(
        "train", help="Train the LightGBM signal model per symbol."
    )
    _add_symbol_arg(train_parser)
    train_parser.set_defaults(func=cmd_train)

    backtest_parser = subparsers.add_parser(
        "backtest", help="Backtest one symbol (no risk management)."
    )
    _add_symbol_arg(backtest_parser)
    backtest_parser.add_argument(
        "--segment",
        choices=("train", "valid", "test"),
        default="test",
        help="Which chronological split to evaluate (default: test).",
    )
    backtest_parser.set_defaults(func=cmd_backtest)

    backtest_rm_parser = subparsers.add_parser(
        "backtest-rm",
        help="Risk-managed backtest v2 for one symbol (SL/TP, sizing, threshold).",
    )
    _add_symbol_arg(backtest_rm_parser)
    backtest_rm_parser.set_defaults(func=cmd_backtest_rm)

    signal_parser = subparsers.add_parser(
        "signal", help="One-shot signal(s) for the latest closed bar (Telegram)."
    )
    _add_symbol_arg(signal_parser)
    signal_parser.add_argument(
        "--notify-no-trade",
        action="store_true",
        help="Also push 'no-trade' signals to Telegram (default: log only).",
    )
    signal_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message instead of sending it to Telegram.",
    )
    signal_parser.set_defaults(func=cmd_signal)

    live_parser = subparsers.add_parser(
        "live", help="Run the live loop over the universe after each bar closes."
    )
    _add_symbol_arg(live_parser)
    live_parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sweep and exit (e.g. for cron).",
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
