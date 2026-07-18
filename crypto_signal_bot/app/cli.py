"""CLI orchestration for crypto_signal_bot (relocated verbatim from main.py).

Multi-symbol: fetch/build/train/signal/live operate over the whole universe
(top-N liquid Bybit perps) by default; pass --symbol to target a single pair.
The ``shadow`` command runs the platform daily pipeline (cross-sectional book,
virtual PnL only). Business logic lives in Production Core and the platform
layers; this module only wires commands to them.
"""

from __future__ import annotations

import argparse

from loguru import logger

from crypto_signal_bot.config import HISTORY_DAYS, INTERVAL, SYMBOL
from crypto_signal_bot.data.universe import get_universe, refresh_universe
from crypto_signal_bot.backtest.portfolio import log_portfolio, run_portfolio_backtest
from crypto_signal_bot.backtest.runner import run as run_backtest
from crypto_signal_bot.backtest.runner import run_risk_managed
from crypto_signal_bot.backtest.walkforward import log_walkforward, run_walkforward
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
    fetch_all(_resolve_symbols(args), interval=args.interval, days=args.days)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build features+labels datasets for the resolved symbol(s)."""
    build_all(_resolve_symbols(args), interval=args.interval)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model per resolved symbol."""
    train_all(_resolve_symbols(args), interval=args.interval)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Backtest one symbol on a held-out split (no risk management)."""
    symbol = args.symbol or SYMBOL
    logger.info("Backtesting {} {}m on '{}' segment", symbol, args.interval, args.segment)
    stats = run_backtest(symbol, args.interval, segment=args.segment)
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
    logger.info("Risk-managed backtest for {} {}m", symbol, args.interval)
    stats = run_risk_managed(symbol, args.interval)
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


def cmd_walkforward(args: argparse.Namespace) -> int:
    """Purged walk-forward CV of the risk-managed strategy for one symbol."""
    symbol = args.symbol or SYMBOL
    logger.info("Walk-forward CV for {} {}m ({} splits)", symbol, args.interval, args.splits)
    summary = run_walkforward(symbol, args.interval, n_splits=args.splits)
    log_walkforward(summary)
    logger.success(
        "Walk-forward done — mean_sharpe={:.2f} positive_folds={:.0%}",
        summary["mean_sharpe"],
        summary["frac_positive"],
    )
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Cross-sectional market-neutral portfolio backtest over the universe."""
    logger.info("Portfolio backtest (top/bottom k={})", args.k)
    stats = run_portfolio_backtest(k=args.k)
    log_portfolio(stats)
    logger.success(
        "Portfolio done — gross_sharpe={:.2f} net_sharpe={:.2f}",
        stats["gross_sharpe"],
        stats["net_sharpe"],
    )
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


def cmd_shadow(args: argparse.Namespace) -> int:
    """Run one day of the platform pipeline (virtual book; no real trades)."""
    import pandas as pd

    from crypto_signal_bot.platform.pipeline import build_default_pipeline

    pipeline = build_default_pipeline(signal_names=tuple(s for s in args.signals.split(",") if s))
    asof = pd.Timestamp(args.asof, tz="UTC") if args.asof else pd.Timestamp.utcnow().normalize()
    result = pipeline.run_once(asof)
    logger.success(
        "Shadow {} — signals=[{}] gross={:.2f} net={:+.4f} daily_pnl={:+.5f}",
        result.asof.date(),
        args.signals,
        result.book.gross,
        result.book.net,
        result.shadow.daily_pnl,
    )
    return 0


def cmd_trade(args: argparse.Namespace) -> int:
    """Run one daily platform cycle in the .env-selected mode (SHADOW/PAPER/LIVE).

    Mode and all Bybit/Telegram settings come from the environment only; this
    reuses the existing ``build_default_pipeline`` — no separate pipeline.
    """
    import pandas as pd

    from crypto_signal_bot.app.trade_runner import run_trade

    signals = tuple(s for s in args.signals.split(",") if s)
    asof = pd.Timestamp(args.asof, tz="UTC") if args.asof else None
    return run_trade(signals=signals, asof=asof)


def cmd_bot(args: argparse.Namespace) -> int:
    """Run the Telegram bot listener (long-polling): /start, /language, buttons."""
    import os

    from crypto_signal_bot.config import (
        TELEGRAM_LEDGER_PATH,
        TELEGRAM_PENDING_PATH,
        TELEGRAM_PREFS_PATH,
        TELEGRAM_RISK_PREFS_PATH,
        TELEGRAM_SUBS_PATH,
        TELEGRAM_USERS_PATH,
    )
    from crypto_signal_bot.platform.accounting.ledger import AccountingStore
    from crypto_signal_bot.platform.notify.listener import (
        RequestsBotClient,
        TelegramListener,
    )
    from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
    from crypto_signal_bot.platform.notify.riskprefs import RiskPrefsStore
    from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore
    from crypto_signal_bot.platform.notify.users import UsersStore

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set — cannot start the bot listener.")
        return 2

    admin_id = os.getenv("TELEGRAM_ADMIN_ID", "")
    if not admin_id:
        logger.warning("TELEGRAM_ADMIN_ID is not set — admin commands are disabled.")

    prefs = LanguagePrefsStore(TELEGRAM_PREFS_PATH)
    ledger_store = AccountingStore(TELEGRAM_LEDGER_PATH)
    # Owner-only semi-auto approval: build it only when an owner (== admin) and a
    # broker mode exist. Owner == TELEGRAM_ADMIN_ID; the whole surface is owner-gated
    # inside the handler, so it stays invisible and inert for every other user.
    owner_handler = _build_owner_handler(
        admin_id or None, PendingPaths(TELEGRAM_PENDING_PATH, prefs, ledger_store)
    )

    client = RequestsBotClient(token)
    listener = TelegramListener(
        client,
        prefs,
        subscriptions=SubscriptionStore(TELEGRAM_SUBS_PATH),
        users=UsersStore(TELEGRAM_USERS_PATH),
        admin_id=admin_id or None,
        ledger_store=ledger_store,
        risk_prefs=RiskPrefsStore(TELEGRAM_RISK_PREFS_PATH),
        owner_handler=owner_handler,
    )
    listener.run_forever(timeout=args.poll_timeout)
    return 0


class PendingPaths:
    """Small bundle of the stores the owner-approval handler needs."""

    def __init__(self, pending_path, prefs, ledger_store) -> None:
        self.pending_path = pending_path
        self.prefs = prefs
        self.ledger_store = ledger_store


def _build_owner_handler(owner_id, paths: "PendingPaths"):
    """Construct the OwnerApprovalHandler, or None when it can't place orders.

    The order-placing broker follows the env trading mode (``BYBIT_TRADING_MODE``):
    PAPER/LIVE give a real BybitBroker; SHADOW has no broker, so approvals can't
    place and the handler is disabled (a warning is logged).
    """
    import os

    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig
    from crypto_signal_bot.platform.execution.bybit_broker import build_broker
    from crypto_signal_bot.platform.execution.engine import ExecutionEngine
    from crypto_signal_bot.platform.notify.owner import OwnerApprovalHandler
    from crypto_signal_bot.platform.notify.pending import PendingSignalStore
    from crypto_signal_bot.platform.risk_control.control import (
        ProductionRiskControl,
        RiskControlConfig,
    )
    from crypto_signal_bot.platform.risk_control.state import RiskStateStore

    if not owner_id:
        return None
    cfg = BybitConfig.from_env()
    broker = build_broker(cfg)  # None for SHADOW
    if broker is None:
        logger.warning(
            "Owner semi-auto approval disabled: BYBIT_TRADING_MODE={} has no broker "
            "(set paper/live to place orders on Accept).", cfg.mode.value,
        )
        return None
    return OwnerApprovalHandler(
        owner_id=owner_id,
        pending=PendingSignalStore(paths.pending_path),
        engine=ExecutionEngine(broker),
        risk_control=ProductionRiskControl(RiskControlConfig.from_env()),
        risk_state=RiskStateStore(os.getenv("RISK_STATE_PATH", "data/risk_state.json")),
        ledger_store=paths.ledger_store,
        prefs=paths.prefs,
    )


def _add_symbol_arg(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--symbol", default=None,
                     help="Target a single symbol instead of the whole universe.")


def _add_interval_arg(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--interval", default=INTERVAL,
                     help=f"Kline interval in minutes (default: {INTERVAL}). E.g. 60 for 1h.")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="crypto_signal_bot",
        description="MVP crypto signal bot for Bybit (no order execution).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    symbols_parser = subparsers.add_parser(
        "symbols", help="Discover and cache the top-N liquid Bybit universe.")
    symbols_parser.set_defaults(func=cmd_symbols)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Download OHLCV history from Bybit and save to parquet.")
    _add_symbol_arg(fetch_parser)
    _add_interval_arg(fetch_parser)
    fetch_parser.add_argument("--days", type=int, default=HISTORY_DAYS,
                              help=f"How many days of history to fetch (default: {HISTORY_DAYS}).")
    fetch_parser.set_defaults(func=cmd_fetch)

    build_parser_ = subparsers.add_parser(
        "build", help="Build features + triple-barrier labels from raw OHLCV.")
    _add_symbol_arg(build_parser_)
    _add_interval_arg(build_parser_)
    build_parser_.set_defaults(func=cmd_build)

    train_parser = subparsers.add_parser(
        "train", help="Train the LightGBM signal model per symbol.")
    _add_symbol_arg(train_parser)
    _add_interval_arg(train_parser)
    train_parser.set_defaults(func=cmd_train)

    backtest_parser = subparsers.add_parser(
        "backtest", help="Backtest one symbol (no risk management).")
    _add_symbol_arg(backtest_parser)
    _add_interval_arg(backtest_parser)
    backtest_parser.add_argument("--segment", choices=("train", "valid", "test"),
                                 default="test",
                                 help="Which chronological split to evaluate (default: test).")
    backtest_parser.set_defaults(func=cmd_backtest)

    backtest_rm_parser = subparsers.add_parser(
        "backtest-rm",
        help="Risk-managed backtest v2 for one symbol (SL/TP, sizing, threshold).")
    _add_symbol_arg(backtest_rm_parser)
    _add_interval_arg(backtest_rm_parser)
    backtest_rm_parser.set_defaults(func=cmd_backtest_rm)

    walkforward_parser = subparsers.add_parser(
        "walkforward",
        help="Purged walk-forward CV of the risk-managed strategy (one symbol).")
    _add_symbol_arg(walkforward_parser)
    _add_interval_arg(walkforward_parser)
    walkforward_parser.add_argument("--splits", type=int, default=6,
                                    help="Number of walk-forward test folds (default: 6).")
    walkforward_parser.set_defaults(func=cmd_walkforward)

    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="Cross-sectional market-neutral top/bottom-k portfolio backtest.")
    portfolio_parser.add_argument("--k", type=int, default=5,
                                  help="Number of longs and of shorts per bar (default: 5).")
    portfolio_parser.set_defaults(func=cmd_portfolio)

    signal_parser = subparsers.add_parser(
        "signal", help="One-shot signal(s) for the latest closed bar (Telegram).")
    _add_symbol_arg(signal_parser)
    signal_parser.add_argument("--notify-no-trade", action="store_true",
                               help="Also push 'no-trade' signals to Telegram (default: log only).")
    signal_parser.add_argument("--dry-run", action="store_true",
                               help="Print the message instead of sending it to Telegram.")
    signal_parser.set_defaults(func=cmd_signal)

    live_parser = subparsers.add_parser(
        "live", help="Run the live loop over the universe after each bar closes.")
    _add_symbol_arg(live_parser)
    live_parser.add_argument("--once", action="store_true",
                             help="Run a single sweep and exit (e.g. for cron).")
    live_parser.add_argument("--notify-no-trade", action="store_true",
                             help="Also push 'no-trade' bars to Telegram (default: log only).")
    live_parser.add_argument("--dry-run", action="store_true",
                             help="Print messages instead of sending them to Telegram.")
    live_parser.set_defaults(func=cmd_live)

    shadow_parser = subparsers.add_parser(
        "shadow", help="Run one day of the platform pipeline (virtual book, no trades).")
    shadow_parser.add_argument("--signals", default="alx",
                               help="Comma-separated signal names to combine (default: alx).")
    shadow_parser.add_argument("--asof", default=None,
                               help="As-of date YYYY-MM-DD (default: today, UTC).")
    shadow_parser.set_defaults(func=cmd_shadow)

    trade_parser = subparsers.add_parser(
        "trade",
        help="Run one platform cycle in the .env-selected mode (SHADOW/PAPER/LIVE).")
    trade_parser.add_argument("--signals", default="alx",
                              help="Comma-separated signal names to combine (default: alx).")
    trade_parser.add_argument("--asof", default=None,
                              help="As-of date YYYY-MM-DD (default: today, UTC).")
    trade_parser.set_defaults(func=cmd_trade)

    bot_parser = subparsers.add_parser(
        "bot",
        help="Run the Telegram bot listener (long-polling): /start, /language, buttons.")
    bot_parser.add_argument("--poll-timeout", type=int, default=30,
                            help="getUpdates long-poll timeout in seconds (default: 30).")
    bot_parser.set_defaults(func=cmd_bot)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
