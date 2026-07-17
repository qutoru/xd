"""Production trade runner — wires the existing pipeline for SHADOW/PAPER/LIVE.

Thin glue only: it reads configuration from the environment (BybitConfig.from_env
+ TelegramNotifier.from_env), builds the single existing ``build_default_pipeline``
(no separate pipeline), runs pre-flight connectivity checks for exchange modes,
then executes one daily cycle. It creates no strategy/execution logic of its own
and never touches the Bybit API directly — all exchange access stays inside
BybitBroker.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

import pandas as pd
from loguru import logger

from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
from crypto_signal_bot.platform.notify.notifier import TelegramNotifier
from crypto_signal_bot.platform.observability.summary import CycleSummary
from crypto_signal_bot.platform.pipeline import DailyPipeline, PipelineResult, build_default_pipeline
from crypto_signal_bot.platform.risk_control.control import RiskControlConfig
from crypto_signal_bot.platform.risk_control.state import RiskStateStore


def parse_symbols(raw: str | None) -> list[str]:
    """Parse ``BYBIT_SYMBOLS`` (comma-separated) into an upper-cased list."""
    if not raw:
        return []
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def preflight(cfg: BybitConfig, broker) -> str | None:
    """Validate credentials and connectivity before any trading cycle.

    Returns an error message if the run must abort, else None. Only exercised for
    exchange modes (PAPER/LIVE); SHADOW needs no exchange.
    """
    if cfg.mode is TradingMode.LIVE:
        if not cfg.api_key:
            return "BYBIT_API_KEY is not set (required for LIVE)"
        if not cfg.api_secret:
            return "BYBIT_API_SECRET is not set (required for LIVE)"
    if not broker.check_connection():
        return (
            f"cannot reach Bybit (testnet={cfg.testnet}); "
            "check BYBIT_API_KEY/BYBIT_API_SECRET and network"
        )
    return None


def build_trade_pipeline(
    cfg: BybitConfig,
    *,
    signals: Sequence[str],
    symbols: Sequence[str],
    notifier: TelegramNotifier | None,
    session=None,
) -> DailyPipeline:
    """Build the one existing pipeline for the configured mode.

    Risk-control limits come from ``RISK_*`` env vars (RiskControlConfig.from_env);
    the emergency-stop latch and daily-loss baseline persist to ``RISK_STATE_PATH``
    so they survive the one-shot process.
    """
    return build_default_pipeline(
        signal_names=tuple(signals),
        bybit_config=cfg,
        bybit_session=session,
        notifier=notifier,
        symbols=list(symbols) or None,
        risk_control_config=RiskControlConfig.from_env(),
        risk_state=RiskStateStore(os.getenv("RISK_STATE_PATH", "data/risk_state.json")),
    )


def _log_result(cfg: BybitConfig, result: PipelineResult, duration_s: float | None = None) -> None:
    logger.success(
        "{} {} — gross={:.2f} net={:+.4f} daily_pnl={:+.5f}",
        cfg.mode.value.upper(), result.asof.date(),
        result.book.gross, result.book.net, result.shadow.daily_pnl,
    )
    if result.execution is not None:
        logger.info("Execution — filled={} rejected={} traded_notional={:.2f}",
                    result.execution.n_filled, result.execution.n_rejected,
                    result.execution.traded_notional)
    if result.reconciliation is not None:
        rec = result.reconciliation
        if rec.failed:
            logger.warning("Reconciliation FAILED: {}", rec.message)
        elif rec.ok:
            logger.info("Reconciliation OK — platform matches exchange")
        else:
            logger.warning("Reconciliation — {} discrepancy(ies): {}",
                           rec.n_discrepancies, sorted(k.value for k in rec.kinds()))
    if result.notify is not None:
        logger.info("Telegram — sent={} failed={} skipped={}",
                    result.notify.sent, result.notify.failed, result.notify.skipped)
    # Additional structured cycle summary (observability only; existing lines above
    # are unchanged). Read-only projection over the result — computes nothing.
    for line in CycleSummary.from_result(result, duration_s=duration_s).format_lines():
        logger.info(line)


def run_trade(
    *,
    signals: Sequence[str],
    asof: pd.Timestamp | None = None,
    session=None,
) -> int:
    """Run one daily cycle in the env-selected mode. Returns a process exit code."""
    cfg = BybitConfig.from_env()
    symbols = parse_symbols(os.getenv("BYBIT_SYMBOLS"))
    notifier = TelegramNotifier.from_env()

    logger.info("Trade mode={} testnet={} symbols={}",
                cfg.mode.value, cfg.testnet, symbols or "<mainnet top-N>")

    try:
        pipeline = build_trade_pipeline(
            cfg, signals=signals, symbols=symbols, notifier=notifier, session=session
        )
    except ValueError as exc:  # e.g. LIVE without credentials
        logger.error("Configuration error, aborting before trading: {}", exc)
        return 2

    if cfg.needs_exchange:
        error = preflight(cfg, pipeline.execution_engine.broker)
        if error is not None:
            logger.error("Pre-flight failed, aborting before trading: {}", error)
            return 1
        logger.success("Pre-flight OK — connected to Bybit (testnet={})", cfg.testnet)

    t0 = time.perf_counter()
    result = pipeline.run_once(asof if asof is not None else pd.Timestamp.now(tz="UTC").normalize())
    _log_result(cfg, result, duration_s=time.perf_counter() - t0)
    return 0
