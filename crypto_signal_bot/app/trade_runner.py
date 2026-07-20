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


def _env_flag(name: str) -> bool:
    """True when env var ``name`` is set to a truthy value (1/true/yes/on)."""
    raw = os.getenv(name)
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}


def _semi_auto_enabled() -> bool:
    """True when ``TRADE_SEMI_AUTO`` opts into owner semi-auto delivery (owner buttons)."""
    return _env_flag("TRADE_SEMI_AUTO")


def compute_signals(*, signals: Sequence[str], asof: pd.Timestamp | None = None, session=None):
    """Run one ALX cycle without executing and return the quality-filtered signals.

    The generation half of semi-auto, factored out so the in-process signal driver
    (started with the bot) can produce signals itself. Builds the same pipeline as
    ``run_trade`` but forces ``execute=False`` (the owner approves each entry by
    hand), runs one cycle, and returns only the strongest ``SIGNAL_TOP_N`` signals
    at or above ``SIGNAL_MIN_CONFIDENCE``. Delivery (owner buttons + subscriber
    broadcast) is the caller's job — this places nothing and sends nothing.
    """
    cfg = BybitConfig.from_env()
    symbols = parse_symbols(os.getenv("BYBIT_SYMBOLS"))
    pipeline = build_trade_pipeline(
        cfg, signals=signals, symbols=symbols, notifier=None, session=session
    )
    pipeline.execute = False  # semi-auto: signals only, owner places on Accept
    result = pipeline.run_once(
        asof if asof is not None else pd.Timestamp.now(tz="UTC").normalize()
    )
    top_n = int(os.getenv("SIGNAL_TOP_N", "5"))
    min_conf = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.0"))
    candidates = _restrict_to_liquid(result.intents)
    return select_quality_signals(candidates, top_n=top_n, min_confidence=min_conf)


def _restrict_to_liquid(intents):
    """Drop intents whose symbol is outside the ``TRADE_LIQUID_SYMBOLS`` allowlist.

    ALX is cross-sectional and ranks across the full universe, but on some venues
    (notably Bybit testnet) illiquid/exotic symbols have no fillable order book, so
    a market entry is rejected by price-band protection (retCode 30208) and the TP/SL
    brackets cascade. When ``TRADE_LIQUID_SYMBOLS`` is set (comma-separated), only
    signals on those symbols are surfaced for execution — the full universe is still
    used for ranking, only delivery is gated. An empty/unset value disables the
    filter (every symbol is delivered), preserving prior behavior.
    """
    intents = list(intents)
    allow = set(parse_symbols(os.getenv("TRADE_LIQUID_SYMBOLS")))
    if not allow:
        return intents
    kept = [i for i in intents if getattr(i, "symbol", None) in allow]
    if len(kept) != len(intents):
        logger.info(
            "Liquidity filter: {} of {} signal(s) within TRADE_LIQUID_SYMBOLS allowlist",
            len(kept), len(intents),
        )
    return kept


def _conviction(intent) -> float:
    """Sort key for signal quality: the normalized confidence, or -inf if absent."""
    return intent.confidence if intent.confidence is not None else float("-inf")


def select_quality_signals(intents, *, top_n: int, min_confidence: float = 0.0):
    """Keep only the strongest ``top_n`` intents, highest-conviction first.

    ALX is cross-sectional: one rebalance proposes the whole book (roughly the
    universe size), which is why raw semi-auto delivery pushed ~30 signals at once.
    Semi-auto should instead surface only the highest-quality entries, so this ranks
    by ``confidence`` (the normalized signal-strength proxy) descending, drops any
    below ``min_confidence`` (an intent with no confidence never clears a floor of
    0), and returns the top ``top_n``. ``top_n <= 0`` disables the cap (floor only).
    """
    ranked = sorted(
        (i for i in intents if _conviction(i) >= min_confidence),
        key=_conviction,
        reverse=True,
    )
    return ranked if top_n <= 0 else ranked[:top_n]


def push_owner_approvals(intents, *, owner_id, pending, client, prefs=None, risk_prefs=None) -> int:
    """Persist each intent as pending and push the owner an Accept/Ignore message.

    Semi-auto mode: the pipeline runs with execution disabled and hands each
    proposed signal to the owner here. The owner's tap is handled in the separate
    ``cmd_bot`` listener via ``OwnerApprovalHandler``, which risk-gates and places
    the order. Best-effort delivery — a send failure is logged, never raised.
    Returns the number of signals pushed.
    """
    if not owner_id or not intents:
        return 0
    from crypto_signal_bot.platform.notify.i18n import DEFAULT_LANGUAGE, t
    from crypto_signal_bot.platform.notify.owner import owner_only_keyboard
    from crypto_signal_bot.platform.notify.telegram import TelegramFormatter

    formatter = TelegramFormatter()
    lang = prefs.get(owner_id) if prefs is not None else DEFAULT_LANGUAGE
    # The owner holds all VIP privileges, so their signal carries the VIP-only
    # recommended risk-per-trade line (from the owner's personal /risk level).
    owner_risk = risk_prefs.get(owner_id) if risk_prefs is not None else None
    pushed = 0
    for intent in intents:
        signal = pending.add(intent, chat_id=owner_id)
        text = f"{formatter.format(intent, lang, risk_level=owner_risk)}\n\n{t(lang, 'owner_prompt')}"
        # The Accept/Ignore buttons attach only when the recipient IS the owner
        # (owner_only_keyboard returns None otherwise), so they can never be shown
        # to any other user — the message here is always addressed to the owner.
        markup = owner_only_keyboard(signal.token, lang, recipient=owner_id, owner_id=owner_id)
        try:
            client.send_message(chat_id=owner_id, text=text, reply_markup=markup)
            pushed += 1
        except Exception as exc:  # delivery is best-effort; the pending row survives
            logger.error("Failed to push owner approval for {}: {}", intent.symbol, exc)
    return pushed


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
    semi_auto = _semi_auto_enabled()
    # In semi-auto the owner receives one buttoned Accept/Ignore approval per
    # signal; the plain pipeline notifier would double-send the same signals
    # without buttons, so it is suppressed here (the buttoned owner push is the
    # only delivery). It stays enabled in fully-auto modes.
    notifier = None if semi_auto else TelegramNotifier.from_env()

    logger.info("Trade mode={} testnet={} symbols={}",
                cfg.mode.value, cfg.testnet, symbols or "<mainnet top-N>")

    try:
        pipeline = build_trade_pipeline(
            cfg, signals=signals, symbols=symbols, notifier=notifier, session=session
        )
    except ValueError as exc:  # e.g. LIVE without credentials
        logger.error("Configuration error, aborting before trading: {}", exc)
        return 2

    if semi_auto:
        # Owner approves each signal by hand; the pipeline must not auto-place any
        # order. It still computes the intents that are pushed for approval.
        pipeline.execute = False
        logger.info("Semi-auto mode ON — orders await owner approval; no auto-execution")

    if cfg.needs_exchange:
        error = preflight(cfg, pipeline.execution_engine.broker)
        if error is not None:
            logger.error("Pre-flight failed, aborting before trading: {}", error)
            return 1
        logger.success("Pre-flight OK — connected to Bybit (testnet={})", cfg.testnet)

    t0 = time.perf_counter()
    result = pipeline.run_once(asof if asof is not None else pd.Timestamp.now(tz="UTC").normalize())
    _log_result(cfg, result, duration_s=time.perf_counter() - t0)

    if semi_auto:
        _push_semi_auto(result)
    return 0


def deliver_semi_auto(
    intents, *, owner_id, client, pending, subs, prefs=None, risk_prefs=None,
    broadcast_subscribers=True,
):
    """Deliver one cycle's signals: buttoned approvals to the owner, plain to subs.

    The owner (and only the owner) gets the Accept/Ignore approval messages. When
    ``broadcast_subscribers`` is set, every active subscriber also gets the plain,
    tier-shaped signal — with the owner excluded from that fan-out so they never
    receive a duplicate without buttons. The two deliveries are independent: the
    owner buttons can be turned on without also fanning out to real subscribers.
    Returns ``(owner_pushed, broadcast_result)``.
    """
    from crypto_signal_bot.platform.notify.broadcast import BroadcastResult, SubscriberBroadcaster

    owner_pushed = 0
    if owner_id:
        owner_pushed = push_owner_approvals(
            intents, owner_id=owner_id, pending=pending, client=client,
            prefs=prefs, risk_prefs=risk_prefs,
        )
    if not broadcast_subscribers:
        return owner_pushed, BroadcastResult()
    broadcaster = SubscriberBroadcaster(client, subs, prefs=prefs, risk_prefs=risk_prefs)
    result = broadcaster.broadcast(intents, exclude={owner_id} if owner_id else None)
    return owner_pushed, result


def _push_semi_auto(result) -> None:
    """Deliver the cycle's signals: owner buttons + plain broadcast to subscribers."""
    owner_id = os.getenv("TELEGRAM_ADMIN_ID", "")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.warning("TRADE_SEMI_AUTO set but TELEGRAM_BOT_TOKEN missing — cannot deliver.")
        return
    if not owner_id:
        logger.warning("TRADE_SEMI_AUTO set but TELEGRAM_ADMIN_ID missing — no owner buttons "
                       "(subscribers still get the plain signal).")
    from crypto_signal_bot.config import (
        TELEGRAM_PENDING_PATH,
        TELEGRAM_PREFS_PATH,
        TELEGRAM_RISK_PREFS_PATH,
        TELEGRAM_SUBS_PATH,
    )
    from crypto_signal_bot.platform.notify.listener import RequestsBotClient
    from crypto_signal_bot.platform.notify.pending import PendingSignalStore
    from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
    from crypto_signal_bot.platform.notify.riskprefs import RiskPrefsStore
    from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore

    # Quality filter: instead of pushing the whole cross-sectional book, surface only
    # the strongest ``SIGNAL_TOP_N`` signals (>= ``SIGNAL_MIN_CONFIDENCE``). Applied
    # only to owner/subscriber delivery — the pipeline's book/shadow/accounting are
    # untouched. SIGNAL_TOP_N=0 disables the cap (deliver every signal above the floor).
    top_n = int(os.getenv("SIGNAL_TOP_N", "5"))
    min_conf = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.0"))
    candidates = _restrict_to_liquid(result.intents)
    selected = select_quality_signals(candidates, top_n=top_n, min_confidence=min_conf)
    logger.info(
        "Quality filter: delivering {} of {} signal(s) (top_n={}, min_confidence={})",
        len(selected), len(result.intents), top_n, min_conf,
    )

    # Owner buttons are always delivered in semi-auto; the plain subscriber fan-out
    # is a SEPARATE opt-in (TRADE_BROADCAST_SUBSCRIBERS) so enabling owner approvals
    # does not silently start messaging real subscribers on every scheduled run.
    broadcast_subs = _env_flag("TRADE_BROADCAST_SUBSCRIBERS")
    owner_pushed, bcast = deliver_semi_auto(
        selected,
        owner_id=owner_id,
        client=RequestsBotClient(token),
        pending=PendingSignalStore(TELEGRAM_PENDING_PATH),
        subs=SubscriptionStore(TELEGRAM_SUBS_PATH),
        prefs=LanguagePrefsStore(TELEGRAM_PREFS_PATH),
        risk_prefs=RiskPrefsStore(TELEGRAM_RISK_PREFS_PATH),
        broadcast_subscribers=broadcast_subs,
    )
    logger.success(
        "Semi-auto delivered: {} approval(s) to owner; subscriber broadcast {} "
        "({} sent to {} subscriber(s))",
        owner_pushed, "ON" if broadcast_subs else "OFF", bcast.sent, bcast.recipients,
    )
