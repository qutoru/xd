"""Integration pipeline — wires Stage 1-5 layers into one daily lifecycle.

    MarketSnapshot -> SignalProvider(s) -> PortfolioBuilder -> TargetBook
        -> ExecutionEngine -> Broker -> ShadowRunner

Alpha-agnostic: signals are resolved by name through the registry, so ALX is just
one plugin. Data comes from an injected snapshot provider (Bybit in production,
a fake in tests), so this reuses Production Core without duplicating it and adds
no second data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from loguru import logger

from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.domain import ExecutionReport, OrderStatus, Side
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.execution.intent_builder import IntentParams, build_trade_intents
from crypto_signal_bot.platform.execution.reconciler import Reconciler, ReconciliationReport
from crypto_signal_bot.platform.notify.notifier import NotifyResult, TelegramNotifier
from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig, build_target_book
from crypto_signal_bot.platform.shadow.report import ShadowReport
from crypto_signal_bot.platform.shadow.runner import ShadowRunner
from crypto_signal_bot.platform.signals.registry import load_signal


@dataclass(frozen=True)
class PipelineResult:
    """Everything produced by one daily run."""

    asof: pd.Timestamp
    book: TargetBook
    shadow: ShadowReport
    execution: ExecutionReport | None
    intents: list[TradeIntent]
    notify: NotifyResult | None = None
    reconciliation: ReconciliationReport | None = None


def _bracket_expected_weights(intents: list[TradeIntent], nav: float) -> pd.Series:
    """Expected exposure (as weights) implied by the bracket entries.

    The bracket path sizes by ``intent.target_notional`` (not portfolio weight), so
    reconciliation must compare the actual account against *that* target, not
    ``book.weights``. The Reconciler itself is unchanged — only its input differs.
    """
    denom = nav if nav > 0 else 1.0
    acc: dict[str, float] = {}
    for intent in intents:
        signed = intent.target_notional if intent.side is Side.BUY else -intent.target_notional
        acc[intent.symbol] = acc.get(intent.symbol, 0.0) + signed
    return pd.Series(acc, dtype="float64") / denom


def _warn_on_unprotected(exec_report: ExecutionReport) -> None:
    """Log a warning if any reduce-only TP/SL leg was rejected (position left open)."""
    rejected = [
        o for o in exec_report.orders
        if o.request.reduce_only and o.status is OrderStatus.REJECTED
    ]
    if rejected:
        logger.warning(
            "{} reduce-only TP/SL leg(s) rejected; positions left UNPROTECTED "
            "(not auto-closing): {}",
            len(rejected), [o.request.client_id for o in rejected],
        )


class DailyPipeline:
    """One daily run: snapshot -> signals -> book -> execution -> shadow."""

    def __init__(
        self,
        snapshot_provider,
        *,
        signal_names: Sequence[str],
        execution_engine: ExecutionEngine,
        shadow_runner: ShadowRunner,
        portfolio_config: PortfolioConfig | None = None,
        budgets: dict[str, float] | None = None,
        signal_configs: Mapping[str, Mapping[str, Any]] | None = None,
        intent_params: IntentParams | None = None,
        notifier: TelegramNotifier | None = None,
        reconciler: Reconciler | None = None,
        use_brackets: bool = False,
        lookback_days: int = 90,
        execute: bool = True,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.signal_names = list(signal_names)
        self.execution_engine = execution_engine
        self.shadow_runner = shadow_runner
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.budgets = budgets
        self.signal_configs = signal_configs or {}
        self.intent_params = intent_params or IntentParams()
        self.notifier = notifier
        self.reconciler = reconciler
        # When True, open positions via the entry + reduce-only TP/SL bracket
        # (execute_bracket_intents); when False, use the legacy weight-rebalance
        # (execute(book)). A plain flag — the pipeline stays venue-agnostic.
        self.use_brackets = use_brackets
        self.lookback_days = lookback_days
        self.execute = execute
        self._prev_book: TargetBook | None = None

    def run_once(self, asof: pd.Timestamp) -> PipelineResult:
        snapshot = self.snapshot_provider.snapshot(asof, self.lookback_days)

        signals = {}
        for name in self.signal_names:
            provider = load_signal(name)
            signals[name] = provider.generate(snapshot, self.signal_configs.get(name))

        book = build_target_book(
            signals,
            budgets=self.budgets,
            prev_book=self._prev_book,
            config=self.portfolio_config,
        )

        intents = build_trade_intents(
            book, snapshot.closes, snapshot.returns, params=self.intent_params
        )

        # Trade + reconcile. A broker/API failure must never abort the run: it is
        # logged and surfaced as a failed reconciliation, and the cycle continues.
        exec_report = None
        reconciliation = None
        if self.execute:
            try:
                if self.use_brackets:
                    # Open the position AND place its reduce-only TP/SL in one existing
                    # call (entry + reduce-only LIMIT TP + reduce-only STOP SL). Same-day
                    # re-runs are idempotent via the date-stamped orderLinkId.
                    exec_report = self.execution_engine.execute_bracket_intents(intents)
                    expected = _bracket_expected_weights(intents, exec_report.resulting_state.value)
                else:
                    exec_report = self.execution_engine.execute(book)
                    expected = book.weights
                _warn_on_unprotected(exec_report)
                if self.reconciler is not None:
                    # resulting_state is already broker.get_portfolio_state() — the
                    # actual account. Reconcile it against what we targeted.
                    reconciliation = self.reconciler.reconcile(
                        expected, exec_report.resulting_state, asof=book.asof
                    )
            except Exception as exc:
                logger.warning("Execution/reconciliation failed, continuing: {}", exc)
                reconciliation = ReconciliationReport.errored(book.asof, str(exc))

        shadow_report = self.shadow_runner.step(book, snapshot, intents=intents)

        # Notify only after reconciliation, so alerts reflect the synchronized state.
        notify_result = None
        if self.notifier is not None:
            notify_result = self.notifier.notify_intents(intents)

        self._prev_book = book
        return PipelineResult(
            asof=book.asof,
            book=book,
            shadow=shadow_report,
            execution=exec_report,
            intents=intents,
            notify=notify_result,
            reconciliation=reconciliation,
        )


def build_default_pipeline(
    signal_names: Sequence[str] = ("alx",),
    *,
    portfolio_config: PortfolioConfig | None = None,
    shadow_params=None,
    broker=None,
    budgets: dict[str, float] | None = None,
    notifier: TelegramNotifier | None = None,
    bybit_config=None,
    bybit_session=None,
    symbols: Sequence[str] | None = None,
    lookback_days: int = 90,
) -> DailyPipeline:
    """Production wiring: Bybit data providers + mode-selected broker + shadow.

    ``bybit_config`` is the single point that selects the trading mode (SHADOW /
    PAPER / LIVE). ``build_broker`` maps the mode to a broker (None for SHADOW) and
    ``needs_exchange`` decides whether orders are actually routed — there is no
    mode branching anywhere else. When it is omitted, wiring falls back to the
    legacy in-memory broker so existing callers are unchanged.

    ``symbols``, when given, pins the traded universe to that explicit list (data
    is still pulled from mainnet), keeping the symbol set compatible with the
    execution venue (e.g. Bybit testnet). When omitted, the live mainnet top-N
    universe is used, so existing callers are unchanged.

    Bybit providers/broker are imported lazily so this module stays
    offline-importable; they are thin wrappers over Production Core.
    """
    from crypto_signal_bot.platform.data.bybit_source import (
        BybitDailyBarProvider,
        BybitFundingProvider,
        BybitUniverseProvider,
    )
    from crypto_signal_bot.platform.data.snapshot import DailySnapshotProvider
    from crypto_signal_bot.platform.execution.bybit_broker import build_broker
    from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
    from crypto_signal_bot.platform.shadow.report import ShadowParams

    if symbols:
        from crypto_signal_bot.platform.data.static_universe import StaticUniverseProvider
        universe_provider = StaticUniverseProvider(list(symbols))
    else:
        universe_provider = BybitUniverseProvider()

    snapshot_provider = DailySnapshotProvider(
        universe_provider, BybitDailyBarProvider(), BybitFundingProvider()
    )

    # --- single mode-selection point --------------------------------------
    reconciler: Reconciler | None = None
    execute = True
    use_brackets = False
    if bybit_config is not None:
        selected = build_broker(bybit_config, session=bybit_session)  # None for SHADOW
        execute = bybit_config.needs_exchange
        if selected is not None:
            broker = selected
            reconciler = Reconciler()
            # On a real venue, open positions with reduce-only TP/SL brackets.
            use_brackets = True

    engine = ExecutionEngine(broker or InMemoryBroker(value=1.0))
    runner = ShadowRunner(shadow_params or ShadowParams())
    return DailyPipeline(
        snapshot_provider,
        signal_names=signal_names,
        execution_engine=engine,
        shadow_runner=runner,
        portfolio_config=portfolio_config,
        budgets=budgets,
        notifier=notifier,
        reconciler=reconciler,
        use_brackets=use_brackets,
        execute=execute,
        lookback_days=lookback_days,
    )
