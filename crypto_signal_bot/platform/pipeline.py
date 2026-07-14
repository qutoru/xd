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
from crypto_signal_bot.platform.execution.domain import ExecutionReport
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
                exec_report = self.execution_engine.execute(book)
                if self.reconciler is not None:
                    # resulting_state is already broker.get_portfolio_state() — the
                    # actual account. Reconcile it against the weights we targeted.
                    reconciliation = self.reconciler.reconcile(
                        book.weights, exec_report.resulting_state, asof=book.asof
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
    if bybit_config is not None:
        selected = build_broker(bybit_config, session=bybit_session)  # None for SHADOW
        execute = bybit_config.needs_exchange
        if selected is not None:
            broker = selected
            reconciler = Reconciler()

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
        execute=execute,
        lookback_days=lookback_days,
    )
