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

from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.domain import ExecutionReport
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.execution.intent_builder import IntentParams, build_trade_intents
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

        exec_report = None
        if self.execute:
            exec_report = self.execution_engine.execute(book)

        shadow_report = self.shadow_runner.step(book, snapshot, intents=intents)
        self._prev_book = book
        return PipelineResult(
            asof=book.asof,
            book=book,
            shadow=shadow_report,
            execution=exec_report,
            intents=intents,
        )


def build_default_pipeline(
    signal_names: Sequence[str] = ("alx",),
    *,
    portfolio_config: PortfolioConfig | None = None,
    shadow_params=None,
    broker=None,
    budgets: dict[str, float] | None = None,
    lookback_days: int = 90,
) -> DailyPipeline:
    """Production wiring: Bybit data providers + in-memory broker + shadow runner.

    Bybit providers are imported lazily so this module stays offline-importable;
    they are thin wrappers over Production Core (``crypto_signal_bot.data``).
    """
    from crypto_signal_bot.platform.data.bybit_source import (
        BybitDailyBarProvider,
        BybitFundingProvider,
        BybitUniverseProvider,
    )
    from crypto_signal_bot.platform.data.snapshot import DailySnapshotProvider
    from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
    from crypto_signal_bot.platform.shadow.report import ShadowParams

    snapshot_provider = DailySnapshotProvider(
        BybitUniverseProvider(), BybitDailyBarProvider(), BybitFundingProvider()
    )
    engine = ExecutionEngine(broker or InMemoryBroker(value=1.0))
    runner = ShadowRunner(shadow_params or ShadowParams())
    return DailyPipeline(
        snapshot_provider,
        signal_names=signal_names,
        execution_engine=engine,
        shadow_runner=runner,
        portfolio_config=portfolio_config,
        budgets=budgets,
        lookback_days=lookback_days,
    )
