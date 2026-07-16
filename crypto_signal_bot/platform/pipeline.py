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
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.execution.intent_builder import IntentParams, build_trade_intents
from crypto_signal_bot.platform.execution.reconciler import Reconciler, ReconciliationReport
from crypto_signal_bot.platform.notify.notifier import NotifyResult, TelegramNotifier
from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig, build_target_book
from crypto_signal_bot.platform.risk.manager import RiskManager
from crypto_signal_bot.platform.risk_control.control import (
    ProductionRiskControl,
    RiskControlDecision,
)
from crypto_signal_bot.platform.risk_control.state import RiskStateStore
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
    risk_control: RiskControlDecision | None = None


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
        risk_manager: RiskManager | None = None,
        risk_control: ProductionRiskControl | None = None,
        risk_state: RiskStateStore | None = None,
        nav: float = 10_000.0,
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
        # Risk sizing: TargetBook weights -> per-symbol target notional. When None,
        # IntentBuilder falls back to a flat notional (legacy). ``nav`` is only a
        # *fallback* account equity: with a real broker (PAPER/LIVE) sizing reads
        # the true equity from ``broker.get_portfolio_state().value`` instead (see
        # ``_resolve_nav``), so risk sizing and reconciliation share one NAV.
        self.risk_manager = risk_manager
        # Stage 15 — Production Risk Control: the final gate on *new* entries, run
        # after sizing and before execution. When None the pipeline behaves exactly
        # as before (no gate). ``_risk_nav_anchor`` holds (date, NAV) at the start
        # of the trading day for the daily-loss check.
        self.risk_control = risk_control
        # Optional file-backed persistence so the emergency-stop latch and the
        # daily-loss day-start NAV survive process restarts (run_trade is one-shot).
        # When None, the controls keep their in-memory-only behaviour.
        self.risk_state = risk_state
        self._risk_nav_anchor: tuple | None = None
        self.nav = nav
        # When True, open positions via the entry + reduce-only TP/SL bracket
        # (execute_bracket_intents); when False, use the legacy weight-rebalance
        # (execute(book)). A plain flag — the pipeline stays venue-agnostic.
        self.use_brackets = use_brackets
        self.lookback_days = lookback_days
        self.execute = execute
        self._prev_book: TargetBook | None = None

    def _resolve_nav(self) -> float:
        """NAV used for risk sizing — the single source of truth for account size.

        A real broker's ``get_portfolio_state().value`` (Bybit ``totalEquity`` in
        LIVE, ``paper_equity`` in PAPER) *is* that source of truth, so sizing and
        reconciliation scale against the same equity. ``self.nav`` is used only as
        a fallback where there is no real account to read: an ``InMemoryBroker``
        (offline/legacy) or no broker at all (Shadow). A broker read that raises
        must never abort the run, so it too degrades to the fallback.
        """
        broker = getattr(self.execution_engine, "broker", None)
        if broker is None or isinstance(broker, InMemoryBroker):
            return self.nav
        try:
            return float(broker.get_portfolio_state().value)
        except Exception as exc:  # network/credential failure -> fallback, never abort
            logger.warning("Broker NAV read failed; using fallback nav {}: {}", self.nav, exc)
            return self.nav

    def _current_positions_notional(self) -> pd.Series:
        """Signed notional per currently-held symbol (empty if unavailable).

        Read-only view of the broker's pre-trade state for the risk-control gate.
        A broker/API failure must never abort the run, so it degrades to empty
        (the gate then sees no held positions, i.e. treats all entries as new).
        """
        broker = getattr(self.execution_engine, "broker", None)
        if broker is None:
            return pd.Series(dtype="float64")
        try:
            state = broker.get_portfolio_state()
        except Exception as exc:  # network/credential failure -> empty, never abort
            logger.warning("Position read failed for risk control; assuming flat: {}", exc)
            return pd.Series(dtype="float64")
        return pd.Series(
            {s: p.quantity for s, p in state.positions.items()}, dtype="float64"
        )

    def _risk_day_anchor(self, asof: pd.Timestamp, nav: float, state=None) -> float:
        """NAV at the start of ``asof``'s day — the baseline for the daily-loss halt.

        With a RiskStateStore the baseline is persisted, so a later run on the same
        day (even in a new process) measures the loss against the *first* run's NAV
        rather than against itself.
        """
        date = pd.Timestamp(asof).date()
        if self.risk_state is not None and state is not None:
            iso = date.isoformat()
            if state.day == iso and state.day_start_nav is not None:
                return float(state.day_start_nav)
            state.day = iso
            state.day_start_nav = nav
            self.risk_state.save(state)
            return nav
        # in-memory fallback (unchanged when no store is wired)
        if self._risk_nav_anchor is None or self._risk_nav_anchor[0] != date:
            self._risk_nav_anchor = (date, nav)
        return self._risk_nav_anchor[1]

    def _persist_emergency_stop(self) -> None:
        """Latch the emergency stop on disk so it survives a process restart."""
        if self.risk_state is None:
            return
        state = self.risk_state.load()
        state.emergency_stopped = True
        self.risk_state.save(state)

    def _apply_production_risk_control(
        self, intents: list[TradeIntent], asof: pd.Timestamp
    ) -> tuple[RiskControlDecision, list[TradeIntent]]:
        """Gate new-entry intents; return (decision, intents that may execute).

        Only the executed intents are filtered — shadow, notify and the result keep
        the full intent list. Blocked/halted means fewer (or no) new entries; open
        positions are never closed here.
        """
        nav = self._resolve_nav()
        positions = self._current_positions_notional()
        # Persisted state (when a store is wired) carries the emergency latch and
        # the daily-loss baseline across separate program runs.
        state = self.risk_state.load() if self.risk_state is not None else None
        if state is not None and state.emergency_stopped:
            self.risk_control.trip_emergency_stop()
        proposed = pd.Series(
            {
                i.symbol: (i.target_notional if i.side is Side.BUY else -i.target_notional)
                for i in intents
            },
            dtype="float64",
        )
        decision = self.risk_control.evaluate(
            proposed, positions, nav, day_start_nav=self._risk_day_anchor(asof, nav, state)
        )
        if decision.halted:
            logger.warning(
                "Production risk control HALTED new entries: {}",
                decision.halt_reason.value if decision.halt_reason else "unknown",
            )
        elif decision.blocked:
            logger.warning(
                "Production risk control blocked {} new entry(ies): {}",
                decision.n_blocked, [(s, r.value) for s, r in decision.blocked],
            )
        allowed = set(decision.allowed)
        exec_intents = [i for i in intents if i.symbol in allowed]
        return decision, exec_intents

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

        # Risk layer sits between Portfolio and IntentBuilder: it turns weights into
        # per-symbol target notionals; IntentBuilder only renders them into intents.
        # NAV comes from the real broker when one exists (see _resolve_nav).
        notionals = None
        if self.risk_manager is not None:
            notionals = self.risk_manager.size(book, self._resolve_nav())
        intents = build_trade_intents(
            book, snapshot.closes, snapshot.returns,
            params=self.intent_params, notionals=notionals,
        )

        # Stage 15 — Production Risk Control: the final gate on *new* entries, after
        # sizing and before execution. It filters the intents that reach the broker
        # (exec_intents); shadow/notify/result keep the full intent list. A global
        # halt (kill switch / daily-loss / emergency) opens nothing this cycle and
        # never closes existing positions.
        rc_decision: RiskControlDecision | None = None
        exec_intents = intents
        if self.risk_control is not None:
            rc_decision, exec_intents = self._apply_production_risk_control(intents, asof)
        halted = rc_decision is not None and rc_decision.halted

        # Trade + reconcile. A broker/API failure must never abort the run: it is
        # logged and surfaced as a failed reconciliation, and the cycle continues.
        exec_report = None
        reconciliation = None
        if self.execute and not halted:
            try:
                if self.use_brackets:
                    # Open the position AND place its reduce-only TP/SL in one existing
                    # call (entry + reduce-only LIMIT TP + reduce-only STOP SL). Same-day
                    # re-runs are idempotent via the date-stamped orderLinkId.
                    exec_report = self.execution_engine.execute_bracket_intents(exec_intents)
                    expected = _bracket_expected_weights(exec_intents, exec_report.resulting_state.value)
                else:
                    # Legacy weight-rebalance. NOTE: this path sizes orders from
                    # book.weights x broker equity and does NOT consume the risk
                    # notionals — RiskManager governs only the bracket path (and the
                    # intents used by shadow/notify). It is reachable in production
                    # solely with the offline InMemoryBroker (real venues force
                    # use_brackets=True in build_default_pipeline), so no real
                    # account is ever sized outside RiskManager.
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
                # A critical execution failure trips the emergency stop: subsequent
                # cycles open no new positions until it is explicitly reset. The app
                # keeps running; only new entries are refused. Persisted so it also
                # survives a process restart.
                if self.risk_control is not None:
                    self.risk_control.trip_emergency_stop()
                    self._persist_emergency_stop()
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
            risk_control=rc_decision,
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
    risk_config=None,
    risk_control_config=None,
    risk_state: RiskStateStore | None = None,
    nav: float = 10_000.0,
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

    from crypto_signal_bot.platform.risk.manager import RiskConfig
    from crypto_signal_bot.platform.risk_control.control import (
        ProductionRiskControl,
        RiskControlConfig,
    )

    engine = ExecutionEngine(broker or InMemoryBroker(value=1.0))
    runner = ShadowRunner(shadow_params or ShadowParams())
    # Production Risk Control is always present so the emergency stop exists; the
    # default (no-limit) config is a no-op gate, tuned via ``risk_control_config``.
    risk_control = ProductionRiskControl(risk_control_config or RiskControlConfig())
    return DailyPipeline(
        snapshot_provider,
        signal_names=signal_names,
        execution_engine=engine,
        shadow_runner=runner,
        portfolio_config=portfolio_config,
        budgets=budgets,
        notifier=notifier,
        reconciler=reconciler,
        risk_manager=RiskManager(risk_config or RiskConfig()),
        risk_control=risk_control,
        risk_state=risk_state,
        nav=nav,
        use_brackets=use_brackets,
        execute=execute,
        lookback_days=lookback_days,
    )
