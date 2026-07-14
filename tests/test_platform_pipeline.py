"""Stage 6 — integration pipeline (offline, injected fake data source)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
from crypto_signal_bot.platform.pipeline import DailyPipeline, PipelineResult
from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig
from crypto_signal_bot.platform.shadow.runner import ShadowRunner

SYMS = [f"C{i}" for i in range(8)]


def _snapshot(asof, n=15):
    idx = pd.date_range(end=asof, periods=n, freq="D")
    rng = np.random.default_rng(7)
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (n, 8)), axis=0),
                          index=idx, columns=SYMS)
    # per-symbol funding means so the ALX score is non-degenerate
    means = np.linspace(-1e-4, 1e-4, 8)
    funding = pd.DataFrame(means + rng.normal(0, 1e-5, (n, 8)), index=idx, columns=SYMS)
    return MarketSnapshot(asof, SYMS, closes, closes.pct_change(), funding)


class _FakeSnapshotProvider:
    def __init__(self, snap): self.snap = snap
    def snapshot(self, asof, lookback_days): return self.snap


def _pipeline(snap, execute=True, notifier=None):
    return DailyPipeline(
        _FakeSnapshotProvider(snap),
        signal_names=["alx"],
        execution_engine=ExecutionEngine(InMemoryBroker(value=100.0)),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        notifier=notifier,
        lookback_days=10,
        execute=execute,
    )


def test_pipeline_runs_full_chain():
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    snap = _snapshot(asof)
    result = _pipeline(snap).run_once(asof)

    assert isinstance(result, PipelineResult)
    # Portfolio: dollar-neutral book
    assert np.isclose(result.book.net, 0.0, atol=1e-9)
    assert np.isclose(result.book.gross, 1.0, atol=1e-9)
    # Execution ran through the (fake) broker
    assert result.execution is not None and result.execution.n_filled > 0
    # Shadow produced a finite virtual PnL
    assert np.isfinite(result.shadow.daily_pnl)
    assert result.shadow.asof == result.book.asof
    # Futures TradeIntents were generated for the traded names
    assert result.intents
    assert all(i.entry is not None and i.confidence is not None for i in result.intents)


def test_pipeline_carries_prev_book_for_turnover():
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    pipe = _pipeline(_snapshot(asof))
    r1 = pipe.run_once(asof)
    r2 = pipe.run_once(asof + pd.Timedelta(days=1))
    # second day's shadow turnover is measured vs the first book (small since
    # the signal is unchanged here) — the pipeline threaded prev_book through.
    assert r2.shadow.turnover <= r1.shadow.turnover + 1e-9


def test_pipeline_execute_false_skips_orders():
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    result = _pipeline(_snapshot(asof), execute=False).run_once(asof)
    assert result.execution is None
    assert np.isfinite(result.shadow.daily_pnl)  # shadow still runs


def test_pipeline_delivers_intents_to_notifier():
    from crypto_signal_bot.platform.notify.notifier import TelegramConfig, TelegramNotifier

    class _FakeClient:
        def __init__(self): self.sent = []
        def send_message(self, *, chat_id, text): self.sent.append(text)

    asof = pd.Timestamp("2023-05-01", tz="UTC")
    client = _FakeClient()
    notifier = TelegramNotifier(
        TelegramConfig(bot_token="T", chat_id="C"), client=client
    )
    result = _pipeline(_snapshot(asof), notifier=notifier).run_once(asof)

    # every generated intent was delivered, and the result carries the outcome
    assert result.notify is not None
    assert result.notify.sent == len(result.intents) > 0
    assert len(client.sent) == len(result.intents)


def test_pipeline_without_notifier_leaves_notify_none():
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    result = _pipeline(_snapshot(asof)).run_once(asof)
    assert result.notify is None


# --- Stage 11: mode selection + reconciliation -----------------------------
class _FakeBybitSession:
    """Minimal Bybit v5 mock for PAPER pipeline runs (instruments + tickers)."""

    def get_server_time(self):
        return {"retCode": 0, "result": {"timeSecond": "1700000000"}}

    def get_instruments_info(self, **kw):
        return {"result": {"list": [{
            "symbol": kw.get("symbol", ""),
            "priceFilter": {"tickSize": "0.01"},
            "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001", "minNotionalValue": "1"},
        }]}}

    def get_tickers(self, **kw):
        return {"result": {"list": [{"lastPrice": "100"}]}}


def _bybit_pipeline(snap, config, session=None, notifier=None, use_brackets=False,
                    intent_params=None):
    from crypto_signal_bot.platform.execution.bybit_broker import build_broker
    from crypto_signal_bot.platform.execution.reconciler import Reconciler

    broker = build_broker(config, session=session)
    return DailyPipeline(
        _FakeSnapshotProvider(snap),
        signal_names=["alx"],
        execution_engine=ExecutionEngine(broker or InMemoryBroker(value=100.0)),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        intent_params=intent_params,
        notifier=notifier,
        reconciler=Reconciler() if broker is not None else None,
        use_brackets=use_brackets,
        execute=config.needs_exchange,
        lookback_days=10,
    )


def test_symbols_pin_the_universe_to_a_static_compatible_set():
    from crypto_signal_bot.platform.data.static_universe import StaticUniverseProvider
    from crypto_signal_bot.platform.pipeline import build_default_pipeline

    pipe = build_default_pipeline(symbols=["BTCUSDT", "ETHUSDT"])
    provider = pipe.snapshot_provider.universe
    assert isinstance(provider, StaticUniverseProvider)
    assert provider.universe(pd.Timestamp.now(tz="UTC")) == ["BTCUSDT", "ETHUSDT"]


def test_build_default_pipeline_is_the_single_mode_selection_point():
    from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
    from crypto_signal_bot.platform.pipeline import build_default_pipeline

    session = _FakeBybitSession()

    shadow = build_default_pipeline(bybit_config=BybitConfig(mode=TradingMode.SHADOW))
    assert shadow.execute is False and shadow.reconciler is None

    for mode in (TradingMode.PAPER, TradingMode.LIVE):
        cfg = BybitConfig(api_key="k", api_secret="s", mode=mode)
        pipe = build_default_pipeline(bybit_config=cfg, bybit_session=session)
        assert pipe.execute is True and pipe.reconciler is not None
        assert isinstance(pipe.execution_engine.broker, BybitBroker)
        assert pipe.execution_engine.broker.config.mode is mode


def test_paper_mode_executes_reconciles_and_notifies_in_sync():
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
    from crypto_signal_bot.platform.notify.notifier import TelegramConfig, TelegramNotifier

    class _FakeClient:
        def __init__(self): self.sent = []
        def send_message(self, *, chat_id, text): self.sent.append(text)

    asof = pd.Timestamp("2023-05-01", tz="UTC")
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER)
    client = _FakeClient()
    notifier = TelegramNotifier(TelegramConfig(bot_token="T", chat_id="C"), client=client)
    result = _bybit_pipeline(_snapshot(asof), cfg, session=_FakeBybitSession(),
                             notifier=notifier).run_once(asof)

    assert result.execution is not None and result.execution.n_filled > 0
    assert result.reconciliation is not None and result.reconciliation.ok  # paper fills match target
    assert result.notify is not None and result.notify.sent == len(result.intents) > 0


def test_shadow_mode_skips_execution_and_reconciliation():
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode

    asof = pd.Timestamp("2023-05-01", tz="UTC")
    result = _bybit_pipeline(_snapshot(asof), BybitConfig(mode=TradingMode.SHADOW)).run_once(asof)
    assert result.execution is None and result.reconciliation is None
    assert np.isfinite(result.shadow.daily_pnl)  # shadow benchmark still runs


def test_reconciliation_survives_broker_failure_without_crashing():
    from crypto_signal_bot.platform.execution.reconciler import Reconciler

    class _BrokenBroker(InMemoryBroker):
        def get_portfolio_state(self):
            raise RuntimeError("bybit unreachable")

    asof = pd.Timestamp("2023-05-01", tz="UTC")
    pipe = DailyPipeline(
        _FakeSnapshotProvider(_snapshot(asof)),
        signal_names=["alx"],
        execution_engine=ExecutionEngine(_BrokenBroker(value=100.0)),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        reconciler=Reconciler(),
        execute=True,
        lookback_days=10,
    )
    result = pipe.run_once(asof)  # must not raise
    assert result.execution is None
    assert result.reconciliation is not None and result.reconciliation.failed
    assert np.isfinite(result.shadow.daily_pnl)  # cycle continued


# --- Stage 13: reduce-only TP/SL brackets in the daily cycle ----------------
class _LiveSession(_FakeBybitSession):
    """LIVE mock: records placed orders; can reject reduce-only legs."""

    def __init__(self, reject_reduce_only=False):
        self.orders = []
        self.reject_reduce_only = reject_reduce_only

    def get_instruments_info(self, **kw):  # no min-notional floor for this test
        return {"result": {"list": [{
            "symbol": kw.get("symbol", ""),
            "priceFilter": {"tickSize": "0.01"},
            "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001", "minNotionalValue": "0"},
        }]}}

    def place_order(self, **params):
        self.orders.append(params)
        if self.reject_reduce_only and params.get("reduceOnly"):
            return {"retCode": 10001, "retMsg": "reduce-only rejected", "result": {}}
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": "oid"}}

    def get_positions(self, **kw):
        return {"result": {"list": []}}

    def get_wallet_balance(self, **kw):
        return {"result": {"list": [{"totalEquity": "10000"}]}}


def _bracket_legs(exec_report):
    """Group an exec report's orders by their client_id suffix (entry/tp/sl)."""
    from collections import Counter
    return Counter(o.request.client_id.rsplit("-", 1)[1] for o in exec_report.orders)


def test_paper_bracket_opens_position_and_places_reduce_only_tp_sl():
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
    from crypto_signal_bot.platform.execution.domain import OrderStatus, OrderType
    from crypto_signal_bot.platform.execution.intent_builder import IntentParams

    asof = pd.Timestamp("2023-05-01", tz="UTC")
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER)
    result = _bybit_pipeline(_snapshot(asof), cfg, session=_FakeBybitSession(),
                             use_brackets=True,
                             intent_params=IntentParams(notional_per_name=100.0)).run_once(asof)

    orders = result.execution.orders
    legs = _bracket_legs(result.execution)
    # one entry + one TP + one SL per traded intent
    assert legs["entry"] == legs["tp"] == legs["sl"] == len(result.intents) > 0
    # entries fill and open the position; TP/SL are reduce-only resting orders
    for o in orders:
        if o.request.order_type is OrderType.MARKET:
            assert o.status is OrderStatus.FILLED and not o.request.reduce_only
        else:
            assert o.request.reduce_only and o.status is OrderStatus.PENDING
    assert result.reconciliation is not None and result.reconciliation.ok
    assert result.execution.resulting_state.positions  # a position was opened


def test_bracket_orderlinkids_are_deterministic_for_idempotency():
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode

    asof = pd.Timestamp("2023-05-01", tz="UTC")
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER)
    ids1 = {o.request.client_id for o in
            _bybit_pipeline(_snapshot(asof), cfg, session=_FakeBybitSession(),
                            use_brackets=True).run_once(asof).execution.orders}
    ids2 = {o.request.client_id for o in
            _bybit_pipeline(_snapshot(asof), cfg, session=_FakeBybitSession(),
                            use_brackets=True).run_once(asof).execution.orders}
    # same as-of => identical orderLinkIds => Bybit dedupes a same-day re-run
    assert ids1 == ids2 and all(cid.endswith(("-entry", "-tp", "-sl")) for cid in ids1)


def test_bracket_tp_sl_rejection_leaves_position_and_continues():
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
    from crypto_signal_bot.platform.execution.domain import OrderStatus
    from crypto_signal_bot.platform.execution.intent_builder import IntentParams

    asof = pd.Timestamp("2023-05-01", tz="UTC")
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.LIVE)
    result = _bybit_pipeline(_snapshot(asof), cfg,
                             session=_LiveSession(reject_reduce_only=True),
                             use_brackets=True,
                             intent_params=IntentParams(notional_per_name=100.0)).run_once(asof)

    rejected = [o for o in result.execution.orders
                if o.request.reduce_only and o.status is OrderStatus.REJECTED]
    entries = [o for o in result.execution.orders if not o.request.reduce_only]
    assert rejected  # TP/SL failed and were recorded
    assert all(o.status is OrderStatus.FILLED for o in entries)  # entry NOT closed
    assert np.isfinite(result.shadow.daily_pnl)  # cycle continued


def test_build_default_pipeline_enables_brackets_only_for_exchange_modes():
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
    from crypto_signal_bot.platform.pipeline import build_default_pipeline

    assert build_default_pipeline().use_brackets is False  # legacy default
    assert build_default_pipeline(
        bybit_config=BybitConfig(mode=TradingMode.SHADOW)).use_brackets is False
    for mode in (TradingMode.PAPER, TradingMode.LIVE):
        pipe = build_default_pipeline(
            bybit_config=BybitConfig(api_key="k", api_secret="s", mode=mode),
            bybit_session=_FakeBybitSession())
        assert pipe.use_brackets is True
