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


def _bybit_pipeline(snap, config, session=None, notifier=None):
    from crypto_signal_bot.platform.execution.bybit_broker import build_broker
    from crypto_signal_bot.platform.execution.reconciler import Reconciler

    broker = build_broker(config, session=session)
    return DailyPipeline(
        _FakeSnapshotProvider(snap),
        signal_names=["alx"],
        execution_engine=ExecutionEngine(broker or InMemoryBroker(value=100.0)),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        notifier=notifier,
        reconciler=Reconciler() if broker is not None else None,
        execute=config.needs_exchange,
        lookback_days=10,
    )


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
