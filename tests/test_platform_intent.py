"""Stage 8-prep — TradeIntent domain + intent->order conversion (offline)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.execution.domain import OrderRequest, OrderStatus, Side
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
from crypto_signal_bot.platform.execution.intent import TradeIntent

TS = pd.Timestamp("2023-07-01", tz="UTC")


def _intent(**kw):
    base = dict(symbol="BTCUSDT", side=Side.BUY, target_notional=50.0, entry=100.0,
                stop_loss=95.0, take_profit=110.0, strategy="alx", reason="crowded funding",
                timestamp=TS)
    base.update(kw)
    return TradeIntent(**base)


def test_risk_reward_known_answer():
    # entry 100, SL 95 (risk 5), TP 110 (reward 10) -> RR 2.0
    assert np.isclose(_intent().risk_reward, 2.0)
    assert _intent(take_profit=None).risk_reward is None


def test_format_contains_futures_fields():
    msg = _intent().format()
    for token in ("BTCUSDT", "BUY", "Entry: 100", "TP: 110", "SL: 95",
                  "Risk/Reward: 2.00", "Size: 50", "Reason: crowded funding"):
        assert token in msg


def test_build_orders_strips_strategy_from_broker_request():
    eng = ExecutionEngine(InMemoryBroker(value=1.0))
    reqs = eng.build_orders([_intent()])
    assert len(reqs) == 1
    req = reqs[0]
    assert req.symbol == "BTCUSDT" and req.side == Side.BUY and np.isclose(req.quantity, 50.0)
    # OrderRequest must NOT carry strategy/reason/SL/TP (broker is strategy-agnostic)
    fields = {f.name for f in dataclasses.fields(OrderRequest)}
    assert "strategy" not in fields and "reason" not in fields
    assert "stop_loss" not in fields and "take_profit" not in fields


def test_execute_intents_routes_through_broker():
    broker = InMemoryBroker(value=1.0)
    eng = ExecutionEngine(broker)
    report = eng.execute_intents([
        _intent(symbol="BTCUSDT", side=Side.BUY, target_notional=50.0),
        _intent(symbol="ETHUSDT", side=Side.SELL, target_notional=30.0),
    ])
    assert report.n_filled == 2
    assert {o.status for o in report.orders} == {OrderStatus.FILLED}
    state = report.resulting_state
    assert np.isclose(state.notional("BTCUSDT"), 50.0)
    assert np.isclose(state.notional("ETHUSDT"), -30.0)
    assert report.asof == TS


def test_zero_size_intent_skipped():
    eng = ExecutionEngine(InMemoryBroker(value=1.0))
    assert eng.build_orders([_intent(target_notional=0.0)]) == []
