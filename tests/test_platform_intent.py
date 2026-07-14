"""TradeIntent domain: data model, serialization, engine conversion (offline)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.execution.domain import OrderRequest, OrderStatus, Side
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
from crypto_signal_bot.platform.execution.intent import TradeIntent

TS = pd.Timestamp("2023-07-01 12:00", tz="UTC")


def _intent(**kw):
    base = dict(symbol="BTCUSDT", side=Side.BUY, target_notional=50.0, entry=105400.0,
                take_profit=108100.0, stop_loss=104200.0, confidence=0.78,
                strategy="platform", timestamp=TS)
    base.update(kw)
    return TradeIntent(**base)


def test_no_leverage_reason_or_rr_fields():
    fields = {f.name for f in dataclasses.fields(TradeIntent)}
    assert "leverage" not in fields and "reason" not in fields
    assert not hasattr(_intent(), "risk_reward")


def test_intent_is_pure_data_model_without_rendering():
    # Display belongs to the notify layer, not the entity.
    assert not hasattr(_intent(), "format")


def test_to_dict_serialization():
    d = _intent().to_dict()
    assert d["symbol"] == "BTCUSDT" and d["side"] == "buy"
    assert d["entry"] == 105400.0 and d["take_profit"] == 108100.0
    assert d["confidence"] == 0.78 and d["timestamp"] == TS.isoformat()


def test_build_orders_strips_strategy_from_broker_request():
    reqs = ExecutionEngine(InMemoryBroker(value=1.0)).build_orders([_intent()])
    assert len(reqs) == 1 and reqs[0].symbol == "BTCUSDT" and reqs[0].side == Side.BUY
    fields = {f.name for f in dataclasses.fields(OrderRequest)}
    assert "strategy" not in fields and "stop_loss" not in fields and "take_profit" not in fields


def test_execute_intents_routes_through_broker():
    eng = ExecutionEngine(InMemoryBroker(value=1.0))
    report = eng.execute_intents([
        _intent(symbol="BTCUSDT", side=Side.BUY, target_notional=50.0),
        _intent(symbol="ETHUSDT", side=Side.SELL, target_notional=30.0),
    ])
    assert report.n_filled == 2
    assert {o.status for o in report.orders} == {OrderStatus.FILLED}
    assert np.isclose(report.resulting_state.notional("BTCUSDT"), 50.0)
    assert np.isclose(report.resulting_state.notional("ETHUSDT"), -30.0)
