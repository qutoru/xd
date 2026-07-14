"""TradeIntent generation from a TargetBook: TP/SL, confidence, sides (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
from crypto_signal_bot.platform.execution.intent_builder import IntentParams, build_trade_intents
from crypto_signal_bot.platform.portfolio.book import TargetBook

TS = pd.Timestamp("2023-07-01 12:00", tz="UTC")
SYMS = ["A", "F"]


def _frames(n=12):
    idx = pd.date_range(end=TS, periods=n, freq="D")
    # A: alternating ±1% (known vol); F: alternating ±2%
    a = np.tile([0.01, -0.01], n // 2 + 1)[:n]
    f = np.tile([0.02, -0.02], n // 2 + 1)[:n]
    returns = pd.DataFrame({"A": a, "F": f}, index=idx)
    closes = pd.DataFrame({"A": 100.0, "F": 200.0}, index=idx)
    return closes, returns


def _book():
    return TargetBook(
        asof=TS,
        weights=pd.Series({"A": 0.5, "F": -0.5}),
        contributions={"sig": pd.Series({"A": 2.0, "F": -1.0})},
    )


def test_tp_sl_from_volatility_known_answer():
    closes, returns = _frames()
    params = IntentParams(vol_window=12, tp_mult=2.0, sl_mult=1.0)
    intents = {i.symbol: i for i in build_trade_intents(_book(), closes, returns, params=params)}

    vol_a = returns["A"].tail(12).std()
    ia = intents["A"]
    assert ia.side == Side.BUY and np.isclose(ia.entry, 100.0)
    assert np.isclose(ia.take_profit, 100.0 * (1 + 2.0 * vol_a))  # above entry
    assert np.isclose(ia.stop_loss, 100.0 * (1 - 1.0 * vol_a))  # below entry
    assert ia.timestamp == TS

    ifn = intents["F"]
    assert ifn.side == Side.SELL
    # SHORT: TP below entry, SL above entry
    assert ifn.take_profit < ifn.entry < ifn.stop_loss
    # higher-vol name has wider stops
    assert (ifn.entry - ifn.take_profit) / ifn.entry > (ia.take_profit - ia.entry) / ia.entry


def test_confidence_in_unit_range_and_not_fixed():
    closes, returns = _frames()
    intents = build_trade_intents(_book(), closes, returns)
    for i in intents:
        assert 0.0 <= i.confidence <= 1.0
    # combined |score| A=2, F=1 -> A is the max -> confidence 1.0; F < 1.0
    by = {i.symbol: i.confidence for i in intents}
    assert np.isclose(by["A"], 1.0) and by["F"] < 1.0


def test_intents_feed_execution_engine():
    closes, returns = _frames()
    intents = build_trade_intents(_book(), closes, returns)
    report = ExecutionEngine(InMemoryBroker(value=1.0)).execute_intents(intents)
    assert report.n_filled == 2


def test_empty_book_yields_no_intents():
    closes, returns = _frames()
    empty = TargetBook(asof=TS, weights=pd.Series({"A": 0.0, "F": 0.0}))
    assert build_trade_intents(empty, closes, returns) == []
