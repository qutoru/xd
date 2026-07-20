"""select_quality_signals: cross-sectional book -> only the strongest signals.

ALX proposes the whole book each rebalance; semi-auto delivery should surface only
the highest-conviction entries. These lock the ranking, the top-N cap and the
confidence floor (including how a missing confidence is treated).
"""

from __future__ import annotations

from crypto_signal_bot.app.trade_runner import select_quality_signals
from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.intent import TradeIntent


def _intent(symbol: str, conf: float | None) -> TradeIntent:
    return TradeIntent(symbol=symbol, side=Side.BUY, target_notional=100.0, confidence=conf)


def test_top_n_keeps_strongest_highest_first():
    intents = [_intent("A", 0.2), _intent("B", 0.9), _intent("C", 0.5), _intent("D", 0.7)]
    out = select_quality_signals(intents, top_n=2)
    assert [i.symbol for i in out] == ["B", "D"]


def test_min_confidence_floor_drops_weak_and_none():
    intents = [_intent("A", 0.2), _intent("B", 0.9), _intent("N", None)]
    out = select_quality_signals(intents, top_n=0, min_confidence=0.5)
    assert [i.symbol for i in out] == ["B"]


def test_top_n_zero_disables_cap_but_keeps_order():
    intents = [_intent("A", 0.2), _intent("B", 0.9), _intent("C", 0.5)]
    out = select_quality_signals(intents, top_n=0)
    assert [i.symbol for i in out] == ["B", "C", "A"]


def test_missing_confidence_excluded_by_default_zero_floor():
    intents = [_intent("N", None), _intent("A", 0.1)]
    out = select_quality_signals(intents, top_n=5)  # default min_confidence=0.0
    assert [i.symbol for i in out] == ["A"]


def test_empty_input_returns_empty():
    assert select_quality_signals([], top_n=5) == []
