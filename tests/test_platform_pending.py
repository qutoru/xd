"""PendingSignalStore: durable hand-off of proposed signals for owner approval."""

from __future__ import annotations

import pandas as pd

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.pending import (
    STATUS_ACCEPTED,
    STATUS_IGNORED,
    STATUS_PENDING,
    PendingSignal,
    PendingSignalStore,
)


def _intent(symbol: str = "BTCUSDT", notional: float = 100.0) -> TradeIntent:
    return TradeIntent(
        symbol=symbol, side=Side.BUY, target_notional=notional,
        entry=100.0, stop_loss=95.0, take_profit=110.0, confidence=0.7,
        strategy="alx", timestamp=pd.Timestamp("2026-07-18", tz="UTC"),
    )


def test_add_persists_and_roundtrips(tmp_path):
    store = PendingSignalStore(tmp_path / "pending.json")
    sig = store.add(_intent(), chat_id="42")

    assert sig.status == STATUS_PENDING
    assert sig.chat_id == "42"
    assert sig.created_at is not None
    loaded = store.get(sig.token)
    assert loaded is not None
    assert loaded.intent["symbol"] == "BTCUSDT"
    assert loaded.intent["target_notional"] == 100.0
    assert loaded.is_pending()


def test_tokens_are_unique(tmp_path):
    store = PendingSignalStore(tmp_path / "pending.json")
    a = store.add(_intent("BTCUSDT"))
    b = store.add(_intent("ETHUSDT"))
    assert a.token != b.token
    assert {s.token for s in store.all_pending()} == {a.token, b.token}


def test_resolve_sets_terminal_status(tmp_path):
    store = PendingSignalStore(tmp_path / "pending.json")
    sig = store.add(_intent())
    resolved = store.resolve(sig.token, STATUS_ACCEPTED, "filled 1")

    assert resolved is not None
    assert resolved.status == STATUS_ACCEPTED
    assert resolved.result == "filled 1"
    assert store.get(sig.token).status == STATUS_ACCEPTED
    assert store.all_pending() == []  # no longer pending


def test_resolve_unknown_token_is_noop(tmp_path):
    store = PendingSignalStore(tmp_path / "pending.json")
    assert store.resolve("nope", STATUS_IGNORED) is None


def test_get_missing_returns_none(tmp_path):
    store = PendingSignalStore(tmp_path / "missing.json")
    assert store.get("x") is None
    assert store.all_pending() == []


def test_age_seconds_uses_created_at():
    created = pd.Timestamp("2026-07-18T00:00:00Z")
    sig = PendingSignal(token="t", intent={}, created_at=created.isoformat())
    age = sig.age_seconds(now=created + pd.Timedelta(seconds=90))
    assert age == 90.0


def test_age_seconds_none_without_timestamp():
    assert PendingSignal(token="t", intent={}).age_seconds() is None
