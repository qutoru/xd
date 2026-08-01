"""TradeOutcomeReporter: WIN/LOSE messages on close, de-duped and baseline-safe.

Fully offline — a fake broker feeds fills and a fake client records every send, so
the whole reporting path runs with no network, no token and no Bybit.
"""

from __future__ import annotations

import pandas as pd

from crypto_signal_bot.platform.accounting.ledger import AccountingStore
from crypto_signal_bot.platform.execution.domain import Fill, Side
from crypto_signal_bot.platform.notify.outcomes import (
    NotifiedOutcomeStore,
    TradeOutcomeReporter,
)

OWNER = "7"


class _FakeClient:
    def __init__(self):
        self.sent: list[dict] = []

    def send_message(self, *, chat_id, text, reply_markup=None):
        self.sent.append({"chat_id": chat_id, "text": text})


class _FakeBroker:
    """Returns the same fills on every poll (models an overlapping ``since`` window)."""

    def __init__(self, fills):
        self._fills = list(fills)

    def get_fills(self, since=None):
        return list(self._fills)


def _fill(symbol, exec_id, *, realized=0.0, closed=0.0, link=None, fee=0.0):
    return Fill(
        symbol=symbol, side=Side.SELL, quantity=1.0, price=100.0, fee=fee,
        timestamp=pd.Timestamp("2026-07-20T00:00:00Z"),
        order_link_id=link, exec_id=exec_id, realized_pnl=realized, closed_quantity=closed,
    )


def _reporter(tmp_path, fills):
    ledger_store = AccountingStore(tmp_path / "ledger.json")
    state = NotifiedOutcomeStore(tmp_path / "notified.json")
    client = _FakeClient()
    reporter = TradeOutcomeReporter(
        broker=_FakeBroker(fills), ledger_store=ledger_store, client=client,
        chat_id=OWNER, state=state,
    )
    return reporter, client, ledger_store, state


def test_win_and_lose_announced_entry_ignored(tmp_path):
    fills = [
        _fill("BTCUSDT", "e1", closed=0.0),                                    # entry
        _fill("BTCUSDT", "c1", realized=12.0, fee=0.5, closed=1.0, link="alx-BTCUSDT-tp"),
        _fill("ETHUSDT", "c2", realized=-8.0, fee=0.5, closed=1.0, link="x-ETHUSDT-sl"),
    ]
    reporter, client, *_ = _reporter(tmp_path, fills)

    sent = reporter.poll()

    assert sent == 2  # the entry produces no outcome
    texts = "\n".join(m["text"] for m in client.sent)
    assert "BTCUSDT — WIN" in texts and "+11.50 USDT" in texts   # 12.0 - 0.5 fee
    assert "ETHUSDT — LOSE" in texts and "-8.50 USDT" in texts   # -8.0 - 0.5 fee
    assert all(m["chat_id"] == OWNER for m in client.sent)


def test_no_double_send_across_polls(tmp_path):
    fills = [_fill("BTCUSDT", "c1", realized=5.0, closed=1.0, link="a-BTCUSDT-tp")]
    reporter, client, *_ = _reporter(tmp_path, fills)

    assert reporter.poll() == 1
    assert reporter.poll() == 0  # same close, already announced
    assert len(client.sent) == 1


def test_seed_if_new_baselines_existing_closes(tmp_path):
    fills = [_fill("BTCUSDT", "c1", realized=5.0, closed=1.0, link="a-BTCUSDT-tp")]
    reporter, client, ledger_store, _ = _reporter(tmp_path, fills)
    # Pre-populate the ledger with the close, as if it happened before the reporter ran.
    ledger = ledger_store.load()
    ledger.record(_FakeBroker(fills).get_fills())
    ledger_store.save(ledger)

    seeded = reporter.seed_if_new()

    assert seeded == 1
    assert reporter.poll() == 0  # pre-existing close is not re-announced
    assert client.sent == []


def test_partial_fills_aggregated_into_one_message(tmp_path):
    # One stop-loss that settles in many partial executions (distinct execIds,
    # each with its own slice of the loss) must yield ONE LOSE message carrying
    # the summed net PnL — not one message per partial fill.
    fills = [_fill("SOLUSDT", "entry", closed=0.0, link="alx-SOLUSDT-entry")]
    fills += [
        _fill("SOLUSDT", f"p{i}", realized=-2.0, fee=0.1, closed=1.0, link="alx-SOLUSDT-sl")
        for i in range(20)
    ]
    reporter, client, *_ = _reporter(tmp_path, fills)

    sent = reporter.poll()

    assert sent == 1
    assert len(client.sent) == 1
    text = client.sent[0]["text"]
    assert "SOLUSDT — LOSE" in text
    assert "-42.00 USDT" in text          # 20 * (-2.0 - 0.1) = -42.00
    assert reporter.poll() == 0           # nothing re-announced on the next tick


def test_breakeven_close_reports_zero(tmp_path):
    fills = [_fill("BTCUSDT", "c1", realized=0.0, fee=0.0, closed=1.0, link="a-BTCUSDT-close")]
    reporter, client, *_ = _reporter(tmp_path, fills)

    assert reporter.poll() == 1
    assert "BREAK-EVEN" in client.sent[0]["text"]
    assert "+0.00 USDT" in client.sent[0]["text"]
