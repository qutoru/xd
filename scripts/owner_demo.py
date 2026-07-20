"""Offline PAPER demo of the owner semi-auto approval flow.

Renders exactly what the owner would receive in Telegram (signal text + Accept /
Ignore buttons), then simulates taps and shows the outcome — using the real PAPER
BybitBroker (mock HTTP session, no network) so the order path is genuine. It also
shows a non-owner tap being refused, so you can see the buttons/flow are owner-only.

Run from anywhere:  py scripts/owner_demo.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

# Make the repo importable and force UTF-8 so the emoji render on Windows consoles.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # py3.7+; no-op if already utf-8
except Exception:
    pass

import pandas as pd

from crypto_signal_bot.app.trade_runner import push_owner_approvals
from crypto_signal_bot.platform.accounting.ledger import AccountingStore
from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker
from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.owner import OWNER_PREFIX, OwnerApprovalHandler
from crypto_signal_bot.platform.notify.pending import PendingSignalStore
from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
from crypto_signal_bot.platform.risk_control.control import (
    ProductionRiskControl,
    RiskControlConfig,
)

OWNER = "123456789"     # pretend this is your Telegram id
STRANGER = "999000999"  # some other user
LANG = "ru"
PRICES = {"BTCUSDT": 60000.0, "ETHUSDT": 3000.0}


class PrintingClient:
    """Fake Telegram transport that prints every call like a chat transcript."""

    def __init__(self):
        self._next_id = 1000
        self.token_to_msg: dict[str, int] = {}

    def send_message(self, *, chat_id, text, reply_markup=None):
        self._next_id += 1
        mid = self._next_id
        buttons = ""
        if reply_markup:
            row = reply_markup["inline_keyboard"][0]
            buttons = "   ".join(f"[ {b['text']} ]" for b in row)
            for b in row:
                self.token_to_msg[b["callback_data"].split(":")[-1]] = mid
        tag = "with BUTTONS" if reply_markup else "NO buttons"
        print(f"\n┌─ Telegram → chat {chat_id} (msg #{mid}, {tag})")
        for line in text.splitlines():
            print(f"│ {line}")
        if buttons:
            print(f"│ {buttons}")
        print("└" + "─" * 48)

    def edit_message_text(self, *, chat_id, message_id, text, reply_markup=None):
        print(f"\n  ✏️  msg #{message_id} edited → {text}")

    def answer_callback_query(self, *, callback_query_id, text=""):
        print(f'  🔔 toast: "{text}"' if text else "  🔔 (spinner stopped)")


class MockSession:
    """Bybit public endpoints a PAPER broker needs (instrument info + last price)."""

    def get_instruments_info(self, **kw):
        return {"result": {"list": [{
            "priceFilter": {"tickSize": "0.1"},
            "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001", "minNotionalValue": "5"},
        }]}}

    def get_tickers(self, **kw):
        sym = kw.get("symbol", "BTCUSDT")
        return {"result": {"list": [{"lastPrice": str(PRICES.get(sym, 100.0))}]}}


def sample_intents():
    ts = pd.Timestamp("2026-07-18", tz="UTC")
    return [
        TradeIntent(symbol="BTCUSDT", side=Side.BUY, target_notional=300.0,
                    entry=60000.0, stop_loss=58200.0, take_profit=63000.0,
                    confidence=0.72, strategy="alx", timestamp=ts),
        TradeIntent(symbol="ETHUSDT", side=Side.SELL, target_notional=200.0,
                    entry=3000.0, stop_loss=3090.0, take_profit=2850.0,
                    confidence=0.61, strategy="alx", timestamp=ts),
    ]


def tap(handler, client, token, action, *, from_id):
    handler.handle(
        data=f"{OWNER_PREFIX}{action}:{token}", from_id=from_id, chat_id=OWNER,
        message_id=client.token_to_msg.get(token), callback_id="cbdemo", client=client,
    )


def main():
    tmp = pathlib.Path(tempfile.mkdtemp())
    prefs = LanguagePrefsStore(tmp / "prefs.json")
    prefs.set(OWNER, LANG)
    pending = PendingSignalStore(tmp / "pending.json")
    ledger_store = AccountingStore(tmp / "ledger.json")

    broker = BybitBroker(BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER,
                                     paper_equity=10_000.0), session=MockSession())
    handler = OwnerApprovalHandler(
        owner_id=OWNER, pending=pending, engine=ExecutionEngine(broker),
        risk_control=ProductionRiskControl(RiskControlConfig(max_position_size=0.10)),
        ledger_store=ledger_store, prefs=prefs,
    )
    client = PrintingClient()

    print("=" * 50)
    print("STEP 1 — pipeline pushes signals (buttons go to OWNER only)")
    print("=" * 50)
    push_owner_approvals(sample_intents(), owner_id=OWNER, pending=pending,
                         client=client, prefs=prefs)
    btc, eth = list(client.token_to_msg)[:2]

    print("\n" + "=" * 50)
    print("STEP 2 — OWNER taps ✅ Принять on BTCUSDT")
    print("=" * 50)
    tap(handler, client, btc, "acc", from_id=OWNER)

    print("\n" + "=" * 50)
    print("STEP 3 — OWNER taps 🚫 Игнорировать on ETHUSDT")
    print("=" * 50)
    tap(handler, client, eth, "ign", from_id=OWNER)

    print("\n" + "=" * 50)
    print("STEP 4 — a STRANGER taps Принять on BTCUSDT (must be refused)")
    print("=" * 50)
    tap(handler, client, btc, "acc", from_id=STRANGER)

    print("\n" + "=" * 50)
    print("RESULT — PAPER account")
    print("=" * 50)
    state = broker.get_portfolio_state()
    print(f"NAV: {state.value:.2f} USDT | positions:")
    for sym, pos in state.positions.items():
        print(f"  {sym}: {pos.quantity:+.2f} @ {pos.avg_price:.2f}")
    print(f"Ledger entries: {len(ledger_store.load())}")
    for tok in (btc, eth):
        sig = pending.get(tok)
        print(f"  {sig.intent['symbol']:8} → {sig.status}")


if __name__ == "__main__":
    main()
