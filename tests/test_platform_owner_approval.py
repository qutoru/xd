"""OwnerApprovalHandler: owner-gated Accept/Ignore that risk-gates then places.

Fully offline — an InMemoryBroker places the order and a fake bot client records
every Bot API call, so the whole approval flow runs with no network, no token and
no Bybit.
"""

from __future__ import annotations

import pandas as pd

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.owner import (
    OWNER_PREFIX,
    OwnerApprovalHandler,
    approval_keyboard,
)
from crypto_signal_bot.platform.notify.pending import (
    STATUS_ACCEPTED,
    STATUS_BLOCKED,
    STATUS_EXPIRED,
    STATUS_IGNORED,
    STATUS_PENDING,
    PendingSignalStore,
)
from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
from crypto_signal_bot.platform.risk_control.control import (
    ProductionRiskControl,
    RiskControlConfig,
)

OWNER = "7"


class _FakeBotClient:
    def __init__(self):
        self.edited: list[dict] = []
        self.answered: list[dict] = []

    def edit_message_text(self, *, chat_id, message_id, text, reply_markup=None):
        self.edited.append({"chat_id": chat_id, "message_id": message_id, "text": text})

    def answer_callback_query(self, *, callback_query_id, text=""):
        self.answered.append({"id": callback_query_id, "text": text})


def _intent(symbol="BTCUSDT", notional=100.0):
    return TradeIntent(
        symbol=symbol, side=Side.BUY, target_notional=notional,
        entry=100.0, stop_loss=95.0, take_profit=110.0, confidence=0.7,
        strategy="alx", timestamp=pd.Timestamp("2026-07-18", tz="UTC"),
    )


def _handler(tmp_path, *, risk_config=None, broker_value=10_000.0, ttl=24 * 3600,
             broker=None):
    broker = broker or InMemoryBroker(value=broker_value)
    pending = PendingSignalStore(tmp_path / "pending.json")
    handler = OwnerApprovalHandler(
        owner_id=OWNER,
        pending=pending,
        engine=ExecutionEngine(broker),
        risk_control=ProductionRiskControl(risk_config or RiskControlConfig()),
        prefs=LanguagePrefsStore(tmp_path / "prefs.json"),
        ttl_seconds=ttl,
    )
    return handler, pending, broker


def _call(handler, token, action, *, from_id=OWNER, client=None):
    client = client or _FakeBotClient()
    handler.handle(
        data=f"{OWNER_PREFIX}{action}:{token}",
        from_id=from_id, chat_id=OWNER, message_id=55,
        callback_id="cb1", client=client,
    )
    return client


# -- keyboard -----------------------------------------------------------------

def test_approval_keyboard_carries_accept_and_ignore():
    kb = approval_keyboard("abc123", "en")
    row = kb["inline_keyboard"][0]
    assert [b["callback_data"] for b in row] == ["own:acc:abc123", "own:ign:abc123"]


def test_handles_only_own_namespace(tmp_path):
    handler, _, _ = _handler(tmp_path)
    assert handler.handles("own:acc:x")
    assert not handler.handles("lang:en:greet")


# -- emergency-stop reset (own:reset / /resume / /status) ---------------------

def _handler_with_state(tmp_path, *, halted=False):
    from crypto_signal_bot.platform.risk_control.state import RiskState, RiskStateStore

    rc = ProductionRiskControl(RiskControlConfig())
    rs = RiskStateStore(tmp_path / "risk_state.json")
    if halted:
        rc.trip_emergency_stop()
        rs.save(RiskState(emergency_stopped=True))
    handler = OwnerApprovalHandler(
        owner_id=OWNER,
        pending=PendingSignalStore(tmp_path / "pending.json"),
        engine=ExecutionEngine(InMemoryBroker(value=10_000.0)),
        risk_control=rc,
        risk_state=rs,
        prefs=LanguagePrefsStore(tmp_path / "prefs.json"),
    )
    return handler, rs


def test_reset_button_clears_emergency_stop_in_memory_and_on_disk(tmp_path):
    handler, rs = _handler_with_state(tmp_path, halted=True)
    assert handler.is_halted() is True

    client = _FakeBotClient()
    handler.handle(data="own:reset", from_id=OWNER, chat_id=OWNER, message_id=55,
                   callback_id="cb", client=client)

    assert handler.is_halted() is False
    assert rs.load().emergency_stopped is False  # persisted latch cleared too
    assert client.edited  # the owner got a confirmation


def test_reset_button_refused_for_non_owner(tmp_path):
    handler, rs = _handler_with_state(tmp_path, halted=True)
    client = _FakeBotClient()
    handler.handle(data="own:reset", from_id="999", chat_id="999", message_id=1,
                   callback_id="cb", client=client)
    assert handler.is_halted() is True            # still halted
    assert rs.load().emergency_stopped is True


def test_status_keyboard_shows_reset_only_when_halted(tmp_path):
    halted, _ = _handler_with_state(tmp_path, halted=True)
    kb = halted.status_keyboard("en")
    assert kb is not None
    assert kb["inline_keyboard"][0][0]["callback_data"] == "own:reset"

    active, _ = _handler_with_state(tmp_path / "active", halted=False)
    assert active.status_keyboard("en") is None


def test_reset_when_not_halted_reports_nothing_to_reset(tmp_path):
    handler, _ = _handler_with_state(tmp_path, halted=False)
    assert handler.reset_emergency_stop() is False  # was not halted


# -- buttons are owner-only ---------------------------------------------------

def test_owner_only_keyboard_present_for_owner():
    from crypto_signal_bot.platform.notify.owner import owner_only_keyboard

    kb = owner_only_keyboard("tok", "en", recipient=OWNER, owner_id=OWNER)
    assert kb is not None
    assert kb["inline_keyboard"][0][0]["callback_data"] == "own:acc:tok"


def test_owner_only_keyboard_absent_for_non_owner():
    from crypto_signal_bot.platform.notify.owner import owner_only_keyboard

    assert owner_only_keyboard("tok", "en", recipient="999", owner_id=OWNER) is None
    assert owner_only_keyboard("tok", "en", recipient=None, owner_id=OWNER) is None
    assert owner_only_keyboard("tok", "en", recipient=OWNER, owner_id=None) is None


def test_push_attaches_buttons_only_to_owner(tmp_path):
    """push_owner_approvals must address every buttoned message to the owner only."""
    from crypto_signal_bot.app.trade_runner import push_owner_approvals

    class _Recorder:
        def __init__(self):
            self.sent = []

        def send_message(self, *, chat_id, text, reply_markup=None):
            self.sent.append({"chat_id": chat_id, "reply_markup": reply_markup})

    pending = PendingSignalStore(tmp_path / "pending.json")
    client = _Recorder()
    n = push_owner_approvals(
        [_intent("BTCUSDT"), _intent("ETHUSDT")],
        owner_id=OWNER, pending=pending, client=client,
        prefs=LanguagePrefsStore(tmp_path / "prefs.json"),
    )
    assert n == 2
    # Every message with approval buttons is addressed to the owner and no one else.
    for msg in client.sent:
        assert msg["chat_id"] == OWNER
        assert msg["reply_markup"] is not None
    assert {m["chat_id"] for m in client.sent} == {OWNER}


# -- accept -------------------------------------------------------------------

def test_accept_places_order_and_marks_accepted(tmp_path):
    handler, pending, broker = _handler(tmp_path)
    sig = pending.add(_intent(), chat_id=OWNER)

    client = _call(handler, sig.token, "acc")

    assert broker.get_portfolio_state().positions  # a position was opened
    assert pending.get(sig.token).status == STATUS_ACCEPTED
    assert client.edited and "BTCUSDT" in client.edited[0]["text"]
    assert client.answered  # spinner stopped


def test_accept_books_fills_into_ledger(tmp_path):
    from crypto_signal_bot.platform.accounting.ledger import AccountingStore
    from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode

    # PAPER broker with a mock session -> real simulated fills carry a realized-PnL
    # exec feed, so the accept path books them into the ledger.
    class _Session:
        def get_instruments_info(self, **_):
            return {"result": {"list": [{"priceFilter": {"tickSize": "0.1"},
                    "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001",
                                      "minNotionalValue": "1"}}]}}

        def get_tickers(self, **_):
            return {"result": {"list": [{"lastPrice": "100"}]}}

    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER)
    broker = BybitBroker(cfg, session=_Session())
    ledger_store = AccountingStore(tmp_path / "ledger.json")
    pending = PendingSignalStore(tmp_path / "pending.json")
    handler = OwnerApprovalHandler(
        owner_id=OWNER, pending=pending, engine=ExecutionEngine(broker),
        risk_control=ProductionRiskControl(RiskControlConfig()),
        ledger_store=ledger_store,
    )
    sig = pending.add(_intent(notional=100.0), chat_id=OWNER)

    _call(handler, sig.token, "acc")

    assert pending.get(sig.token).status == STATUS_ACCEPTED
    assert len(ledger_store.load()) >= 1  # the entry fill was booked


# -- ignore -------------------------------------------------------------------

def test_ignore_places_nothing(tmp_path):
    handler, pending, broker = _handler(tmp_path)
    sig = pending.add(_intent(), chat_id=OWNER)

    client = _call(handler, sig.token, "ign")

    assert not broker.get_portfolio_state().positions
    assert pending.get(sig.token).status == STATUS_IGNORED
    assert client.edited


# -- owner gate ---------------------------------------------------------------

def test_non_owner_is_refused_without_side_effects(tmp_path):
    handler, pending, broker = _handler(tmp_path)
    sig = pending.add(_intent(), chat_id=OWNER)

    client = _call(handler, sig.token, "acc", from_id="999")

    assert not broker.get_portfolio_state().positions
    assert pending.get(sig.token).status == STATUS_PENDING  # untouched
    assert not client.edited                                 # message not changed
    assert client.answered[0]["text"]                        # generic denial toast


# -- risk gate ----------------------------------------------------------------

def test_risk_control_block_prevents_order(tmp_path):
    # Per-order cap far below the intent's notional -> MAX_POSITION_SIZE.
    handler, pending, broker = _handler(
        tmp_path, risk_config=RiskControlConfig(max_position_size=0.0001),
    )
    sig = pending.add(_intent(notional=100.0), chat_id=OWNER)

    client = _call(handler, sig.token, "acc")

    assert not broker.get_portfolio_state().positions
    assert pending.get(sig.token).status == STATUS_BLOCKED
    assert "max_position_size" in client.edited[0]["text"]


def test_kill_switch_halts_accept(tmp_path):
    handler, pending, broker = _handler(
        tmp_path, risk_config=RiskControlConfig(kill_switch=True),
    )
    sig = pending.add(_intent(), chat_id=OWNER)

    _call(handler, sig.token, "acc")

    assert not broker.get_portfolio_state().positions
    assert pending.get(sig.token).status == STATUS_BLOCKED


# -- lifecycle edges ----------------------------------------------------------

def test_expired_signal_is_refused(tmp_path):
    handler, pending, broker = _handler(tmp_path, ttl=-1)  # any age exceeds ttl
    sig = pending.add(_intent(), chat_id=OWNER)

    _call(handler, sig.token, "acc")

    assert not broker.get_portfolio_state().positions
    assert pending.get(sig.token).status == STATUS_EXPIRED


def test_unknown_token_reports_gone(tmp_path):
    handler, _, broker = _handler(tmp_path)
    client = _call(handler, "nope", "acc")
    assert not broker.get_portfolio_state().positions
    assert client.edited and client.answered


def test_double_accept_is_idempotent(tmp_path):
    handler, pending, broker = _handler(tmp_path)
    sig = pending.add(_intent(), chat_id=OWNER)

    _call(handler, sig.token, "acc")
    positions_after_first = dict(broker.get_portfolio_state().positions)
    _call(handler, sig.token, "acc")  # already accepted -> no second order

    assert pending.get(sig.token).status == STATUS_ACCEPTED
    assert set(broker.get_portfolio_state().positions) == set(positions_after_first)


# -- listener routing ---------------------------------------------------------

class _StubOwner:
    def __init__(self):
        self.calls: list[dict] = []

    def handles(self, data):
        return data.startswith(OWNER_PREFIX)

    def handle(self, **kwargs):
        self.calls.append(kwargs)


def _owner_callback(data, from_id):
    return {
        "update_id": 9,
        "callback_query": {
            "id": "cb9", "data": data,
            "from": {"id": from_id},
            "message": {"chat": {"id": OWNER}, "message_id": 3},
        },
    }


def test_listener_routes_own_callbacks_to_owner_handler(tmp_path):
    from crypto_signal_bot.platform.notify.listener import TelegramListener

    stub = _StubOwner()
    listener = TelegramListener(
        _FakeBotClient(), LanguagePrefsStore(tmp_path / "p.json"), owner_handler=stub,
    )

    listener.handle_update(_owner_callback("own:acc:tok", from_id=OWNER))
    assert len(stub.calls) == 1
    assert stub.calls[0]["from_id"] == OWNER
    assert stub.calls[0]["data"] == "own:acc:tok"

    # A non-owner tap still reaches the handler, which enforces the owner gate.
    listener.handle_update(_owner_callback("own:acc:tok", from_id="999"))
    assert stub.calls[1]["from_id"] == "999"


def test_listener_leaves_non_owner_callbacks_alone(tmp_path):
    from crypto_signal_bot.platform.notify.listener import TelegramListener

    stub = _StubOwner()
    listener = TelegramListener(
        _FakeBotClient(), LanguagePrefsStore(tmp_path / "p.json"), owner_handler=stub,
    )
    listener.handle_update(_owner_callback("lang:en:greet", from_id=OWNER))
    assert stub.calls == []  # language callback not delegated to the owner handler


# -- delivery split: owner buttons, subscribers plain -------------------------

class _SendRecorder:
    def __init__(self):
        self.sent: list[dict] = []

    def send_message(self, *, chat_id, text, reply_markup=None):
        self.sent.append({"chat_id": chat_id, "reply_markup": reply_markup})


def test_deliver_semi_auto_owner_gets_buttons_subscribers_get_plain(tmp_path):
    from crypto_signal_bot.app.trade_runner import deliver_semi_auto
    from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore

    subs = SubscriptionStore(tmp_path / "subs.json")
    subs.grant(OWNER, 30, "VIP")   # owner also happens to hold a subscription
    subs.grant("8", 30, "PRO")     # a genuine subscriber
    client = _SendRecorder()

    owner_pushed, bcast = deliver_semi_auto(
        [_intent("BTCUSDT"), _intent("ETHUSDT")],
        owner_id=OWNER, client=client,
        pending=PendingSignalStore(tmp_path / "pending.json"),
        subs=subs, prefs=LanguagePrefsStore(tmp_path / "prefs.json"),
    )

    owner_msgs = [m for m in client.sent if m["chat_id"] == OWNER]
    sub_msgs = [m for m in client.sent if m["chat_id"] == "8"]

    # Owner: only buttoned approval messages (and never a plain broadcast copy).
    assert owner_pushed == 2
    assert owner_msgs and all(m["reply_markup"] is not None for m in owner_msgs)
    # Subscriber: only plain signals, never buttons.
    assert sub_msgs and all(m["reply_markup"] is None for m in sub_msgs)
    # The owner is excluded from the fan-out, so only the real subscriber counts.
    assert bcast.recipients == 1


def test_deliver_semi_auto_owner_only_when_broadcast_disabled(tmp_path):
    from crypto_signal_bot.app.trade_runner import deliver_semi_auto
    from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore

    subs = SubscriptionStore(tmp_path / "subs.json")
    subs.grant("8", 30, "PRO")   # a real subscriber that must NOT be messaged
    client = _SendRecorder()

    owner_pushed, bcast = deliver_semi_auto(
        [_intent("BTCUSDT")], owner_id=OWNER, client=client,
        pending=PendingSignalStore(tmp_path / "pending.json"),
        subs=subs, prefs=LanguagePrefsStore(tmp_path / "prefs.json"),
        broadcast_subscribers=False,
    )
    assert owner_pushed == 1
    # Only the owner was contacted (with buttons); the subscriber got nothing.
    assert {m["chat_id"] for m in client.sent} == {OWNER}
    assert bcast.sent == 0 and bcast.recipients == 0


def test_deliver_semi_auto_without_owner_still_broadcasts(tmp_path):
    from crypto_signal_bot.app.trade_runner import deliver_semi_auto
    from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore

    subs = SubscriptionStore(tmp_path / "subs.json")
    subs.grant("8", 30, "PRO")
    client = _SendRecorder()

    owner_pushed, bcast = deliver_semi_auto(
        [_intent("BTCUSDT")], owner_id="", client=client,
        pending=PendingSignalStore(tmp_path / "pending.json"),
        subs=subs, prefs=LanguagePrefsStore(tmp_path / "prefs.json"),
    )
    assert owner_pushed == 0
    assert bcast.sent == 1 and [m["chat_id"] for m in client.sent] == ["8"]
