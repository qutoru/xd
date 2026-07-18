"""SubscriberBroadcaster: tier-gated fan-out to active subscribers (offline).

A fake client records every send, so the whole fan-out is exercised with no
network and no token.
"""

from __future__ import annotations

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.broadcast import SubscriberBroadcaster
from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore


class _FakeClient:
    """Records every delivered (chat_id, text)."""

    def __init__(self):
        self.sent: list[dict] = []

    def send_message(self, *, chat_id, text):
        self.sent.append({"chat_id": chat_id, "text": text})


def _intent(symbol: str, conf: float, strategy: str = "platform") -> TradeIntent:
    return TradeIntent(
        symbol=symbol, side=Side.BUY, target_notional=50.0, entry=100.0,
        take_profit=110.0, stop_loss=95.0, confidence=conf, strategy=strategy,
    )


def _symbols_sent(client: _FakeClient) -> list[str]:
    # The formatter renders "<SYMBOL> | <side> ..." so the first token is the pair.
    return [m["text"].split(" ")[0] for m in client.sent]


def _subs(tmp_path):
    return SubscriptionStore(tmp_path / "subs.json")


def test_broadcast_fans_out_to_all_active_subscribers(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("1", 30, "PRO")
    subs.grant("2", 30, "VIP")
    client = _FakeClient()
    res = SubscriberBroadcaster(client, subs).broadcast([_intent("BTCUSDT", 0.9)])
    assert res.recipients == 2 and res.sent == 2 and res.ok
    assert {m["chat_id"] for m in client.sent} == {"1", "2"}


def test_start_capped_to_three_highest_confidence_pairs(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("1", 30, "START")
    client = _FakeClient()
    intents = [
        _intent("A", 0.1), _intent("B", 0.9), _intent("C", 0.5),
        _intent("D", 0.7), _intent("E", 0.3),
    ]
    SubscriberBroadcaster(client, subs).broadcast(intents)
    assert _symbols_sent(client) == ["B", "D", "C"]  # top-3 by confidence


def test_pro_receives_all_pairs(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("1", 30, "PRO")
    client = _FakeClient()
    intents = [_intent("A", 0.1), _intent("B", 0.9), _intent("C", 0.5), _intent("D", 0.7)]
    SubscriberBroadcaster(client, subs).broadcast(intents)
    assert len(client.sent) == 4


def test_beta_strategy_reaches_vip_only(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("pro", 30, "PRO")
    subs.grant("vip", 30, "VIP")
    client = _FakeClient()
    b = SubscriberBroadcaster(client, subs, beta_strategies=frozenset({"exp1"}))
    b.broadcast([_intent("BTCUSDT", 0.9, strategy="exp1")])
    assert [m["chat_id"] for m in client.sent] == ["vip"]


def test_priority_delivery_orders_richer_tier_first(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("s", 30, "START")
    subs.grant("v", 30, "VIP")
    client = _FakeClient()
    SubscriberBroadcaster(client, subs).broadcast([_intent("BTCUSDT", 0.9)])
    assert [m["chat_id"] for m in client.sent] == ["v", "s"]  # VIP before START


def test_lapsed_subscription_is_excluded(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("active", 30, "PRO")
    subs.grant("lapsed", -1, "PRO")  # already expired
    client = _FakeClient()
    res = SubscriberBroadcaster(client, subs).broadcast([_intent("BTCUSDT", 0.9)])
    assert res.recipients == 1
    assert [m["chat_id"] for m in client.sent] == ["active"]


def test_renders_in_each_subscribers_language(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("ru", 30, "PRO")
    subs.grant("en", 30, "PRO")
    prefs = LanguagePrefsStore(tmp_path / "prefs.json")
    prefs.set("ru", "ru")
    prefs.set("en", "en")
    client = _FakeClient()
    SubscriberBroadcaster(client, subs, prefs=prefs).broadcast([_intent("BTCUSDT", 0.9)])
    by_chat = {m["chat_id"]: m["text"] for m in client.sent}
    assert "ЛОНГ" in by_chat["ru"]
    assert "LONG" in by_chat["en"]


def test_active_sub_with_unknown_plan_defaults_to_start(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("u", 30, "test")  # unrecognised label -> base tier
    client = _FakeClient()
    intents = [_intent("A", 0.1), _intent("B", 0.9), _intent("C", 0.5), _intent("D", 0.7)]
    SubscriberBroadcaster(client, subs).broadcast(intents)
    assert len(client.sent) == 3  # capped like START


def test_send_failure_does_not_stop_the_fanout(tmp_path):
    subs = _subs(tmp_path)
    subs.grant("bad", 30, "PRO")   # sorts before "good"; raises on send
    subs.grant("good", 30, "PRO")
    client = _FakeClient()

    class _Flaky(_FakeClient):
        def send_message(self, *, chat_id, text):
            if chat_id == "bad":
                raise RuntimeError("boom")
            super().send_message(chat_id=chat_id, text=text)

    client = _Flaky()
    res = SubscriberBroadcaster(client, subs).broadcast([_intent("BTCUSDT", 0.9)])
    assert res.failed == 1 and res.sent == 1 and not res.ok
    assert [m["chat_id"] for m in client.sent] == ["good"]


def test_no_active_subscribers_is_a_noop(tmp_path):
    subs = _subs(tmp_path)
    client = _FakeClient()
    res = SubscriberBroadcaster(client, subs).broadcast([_intent("BTCUSDT", 0.9)])
    assert res == res.__class__() or (res.sent == 0 and res.recipients == 0)
    assert client.sent == []


def test_broadcast_excludes_given_chats(tmp_path):
    # The owner is kept out of the subscriber fan-out (they get the buttoned copy).
    subs = _subs(tmp_path)
    subs.grant("7", 30, "VIP")   # owner, also happens to hold a subscription
    subs.grant("8", 30, "PRO")   # a real subscriber
    client = _FakeClient()
    res = SubscriberBroadcaster(client, subs).broadcast(
        [_intent("BTCUSDT", 0.9)], exclude={"7"}
    )
    assert {m["chat_id"] for m in client.sent} == {"8"}
    assert res.recipients == 1 and res.sent == 1


def test_vip_gets_risk_line_lower_tiers_do_not(tmp_path):
    from crypto_signal_bot.platform.notify.riskprefs import RiskPrefsStore

    subs = _subs(tmp_path)
    subs.grant("vip", 30, "VIP")
    subs.grant("pro", 30, "PRO")
    rp = RiskPrefsStore(tmp_path / "risk.json")
    rp.set("vip", "high")
    client = _FakeClient()
    SubscriberBroadcaster(client, subs, risk_prefs=rp).broadcast([_intent("BTCUSDT", 0.9)])

    by_chat = {m["chat_id"]: m["text"] for m in client.sent}
    assert "Risk per trade" in by_chat["vip"] and "3% (High)" in by_chat["vip"]
    assert "Risk per trade" not in by_chat["pro"]  # only VIP (custom_risk) gets it
