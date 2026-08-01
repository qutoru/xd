"""TelegramListener: /start, /language, and language-switch button flow (offline).

A fake bot client records every Bot API call, so the whole interactive surface is
exercised with no network and no token.
"""

from __future__ import annotations

from typing import Any

from crypto_signal_bot.platform.notify.i18n import MENU_PROMPT, t
from crypto_signal_bot.platform.notify.listener import (
    TelegramListener,
    back_keyboard,
    language_keyboard,
)
from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore
from crypto_signal_bot.platform.notify.users import UsersStore


class _FakeBotClient:
    """Records sent/edited messages and answered callbacks."""

    def __init__(self):
        self.sent: list[dict] = []
        self.edited: list[dict] = []
        self.answered: list[dict] = []
        self.commands: list[dict] = []

    def get_updates(self, *, offset, timeout):  # pragma: no cover - not used here
        return []

    def send_message(self, *, chat_id, text, reply_markup=None):
        self.sent.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})

    def edit_message_text(self, *, chat_id, message_id, text, reply_markup=None):
        self.edited.append({"chat_id": chat_id, "message_id": message_id,
                            "text": text, "reply_markup": reply_markup})

    def answer_callback_query(self, *, callback_query_id, text=""):
        self.answered.append({"id": callback_query_id, "text": text})

    def set_my_commands(self, *, commands, scope=None):
        self.commands.append({"commands": commands, "scope": scope})


def _listener(tmp_path):
    client = _FakeBotClient()
    prefs = LanguagePrefsStore(tmp_path / "prefs.json")
    return TelegramListener(client, prefs), client, prefs


def _message(chat_id: int, text: str) -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "text": text}}


def _callback(chat_id: int, message_id: int, data: str) -> dict[str, Any]:
    return {
        "update_id": 2,
        "callback_query": {
            "id": "cb1",
            "data": data,
            "message": {"chat": {"id": chat_id}, "message_id": message_id},
        },
    }


def test_language_keyboard_layout_and_flags():
    kb = language_keyboard(action="greet")
    row = kb["inline_keyboard"][0]
    assert [b["text"] for b in row] == ["🇺🇸 English", "🇷🇺 Русский"]
    assert [b["callback_data"] for b in row] == ["lang:en:greet", "lang:ru:greet"]


def test_language_keyboard_marks_current():
    row = language_keyboard(current="ru", action="switch")["inline_keyboard"][0]
    assert row[0]["text"] == "🇺🇸 English"
    assert row[1]["text"] == "✅ 🇷🇺 Русский"
    assert row[1]["callback_data"] == "lang:ru:switch"


def test_start_asks_language_first_no_greeting(tmp_path):
    listener, client, _ = _listener(tmp_path)
    listener.handle_update(_message(100, "/start"))
    assert len(client.sent) == 1
    msg = client.sent[0]
    assert msg["chat_id"] == "100"
    # /start asks the language first — the bilingual prompt, not the greeting
    assert msg["text"] == MENU_PROMPT
    assert "Bybit Smart Signals" not in msg["text"]
    assert msg["reply_markup"] == language_keyboard(action="greet")


def test_start_prompt_is_language_agnostic(tmp_path):
    # Even with a saved language, /start still asks first (fresh onboarding).
    listener, client, prefs = _listener(tmp_path)
    prefs.set("100", "ru")
    listener.handle_update(_message(100, "/start"))
    assert client.sent[0]["text"] == MENU_PROMPT
    assert client.sent[0]["reply_markup"] == language_keyboard(action="greet")


def test_language_command_shows_menu_with_current_ticked(tmp_path):
    listener, client, prefs = _listener(tmp_path)
    prefs.set("55", "ru")
    listener.handle_update(_message(55, "/language"))
    msg = client.sent[0]
    assert "Select language" in msg["text"]
    assert msg["reply_markup"] == language_keyboard(current="ru", action="switch")


def test_start_language_pick_shows_greeting_in_chosen_language(tmp_path):
    listener, client, prefs = _listener(tmp_path)
    listener.handle_update(_callback(77, 900, "lang:ru:greet"))

    assert prefs.get("77") == "ru"  # persisted
    assert len(client.edited) == 1
    edit = client.edited[0]
    assert edit["chat_id"] == "77" and edit["message_id"] == 900
    # First-ever /start: greeting followed by the risk disclaimer.
    assert edit["text"] == t("ru", "welcome") + "\n\n" + t("ru", "disclaimer")
    assert client.answered == [{"id": "cb1", "text": t("ru", "toast")}]


def test_repeat_start_greeting_omits_disclaimer(tmp_path):
    # First onboarding stores the language and shows the disclaimer; a later
    # /start greets again but without repeating the disclaimer.
    listener, client, prefs = _listener(tmp_path)
    listener.handle_update(_callback(77, 900, "lang:ru:greet"))
    assert t("ru", "disclaimer") in client.edited[-1]["text"]

    listener.handle_update(_callback(77, 901, "lang:ru:greet"))
    assert client.edited[-1]["text"] == t("ru", "welcome")
    assert t("ru", "disclaimer") not in client.edited[-1]["text"]


def test_language_menu_pick_shows_short_confirmation(tmp_path):
    listener, client, prefs = _listener(tmp_path)
    prefs.set("77", "ru")
    listener.handle_update(_callback(77, 901, "lang:en:switch"))
    assert prefs.get("77") == "en"
    assert client.edited[0]["text"] == t("en", "switched")


def test_unknown_callback_still_answered_no_state_change(tmp_path):
    listener, client, prefs = _listener(tmp_path)
    listener.handle_update(_callback(77, 902, "noise"))
    assert prefs.get("77") == "en"  # unchanged / default
    assert client.edited == []
    assert client.answered == [{"id": "cb1", "text": ""}]


def test_subscribe_sends_plans_with_back_button(tmp_path):
    listener, client, prefs = _listener(tmp_path)
    prefs.set("30", "ru")
    listener.handle_update(_message(30, "/subscribe"))
    assert len(client.sent) == 1
    msg = client.sent[0]
    assert msg["text"] == t("ru", "subscribe")
    assert msg["reply_markup"] == back_keyboard("ru")
    assert "@Daxakson" in msg["text"]


def test_subscribe_defaults_to_english(tmp_path):
    listener, client, _ = _listener(tmp_path)
    listener.handle_update(_message(31, "/subscribe"))
    assert client.sent[0]["text"] == t("en", "subscribe")
    assert client.sent[0]["reply_markup"] == back_keyboard("en")


def test_back_button_returns_to_greeting(tmp_path):
    listener, client, prefs = _listener(tmp_path)
    prefs.set("30", "ru")
    listener.handle_update(_callback(30, 500, "back"))
    assert len(client.edited) == 1
    edit = client.edited[0]
    assert edit["chat_id"] == "30" and edit["message_id"] == 500
    assert edit["text"] == t("ru", "welcome")  # back to the greeting
    assert client.answered == [{"id": "cb1", "text": ""}]


def test_help_returns_help_text_localized(tmp_path):
    listener, client, prefs = _listener(tmp_path)
    prefs.set("30", "ru")
    listener.handle_update(_message(30, "/help"))
    assert client.sent[-1]["text"] == t("ru", "help")
    # available to everyone (no subscription needed)
    listener.handle_update(_message(31, "/help"))
    assert client.sent[-1]["text"] == t("en", "help")


def test_non_command_message_ignored(tmp_path):
    listener, client, _ = _listener(tmp_path)
    listener.handle_update(_message(1, "hello there"))
    assert client.sent == []


def test_poll_once_advances_offset_past_handled_updates(tmp_path):
    listener, client, _ = _listener(tmp_path)

    class _Batch(_FakeBotClient):
        def get_updates(self, *, offset, timeout):
            return [_message(1, "/start"), _callback(1, 5, "lang:ru")]

    batch = _Batch()
    listener = TelegramListener(batch, LanguagePrefsStore(tmp_path / "p.json"))
    next_offset = listener.poll_once(None)
    assert next_offset == 3  # max update_id (2) + 1


def test_poll_once_survives_network_error(tmp_path, monkeypatch):
    # A getUpdates failure must not crash the loop: same offset, no raise.
    import crypto_signal_bot.platform.notify.listener as mod
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)  # no real backoff

    class _Boom(_FakeBotClient):
        def get_updates(self, *, offset, timeout):
            raise TimeoutError("read timed out")

    listener = TelegramListener(_Boom(), LanguagePrefsStore(tmp_path / "p.json"))
    assert listener.poll_once(42) == 42  # unchanged, and did not raise


# -- admin commands -------------------------------------------------------

ADMIN = "999"


def _admin_listener(tmp_path):
    client = _FakeBotClient()
    prefs = LanguagePrefsStore(tmp_path / "prefs.json")
    subs = SubscriptionStore(tmp_path / "subs.json")
    users = UsersStore(tmp_path / "users.json")
    listener = TelegramListener(
        client, prefs, subscriptions=subs, users=users, admin_id=ADMIN
    )
    return listener, client, subs, users


def _msg_from(from_id, text):
    return {"update_id": 1, "message": {
        "chat": {"id": from_id},
        "from": {"id": from_id, "username": "u" + str(from_id)},
        "text": text,
    }}


def test_records_users_on_message(tmp_path):
    listener, _, _, users = _admin_listener(tmp_path)
    listener.handle_update(_msg_from(1234, "/start"))
    seen = users.all()
    assert [u.chat_id for u in seen] == ["1234"]
    assert seen[0].username == "u1234"


class _FakeOwnerHandler:
    """Minimal owner handler exposing the risk-control surface the listener uses."""

    def __init__(self, halted):
        self._halted = halted
        self.reset_called = False

    def handles(self, data):
        return data.startswith("own:")

    def handle(self, **kwargs):
        pass

    def is_halted(self):
        return self._halted

    def reset_emergency_stop(self):
        was, self._halted, self.reset_called = self._halted, False, True
        return was

    def reset_result_text(self, lang, was):
        return "reset done" if was else "nothing to reset"

    def status_text(self, lang):
        return "HALTED" if self._halted else "ACTIVE"

    def status_keyboard(self, lang):
        return {"inline_keyboard": [[{"callback_data": "own:reset"}]]} if self._halted else None


def _admin_listener_with_owner(tmp_path, *, halted):
    client = _FakeBotClient()
    prefs = LanguagePrefsStore(tmp_path / "prefs.json")
    oh = _FakeOwnerHandler(halted)
    listener = TelegramListener(
        client, prefs,
        subscriptions=SubscriptionStore(tmp_path / "subs.json"),
        users=UsersStore(tmp_path / "users.json"),
        admin_id=ADMIN, owner_handler=oh,
    )
    return listener, client, oh


def test_admin_status_shows_halt_with_reset_button(tmp_path):
    listener, client, _ = _admin_listener_with_owner(tmp_path, halted=True)
    listener.handle_update(_msg_from(ADMIN, "/status"))
    assert client.sent[-1]["text"] == "HALTED"
    assert client.sent[-1]["reply_markup"] is not None  # Reset button attached


def test_admin_resume_clears_the_emergency_stop(tmp_path):
    listener, client, oh = _admin_listener_with_owner(tmp_path, halted=True)
    listener.handle_update(_msg_from(ADMIN, "/resume"))
    assert oh.reset_called is True and oh.is_halted() is False


def test_status_resume_ignored_from_non_admin(tmp_path):
    listener, _, oh = _admin_listener_with_owner(tmp_path, halted=True)
    listener.handle_update(_msg_from(9999, "/resume"))
    assert oh.reset_called is False  # non-admin cannot reset


def test_admin_grant_creates_active_subscription(tmp_path):
    listener, client, subs, _ = _admin_listener(tmp_path)
    listener.handle_update(_msg_from(ADMIN, "/grant 1234 30 START"))
    assert subs.is_active("1234") is True
    assert subs.get("1234").plan == "START"
    assert "1234" in client.sent[-1]["text"]


def test_admin_grant_bad_args_shows_usage(tmp_path):
    listener, client, subs, _ = _admin_listener(tmp_path)
    listener.handle_update(_msg_from(ADMIN, "/grant 1234 notanumber"))
    assert subs.get("1234") is None
    assert client.sent[-1]["text"] == t("en", "admin_grant_usage")


def test_admin_revoke(tmp_path):
    listener, client, subs, _ = _admin_listener(tmp_path)
    subs.grant("1234", 30)
    listener.handle_update(_msg_from(ADMIN, "/revoke 1234"))
    assert subs.get("1234") is None
    assert client.sent[-1]["text"] == t("en", "admin_revoke_ok").format(id="1234")


def test_admin_grant_by_username_resolves_via_registry(tmp_path):
    listener, client, subs, users = _admin_listener(tmp_path)
    users.record("1234", username="alice")  # alice has messaged the bot before
    listener.handle_update(_msg_from(ADMIN, "/grant @alice 30 PRO"))
    assert subs.is_active("1234") is True   # resolved @alice -> 1234
    assert subs.get("1234").plan == "PRO"


def test_admin_grant_username_case_insensitive_without_at(tmp_path):
    listener, client, subs, users = _admin_listener(tmp_path)
    users.record("1234", username="Alice")
    listener.handle_update(_msg_from(ADMIN, "/grant alice 30"))  # no @, other case
    assert subs.is_active("1234") is True


def test_admin_grant_unknown_username_reports_not_found(tmp_path):
    listener, client, subs, _ = _admin_listener(tmp_path)
    listener.handle_update(_msg_from(ADMIN, "/grant @ghost 30"))
    assert subs.active() == []  # nothing granted
    assert client.sent[-1]["text"] == t("en", "admin_user_not_found").format(name="@ghost")


def test_admin_revoke_by_username(tmp_path):
    listener, client, subs, users = _admin_listener(tmp_path)
    users.record("1234", username="alice")
    subs.grant("1234", 30)
    listener.handle_update(_msg_from(ADMIN, "/revoke @alice"))
    assert subs.get("1234") is None


def test_non_admin_cannot_use_admin_commands(tmp_path):
    listener, client, subs, _ = _admin_listener(tmp_path)
    listener.handle_update(_msg_from(1234, "/grant 5678 30"))  # not the admin
    assert subs.get("5678") is None       # nothing granted
    assert client.sent == []              # silently ignored, no reply


def test_admin_subs_lists_active_with_username(tmp_path):
    listener, client, subs, users = _admin_listener(tmp_path)
    users.record("1234", username="alice")
    subs.grant("1234", 30, plan="PRO")
    listener.handle_update(_msg_from(ADMIN, "/subs"))
    text = client.sent[-1]["text"]
    assert "1234" in text and "@alice" in text and "PRO" in text


def test_admin_users_lists_known_users(tmp_path):
    listener, client, subs, users = _admin_listener(tmp_path)
    users.record("1234", username="alice")
    subs.grant("1234", 30)
    listener.handle_update(_msg_from(ADMIN, "/users"))
    text = client.sent[-1]["text"]
    assert "1234" in text and "@alice" in text and "✅" in text  # active mark


def test_admin_disabled_when_no_admin_id(tmp_path):
    # No admin_id configured -> admin commands are inert even for that chat.
    client = _FakeBotClient()
    prefs = LanguagePrefsStore(tmp_path / "prefs.json")
    subs = SubscriptionStore(tmp_path / "subs.json")
    listener = TelegramListener(client, prefs, subscriptions=subs)
    listener.handle_update(_msg_from(ADMIN, "/grant 1234 30"))
    assert subs.get("1234") is None
    assert client.sent == []


# -- tier-gated subscriber commands: /stats and /risk ---------------------

from crypto_signal_bot.platform.accounting.ledger import (  # noqa: E402
    AccountingLedger,
    AccountingStore,
    LedgerEntry,
)
from crypto_signal_bot.platform.notify.riskprefs import RiskPrefsStore  # noqa: E402


def _tier_listener(tmp_path, *, ledger=None):
    client = _FakeBotClient()
    prefs = LanguagePrefsStore(tmp_path / "prefs.json")
    subs = SubscriptionStore(tmp_path / "subs.json")
    risk = RiskPrefsStore(tmp_path / "risk.json")
    store = AccountingStore(tmp_path / "ledger.json")
    if ledger is not None:
        store.save(ledger)
    listener = TelegramListener(
        client, prefs, subscriptions=subs, users=UsersStore(tmp_path / "u.json"),
        admin_id=ADMIN, ledger_store=store, risk_prefs=risk,
    )
    return listener, client, subs, risk


def _closed_ledger():
    return AccountingLedger([
        LedgerEntry(exec_id="1", timestamp="2026-07-01T00:00:00+00:00", symbol="BTCUSDT",
                    side="sell", quantity=1.0, price=100.0, fee=0.0, realized_pnl=30.0,
                    attribution="take_profit"),
        LedgerEntry(exec_id="2", timestamp="2026-07-02T00:00:00+00:00", symbol="ETHUSDT",
                    side="sell", quantity=1.0, price=100.0, fee=0.0, realized_pnl=-10.0,
                    attribution="stop_loss"),
    ])


def test_stats_locked_for_non_subscriber(tmp_path):
    listener, client, _, _ = _tier_listener(tmp_path)
    listener.handle_update(_msg_from(500, "/stats"))
    assert client.sent[-1]["text"] == t("en", "stats_locked")


def test_stats_locked_for_start_tier(tmp_path):
    listener, client, subs, _ = _tier_listener(tmp_path)
    subs.grant("500", 30, "START")
    listener.handle_update(_msg_from(500, "/stats"))
    assert client.sent[-1]["text"] == t("en", "stats_locked")


def test_stats_shown_for_pro_tier(tmp_path):
    listener, client, subs, _ = _tier_listener(tmp_path, ledger=_closed_ledger())
    subs.grant("500", 30, "PRO")
    listener.handle_update(_msg_from(500, "/stats"))
    text = client.sent[-1]["text"]
    assert t("en", "stats_header") in text
    assert "50%" in text and "BTCUSDT" in text


def test_stats_empty_when_no_closed_trades(tmp_path):
    listener, client, subs, _ = _tier_listener(tmp_path)  # empty ledger
    subs.grant("500", 30, "VIP")
    listener.handle_update(_msg_from(500, "/stats"))
    assert client.sent[-1]["text"] == t("en", "stats_empty")


def test_risk_locked_for_pro_tier(tmp_path):
    listener, client, subs, _ = _tier_listener(tmp_path)
    subs.grant("500", 30, "PRO")  # PRO has no personal risk settings
    listener.handle_update(_msg_from(500, "/risk"))
    assert client.sent[-1]["text"] == t("en", "risk_locked")


def test_risk_view_and_set_for_vip(tmp_path):
    listener, client, subs, risk = _tier_listener(tmp_path)
    subs.grant("500", 30, "VIP")
    # no arg -> reports current (default medium)
    listener.handle_update(_msg_from(500, "/risk"))
    assert t("en", "risk_medium") in client.sent[-1]["text"]
    # set -> persisted and confirmed
    listener.handle_update(_msg_from(500, "/risk high"))
    assert risk.get("500") == "high"
    assert client.sent[-1]["text"] == t("en", "risk_set").format(level=t("en", "risk_high"))


def test_risk_invalid_level_shows_usage(tmp_path):
    listener, client, subs, risk = _tier_listener(tmp_path)
    subs.grant("500", 30, "VIP")
    listener.handle_update(_msg_from(500, "/risk bogus"))
    assert client.sent[-1]["text"] == t("en", "risk_usage")
    assert risk.get("500") == "medium"  # unchanged


# -- per-tier '/' command menu (setMyCommands) ----------------------------

def _last_menu(client):
    call = client.commands[-1]
    return [c["command"] for c in call["commands"]], call["scope"]


def test_start_sets_base_command_menu_scoped_to_chat(tmp_path):
    listener, client, _, _ = _tier_listener(tmp_path)
    listener.handle_update(_msg_from(500, "/start"))
    names, scope = _last_menu(client)
    assert names == ["start", "language", "subscribe", "help"]  # base only, no sub
    assert scope == {"type": "chat", "chat_id": "500"}


def test_language_pick_localizes_the_command_menu(tmp_path):
    listener, client, _, _ = _tier_listener(tmp_path)
    listener.handle_update(_callback(500, 900, "lang:ru:greet"))
    call = client.commands[-1]
    descriptions = [c["description"] for c in call["commands"]]
    assert t("ru", "cmd_subscribe") in descriptions  # RU descriptions
    assert call["scope"] == {"type": "chat", "chat_id": "500"}


def test_vip_menu_includes_stats_and_risk(tmp_path):
    listener, client, subs, _ = _tier_listener(tmp_path)
    subs.grant("500", 30, "VIP")
    listener.handle_update(_msg_from(500, "/start"))
    names, _ = _last_menu(client)
    assert names == ["start", "language", "subscribe", "help", "stats", "risk"]


def test_pro_menu_includes_stats_but_not_risk(tmp_path):
    listener, client, subs, _ = _tier_listener(tmp_path)
    subs.grant("500", 30, "PRO")
    listener.handle_update(_msg_from(500, "/start"))
    names, _ = _last_menu(client)
    assert names == ["start", "language", "subscribe", "help", "stats"]


def test_admin_grant_refreshes_target_command_menu(tmp_path):
    listener, client, subs, _ = _tier_listener(tmp_path)
    listener.handle_update(_msg_from(ADMIN, "/grant 500 30 VIP"))
    # the last setMyCommands call targets the granted user with VIP commands
    names, scope = _last_menu(client)
    assert scope == {"type": "chat", "chat_id": "500"}
    assert "stats" in names and "risk" in names


def test_admin_menu_includes_admin_commands(tmp_path):
    listener, client, _, _ = _tier_listener(tmp_path)
    listener.handle_update(_msg_from(ADMIN, "/start"))
    names, scope = _last_menu(client)
    assert scope == {"type": "chat", "chat_id": ADMIN}
    assert names[-6:] == ["grant", "revoke", "subs", "users", "status", "resume"]


def test_regular_user_menu_excludes_admin_commands(tmp_path):
    listener, client, subs, _ = _tier_listener(tmp_path)
    subs.grant("500", 30, "VIP")  # even a VIP is not the admin
    listener.handle_update(_msg_from(500, "/start"))
    names, _ = _last_menu(client)
    assert not ({"grant", "revoke", "subs", "users"} & set(names))


def test_owner_has_vip_entitlements_without_subscription(tmp_path):
    from crypto_signal_bot.platform.notify.plans import ENTITLEMENTS, Tier

    listener, _, _, _ = _tier_listener(tmp_path)  # ADMIN has NO subscription granted
    ent = listener._entitlements(ADMIN)
    assert ent == ENTITLEMENTS[Tier.VIP]           # owner == full VIP
    assert ent.stats and ent.custom_risk
    assert listener._entitlements("12345") is None  # a non-owner still gets nothing


def test_owner_menu_includes_vip_commands(tmp_path):
    listener, client, _, _ = _tier_listener(tmp_path)  # ADMIN, no subscription
    listener.handle_update(_msg_from(ADMIN, "/start"))
    names, _ = _last_menu(client)
    assert {"stats", "risk"} <= set(names)  # VIP commands unlocked for the owner
