"""TelegramListener — the receiving half of the Telegram bot.

The notifier only *sends*; this module *receives*. It long-polls ``getUpdates``
and handles the interactive surface:

    /start      -> ask the language first (buttons only, no greeting yet)
    /language   -> the language menu (current language ticked)
    /subscribe  -> the localized subscription plans
    button tap  -> persist the choice, then show the localized greeting
                   (from /start) or a short confirmation (from /language)

Admin-only commands (gated on the sender's id == ``admin_id``):

    /grant <user_id|@username> <days> [plan] -> grant/extend a subscription
    /revoke <user_id|@username>              -> revoke a subscription
    /subs                                    -> list active subscriptions
    /users                                   -> list everyone who messaged the bot

A @username is resolved against the users registry, so it only works for people
who have already messaged the bot; otherwise the admin is told to have them start
the bot first.

Transport is injected (a ``TelegramBotClient``) so the whole flow is unit-tested
with a fake — no network, no real token. Language choices are persisted via a
:class:`LanguagePrefsStore`, so a restart keeps every chat's language.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from crypto_signal_bot.platform.notify.i18n import (
    DEFAULT_LANGUAGE,
    FLAG,
    LANG_NAME,
    LANGUAGES,
    MENU_PROMPT,
    normalize,
    t,
)
from crypto_signal_bot.platform.accounting.ledger import AccountingLedger, AccountingStore
from crypto_signal_bot.platform.notify.commands import tier_command_menu
from crypto_signal_bot.platform.notify.plans import ENTITLEMENTS, Entitlements, Tier, tier_of
from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore
from crypto_signal_bot.platform.notify.riskprefs import RiskPrefsStore, normalize_risk
from crypto_signal_bot.platform.notify.stats import TradeStats, format_stats
from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore
from crypto_signal_bot.platform.notify.users import UsersStore

_CALLBACK_PREFIX = "lang:"
_BACK_DATA = "back"  # callback_data of the "Back" button under /subscribe
_DEFAULT_API_BASE = "https://api.telegram.org"
_DEFAULT_TIMEOUT = 35.0  # long-poll: must exceed the getUpdates poll window
_ERROR_BACKOFF_S = 3.0   # pause before retrying after a transient getUpdates error


def _handle_str(user: Any) -> str:
    """Render '' / ' @username' / ' (First)' for an optional KnownUser."""
    if user is None:
        return ""
    if getattr(user, "username", None):
        return f" @{user.username}"
    if getattr(user, "first_name", None):
        return f" ({user.first_name})"
    return ""


def language_keyboard(current: str | None = None, action: str = "switch") -> dict[str, Any]:
    """Inline keyboard with one button per language (current one ticked).

    Buttons carry ``callback_data`` ``"lang:<code>:<action>"`` where ``action``
    is ``"greet"`` (tapped from /start → show the greeting next) or ``"switch"``
    (tapped from /language → show a short confirmation). ``current`` (if given
    and supported) is prefixed with a check mark. A single row keeps the two
    flags side by side, matching the agreed layout.
    """
    row = []
    for code in LANGUAGES:
        label = f"{FLAG[code]} {LANG_NAME[code]}"
        if current is not None and normalize(current) == code:
            label = f"✅ {label}"
        row.append({"text": label, "callback_data": f"{_CALLBACK_PREFIX}{code}:{action}"})
    return {"inline_keyboard": [row]}


def back_keyboard(lang: str) -> dict[str, Any]:
    """Single-button keyboard ("Back") that returns to the greeting."""
    return {"inline_keyboard": [[{"text": t(lang, "back_button"),
                                  "callback_data": _BACK_DATA}]]}


class TelegramBotClient(Protocol):
    """Transport for the receiving side of the Bot API (raises on failure)."""

    def get_updates(self, *, offset: int | None, timeout: int) -> list[dict[str, Any]]: ...

    def send_message(
        self, *, chat_id: str, text: str, reply_markup: dict[str, Any] | None = None
    ) -> None: ...

    def edit_message_text(
        self,
        *,
        chat_id: str,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None: ...

    def answer_callback_query(self, *, callback_query_id: str, text: str = "") -> None: ...

    def set_my_commands(
        self, *, commands: list[dict[str, Any]], scope: dict[str, Any] | None = None
    ) -> None: ...


_ADMIN_COMMANDS = ("/grant", "/revoke", "/subs", "/users")


class TelegramListener:
    """Handles incoming Telegram updates: commands, language buttons, admin ops."""

    def __init__(
        self,
        client: TelegramBotClient,
        prefs: LanguagePrefsStore,
        *,
        subscriptions: SubscriptionStore | None = None,
        users: UsersStore | None = None,
        admin_id: str | None = None,
        ledger_store: AccountingStore | None = None,
        risk_prefs: RiskPrefsStore | None = None,
        owner_handler=None,
    ) -> None:
        self._client = client
        self._prefs = prefs
        self._subs = subscriptions
        self._users = users
        self._admin_id = str(admin_id) if admin_id else None
        self._ledger_store = ledger_store  # source for /stats (PRO/VIP)
        self._risk_prefs = risk_prefs       # personal risk level for /risk (VIP)
        # Owner-only semi-auto approval (Accept/Ignore on pushed signals). Optional
        # and owner-gated inside the handler, so it stays invisible to other users.
        self._owner_handler = owner_handler

    # -- individual update handling ---------------------------------------

    def handle_update(self, update: dict[str, Any]) -> None:
        """Dispatch one raw update to the command or callback handler."""
        if "callback_query" in update:
            self._on_callback(update["callback_query"])
        elif "message" in update:
            self._on_message(update["message"])

    def _on_message(self, message: dict[str, Any]) -> None:
        chat_id = str(message.get("chat", {}).get("id", ""))
        sender = message.get("from") or {}
        from_id = str(sender.get("id", "")) or chat_id
        text = (message.get("text") or "").strip()
        if not chat_id:
            return

        # Remember every user so the admin can grant access by id later.
        if self._users is not None and from_id:
            self._users.record(
                from_id,
                username=sender.get("username"),
                first_name=sender.get("first_name"),
            )

        if not text.startswith("/"):
            return
        parts = text.split()
        command = parts[0].split("@")[0].lower()  # strip args and @botname
        args = parts[1:]

        if command in _ADMIN_COMMANDS:
            if self._admin_id is not None and from_id == self._admin_id:
                self._handle_admin(command, args, chat_id, self._prefs.get(chat_id))
            return  # silently ignore admin commands from non-admins

        if command == "/start":
            # Ask the language first; the greeting follows once one is picked.
            self._client.send_message(
                chat_id=chat_id,
                text=MENU_PROMPT,
                reply_markup=language_keyboard(action="greet"),
            )
            self._sync_commands(chat_id)  # refresh the '/' menu for this chat
        elif command == "/language":
            lang = self._prefs.get(chat_id)
            self._client.send_message(
                chat_id=chat_id,
                text=MENU_PROMPT,
                reply_markup=language_keyboard(current=lang, action="switch"),
            )
        elif command == "/subscribe":
            lang = self._prefs.get(chat_id)
            self._client.send_message(
                chat_id=chat_id,
                text=t(lang, "subscribe"),
                reply_markup=back_keyboard(lang),
            )
        elif command == "/help":
            lang = self._prefs.get(chat_id)
            self._client.send_message(chat_id=chat_id, text=t(lang, "help"))
        elif command == "/stats":
            self._on_stats(chat_id, self._prefs.get(chat_id))
        elif command == "/risk":
            self._on_risk(chat_id, args, self._prefs.get(chat_id))

    # -- tier-gated subscriber commands ------------------------------------

    def _entitlements(self, chat_id: str) -> Entitlements | None:
        """Entitlements for a chat's *active* subscription, else None.

        An active subscription with an unrecognised plan label falls back to the
        base (START) tier, so richer features stay gated to explicit PRO/VIP.
        """
        if self._subs is None:
            return None
        sub = self._subs.get(chat_id)
        if sub is None or not sub.is_active():
            return None
        return ENTITLEMENTS[tier_of(sub.plan) or Tier.START]

    def _sync_commands(self, chat_id: str) -> None:
        """Best-effort: set this chat's '/' menu to match its tier + language.

        Called whenever tier or language may have changed (language pick, admin
        grant/revoke). A transport error here must never break message handling.
        """
        lang = self._prefs.get(chat_id)
        is_admin = self._admin_id is not None and chat_id == self._admin_id
        try:
            self._client.set_my_commands(
                commands=tier_command_menu(self._entitlements(chat_id), lang, is_admin=is_admin),
                scope={"type": "chat", "chat_id": chat_id},
            )
        except Exception as exc:  # best-effort: menu is cosmetic, don't crash
            logger.warning("Failed to set command menu for {}: {}", chat_id, exc)

    def _sync_default_commands(self) -> None:
        """Set the fallback '/' menu (base commands) for chats without an override."""
        try:
            self._client.set_my_commands(
                commands=tier_command_menu(None, DEFAULT_LANGUAGE)
            )
        except Exception as exc:  # best-effort
            logger.warning("Failed to set default command menu: {}", exc)

    def _on_stats(self, chat_id: str, lang: str) -> None:
        """/stats — trade statistics, available on PRO and VIP."""
        ent = self._entitlements(chat_id)
        if ent is None or not ent.stats:
            self._client.send_message(chat_id=chat_id, text=t(lang, "stats_locked"))
            return
        ledger = self._ledger_store.load() if self._ledger_store is not None else AccountingLedger()
        self._client.send_message(
            chat_id=chat_id, text=format_stats(TradeStats.from_ledger(ledger), lang)
        )

    def _on_risk(self, chat_id: str, args: list[str], lang: str) -> None:
        """/risk [level] — view or set the personal risk level (VIP only)."""
        ent = self._entitlements(chat_id)
        if ent is None or not ent.custom_risk:
            self._client.send_message(chat_id=chat_id, text=t(lang, "risk_locked"))
            return
        if self._risk_prefs is None:
            return
        if not args:  # no argument: report the current level
            level = self._risk_prefs.get(chat_id)
            self._client.send_message(
                chat_id=chat_id,
                text=t(lang, "risk_current").format(level=t(lang, f"risk_{level}")),
            )
            return
        level = normalize_risk(args[0])
        if level is None:
            self._client.send_message(chat_id=chat_id, text=t(lang, "risk_usage"))
            return
        self._risk_prefs.set(chat_id, level)
        self._client.send_message(
            chat_id=chat_id, text=t(lang, "risk_set").format(level=t(lang, f"risk_{level}"))
        )

    # -- admin operations --------------------------------------------------

    def _handle_admin(self, command: str, args: list[str], chat_id: str, lang: str) -> None:
        """Grant/revoke/list subscriptions — only reached for the admin."""
        if self._subs is None:
            return
        if command == "/grant":
            self._admin_grant(args, chat_id, lang)
        elif command == "/revoke":
            self._admin_revoke(args, chat_id, lang)
        elif command == "/subs":
            self._admin_list_subs(chat_id, lang)
        elif command == "/users":
            self._admin_list_users(chat_id, lang)

    def _resolve_target(self, token: str) -> str | None:
        """Turn a /grant|/revoke target into a chat_id.

        A numeric token is taken as an id directly; anything else (with or
        without a leading @) is looked up as a username in the users registry.
        Returns None if a username can't be resolved (user never messaged us).
        """
        if token.lstrip("-").isdigit():
            return token
        return self._users.find_by_username(token) if self._users else None

    def _admin_grant(self, args: list[str], chat_id: str, lang: str) -> None:
        # /grant <user_id|@username> <days> [plan]
        if len(args) < 2 or not args[1].lstrip("-").isdigit():
            self._client.send_message(chat_id=chat_id, text=t(lang, "admin_grant_usage"))
            return
        target = self._resolve_target(args[0])
        if target is None:
            self._client.send_message(
                chat_id=chat_id, text=t(lang, "admin_user_not_found").format(name=args[0])
            )
            return
        days = int(args[1])
        plan = " ".join(args[2:]) if len(args) > 2 else None
        sub = self._subs.grant(target, days, plan)
        text = t(lang, "admin_grant_ok").format(
            id=target, until=sub.expires_at.date().isoformat(), plan=sub.plan or "—"
        )
        self._client.send_message(chat_id=chat_id, text=text)
        self._sync_commands(target)  # unlock tier commands in the user's '/' menu
        logger.info("Admin granted {} days to {} (plan={})", days, target, sub.plan)

    def _admin_revoke(self, args: list[str], chat_id: str, lang: str) -> None:
        if not args:
            self._client.send_message(chat_id=chat_id, text=t(lang, "admin_revoke_usage"))
            return
        target = self._resolve_target(args[0])
        if target is None:
            self._client.send_message(
                chat_id=chat_id, text=t(lang, "admin_user_not_found").format(name=args[0])
            )
            return
        key = "admin_revoke_ok" if self._subs.revoke(target) else "admin_revoke_none"
        self._client.send_message(chat_id=chat_id, text=t(lang, key).format(id=target))
        self._sync_commands(target)  # drop tier commands from the user's '/' menu
        logger.info("Admin revoke {} -> {}", target, key)

    def _admin_list_subs(self, chat_id: str, lang: str) -> None:
        active = self._subs.active()
        if not active:
            self._client.send_message(chat_id=chat_id, text=t(lang, "admin_subs_empty"))
            return
        names = {u.chat_id: u for u in (self._users.all() if self._users else [])}
        lines = [t(lang, "admin_subs_header")]
        for sub in active:
            handle = _handle_str(names.get(sub.chat_id))
            plan = f" [{sub.plan}]" if sub.plan else ""
            lines.append(f"{sub.chat_id}{handle} — {sub.expires_at.date().isoformat()}{plan}")
        self._client.send_message(chat_id=chat_id, text="\n".join(lines))

    def _admin_list_users(self, chat_id: str, lang: str) -> None:
        users = self._users.all() if self._users else []
        if not users:
            self._client.send_message(chat_id=chat_id, text=t(lang, "admin_users_empty"))
            return
        lines = [t(lang, "admin_users_header")]
        for u in users:
            mark = "✅ " if self._subs and self._subs.is_active(u.chat_id) else ""
            lines.append(f"{mark}{u.chat_id}{_handle_str(u)}")
        self._client.send_message(chat_id=chat_id, text="\n".join(lines))

    def _on_callback(self, callback: dict[str, Any]) -> None:
        data = callback.get("data") or ""
        callback_id = str(callback.get("id", ""))
        message = callback.get("message") or {}
        chat_id = str(message.get("chat", {}).get("id", ""))
        message_id = message.get("message_id")
        from_id = str((callback.get("from") or {}).get("id", "")) or chat_id

        # Owner-only approval buttons (own:acc / own:ign). Owner-gated inside the
        # handler; any tap from a non-owner is refused there without side effects.
        if self._owner_handler is not None and self._owner_handler.handles(data):
            self._owner_handler.handle(
                data=data, from_id=from_id, chat_id=chat_id,
                message_id=message_id, callback_id=callback_id, client=self._client,
            )
            return

        # "Back" button under /subscribe: return to the greeting.
        if data == _BACK_DATA and chat_id and message_id is not None:
            lang = self._prefs.get(chat_id)
            self._client.edit_message_text(
                chat_id=chat_id, message_id=int(message_id), text=t(lang, "welcome")
            )
            if callback_id:
                self._client.answer_callback_query(callback_query_id=callback_id)
            return

        if not data.startswith(_CALLBACK_PREFIX) or not chat_id or message_id is None:
            # Always answer, else the client shows a spinner forever.
            if callback_id:
                self._client.answer_callback_query(callback_query_id=callback_id)
            return

        payload = data[len(_CALLBACK_PREFIX):].split(":")
        lang = normalize(payload[0])
        action = payload[1] if len(payload) > 1 else "switch"
        # First onboarding? Decide before persisting the (first) language choice.
        first_time = not self._prefs.has(chat_id)
        self._prefs.set(chat_id, lang)
        # From /start show the greeting; from /language a short confirmation.
        # The risk disclaimer is appended to the greeting on the first /start only.
        if action == "greet":
            text = t(lang, "welcome")
            if first_time:
                text = f"{text}\n\n{t(lang, 'disclaimer')}"
        else:
            text = t(lang, "switched")
        self._client.edit_message_text(
            chat_id=chat_id, message_id=int(message_id), text=text
        )
        self._client.answer_callback_query(
            callback_query_id=callback_id, text=t(lang, "toast")
        )
        self._sync_commands(chat_id)  # menu descriptions follow the chosen language
        logger.info("Telegram chat {} selected language {} ({})", chat_id, lang, action)

    # -- polling loop ------------------------------------------------------

    def poll_once(self, offset: int | None, *, timeout: int = 30) -> int | None:
        """Fetch one batch of updates, handle each, return the next offset.

        A transient network error on ``getUpdates`` (timeout, dropped
        connection) is logged and swallowed — the loop keeps the same offset and
        retries after a short backoff, so a network hiccup never kills the bot.
        """
        try:
            updates = self._client.get_updates(offset=offset, timeout=timeout)
        except Exception as exc:  # network hiccup: back off and retry, don't crash
            logger.warning("getUpdates failed ({}); retrying after {}s", exc, _ERROR_BACKOFF_S)
            time.sleep(_ERROR_BACKOFF_S)
            return offset
        for update in updates:
            try:
                self.handle_update(update)
            except Exception as exc:  # best-effort: one bad update can't stop the loop
                logger.error("Failed to handle Telegram update {}: {}",
                             update.get("update_id"), exc)
            offset = int(update["update_id"]) + 1
        return offset

    def run_forever(self, *, timeout: int = 30) -> None:  # pragma: no cover - blocking loop
        """Long-poll indefinitely, handling updates as they arrive."""
        logger.info("Telegram listener started (long-polling)")
        self._sync_default_commands()  # base '/' menu for chats without an override
        offset: int | None = None
        while True:
            offset = self.poll_once(offset, timeout=timeout)


@dataclass
class RequestsBotClient:
    """Default receiving transport over the Bot API (``requests``, lazy-imported)."""

    bot_token: str
    api_base: str = _DEFAULT_API_BASE
    timeout: float = _DEFAULT_TIMEOUT

    def _post(self, method: str, payload: dict[str, Any]) -> Any:
        import requests  # lazy: keeps the notify layer offline-importable

        url = f"{self.api_base}/bot{self.bot_token}/{method}"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        return resp.json().get("result")

    def get_updates(self, *, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        return self._post("getUpdates", payload) or []

    def send_message(
        self, *, chat_id: str, text: str, reply_markup: dict[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        self._post("sendMessage", payload)

    def edit_message_text(
        self,
        *,
        chat_id: str,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        self._post("editMessageText", payload)

    def answer_callback_query(self, *, callback_query_id: str, text: str = "") -> None:
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        self._post("answerCallbackQuery", payload)

    def set_my_commands(
        self, *, commands: list[dict[str, Any]], scope: dict[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {"commands": commands}
        if scope is not None:
            payload["scope"] = scope
        self._post("setMyCommands", payload)
