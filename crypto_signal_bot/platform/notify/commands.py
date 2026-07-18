"""Per-tier Telegram command menu — the list shown when a user types '/'.

Telegram renders a bot's command menu from ``setMyCommands``; with a per-chat
scope the menu can differ per user. This module builds that list from a chat's
:class:`~crypto_signal_bot.platform.notify.plans.Entitlements` and language, so a
VIP sees ``/risk`` and ``/stats`` while a free chat sees only the base commands.

Only commands the bot actually handles are listed (e.g. no ``/signals``
placeholder), so the menu never advertises a command that does nothing.
"""

from __future__ import annotations

from typing import Any

from crypto_signal_bot.platform.notify.i18n import t
from crypto_signal_bot.platform.notify.plans import Entitlements


def tier_command_menu(
    ent: Entitlements | None, lang: str, *, is_admin: bool = False
) -> list[dict[str, Any]]:
    """Bot-API command list for a chat's tier + language (base + gated extras).

    ``is_admin`` appends the admin operations, so they surface in the '/' menu of
    the admin chat only (they stay hidden and inert for everyone else).
    """
    menu: list[dict[str, Any]] = [
        {"command": "start", "description": t(lang, "cmd_start")},
        {"command": "language", "description": t(lang, "cmd_language")},
        {"command": "subscribe", "description": t(lang, "cmd_subscribe")},
        {"command": "help", "description": t(lang, "cmd_help")},
    ]
    if ent is not None and ent.stats:  # PRO and up
        menu.append({"command": "stats", "description": t(lang, "cmd_stats")})
    if ent is not None and ent.custom_risk:  # VIP only
        menu.append({"command": "risk", "description": t(lang, "cmd_risk")})
    if is_admin:
        menu += [
            {"command": "grant", "description": t(lang, "cmd_grant")},
            {"command": "revoke", "description": t(lang, "cmd_revoke")},
            {"command": "subs", "description": t(lang, "cmd_subs")},
            {"command": "users", "description": t(lang, "cmd_users")},
        ]
    return menu
