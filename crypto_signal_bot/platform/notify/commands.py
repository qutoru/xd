"""Per-tier Telegram command menu — the list shown when a user types '/'.

Telegram renders a bot's command menu from ``setMyCommands``; with a per-chat
scope the menu can differ per user. This module builds that list from a chat's
:class:`~crypto_signal_bot.platform.notify.plans.Entitlements` and language, so a
VIP sees ``/risk`` and ``/stats`` while a free chat sees only the base commands.

Only commands the bot actually handles are listed (no ``/help``/``/signals``
placeholders), so the menu never advertises a command that does nothing.
"""

from __future__ import annotations

from typing import Any

from crypto_signal_bot.platform.notify.i18n import t
from crypto_signal_bot.platform.notify.plans import Entitlements


def tier_command_menu(ent: Entitlements | None, lang: str) -> list[dict[str, Any]]:
    """Bot-API command list for a chat's tier + language (base + gated extras)."""
    menu: list[dict[str, Any]] = [
        {"command": "start", "description": t(lang, "cmd_start")},
        {"command": "language", "description": t(lang, "cmd_language")},
        {"command": "subscribe", "description": t(lang, "cmd_subscribe")},
    ]
    if ent is not None and ent.stats:  # PRO and up
        menu.append({"command": "stats", "description": t(lang, "cmd_stats")})
    if ent is not None and ent.custom_risk:  # VIP only
        menu.append({"command": "risk", "description": t(lang, "cmd_risk")})
    return menu
