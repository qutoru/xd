"""tier_command_menu: base commands plus tier-gated extras, localized."""

from __future__ import annotations

from crypto_signal_bot.platform.notify.commands import tier_command_menu
from crypto_signal_bot.platform.notify.i18n import t
from crypto_signal_bot.platform.notify.plans import ENTITLEMENTS, Tier


def _names(menu):
    return [c["command"] for c in menu]


def test_no_subscription_lists_base_commands_only():
    assert _names(tier_command_menu(None, "en")) == ["start", "language", "subscribe"]


def test_start_tier_has_no_gated_commands():
    menu = tier_command_menu(ENTITLEMENTS[Tier.START], "en")
    assert _names(menu) == ["start", "language", "subscribe"]


def test_pro_adds_stats_only():
    menu = tier_command_menu(ENTITLEMENTS[Tier.PRO], "en")
    assert _names(menu) == ["start", "language", "subscribe", "stats"]


def test_vip_adds_stats_and_risk():
    menu = tier_command_menu(ENTITLEMENTS[Tier.VIP], "en")
    assert _names(menu) == ["start", "language", "subscribe", "stats", "risk"]


def test_descriptions_are_localized():
    en = tier_command_menu(None, "en")
    ru = tier_command_menu(None, "ru")
    assert en[2]["description"] == t("en", "cmd_subscribe")
    assert ru[2]["description"] == t("ru", "cmd_subscribe")
    assert en[2]["description"] != ru[2]["description"]
