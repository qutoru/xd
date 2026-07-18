"""i18n catalog: language fallback and known-answer label/greeting strings."""

from __future__ import annotations

from crypto_signal_bot.platform.notify.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    normalize,
    t,
)


def test_default_language_is_english_and_first():
    assert DEFAULT_LANGUAGE == "en"
    assert LANGUAGES[0] == "en"


def test_normalize_falls_back_to_default_for_unknown():
    assert normalize("ru") == "ru"
    assert normalize("en") == "en"
    assert normalize(None) == "en"
    assert normalize("de") == "en"  # unsupported -> default


def test_labels_localized():
    assert t("en", "entry") == "Entry"
    assert t("ru", "entry") == "Вход"
    assert t("en", "side_long") == "🟢 LONG"
    assert t("ru", "side_long") == "🟢 ЛОНГ"
    assert t("ru", "side_short") == "🔴 ШОРТ"


def test_unknown_language_uses_english_strings():
    assert t("de", "take_profit") == t("en", "take_profit") == "Take Profit"


def test_greeting_present_in_both_languages():
    assert "Bybit Smart Signals" in t("en", "welcome")
    assert "Bybit Smart Signals" in t("ru", "welcome")
    # commands advertised in the greeting (/status was removed)
    for cmd in ("/subscribe", "/signals", "/help"):
        assert cmd in t("en", "welcome")
        assert cmd in t("ru", "welcome")
    assert "/status" not in t("en", "welcome")
    assert "/status" not in t("ru", "welcome")


def test_subscribe_plans_present_in_both_languages():
    for lang in ("en", "ru"):
        text = t(lang, "subscribe")
        assert "START" in text and "PRO" in text and "VIP" in text
        assert "$20" in text and "$50" in text and "$120" in text
        assert "@Daxakson" in text  # payment contact preserved verbatim
    # risk disclaimer kept as its own key (appended to the first greeting only)
    assert "informational purposes only" in t("en", "disclaimer")
    assert "информационный характер" in t("ru", "disclaimer")
    # ...and no longer baked into the plain greeting
    assert "informational purposes only" not in t("en", "welcome")
    assert "информационный характер" not in t("ru", "welcome")
    # the switch confirmation names the chosen language
    assert "switched to English" in t("en", "switched")
    assert "переключён на русский" in t("ru", "switched")
