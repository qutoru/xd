"""LanguagePrefsStore: per-chat language persistence (default fallback + round-trip)."""

from __future__ import annotations

from crypto_signal_bot.platform.notify.prefs import LanguagePrefsStore


def test_unknown_chat_returns_default(tmp_path):
    store = LanguagePrefsStore(tmp_path / "prefs.json")
    assert store.get("123") == "en"  # nothing saved yet -> default


def test_set_and_get_round_trip(tmp_path):
    store = LanguagePrefsStore(tmp_path / "prefs.json")
    store.set("123", "ru")
    assert store.get("123") == "ru"
    # int/str chat ids are interchangeable
    assert store.get(123) == "ru"


def test_persists_across_instances(tmp_path):
    path = tmp_path / "sub" / "prefs.json"  # parent dir auto-created
    LanguagePrefsStore(path).set("42", "ru")
    assert LanguagePrefsStore(path).get("42") == "ru"


def test_unsupported_language_is_normalized_to_default(tmp_path):
    store = LanguagePrefsStore(tmp_path / "prefs.json")
    store.set("7", "de")
    assert store.get("7") == "en"


def test_corrupt_file_reads_as_empty(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text("not json", encoding="utf-8")
    assert LanguagePrefsStore(path).get("1") == "en"
