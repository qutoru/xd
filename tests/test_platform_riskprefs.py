"""RiskPrefsStore + normalize_risk: per-chat personal risk level (offline)."""

from __future__ import annotations

from crypto_signal_bot.platform.notify.riskprefs import (
    DEFAULT_RISK,
    RiskPrefsStore,
    normalize_risk,
)


def test_normalize_accepts_aliases_case_insensitively():
    assert normalize_risk("LOW") == "low"
    assert normalize_risk(" Med ") == "medium"
    assert normalize_risk("high") == "high"
    assert normalize_risk("агрессивный") == "high"  # RU alias
    assert normalize_risk("низкий") == "low"


def test_normalize_rejects_unknown_and_empty():
    assert normalize_risk(None) is None
    assert normalize_risk("") is None
    assert normalize_risk("insane") is None


def test_get_defaults_until_set(tmp_path):
    store = RiskPrefsStore(tmp_path / "risk.json")
    assert store.get("42") == DEFAULT_RISK


def test_set_persists_normalized_level(tmp_path):
    path = tmp_path / "risk.json"
    RiskPrefsStore(path).set("42", "HIGH")
    assert RiskPrefsStore(path).get("42") == "high"  # survives reload, normalized


def test_set_ignores_invalid_level(tmp_path):
    store = RiskPrefsStore(tmp_path / "risk.json")
    store.set("42", "bogus")
    assert store.get("42") == DEFAULT_RISK  # unchanged
