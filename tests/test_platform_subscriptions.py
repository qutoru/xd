"""SubscriptionStore: grant/extend/revoke/active with time-based expiry (offline)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crypto_signal_bot.platform.notify.subscriptions import SubscriptionStore

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def test_unknown_chat_has_no_subscription(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.json")
    assert store.get("1") is None
    assert store.is_active("1", now=NOW) is False


def test_grant_sets_expiry_and_activates(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.json")
    sub = store.grant("1", 30, plan="START", now=NOW)
    assert sub.plan == "START"
    assert sub.expires_at == NOW + timedelta(days=30)
    assert store.is_active("1", now=NOW) is True


def test_expired_subscription_is_inactive(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.json")
    store.grant("1", 1, now=NOW)
    later = NOW + timedelta(days=2)
    assert store.is_active("1", now=later) is False


def test_grant_extends_from_current_expiry_when_active(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.json")
    store.grant("1", 30, now=NOW)
    sub = store.grant("1", 30, now=NOW)  # still active -> stacks
    assert sub.expires_at == NOW + timedelta(days=60)


def test_grant_after_expiry_starts_from_now(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.json")
    store.grant("1", 1, now=NOW)
    later = NOW + timedelta(days=5)
    sub = store.grant("1", 10, now=later)  # lapsed -> from `later`
    assert sub.expires_at == later + timedelta(days=10)


def test_revoke(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.json")
    store.grant("1", 30, now=NOW)
    assert store.revoke("1") is True
    assert store.get("1") is None
    assert store.revoke("1") is False  # already gone


def test_active_lists_only_current_sorted_by_expiry(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.json")
    store.grant("late", 30, now=NOW)
    store.grant("soon", 10, now=NOW)
    store.grant("dead", 1, now=NOW - timedelta(days=5))  # expired before NOW
    active = store.active(now=NOW)
    assert [s.chat_id for s in active] == ["soon", "late"]


def test_persists_across_instances(tmp_path):
    path = tmp_path / "subs.json"
    SubscriptionStore(path).grant("1", 30, plan="PRO", now=NOW)
    reloaded = SubscriptionStore(path).get("1")
    assert reloaded is not None and reloaded.plan == "PRO"
