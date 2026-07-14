"""TelegramNotifier: delivery, error handling, config gating, mock API (offline).

The Telegram Bot API is mocked with a fake ``TelegramClient`` (captures calls or
raises), so these tests exercise the notifier with no network and no real token.
"""

from __future__ import annotations

import pandas as pd
import pytest

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.notifier import (
    NotifyResult,
    RequestsTelegramClient,
    TelegramConfig,
    TelegramNotifier,
    TelegramSendError,
)
from crypto_signal_bot.platform.notify.telegram import TelegramFormatter

TS = pd.Timestamp("2023-07-01 12:00", tz="UTC")


def _intent(symbol="BTCUSDT", side=Side.BUY, **kw):
    base = dict(symbol=symbol, side=side, target_notional=50.0, entry=105400.0,
                take_profit=108100.0, stop_loss=104200.0, confidence=0.78,
                strategy="platform", timestamp=TS)
    base.update(kw)
    return TradeIntent(**base)


class _FakeClient:
    """Mock Telegram transport: records every delivered message."""

    def __init__(self):
        self.sent: list[dict] = []

    def send_message(self, *, chat_id: str, text: str) -> None:
        self.sent.append({"chat_id": chat_id, "text": text})


class _FlakyClient:
    """Mock transport that fails on specific symbols (by text substring)."""

    def __init__(self, fail_on: tuple[str, ...]):
        self.fail_on = fail_on
        self.attempts = 0

    def send_message(self, *, chat_id: str, text: str) -> None:
        self.attempts += 1
        if any(f in text for f in self.fail_on):
            raise TelegramSendError("boom")


def _live_config():
    return TelegramConfig(bot_token="T", chat_id="C", enabled=True, dry_run=False)


def test_sends_one_message_per_intent_in_formatter_layout():
    client = _FakeClient()
    notifier = TelegramNotifier(_live_config(), client=client)
    intents = [_intent(symbol="BTCUSDT"), _intent(symbol="ETHUSDT", side=Side.SELL)]

    result = notifier.notify_intents(intents)

    assert result == NotifyResult(sent=2, failed=0)
    assert result.ok
    assert [m["chat_id"] for m in client.sent] == ["C", "C"]
    # exact bytes match the formatter, i.e. notifier delegates rendering
    assert client.sent[0]["text"] == TelegramFormatter().format(intents[0])
    assert client.sent[1]["text"] == TelegramFormatter().format(intents[1])


def test_send_failure_is_recorded_not_raised_and_does_not_stop_the_batch():
    client = _FlakyClient(fail_on=("ETHUSDT",))
    notifier = TelegramNotifier(_live_config(), client=client)
    intents = [_intent(symbol="BTCUSDT"), _intent(symbol="ETHUSDT"), _intent(symbol="SOLUSDT")]

    result = notifier.notify_intents(intents)

    assert client.attempts == 3  # a mid-batch failure did not abort the rest
    assert result.sent == 2 and result.failed == 1 and not result.ok
    assert len(result.errors) == 1 and "ETHUSDT" in result.errors[0]


def test_missing_credentials_skips_without_touching_client():
    client = _FakeClient()
    notifier = TelegramNotifier(TelegramConfig(bot_token="", chat_id=""), client=client)
    result = notifier.notify_intents([_intent()])
    assert result == NotifyResult(skipped=1)
    assert client.sent == []


def test_dry_run_and_disabled_skip():
    client = _FakeClient()
    dry = TelegramNotifier(TelegramConfig(bot_token="T", chat_id="C", dry_run=True), client=client)
    off = TelegramNotifier(TelegramConfig(bot_token="T", chat_id="C", enabled=False), client=client)
    assert dry.notify_intents([_intent()]).skipped == 1
    assert off.notify_intents([_intent()]).skipped == 1
    assert client.sent == []


def test_empty_intents_is_a_noop():
    client = _FakeClient()
    result = TelegramNotifier(_live_config(), client=client).notify_intents([])
    assert result == NotifyResult()
    assert client.sent == []


def test_from_env_reads_credentials(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    notifier = TelegramNotifier.from_env()
    assert notifier.config.bot_token == "tok" and notifier.config.chat_id == "chat"
    assert notifier.config.is_active


def test_default_client_posts_to_bot_api_and_raises_on_non_200(monkeypatch):
    calls = {}

    class _Resp:
        status_code = 500
        text = "server error"

    def _fake_post(url, json, timeout):
        calls.update(url=url, json=json, timeout=timeout)
        return _Resp()

    import sys, types
    fake_requests = types.ModuleType("requests")
    fake_requests.post = _fake_post
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    client = RequestsTelegramClient("TOKEN", api_base="https://api.telegram.org")
    with pytest.raises(TelegramSendError):
        client.send_message(chat_id="C", text="hi")
    assert calls["url"] == "https://api.telegram.org/botTOKEN/sendMessage"
    assert calls["json"] == {"chat_id": "C", "text": "hi"}
