"""TelegramNotifier — delivers TradeIntents to a Telegram chat.

Sits at the end of the notify layer:

    TradeIntent -> TelegramFormatter -> TelegramNotifier -> Telegram Bot API

The notifier owns *delivery* (config, transport, error handling); the formatter
owns *rendering*. The transport is injected (a ``TelegramClient``) so tests mock
the Telegram API without monkeypatching ``requests`` and without any network.

Delivery is best-effort: a failed send is logged and recorded, never raised, so
one bad message can't abort a daily pipeline run. When disabled, dry-run, or
missing credentials, messages are logged instead of sent (skipped).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.telegram import TelegramFormatter

_DEFAULT_API_BASE = "https://api.telegram.org"
_DEFAULT_TIMEOUT = 15.0


class TelegramSendError(RuntimeError):
    """Raised by a TelegramClient when the Bot API rejects a message."""


@dataclass(frozen=True)
class TelegramConfig:
    """Telegram delivery settings (credentials + toggles)."""

    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    dry_run: bool = False
    api_base: str = _DEFAULT_API_BASE
    timeout: float = _DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls, *, enabled: bool = True, dry_run: bool = False) -> TelegramConfig:
        """Read credentials from ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``."""
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=enabled,
            dry_run=dry_run,
        )

    @property
    def is_active(self) -> bool:
        """True only if a real send should happen (enabled, live, credentialed)."""
        return (
            self.enabled
            and not self.dry_run
            and bool(self.bot_token)
            and bool(self.chat_id)
        )


@dataclass(frozen=True)
class NotifyResult:
    """Outcome of one ``notify_intents`` call."""

    sent: int = 0
    failed: int = 0
    skipped: int = 0  # messages not sent because delivery was inactive
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.failed == 0


class TelegramClient(Protocol):
    """Transport that delivers one rendered message; raises on failure."""

    def send_message(self, *, chat_id: str, text: str) -> None: ...


@dataclass
class RequestsTelegramClient:
    """Default transport over the Telegram Bot API (``requests``, lazy-imported)."""

    bot_token: str
    api_base: str = _DEFAULT_API_BASE
    timeout: float = _DEFAULT_TIMEOUT

    def send_message(self, *, chat_id: str, text: str) -> None:
        import requests  # lazy: keeps the notify layer offline-importable

        url = f"{self.api_base}/bot{self.bot_token}/sendMessage"
        resp = requests.post(
            url, json={"chat_id": chat_id, "text": text}, timeout=self.timeout
        )
        if resp.status_code != 200:
            raise TelegramSendError(f"HTTP {resp.status_code}: {resp.text}")


class TelegramNotifier:
    """Formats TradeIntents and delivers them, one message per intent."""

    def __init__(
        self,
        config: TelegramConfig | None = None,
        *,
        formatter: TelegramFormatter | None = None,
        client: TelegramClient | None = None,
    ) -> None:
        self.config = config or TelegramConfig()
        self.formatter = formatter or TelegramFormatter()
        self._client = client

    @classmethod
    def from_env(cls, *, enabled: bool = True, dry_run: bool = False) -> TelegramNotifier:
        """Build a notifier with credentials pulled from the environment."""
        return cls(TelegramConfig.from_env(enabled=enabled, dry_run=dry_run))

    def _get_client(self) -> TelegramClient:
        if self._client is None:
            self._client = RequestsTelegramClient(
                self.config.bot_token,
                api_base=self.config.api_base,
                timeout=self.config.timeout,
            )
        return self._client

    def notify_intents(self, intents: list[TradeIntent]) -> NotifyResult:
        """Send one Telegram message per intent; never raises on send failure."""
        if not intents:
            return NotifyResult()

        if not self.config.is_active:
            reason = "dry-run" if self.config.dry_run else (
                "disabled" if not self.config.enabled else "no TELEGRAM_* credentials"
            )
            for intent in intents:
                logger.info("[telegram:{}] not sent:\n{}", reason, self.formatter.format(intent))
            return NotifyResult(skipped=len(intents))

        client = self._get_client()
        sent = failed = 0
        errors: list[str] = []
        for intent in intents:
            text = self.formatter.format(intent)
            try:
                client.send_message(chat_id=self.config.chat_id, text=text)
                sent += 1
            except Exception as exc:  # best-effort: log and keep going
                failed += 1
                errors.append(f"{intent.symbol}: {exc}")
                logger.error("Telegram send failed for {}: {}", intent.symbol, exc)
        if sent:
            logger.success("Sent {} Telegram message(s) to chat {}", sent, self.config.chat_id)
        return NotifyResult(sent=sent, failed=failed, errors=tuple(errors))
