"""Send messages to Telegram via the Bot API.

Kept dependency-light (``requests`` only). If credentials are missing or
``dry_run`` is set, the message is logged instead of sent, so the pipeline can
be exercised without a bot token.
"""

from __future__ import annotations

import requests
from loguru import logger

from crypto_signal_bot.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT = 15


def send_telegram(text: str, *, dry_run: bool = False) -> bool:
    """Send ``text`` to the configured Telegram chat.

    Args:
        text: Message body.
        dry_run: If True, log the message instead of sending it.

    Returns:
        True if a message was actually delivered, False if it was only logged.
    """
    if dry_run or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        why = "dry-run" if dry_run else "no TELEGRAM_* credentials"
        logger.info("[{}] Telegram message not sent:\n{}", why, text)
        return False

    resp = requests.post(
        _API_URL.format(token=TELEGRAM_BOT_TOKEN),
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        logger.error("Telegram send failed [{}]: {}", resp.status_code, resp.text)
        resp.raise_for_status()

    logger.success("Sent Telegram message to chat {}", TELEGRAM_CHAT_ID)
    return True
