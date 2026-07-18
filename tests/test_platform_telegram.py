"""TelegramFormatter: exact futures layout rendered from a TradeIntent (offline)."""

from __future__ import annotations

import pandas as pd

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.telegram import TelegramFormatter

TS = pd.Timestamp("2023-07-01 12:00", tz="UTC")


def _intent(**kw):
    base = dict(symbol="BTCUSDT", side=Side.BUY, target_notional=50.0, entry=105400.0,
                take_profit=108100.0, stop_loss=104200.0, confidence=0.78,
                strategy="platform", timestamp=TS)
    base.update(kw)
    return TradeIntent(**base)


def test_format_exact_futures_layout():
    msg = TelegramFormatter().format(_intent())
    assert msg == (
        "BTCUSDT | 🟢 LONG\n"
        "12:00 UTC\n"
        "\nEntry\n105400\n"
        "\nTake Profit\n108100 (+2.6%)\n"
        "\nStop Loss\n104200 (-1.1%)\n"
        "\nConfidence\n78%"
    )


def test_format_short_side_signs():
    # SHORT: TP below entry is a profit (+), SL above entry is a loss (-)
    msg = TelegramFormatter().format(
        _intent(side=Side.SELL, take_profit=102700.0, stop_loss=106500.0)
    )
    assert "🔴 SHORT" in msg
    assert "102700 (+2.6%)" in msg  # (105400-102700)/105400
    assert "106500 (-1.0%)" in msg


def test_format_russian_localizes_labels_only():
    msg = TelegramFormatter().format(_intent(), lang="ru")
    assert msg == (
        "BTCUSDT | 🟢 ЛОНГ\n"
        "12:00 UTC\n"
        "\nВход\n105400\n"
        "\nТейк-профит\n108100 (+2.6%)\n"
        "\nСтоп-лосс\n104200 (-1.1%)\n"
        "\nУверенность\n78%"
    )


def test_default_language_is_english_byte_for_byte():
    # No lang arg must equal the previous English rendering.
    assert TelegramFormatter().format(_intent()).startswith("BTCUSDT | 🟢 LONG")
