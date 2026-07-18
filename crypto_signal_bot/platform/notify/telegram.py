"""TelegramFormatter — renders a TradeIntent into a Telegram message.

Display lives here, not on TradeIntent: the intent is a data model, and turning
it into a human-facing string (side labels, favourable %% moves, layout) is a
presentation concern. This keeps the execution package free of formatting code
and lets the message layout change without touching the domain entity.
"""

from __future__ import annotations

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.notify.i18n import DEFAULT_LANGUAGE, t


def _pct(intent: TradeIntent, level: float) -> float:
    """Signed favourable %% move of ``level`` vs entry, per position side."""
    sign = 1.0 if intent.side == Side.BUY else -1.0
    return sign * (level / intent.entry - 1.0)


# Coarse tiers for the signal-strength field. ``confidence`` is a relative,
# max-normalized conviction proxy (NOT a probability), so it is shown as a stars
# tier rather than a false-precision percentage.
_STRENGTH_STARS = {"low": "★☆☆", "medium": "★★☆", "high": "★★★"}

# Recommended risk per trade (% of deposit) for the VIP/owner personal /risk level.
_RISK_PCT = {"low": 1, "medium": 2, "high": 3}


def _strength_tier(confidence: float) -> str:
    """Bucket a [0, 1] conviction proxy into low / medium / high."""
    if confidence >= 0.66:
        return "high"
    if confidence >= 0.33:
        return "medium"
    return "low"


class TelegramFormatter:
    """Renders TradeIntents into Telegram messages (exact futures layout).

    Only the labels are localized (via the i18n catalog); numbers, symbols and
    percentages are language-independent. ``lang`` defaults to English, so the
    original rendering is preserved byte-for-byte.
    """

    def format(
        self,
        intent: TradeIntent,
        lang: str = DEFAULT_LANGUAGE,
        *,
        risk_level: str | None = None,
    ) -> str:
        """Telegram message for one intent.

        ``risk_level`` (``low``/``medium``/``high``), when supplied, appends the
        VIP/owner-only recommended risk-per-trade line. Callers pass it only for
        VIP subscribers and the owner, so the field stays gated to those tiers.
        """
        side_key = "side_long" if intent.side == Side.BUY else "side_short"
        parts = [f"{intent.symbol} | {t(lang, side_key)}"]
        if intent.timestamp is not None:
            parts.append(intent.timestamp.strftime("%H:%M UTC"))
        if intent.entry is not None:
            parts += ["", t(lang, "entry"), f"{intent.entry:g}"]
        if intent.take_profit is not None and intent.entry:
            parts += ["", t(lang, "take_profit"),
                      f"{intent.take_profit:g} ({_pct(intent, intent.take_profit):+.1%})"]
        if intent.stop_loss is not None and intent.entry:
            parts += ["", t(lang, "stop_loss"),
                      f"{intent.stop_loss:g} ({_pct(intent, intent.stop_loss):+.1%})"]
        if intent.confidence is not None:
            tier = _strength_tier(intent.confidence)
            parts += ["", t(lang, "signal_strength"),
                      f"{_STRENGTH_STARS[tier]} {t(lang, f'strength_{tier}')}"]
        if risk_level in _RISK_PCT:
            parts += ["", t(lang, "risk_per_trade"),
                      f"{_RISK_PCT[risk_level]}% ({t(lang, f'risk_{risk_level}')})"]
        return "\n".join(parts)
