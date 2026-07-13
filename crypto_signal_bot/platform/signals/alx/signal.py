"""ALX funding-rank signal — one plugin among many (NOT part of the core).

Returns cross-sectional ``TargetScores = -(L-day mean daily funding)``.
Persistently rich funding = crowded longs -> expected relative underperformance,
so cheap/negative funding scores high (go long). This module owns all its own
parameters, so removing the ``alx`` folder removes ALX entirely — no other file
references it.
"""

from __future__ import annotations

from typing import Any, Mapping

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.signals.base import SignalProvider, TargetScores
from crypto_signal_bot.platform.signals.registry import register_signal

DEFAULT_LOOKBACK = 7


@register_signal("alx")
class ALXSignalProvider(SignalProvider):
    """Funding-rank cross-sectional score provider."""

    def generate(
        self,
        snapshot: MarketSnapshot,
        config: Mapping[str, Any] | None = None,
        state: Any | None = None,
    ) -> TargetScores:
        cfg = config or {}
        lookback = int(cfg.get("lookback", DEFAULT_LOOKBACK))
        score = (-snapshot.funding.rolling(lookback).mean()).iloc[-1]
        return TargetScores(values=score.astype("float64"), asof=snapshot.asof)
