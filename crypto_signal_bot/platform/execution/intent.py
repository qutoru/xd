"""TradeIntent — a strategy's intention to open a futures position.

Strategy-facing entity (direction, entry, SL, TP, confidence). Kept separate from
the broker-facing OrderRequest: the ExecutionEngine translates a TradeIntent into
concrete OrderRequest(s), so the Broker never sees strategy fields. Pure data
model: rendering it for the user lives in the notify layer (see
``platform.notify.telegram.TelegramFormatter``), never on the entity itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crypto_signal_bot.platform.execution.domain import Side


@dataclass(frozen=True)
class TradeIntent:
    """A directional futures trade intention with auto-computed risk levels."""

    symbol: str
    side: Side
    target_notional: float  # position size as notional (> 0); used by ExecutionEngine
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    # NOT a probability of trade success. Currently a normalized signal-strength
    # proxy in [0, 1] (see intent_builder); a placeholder until confidence is
    # derived from model quality (win rate / class probability / ensemble).
    confidence: float | None = None
    strategy: str = ""  # provenance for the order client_id; never shown to the user
    timestamp: pd.Timestamp | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe serialization for persistence."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry": self.entry,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "target_notional": self.target_notional,
            "strategy": self.strategy,
        }
