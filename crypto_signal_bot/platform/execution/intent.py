"""TradeIntent — a strategy's intention to open a futures position.

This is the strategy-facing entity (direction, entry, SL, TP, size, reason). It
is deliberately separate from the broker-facing ``OrderRequest``: the
ExecutionEngine translates a TradeIntent into concrete OrderRequest(s), so the
Broker never sees strategy/reason. It is also the object a future Telegram
notifier will render for the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crypto_signal_bot.platform.execution.domain import Side


@dataclass(frozen=True)
class TradeIntent:
    """A directional futures trade intention with risk levels and a reason."""

    symbol: str
    side: Side
    target_notional: float  # position size as notional (> 0)
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    leverage: float = 1.0
    strategy: str = ""
    reason: str = ""
    timestamp: pd.Timestamp | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def risk_reward(self) -> float | None:
        """Reward-to-risk ratio from entry/SL/TP (None if any is missing)."""
        if self.entry is None or self.stop_loss is None or self.take_profit is None:
            return None
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return reward / risk if risk > 0 else None

    def format(self) -> str:
        """Human-readable message (basis for the future Telegram notification)."""
        lines = [self.symbol, self.side.value.upper()]
        if self.entry is not None:
            lines.append(f"Entry: {self.entry:g}")
        if self.take_profit is not None:
            lines.append(f"TP: {self.take_profit:g}")
        if self.stop_loss is not None:
            lines.append(f"SL: {self.stop_loss:g}")
        rr = self.risk_reward
        if rr is not None:
            lines.append(f"Risk/Reward: {rr:.2f}")
        lines.append(f"Size: {self.target_notional:g}")
        if self.leverage and self.leverage != 1.0:
            lines.append(f"Leverage: {self.leverage:g}x")
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)
