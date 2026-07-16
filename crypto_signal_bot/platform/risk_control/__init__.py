"""Production Risk Control — the final safety gate on *new* position entries.

A standalone layer that sits *after* RiskManager sizing and *before* execution:

    intents (risk-sized) + current positions + NAV -> ProductionRiskControl
        -> which new entries may open / whether trading is halted

It can only *refuse to open* new positions; it never closes or shrinks existing
ones. A halt (kill switch / daily-loss / emergency stop) therefore opens nothing
new while leaving open positions to be managed by their already-resting brackets.

Deliberately independent: it imports only pandas + stdlib and speaks only in
numbers (notionals + NAV) — no Execution, Broker, Bybit, Research, Portfolio,
Strategy or Telegram.
"""

from crypto_signal_bot.platform.risk_control.control import (
    BlockReason,
    ProductionRiskControl,
    RiskControlConfig,
    RiskControlDecision,
)
from crypto_signal_bot.platform.risk_control.state import RiskState, RiskStateStore

__all__ = [
    "BlockReason",
    "ProductionRiskControl",
    "RiskControlConfig",
    "RiskControlDecision",
    "RiskState",
    "RiskStateStore",
]
