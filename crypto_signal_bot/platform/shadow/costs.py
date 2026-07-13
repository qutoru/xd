"""Trading and financing costs. Each cost is one small linear function."""

from __future__ import annotations


def fees(turnover: float, fee_rate: float) -> float:
    """Commission charged on traded turnover |Δw|."""
    return turnover * fee_rate


def slippage(turnover: float, slippage_rate: float) -> float:
    """Execution slippage charged on traded turnover |Δw|."""
    return turnover * slippage_rate


def cost_of_capital(gross: float, daily_rate: float) -> float:
    """Daily financing charge on gross exposure (both legs financed)."""
    return gross * daily_rate
