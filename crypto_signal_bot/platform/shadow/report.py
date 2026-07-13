"""Shadow accounting data types: parameters and the per-day report."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ShadowParams:
    """Cost/financing assumptions for virtual accounting (no strategy params)."""

    fee_rate: float = 0.00055  # taker fee per unit of turnover
    slippage_rate: float = 0.0002  # slippage per unit of turnover
    cost_of_capital: float = 0.0  # daily financing rate charged on gross exposure
    portfolio_value: float = 1.0  # NAV base to express PnL in currency


@dataclass(frozen=True)
class ShadowReport:
    """One trading day of virtual book economics."""

    asof: pd.Timestamp
    daily_pnl: float  # return-on-NAV: price + funding - fees - slippage - coc
    cum_pnl: float
    price_pnl: float
    funding_pnl: float
    fees: float
    slippage: float
    cost_of_capital: float
    turnover: float
    gross: float
    net: float
    long_exposure: float
    short_exposure: float
    n_long: int
    n_short: int
    portfolio_value: float

    @property
    def daily_pnl_value(self) -> float:
        """Daily PnL expressed in portfolio currency."""
        return self.daily_pnl * self.portfolio_value

    def as_row(self) -> dict:
        """Flat dict for persistence / DataFrame assembly."""
        return asdict(self)
