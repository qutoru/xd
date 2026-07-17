"""Execution domain model — pure, broker-agnostic value types.

No knowledge of exchanges, signals, alphas or portfolio construction. Quantities
are expressed as notional (weight * NAV); prices are optional so the model works
in the abstraction-only Stage 5 without any market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"  # stop-market (used for reduce-only stop-loss brackets)


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class OrderRequest:
    """An abstract instruction to trade ``quantity`` (>0) notional of a symbol.

    TP/SL are not fields here: a bracket is expressed as *separate* reduce-only
    OrderRequests (a LIMIT take-profit and a STOP stop-loss), each with its own
    ``price`` and ``reduce_only=True``. This keeps the request broker-facing and
    venue-agnostic — the Broker places whatever single order it is handed.
    """

    symbol: str
    side: Side
    quantity: float  # absolute notional to trade
    order_type: OrderType = OrderType.MARKET
    price: float | None = None  # limit price (LIMIT) / trigger price (STOP)
    reduce_only: bool = False  # close-only leg (take-profit / stop-loss)
    target_weight: float | None = None  # metadata only
    client_id: str | None = None


@dataclass(frozen=True)
class Fill:
    """A (partial) execution of an order.

    ``quantity`` is in base units (contracts), the broker fill convention — not
    notional. The trailing fields are the accounting layer's inputs: they let a
    fill be attributed to the order that caused it and booked into the realized-PnL
    ledger. They default so pre-accounting call sites are unaffected.
    """

    symbol: str
    side: Side
    quantity: float
    price: float | None = None
    fee: float = 0.0
    timestamp: pd.Timestamp | None = None
    order_link_id: str | None = None  # client_id of the originating order (attribution)
    exec_id: str | None = None        # venue execution id (ledger de-dup key)
    realized_pnl: float = 0.0         # PnL booked by this fill (nonzero on closes only)
    closed_quantity: float = 0.0      # base qty this fill closed (>0 => a closing fill)


@dataclass(frozen=True)
class OrderResult:
    """A broker's response to one :class:`OrderRequest`."""

    request: OrderRequest
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_price: float | None = None
    fills: list[Fill] = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class Order:
    """A tracked order: the request plus its terminal status/result."""

    id: str
    request: OrderRequest
    status: OrderStatus
    result: OrderResult | None = None


@dataclass(frozen=True)
class Position:
    """A held position, as signed notional (long > 0, short < 0)."""

    symbol: str
    quantity: float
    avg_price: float | None = None


@dataclass(frozen=True)
class PortfolioState:
    """Current holdings and NAV. Broker-owned; alpha/portfolio-agnostic."""

    value: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    def notional(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity if pos is not None else 0.0

    def weights(self) -> pd.Series:
        """Per-symbol weights = position notional / NAV (empty if NAV == 0)."""
        if self.value == 0:
            return pd.Series(dtype="float64")
        return pd.Series(
            {s: p.quantity / self.value for s, p in self.positions.items()},
            dtype="float64",
        )


@dataclass(frozen=True)
class ExecutionReport:
    """Outcome of translating one TargetBook into routed orders."""

    asof: pd.Timestamp
    orders: list[Order]
    resulting_state: PortfolioState

    @property
    def n_filled(self) -> int:
        return sum(o.status == OrderStatus.FILLED for o in self.orders)

    @property
    def n_rejected(self) -> int:
        return sum(o.status == OrderStatus.REJECTED for o in self.orders)

    @property
    def traded_notional(self) -> float:
        return float(sum(o.request.quantity for o in self.orders))
