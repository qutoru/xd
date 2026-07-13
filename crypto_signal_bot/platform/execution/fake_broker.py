"""InMemoryBroker — a deterministic offline broker for tests.

Fills every order in full at no price (notional bookkeeping only), updating an
in-memory position book. Fully offline; knows nothing about signals or alphas.
Can be configured to reject specific symbols to exercise error paths.
"""

from __future__ import annotations

from crypto_signal_bot.platform.execution.broker import Broker
from crypto_signal_bot.platform.execution.domain import (
    Fill,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    PortfolioState,
    Side,
)


class InMemoryBroker(Broker):
    """A fake broker that books fills in memory (no network, no prices)."""

    def __init__(self, value: float = 1.0, reject: set[str] | None = None) -> None:
        self.value = value
        self._notional: dict[str, float] = {}
        self._reject = reject or set()

    def submit(self, request: OrderRequest) -> OrderResult:
        if request.symbol in self._reject:
            return OrderResult(request=request, status=OrderStatus.REJECTED,
                               message=f"symbol {request.symbol} rejected")
        signed = request.quantity if request.side == Side.BUY else -request.quantity
        self._notional[request.symbol] = self._notional.get(request.symbol, 0.0) + signed
        fill = Fill(symbol=request.symbol, side=request.side, quantity=request.quantity)
        return OrderResult(
            request=request,
            status=OrderStatus.FILLED,
            filled_quantity=request.quantity,
            fills=[fill],
            message="filled",
        )

    def get_portfolio_state(self) -> PortfolioState:
        positions = {
            s: Position(symbol=s, quantity=q)
            for s, q in self._notional.items()
            if q != 0.0
        }
        return PortfolioState(value=self.value, positions=positions)
