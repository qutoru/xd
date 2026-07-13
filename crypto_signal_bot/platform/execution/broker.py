"""The Broker abstraction.

A Broker only accepts an OrderRequest and returns an OrderResult, and can report
current holdings. It knows nothing about signals, alphas or portfolios. Concrete
brokers (fake for tests, exchange adapters for later) implement this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from crypto_signal_bot.platform.execution.domain import (
    OrderRequest,
    OrderResult,
    PortfolioState,
)


class Broker(ABC):
    """Order sink: submit an OrderRequest, get an OrderResult."""

    @abstractmethod
    def submit(self, request: OrderRequest) -> OrderResult:
        """Execute (or reject) one order and return its result."""
        ...

    @abstractmethod
    def get_portfolio_state(self) -> PortfolioState:
        """Return the broker's current holdings and NAV."""
        ...
