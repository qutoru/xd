"""The Broker abstraction.

A Broker only accepts an OrderRequest and returns an OrderResult, and can report
current holdings. It knows nothing about signals, alphas or portfolios. Concrete
brokers (fake for tests, exchange adapters for later) implement this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from crypto_signal_bot.platform.execution.domain import (
    Fill,
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

    def cancel_open_orders(self, symbol: str) -> None:
        """Cancel all resting orders for ``symbol`` (default: no-op).

        Brokers with no resting orders (fakes, offline) keep the default. A venue
        adapter overrides this to clear prior TP/SL brackets before new ones are
        placed, so stale close-only orders can't fire at outdated levels or pile up.
        """
        return None

    def get_fills(self, since: pd.Timestamp | None = None) -> list[Fill]:
        """Return executions at or after ``since`` (default: none).

        The accounting layer polls this each cycle to book realized PnL. Brokers
        that keep no execution history (offline fakes) keep the empty default; a
        venue adapter overrides it to page the exchange's execution feed. ``since``
        is an inclusive lower bound on the fill timestamp; ``None`` means all known
        fills. Each fill carries its ``exec_id`` so the ledger can de-duplicate
        across overlapping polls.
        """
        return []
