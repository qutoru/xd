"""BybitBroker — thin adapter placeholder. Execution intentionally DISABLED.

Stage 5 is abstractions only. This class exists to prove the Broker contract can
be adapted to a real venue, but it performs NO trades: ``submit`` refuses. Any
exchange SDK (pybit) is imported lazily inside methods so the module stays
offline-importable, and it is never invoked from tests.
"""

from __future__ import annotations

from crypto_signal_bot.platform.execution.broker import Broker
from crypto_signal_bot.platform.execution.domain import (
    OrderRequest,
    OrderResult,
    PortfolioState,
)

_DISABLED = (
    "BybitBroker execution is disabled in Stage 5 (abstractions only). "
    "Real order routing is not implemented."
)


class BybitBroker(Broker):
    """Adapter shell over a Bybit session; does not trade."""

    def __init__(self, *, testnet: bool = True) -> None:
        self.testnet = testnet
        self._session = None  # lazily created; never used to trade in Stage 5

    def _get_session(self):  # pragma: no cover - not exercised in tests
        if self._session is None:
            from pybit.unified_trading import HTTP  # lazy import; no top-level dep

            self._session = HTTP(testnet=self.testnet)
        return self._session

    def submit(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError(_DISABLED)

    def get_portfolio_state(self) -> PortfolioState:
        raise NotImplementedError(_DISABLED)
