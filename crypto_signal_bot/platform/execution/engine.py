"""ExecutionEngine — the only bridge from a TargetBook to routed orders.

Diffs the target weights against current holdings, emits abstract OrderRequests
for the deltas, and routes them through a Broker. It consumes a ready TargetBook
(never builds one) and knows nothing about exchanges, alphas or signals.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from crypto_signal_bot.platform.execution.broker import Broker
from crypto_signal_bot.platform.execution.domain import (
    ExecutionReport,
    Order,
    OrderRequest,
    OrderType,
    Side,
)
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.portfolio.book import TargetBook


class ExecutionEngine:
    """Translate a TargetBook into OrderRequests and route them via a Broker."""

    def __init__(self, broker: Broker, *, min_notional: float = 1e-9) -> None:
        self.broker = broker
        self.min_notional = min_notional

    def plan(self, target_book: TargetBook, state) -> list[OrderRequest]:
        """Compute the OrderRequests needed to move ``state`` to ``target_book``."""
        current = state.weights()
        target = target_book.weights
        index = target.index.union(current.index)
        tgt = target.reindex(index).fillna(0.0)
        cur = current.reindex(index).fillna(0.0)
        delta_w = tgt - cur

        requests: list[OrderRequest] = []
        for symbol in sorted(index):
            notional = float(delta_w[symbol] * state.value)
            if abs(notional) <= self.min_notional:
                continue
            side = Side.BUY if notional > 0 else Side.SELL
            requests.append(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    quantity=abs(notional),
                    order_type=OrderType.MARKET,
                    target_weight=float(tgt[symbol]),
                    client_id=f"{pd.Timestamp(target_book.asof).date()}-{symbol}",
                )
            )
        return requests

    def execute(self, target_book: TargetBook, state=None) -> ExecutionReport:
        """Plan and route orders; return an ExecutionReport (no strategy logic)."""
        if state is None:
            state = self.broker.get_portfolio_state()
        requests = self.plan(target_book, state)

        orders: list[Order] = []
        for i, req in enumerate(requests):
            result = self.broker.submit(req)
            orders.append(
                Order(
                    id=req.client_id or f"ord-{i}",
                    request=req,
                    status=result.status,
                    result=result,
                )
            )
        return ExecutionReport(
            asof=target_book.asof,
            orders=orders,
            resulting_state=self.broker.get_portfolio_state(),
        )

    def build_orders(self, intents: Sequence[TradeIntent]) -> list[OrderRequest]:
        """Translate strategy TradeIntents into broker-facing OrderRequests.

        Strategy/reason/SL/TP stay on the TradeIntent; the OrderRequest carries
        only what a broker needs (symbol, side, notional). SL/TP become separate
        bracket orders in a later execution stage.
        """
        requests: list[OrderRequest] = []
        for intent in intents:
            if intent.target_notional <= self.min_notional:
                continue
            ts = "" if intent.timestamp is None else pd.Timestamp(intent.timestamp).date()
            requests.append(
                OrderRequest(
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=abs(intent.target_notional),
                    order_type=OrderType.MARKET,
                    client_id=f"{intent.strategy or 'intent'}-{intent.symbol}-{ts}",
                )
            )
        return requests

    def execute_intents(
        self, intents: Sequence[TradeIntent], *, asof: pd.Timestamp | None = None
    ) -> ExecutionReport:
        """Convert TradeIntents to OrderRequests and route them via the Broker."""
        requests = self.build_orders(intents)
        orders: list[Order] = []
        for i, req in enumerate(requests):
            result = self.broker.submit(req)
            orders.append(
                Order(id=req.client_id or f"ord-{i}", request=req,
                      status=result.status, result=result)
            )
        if asof is None:
            asof = next(
                (i.timestamp for i in intents if i.timestamp is not None),
                pd.Timestamp.now(tz="UTC"),
            )
        return ExecutionReport(
            asof=asof, orders=orders, resulting_state=self.broker.get_portfolio_state()
        )
