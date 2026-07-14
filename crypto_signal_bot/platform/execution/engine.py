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

_OPPOSITE: dict[Side, Side] = {Side.BUY: Side.SELL, Side.SELL: Side.BUY}
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

    def _route(self, requests: list[OrderRequest], asof: pd.Timestamp) -> ExecutionReport:
        """Submit each OrderRequest through the Broker and collect the report."""
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
            asof=asof, orders=orders, resulting_state=self.broker.get_portfolio_state()
        )

    def execute(self, target_book: TargetBook, state=None) -> ExecutionReport:
        """Plan and route orders; return an ExecutionReport (no strategy logic)."""
        if state is None:
            state = self.broker.get_portfolio_state()
        requests = self.plan(target_book, state)
        return self._route(requests, target_book.asof)

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

    def _intents_asof(
        self, intents: Sequence[TradeIntent], asof: pd.Timestamp | None
    ) -> pd.Timestamp:
        if asof is not None:
            return asof
        return next(
            (i.timestamp for i in intents if i.timestamp is not None),
            pd.Timestamp.now(tz="UTC"),
        )

    def execute_intents(
        self, intents: Sequence[TradeIntent], *, asof: pd.Timestamp | None = None
    ) -> ExecutionReport:
        """Convert TradeIntents to OrderRequests and route them via the Broker."""
        requests = self.build_orders(intents)
        return self._route(requests, self._intents_asof(intents, asof))

    def build_bracket_orders(self, intent: TradeIntent) -> list[OrderRequest]:
        """Turn one TradeIntent into an entry + reduce-only TP/SL bracket.

        Emits a MARKET entry and, when the intent carries them, a reduce-only
        LIMIT take-profit and a reduce-only STOP stop-loss — each a plain
        OrderRequest on the *opposite* side. The bracket levels live only on
        these close-only requests, so the Broker still receives nothing but
        OrderRequests and stays venue-agnostic. TP/SL translation (contracts,
        trigger direction) is the Broker's job.
        """
        ts = "" if intent.timestamp is None else pd.Timestamp(intent.timestamp).date()
        tag = intent.strategy or "intent"
        notional = abs(intent.target_notional)
        close_side = _OPPOSITE[intent.side]

        orders = [
            OrderRequest(
                symbol=intent.symbol,
                side=intent.side,
                quantity=notional,
                order_type=OrderType.MARKET,
                client_id=f"{tag}-{intent.symbol}-{ts}-entry",
            )
        ]
        if intent.take_profit is not None:
            orders.append(
                OrderRequest(
                    symbol=intent.symbol,
                    side=close_side,
                    quantity=notional,
                    order_type=OrderType.LIMIT,
                    price=intent.take_profit,
                    reduce_only=True,
                    client_id=f"{tag}-{intent.symbol}-{ts}-tp",
                )
            )
        if intent.stop_loss is not None:
            orders.append(
                OrderRequest(
                    symbol=intent.symbol,
                    side=close_side,
                    quantity=notional,
                    order_type=OrderType.STOP,
                    price=intent.stop_loss,
                    reduce_only=True,
                    client_id=f"{tag}-{intent.symbol}-{ts}-sl",
                )
            )
        return orders

    def execute_bracket_intents(
        self, intents: Sequence[TradeIntent], *, asof: pd.Timestamp | None = None
    ) -> ExecutionReport:
        """Route entry + reduce-only TP/SL brackets for each intent via the Broker."""
        requests: list[OrderRequest] = []
        for intent in intents:
            if abs(intent.target_notional) <= self.min_notional:
                continue
            requests.extend(self.build_bracket_orders(intent))
        return self._route(requests, self._intents_asof(intents, asof))
