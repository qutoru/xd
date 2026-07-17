"""ExecutionEngine — the only bridge from a TargetBook to routed orders.

Diffs the target weights against current holdings, emits abstract OrderRequests
for the deltas, and routes them through a Broker. It consumes a ready TargetBook
(never builds one) and knows nothing about exchanges, alphas or signals.
"""

from __future__ import annotations

import uuid
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

    # --- delta-based bracket rebalance (no position accumulation) -----------
    def rebalance_bracket_intents(
        self,
        intents: Sequence[TradeIntent],
        *,
        book_symbols: set[str] | None = None,
        asof: pd.Timestamp | None = None,
    ) -> ExecutionReport:
        """Rebalance the broker to the intents' target positions (no accumulation).

        Unlike :meth:`execute_bracket_intents` (which re-opens the *full* target
        every call and so accumulates the position day over day), this reads the
        broker's current positions and trades only the *delta* to each target:

        * ``delta`` ~ 0        -> no entry;
        * open / increase      -> a MARKET entry sized to the delta;
        * decrease (same side) -> a MARKET order netting the position down;
        * target 0 / departed  -> a reduce-only MARKET close of the whole position;
        * side flip            -> reduce-only close of the old side, then a fresh
          MARKET entry into the new side.

        Reduce-only TP/SL are sized to the *final target* position. ``book_symbols``
        is the full set the strategy still wants this cycle (before the risk-control
        gate): a held symbol absent from it has left the book and is closed, while a
        held symbol still in the book but absent from ``intents`` was refused by risk
        control and is left untouched (risk control never closes). Defaults to the
        intents' symbols.
        """
        asof_ts = self._intents_asof(intents, asof)
        date = pd.Timestamp(asof_ts).date()
        state = self.broker.get_portfolio_state()
        current = {s: float(p.quantity) for s, p in state.positions.items()}
        targets = {
            i.symbol: (i.target_notional if i.side is Side.BUY else -i.target_notional)
            for i in intents
        }
        intent_by_symbol = {i.symbol: i for i in intents}
        book = set(targets) if book_symbols is None else set(book_symbols)

        requests: list[OrderRequest] = []
        touched: list[str] = []
        for symbol in sorted(set(current) | set(targets)):
            cur = current.get(symbol, 0.0)
            if symbol in targets:
                sym_reqs = self._rebalance_orders(
                    symbol, cur, targets[symbol], intent_by_symbol[symbol], date
                )
            elif symbol not in book:
                # held but gone from the book -> close the whole position
                sym_reqs = self._close_orders(symbol, cur, date)
            else:
                # held, still wanted, refused by risk control -> leave untouched
                sym_reqs = []
            if sym_reqs:
                touched.append(symbol)
                requests.extend(sym_reqs)

        # Clear any prior resting brackets for the symbols we are about to (re)place,
        # so stale reduce-only TP/SL from earlier cycles cannot fire at outdated
        # levels or accumulate toward the venue's open-order limit. Symbols left
        # untouched (e.g. risk-control-blocked but still held) keep their brackets.
        for symbol in touched:
            self.broker.cancel_open_orders(symbol)
        return self._route(requests, asof_ts)

    def _rebalance_orders(
        self, symbol: str, cur: float, tgt: float, intent: TradeIntent, date
    ) -> list[OrderRequest]:
        """Orders to move ``symbol`` from ``cur`` to ``tgt`` signed notional."""
        tag = intent.strategy or "intent"
        orders: list[OrderRequest] = []

        if tgt == 0.0:
            if abs(cur) > self.min_notional:
                orders.append(self._close_request(symbol, cur, tag, date))
            return orders

        flip = cur != 0.0 and (cur > 0) != (tgt > 0)
        if flip:
            # close the old side fully, then open the new side fresh
            orders.append(self._close_request(symbol, cur, tag, date))
            orders.append(self._market_request(symbol, tgt, "entry", tag, date))
        else:
            delta = tgt - cur
            if abs(delta) > self.min_notional:
                # increase -> ``entry``; decrease -> ``reduce`` (a plain MARKET that
                # nets the position down: the broker's reduce-only path would close
                # the whole position, not just the delta).
                kind = "entry" if abs(tgt) >= abs(cur) else "reduce"
                orders.append(self._market_request(symbol, delta, kind, tag, date))

        orders.extend(self._bracket_tp_sl(symbol, tgt, intent, tag))
        return orders

    def _close_orders(self, symbol: str, cur: float, date) -> list[OrderRequest]:
        """Reduce-only MARKET close for a symbol that has left the book."""
        if abs(cur) <= self.min_notional:
            return []
        return [self._close_request(symbol, cur, "rebalance", date)]

    def _market_request(
        self, symbol: str, signed_notional: float, kind: str, tag: str, date
    ) -> OrderRequest:
        return OrderRequest(
            symbol=symbol,
            side=Side.BUY if signed_notional > 0 else Side.SELL,
            quantity=abs(signed_notional),
            order_type=OrderType.MARKET,
            client_id=f"{tag}-{symbol}-{date}-{kind}",
        )

    def _close_request(self, symbol: str, cur: float, tag: str, date) -> OrderRequest:
        """Reduce-only MARKET order flattening the current position (never flips)."""
        return OrderRequest(
            symbol=symbol,
            side=Side.SELL if cur > 0 else Side.BUY,
            quantity=abs(cur),
            order_type=OrderType.MARKET,
            reduce_only=True,
            client_id=f"{tag}-{symbol}-{date}-close",
        )

    def _bracket_tp_sl(
        self, symbol: str, tgt: float, intent: TradeIntent, tag: str
    ) -> list[OrderRequest]:
        """Reduce-only TP/SL sized to the final target position (opposite side).

        The client_id carries a per-placement unique token (not the date): this
        method runs only in the delta-rebalance path, which cancels the prior
        brackets *before* re-placing. A date-stamped (deterministic) id would then
        collide with the just-cancelled order's orderLinkId — a venue that retains
        cancelled ids (the same de-dup the entry path relies on for idempotency)
        would reject the re-placement, leaving the position with NO stop-loss. A
        fresh token makes the re-placement always accepted; the cancel (not the id)
        is what prevents duplicate resting brackets.
        """
        close_side = Side.SELL if tgt > 0 else Side.BUY
        notional = abs(tgt)
        token = uuid.uuid4().hex[:8]
        legs: list[OrderRequest] = []
        if intent.take_profit is not None:
            legs.append(OrderRequest(
                symbol=symbol, side=close_side, quantity=notional,
                order_type=OrderType.LIMIT, price=intent.take_profit,
                reduce_only=True, client_id=f"{tag}-{symbol}-{token}-tp",
            ))
        if intent.stop_loss is not None:
            legs.append(OrderRequest(
                symbol=symbol, side=close_side, quantity=notional,
                order_type=OrderType.STOP, price=intent.stop_loss,
                reduce_only=True, client_id=f"{tag}-{symbol}-{token}-sl",
            ))
        return legs
