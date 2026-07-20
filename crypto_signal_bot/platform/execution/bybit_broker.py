"""BybitBroker — a real :class:`Broker` for Bybit USDT perpetual futures.

The only exchange-aware node in the execution layer. It receives nothing but an
:class:`OrderRequest` (venue-agnostic), translates it into Bybit v5 order params
(notional -> contracts using instrument precision), and either places it (LIVE)
or simulates the fill locally (PAPER). Reduce-only LIMIT/STOP requests become the
take-profit / stop-loss brackets. It knows nothing about signals, alphas,
portfolios, TradeIntents or Telegram.

The ``pybit`` SDK is imported lazily and the HTTP session is injectable, so the
module stays offline-importable and unit tests run against a mock with no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd
from loguru import logger

from crypto_signal_bot.platform.execution.broker import Broker
from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
from crypto_signal_bot.platform.execution.domain import (
    Fill,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    PortfolioState,
    Side,
)

_SIDE_STR = {Side.BUY: "Buy", Side.SELL: "Sell"}


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _at_or_after(ts: pd.Timestamp | None, since: pd.Timestamp | None) -> bool:
    """True when ``ts`` is at/after the inclusive ``since`` bound (None => keep)."""
    if since is None:
        return True
    if ts is None:
        return False
    return ts >= since


@dataclass(frozen=True)
class InstrumentInfo:
    """Trading rules for one symbol (subset of Bybit instruments-info)."""

    symbol: str
    tick_size: float
    qty_step: float
    min_qty: float
    min_notional: float


@dataclass
class _PaperPosition:
    base_qty: float = 0.0  # signed (long > 0, short < 0)
    avg_price: float = 0.0


class BybitBroker(Broker):
    """Bybit USDT-perp adapter implementing the Broker contract (PAPER/LIVE)."""

    def __init__(self, config: BybitConfig, *, session=None) -> None:
        config.validate()
        if not config.needs_exchange:
            raise ValueError(
                "BybitBroker is for PAPER/LIVE; SHADOW mode uses no broker "
                "(see build_broker)"
            )
        self.config = config
        self._session = session
        self._instruments: dict[str, InstrumentInfo] = {}
        self._paper_book: dict[str, _PaperPosition] = {}
        # PAPER has no venue execution feed, so it records its own simulated fills
        # (with realized PnL) here to back get_fills. LIVE reads the real feed.
        self._paper_fills: list[Fill] = []
        self._paper_exec_seq = 0

    # --- connection / market data ------------------------------------------
    def _get_session(self):
        if self._session is None:  # pragma: no cover - needs network/creds
            from pybit.unified_trading import HTTP  # lazy: no top-level dep

            self._session = HTTP(
                testnet=self.config.testnet,
                api_key=self.config.api_key or None,
                api_secret=self.config.api_secret or None,
                recv_window=self.config.recv_window,
            )
        return self._session

    # Bybit retCodes that mean "your request was a no-op", not a failure:
    #   110043 = leverage not modified (already at target)
    #   110025 = position mode not modified
    _BENIGN_RET = {110043, 110025}

    def set_leverage(self, symbol: str, leverage: float | None = None) -> bool:
        """Set per-symbol buy/sell leverage before entries (idempotent).

        Returns True if leverage is now at the target (including the benign
        "not modified" case), False on a real error. Never raises — a failure is
        logged so the caller can decide whether to proceed. Only meaningful for
        LIVE/PAPER (an exchange session); the ``leverage`` argument overrides the
        configured default.
        """
        lev = self.config.leverage if leverage is None else leverage
        try:
            resp = self._get_session().set_leverage(
                category=self.config.category, symbol=symbol,
                buyLeverage=str(lev), sellLeverage=str(lev),
            )
            ret = int(resp.get("retCode", 0))
            if ret == 0 or ret in self._BENIGN_RET:
                return True
            logger.error("set_leverage {} -> retCode={} {}", symbol, ret, resp.get("retMsg"))
            return False
        except Exception as exc:  # network/parse — surfaced, not swallowed silently
            logger.error("set_leverage {} failed: {}", symbol, exc)
            return False

    def check_connection(self) -> bool:
        """Ping the venue; True if reachable, False on any error."""
        try:
            resp = self._get_session().get_server_time()
            return int(resp.get("retCode", 0)) == 0
        except Exception as exc:  # network/credential/parse failure
            logger.error("Bybit connection check failed: {}", exc)
            return False

    def get_instrument_info(self, symbol: str) -> InstrumentInfo:
        """Fetch (and cache) tick size / qty step / min qty / min notional."""
        cached = self._instruments.get(symbol)
        if cached is not None:
            return cached
        resp = self._get_session().get_instruments_info(
            category=self.config.category, symbol=symbol
        )
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            raise ValueError(f"no instrument info for {symbol}")
        row = rows[0]
        price_f = row.get("priceFilter", {})
        lot_f = row.get("lotSizeFilter", {})
        info = InstrumentInfo(
            symbol=symbol,
            tick_size=_to_float(price_f.get("tickSize"), 0.0),
            qty_step=_to_float(lot_f.get("qtyStep"), 0.0),
            min_qty=_to_float(lot_f.get("minOrderQty"), 0.0),
            min_notional=_to_float(lot_f.get("minNotionalValue"), 0.0),
        )
        self._instruments[symbol] = info
        return info

    def _last_price(self, symbol: str) -> float:
        resp = self._get_session().get_tickers(
            category=self.config.category, symbol=symbol
        )
        rows = resp.get("result", {}).get("list", [])
        return _to_float(rows[0].get("lastPrice")) if rows else 0.0

    # --- rounding helpers ---------------------------------------------------
    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        d_step = Decimal(str(step))
        return float((Decimal(str(value)) // d_step) * d_step)

    @staticmethod
    def _round_to_tick(value: float, tick: float) -> float:
        if tick <= 0:
            return value
        d_tick = Decimal(str(tick))
        return float((Decimal(str(value)) / d_tick).to_integral_value() * d_tick)

    # --- order translation + routing ---------------------------------------
    def _reduce_qty(self, symbol: str, notional: float, ref_price: float, info: InstrumentInfo) -> float:
        """Size a reduce-only leg to close the current position (fallback: notional)."""
        held = abs(self._position_base_qty(symbol))
        if held > 0:
            return self._floor_to_step(held, info.qty_step)
        if ref_price <= 0:
            return 0.0
        return self._floor_to_step(notional / ref_price, info.qty_step)

    def submit(self, request: OrderRequest) -> OrderResult:
        """Translate one OrderRequest into a Bybit order and place/simulate it."""
        try:
            info = self.get_instrument_info(request.symbol)
            if request.reduce_only:
                ref_price = request.price or self._last_price(request.symbol)
                base_qty = self._reduce_qty(request.symbol, request.quantity, ref_price, info)
            else:
                ref_price = request.price or self._last_price(request.symbol)
                if ref_price <= 0:
                    return self._reject(request, "no reference price")
                base_qty = self._floor_to_step(request.quantity / ref_price, info.qty_step)

            if base_qty < info.min_qty or base_qty <= 0:
                return self._reject(request, f"qty {base_qty} below min {info.min_qty}")
            if info.min_notional and base_qty * ref_price < info.min_notional:
                return self._reject(request, f"notional below min {info.min_notional}")

            params = self._build_params(request, info, base_qty)
            if self.config.is_live:
                return self._place_live(request, params, base_qty, ref_price)
            return self._fill_paper(request, base_qty, ref_price)
        except Exception as exc:  # any translation/API failure -> a clean reject
            logger.error("Bybit submit failed for {}: {}", request.symbol, exc)
            return self._reject(request, str(exc))

    def _build_params(self, request: OrderRequest, info: InstrumentInfo, base_qty: float) -> dict:
        params = {
            "category": self.config.category,
            "symbol": request.symbol,
            "side": _SIDE_STR[request.side],
            "qty": str(base_qty),
            "reduceOnly": request.reduce_only,
            "orderLinkId": request.client_id,
            # 0 = one-way mode (the only mode we support). On a hedge-mode account
            # this order will be rejected with a clear retCode rather than silently
            # opening the wrong leg — that mismatch must surface, not be masked.
            "positionIdx": self.config.position_idx,
        }
        if request.order_type is OrderType.LIMIT:
            params["orderType"] = "Limit"
            params["price"] = str(self._round_to_tick(request.price, info.tick_size))
            params["timeInForce"] = "GTC"
        elif request.order_type is OrderType.STOP:
            # stop-market: triggers a Market close when price crosses the level
            params["orderType"] = "Market"
            params["triggerPrice"] = str(self._round_to_tick(request.price, info.tick_size))
            params["triggerDirection"] = 2 if request.side is Side.SELL else 1
            params["triggerBy"] = "LastPrice"
        else:
            params["orderType"] = "Market"
        return params

    def _place_live(self, request, params, base_qty, ref_price) -> OrderResult:
        resp = self._get_session().place_order(**params)
        if int(resp.get("retCode", -1)) != 0:
            return self._reject(request, resp.get("retMsg", "rejected"))
        order_id = resp.get("result", {}).get("orderId", "")
        # A market entry is assumed filled; resting reduce-only brackets are pending
        # (reconciliation via get_portfolio_state is the later source of truth).
        if request.order_type is OrderType.MARKET and not request.reduce_only:
            fill = Fill(symbol=request.symbol, side=request.side, quantity=base_qty, price=ref_price)
            return OrderResult(request=request, status=OrderStatus.FILLED,
                               filled_quantity=base_qty, avg_price=ref_price,
                               fills=[fill], message=order_id)
        return OrderResult(request=request, status=OrderStatus.PENDING, message=order_id)

    def _fill_paper(self, request, base_qty, ref_price) -> OrderResult:
        # Resting brackets don't fill in paper; only the market entry books.
        if request.order_type is not OrderType.MARKET:
            return OrderResult(request=request, status=OrderStatus.PENDING,
                               message="paper: bracket resting")
        signed = base_qty if request.side is Side.BUY else -base_qty
        realized, closed_qty = self._apply_paper_fill(request.symbol, signed, ref_price)
        self._paper_exec_seq += 1
        fill = Fill(
            symbol=request.symbol, side=request.side, quantity=base_qty, price=ref_price,
            timestamp=pd.Timestamp.now(tz="UTC"), order_link_id=request.client_id,
            exec_id=f"paper-{self._paper_exec_seq}", realized_pnl=realized,
            closed_quantity=closed_qty,
        )
        self._paper_fills.append(fill)
        return OrderResult(request=request, status=OrderStatus.FILLED,
                           filled_quantity=base_qty, avg_price=ref_price,
                           fills=[fill], message="paper fill")

    def _apply_paper_fill(self, symbol: str, signed_qty: float, price: float) -> tuple[float, float]:
        """Apply a paper fill; return (realized_pnl, closed_base_qty).

        A fill that opposes the held side realizes PnL on the closed portion
        ``min(|fill|, |held|)`` at ``avg_price``; the remainder (on a flip) opens
        the new side at ``price``. A same-side fill only extends the position.
        """
        pos = self._paper_book.setdefault(symbol, _PaperPosition())
        realized = 0.0
        closed_qty = 0.0
        if pos.base_qty == 0 or (pos.base_qty > 0) == (signed_qty > 0):
            # opening or increasing the same side: volume-weight the average price
            total = abs(pos.base_qty) + abs(signed_qty)
            pos.avg_price = price if total == 0 else (
                abs(pos.base_qty) * pos.avg_price + abs(signed_qty) * price
            ) / total
        else:
            # reducing/closing (possibly flipping): the overlap realizes PnL
            closed_qty = min(abs(signed_qty), abs(pos.base_qty))
            direction = 1.0 if pos.base_qty > 0 else -1.0
            realized = closed_qty * (price - pos.avg_price) * direction
            if abs(signed_qty) > abs(pos.base_qty):  # flip: open remainder at price
                pos.avg_price = price
        pos.base_qty += signed_qty
        if abs(pos.base_qty) < 1e-12:
            self._paper_book.pop(symbol, None)
        return realized, closed_qty

    @staticmethod
    def _reject(request: OrderRequest, message: str) -> OrderResult:
        return OrderResult(request=request, status=OrderStatus.REJECTED, message=message)

    def cancel_open_orders(self, symbol: str) -> None:
        """Cancel all resting orders for ``symbol`` (clears prior TP/SL brackets).

        PAPER holds no resting orders (brackets never rest there), so it is a no-op.
        On a real venue a failed cancel is logged and swallowed — it must not abort
        the trading cycle; reduce-only bounds any surviving stale bracket to closing.
        """
        if self.config.mode is TradingMode.PAPER:
            return
        try:
            self._get_session().cancel_all_orders(
                category=self.config.category, symbol=symbol
            )
        except Exception as exc:  # best-effort hygiene, never abort the cycle
            logger.warning("cancel_all_orders failed for {}: {}", symbol, exc)

    def get_fills(self, since: pd.Timestamp | None = None) -> list[Fill]:
        """Executions at/after ``since`` — PAPER's simulated fills or the LIVE feed.

        LIVE pages Bybit's execution list (``get_executions``), keeping only real
        trades (``execType == 'Trade'`` — funding/settlement rows are excluded) and
        mapping each to a :class:`Fill` with its ``orderLinkId`` (attribution),
        ``execId`` (ledger de-dup), fee, ``closedSize`` and the venue's realized
        PnL. A signalled API error fails loud rather than reporting no fills.
        """
        if self.config.mode is TradingMode.PAPER:
            return [f for f in self._paper_fills if _at_or_after(f.timestamp, since)]

        resp = self._get_session().get_executions(category=self.config.category)
        if int(resp.get("retCode", 0)) != 0:
            raise RuntimeError(
                f"get_executions failed (retCode={resp.get('retCode')} "
                f"retMsg={resp.get('retMsg')!r}); refusing to report no fills"
            )
        # Bybit's execution feed carries closedSize but NOT realized PnL, so a close
        # booked from executions alone would record zero PnL — every close would then
        # look like a loss (net = 0 - fee < 0). Realized PnL lives in the closed-PnL
        # feed; index it by orderId and attach the gross round-trip to each close.
        closed_idx = self._closed_pnl_index()
        fills: list[Fill] = []
        for row in resp.get("result", {}).get("list", []):
            if row.get("execType") != "Trade":
                continue
            ts = pd.Timestamp(int(_to_float(row.get("execTime"))), unit="ms", tz="UTC")
            if not _at_or_after(ts, since):
                continue
            closed_qty = _to_float(row.get("closedSize"))
            realized = 0.0
            if closed_qty > 0:
                gross_total, size_total = closed_idx.get(row.get("orderId"), (None, None))
                if gross_total is None:
                    # The closed-PnL record for this order has not landed yet (venue
                    # eventual consistency). Skip this close now rather than book a
                    # wrong zero PnL: it stays after the ledger watermark, so a later
                    # poll re-reads and books it once the realized PnL is available.
                    continue
                realized = gross_total * (closed_qty / size_total) if size_total else 0.0
            fills.append(Fill(
                symbol=row.get("symbol", ""),
                side=Side.BUY if row.get("side") == "Buy" else Side.SELL,
                quantity=_to_float(row.get("execQty")),
                price=_to_float(row.get("execPrice")),
                fee=_to_float(row.get("execFee")),
                timestamp=ts,
                order_link_id=row.get("orderLinkId") or None,
                exec_id=row.get("execId") or None,
                realized_pnl=realized,
                closed_quantity=closed_qty,
            ))
        return fills

    def _closed_pnl_index(self) -> dict[str, tuple[float, float]]:
        """Map ``orderId -> (gross_realized, closed_size)`` from the closed-PnL feed.

        Bybit's execution list omits realized PnL, so it is read from
        ``/v5/position/closed-pnl``. The gross round-trip PnL is derived from the
        record's authoritative ``avgEntryPrice``/``avgExitPrice``/``closedSize`` — a
        closed long realizes ``size*(exit-entry)``, a closed short the negative
        (``side`` is the closing order's: Sell flattens a long, Buy a short). Bybit's
        own ``closedPnl`` field is deliberately NOT used: it is net of fees/funding,
        while the ledger keeps ``realized_pnl`` gross with fees tracked separately, so
        using it would double-count the ``execFee`` the fills already carry.

        Best-effort: a failed/empty read returns an empty index (closes are then held
        back, not mis-booked at zero — see ``get_fills``).
        """
        index: dict[str, tuple[float, float]] = {}
        try:
            resp = self._get_session().get_closed_pnl(
                category=self.config.category, settleCoin="USDT", limit=100,
            )
        except Exception as exc:  # never break fill polling over the PnL read
            logger.warning("get_closed_pnl failed: {}", exc)
            return index
        if int(resp.get("retCode", 0)) != 0:
            logger.warning(
                "get_closed_pnl retCode={} {}", resp.get("retCode"), resp.get("retMsg")
            )
            return index
        for row in resp.get("result", {}).get("list", []):
            order_id = row.get("orderId")
            size = _to_float(row.get("closedSize") or row.get("qty"))
            if not order_id or size <= 0:
                continue
            gross = size * (
                _to_float(row.get("avgExitPrice")) - _to_float(row.get("avgEntryPrice"))
            )
            if row.get("side") == "Buy":  # a Buy close flattens a short -> invert
                gross = -gross
            prev_g, prev_s = index.get(order_id, (0.0, 0.0))
            index[order_id] = (prev_g + gross, prev_s + size)
        return index

    # --- reconciliation -----------------------------------------------------
    def _position_base_qty(self, symbol: str) -> float:
        if self.config.mode is TradingMode.PAPER:
            pos = self._paper_book.get(symbol)
            return pos.base_qty if pos else 0.0
        for sym, (qty, _price) in self._live_positions().items():
            if sym == symbol:
                return qty
        return 0.0

    def _live_positions(self) -> dict[str, tuple[float, float]]:
        resp = self._get_session().get_positions(
            category=self.config.category, settleCoin="USDT"
        )
        # Fail loud on a signalled API error: a non-zero retCode must NOT be read as
        # an empty (flat) book. Otherwise get_portfolio_state would report no
        # positions and a delta-rebalance would treat every target as a fresh open —
        # re-accumulating the position and skipping the close of departed names.
        if int(resp.get("retCode", 0)) != 0:
            raise RuntimeError(
                f"get_positions failed (retCode={resp.get('retCode')} "
                f"retMsg={resp.get('retMsg')!r}); refusing to treat account as flat"
            )
        out: dict[str, tuple[float, float]] = {}
        for row in resp.get("result", {}).get("list", []):
            size = _to_float(row.get("size"))
            if size == 0:
                continue
            signed = size if row.get("side") == "Buy" else -size
            out[row.get("symbol", "")] = (signed, _to_float(row.get("avgPrice")))
        return out

    def get_portfolio_state(self) -> PortfolioState:
        """Current holdings + NAV, as signed notional (broker source of truth)."""
        if self.config.mode is TradingMode.PAPER:
            positions = {
                s: Position(symbol=s, quantity=p.base_qty * p.avg_price, avg_price=p.avg_price)
                for s, p in self._paper_book.items()
            }
            return PortfolioState(value=self.config.paper_equity, positions=positions)

        session = self._get_session()
        wallet = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows = wallet.get("result", {}).get("list", [])
        nav = _to_float(rows[0].get("totalEquity")) if rows else 0.0
        positions = {}
        for sym, (qty, price) in self._live_positions().items():
            positions[sym] = Position(symbol=sym, quantity=qty * price, avg_price=price)
        return PortfolioState(value=nav, positions=positions)


def build_broker(config: BybitConfig, *, session=None) -> Broker | None:
    """Select the broker for a trading mode: None for SHADOW, BybitBroker else.

    Single wiring seam for the three modes — the ExecutionEngine and pipeline are
    unchanged; only the injected broker differs. SHADOW returns None so the caller
    runs virtual accounting with execution disabled.
    """
    config.validate()
    if config.mode is TradingMode.SHADOW:
        return None
    return BybitBroker(config, session=session)
