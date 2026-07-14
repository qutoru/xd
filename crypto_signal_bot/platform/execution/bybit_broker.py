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
        self._apply_paper_fill(request.symbol, signed, ref_price)
        fill = Fill(symbol=request.symbol, side=request.side, quantity=base_qty, price=ref_price)
        return OrderResult(request=request, status=OrderStatus.FILLED,
                           filled_quantity=base_qty, avg_price=ref_price,
                           fills=[fill], message="paper fill")

    def _apply_paper_fill(self, symbol: str, signed_qty: float, price: float) -> None:
        pos = self._paper_book.setdefault(symbol, _PaperPosition())
        new_qty = pos.base_qty + signed_qty
        if pos.base_qty == 0 or (pos.base_qty > 0) == (signed_qty > 0):
            total = abs(pos.base_qty) + abs(signed_qty)
            pos.avg_price = price if total == 0 else (
                abs(pos.base_qty) * pos.avg_price + abs(signed_qty) * price
            ) / total
        pos.base_qty = new_qty
        if abs(new_qty) < 1e-12:
            self._paper_book.pop(symbol, None)

    @staticmethod
    def _reject(request: OrderRequest, message: str) -> OrderResult:
        return OrderResult(request=request, status=OrderStatus.REJECTED, message=message)

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
