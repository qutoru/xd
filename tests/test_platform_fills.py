"""Stage 16.1 — Broker.get_fills + Fill accounting fields (offline).

Covers the realized-PnL bookkeeping the accounting layer depends on: PAPER
simulates its own fills (with realized PnL on closes) and LIVE maps the Bybit
execution feed. No network, no keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker
from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
from crypto_signal_bot.platform.execution.domain import OrderRequest, OrderType, Side
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker


class _PriceSession:
    """Fake Bybit session with a mutable last price (PAPER market data)."""

    def __init__(self, price: float = 100.0):
        self.price = price

    def get_server_time(self):
        return {"retCode": 0, "result": {"timeSecond": "1"}}

    def get_instruments_info(self, **kw):
        return {"result": {"list": [{
            "symbol": kw.get("symbol", ""),
            "priceFilter": {"tickSize": "0.01"},
            "lotSizeFilter": {"qtyStep": "0.0001", "minOrderQty": "0.0001",
                              "minNotionalValue": "0"},
        }]}}

    def get_tickers(self, **kw):
        return {"result": {"list": [{"lastPrice": str(self.price)}]}}


def _paper_broker(price=100.0):
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER,
                      paper_equity=10_000.0)
    return BybitBroker(cfg, session=_PriceSession(price))


def _mkt(symbol, side, notional, *, reduce_only=False, cid="platform-x-entry"):
    return OrderRequest(symbol=symbol, side=side, quantity=notional,
                        order_type=OrderType.MARKET, reduce_only=reduce_only,
                        client_id=cid)


# --- default -----------------------------------------------------------------
def test_broker_get_fills_defaults_to_empty():
    assert InMemoryBroker().get_fills() == []


# --- PAPER realized PnL ------------------------------------------------------
def test_paper_full_close_realizes_pnl_and_tags_fill():
    b = _paper_broker(100.0)
    b.submit(_mkt("BTCUSDT", Side.BUY, 1000.0, cid="platform-BTCUSDT-a-entry"))  # +10 @100
    b._session.price = 110.0
    b.submit(_mkt("BTCUSDT", Side.SELL, 1000.0, reduce_only=True,
                  cid="platform-BTCUSDT-a-close"))  # close 10 @110

    fills = b.get_fills()
    assert len(fills) == 2
    entry, close = fills
    assert entry.realized_pnl == 0.0 and entry.closed_quantity == 0.0
    assert np.isclose(close.realized_pnl, 100.0)      # 10 * (110 - 100)
    assert np.isclose(close.closed_quantity, 10.0)
    assert close.order_link_id == "platform-BTCUSDT-a-close"
    assert entry.exec_id != close.exec_id             # unique per fill
    assert b.get_portfolio_state().positions == {}    # flat


def test_paper_partial_reduce_realizes_proportional_pnl():
    b = _paper_broker(100.0)
    b.submit(_mkt("BTCUSDT", Side.BUY, 1000.0))       # +10 @100
    b._session.price = 120.0
    b.submit(_mkt("BTCUSDT", Side.SELL, 600.0))       # sell 5 @120 (non-reduce delta)

    close = b.get_fills()[-1]
    assert np.isclose(close.realized_pnl, 100.0)      # 5 * (120 - 100)
    assert np.isclose(close.closed_quantity, 5.0)
    # 5 contracts remain long, still marked at the original average price (100)
    assert np.isclose(b.get_portfolio_state().notional("BTCUSDT"), 5.0 * 100.0)


def test_paper_flip_realizes_only_closed_portion():
    b = _paper_broker(100.0)
    b.submit(_mkt("BTCUSDT", Side.BUY, 1000.0))       # +10 @100
    b._session.price = 120.0
    b.submit(_mkt("BTCUSDT", Side.SELL, 1440.0))      # sell 12 @120 -> flip to -2

    close = b.get_fills()[-1]
    assert np.isclose(close.realized_pnl, 200.0)      # only the 10 closed: 10*(120-100)
    assert np.isclose(close.closed_quantity, 10.0)
    assert np.isclose(b.get_portfolio_state().notional("BTCUSDT"), -2.0 * 120.0)


def test_paper_get_fills_since_is_inclusive_lower_bound():
    b = _paper_broker(100.0)
    b.submit(_mkt("BTCUSDT", Side.BUY, 1000.0))
    b._session.price = 110.0
    b.submit(_mkt("BTCUSDT", Side.SELL, 1000.0, reduce_only=True))

    all_fills = b.get_fills()
    cutoff = all_fills[-1].timestamp
    later = b.get_fills(since=cutoff)
    assert later and all(f.timestamp >= cutoff for f in later)
    assert len(b.get_fills(since=all_fills[0].timestamp)) == 2


# --- LIVE execution feed -----------------------------------------------------
class _ExecSession:
    def __init__(self, rows, ret=0, pnl_rows=None):
        self._rows = rows
        self._ret = ret
        self._pnl_rows = pnl_rows or []

    def get_server_time(self):
        return {"retCode": 0, "result": {"timeSecond": "1"}}

    def get_executions(self, **kw):
        return {"retCode": self._ret, "retMsg": "err" if self._ret else "OK",
                "result": {"list": self._rows}}

    def get_closed_pnl(self, **kw):
        # Execution feed carries no realized PnL; it lives here (see _closed_pnl_index).
        return {"retCode": 0, "result": {"list": self._pnl_rows}}


def _live_broker(session):
    cfg = BybitConfig(api_key="k", api_secret="s", testnet=True, mode=TradingMode.LIVE)
    return BybitBroker(cfg, session=session)


def test_live_get_fills_maps_trade_rows_and_skips_non_trades():
    rows = [
        {"execType": "Trade", "symbol": "BTCUSDT", "side": "Sell", "execQty": "10",
         "execPrice": "110", "execFee": "0.6", "execTime": "1700000000000",
         "orderLinkId": "platform-BTCUSDT-a-close", "orderId": "close-o", "execId": "e1",
         "closedSize": "10"},
        {"execType": "Funding", "symbol": "BTCUSDT", "side": "Sell", "execQty": "0",
         "execPrice": "0", "execFee": "0.1", "execTime": "1700000000001",
         "orderLinkId": "", "execId": "f1", "closedSize": "0"},
    ]
    # Realized PnL comes from the closed-PnL feed (gross = 10*(110-100) = 100), not
    # from the execution row, which carries none.
    pnl = [{"orderId": "close-o", "side": "Sell", "closedSize": "10",
            "avgEntryPrice": "100", "avgExitPrice": "110", "closedPnl": "99.4"}]
    fills = _live_broker(_ExecSession(rows, pnl_rows=pnl)).get_fills()
    assert len(fills) == 1                             # funding row skipped
    f = fills[0]
    assert f.side is Side.SELL and np.isclose(f.quantity, 10.0)
    assert np.isclose(f.realized_pnl, 100.0) and np.isclose(f.closed_quantity, 10.0)
    assert np.isclose(f.fee, 0.6)
    assert f.order_link_id == "platform-BTCUSDT-a-close" and f.exec_id == "e1"
    assert f.timestamp == pd.Timestamp(1700000000000, unit="ms", tz="UTC")


def test_live_get_fills_filters_by_since():
    rows = [
        {"execType": "Trade", "symbol": "BTCUSDT", "side": "Buy", "execQty": "1",
         "execPrice": "100", "execTime": "1700000000000", "execId": "old"},
        {"execType": "Trade", "symbol": "BTCUSDT", "side": "Buy", "execQty": "1",
         "execPrice": "100", "execTime": "1700000600000", "execId": "new"},
    ]
    since = pd.Timestamp(1700000600000, unit="ms", tz="UTC")
    fills = _live_broker(_ExecSession(rows)).get_fills(since=since)
    assert [f.exec_id for f in fills] == ["new"]


def test_live_get_fills_fails_loud_on_api_error():
    with pytest.raises(RuntimeError, match="get_executions failed"):
        _live_broker(_ExecSession([], ret=10001)).get_fills()
