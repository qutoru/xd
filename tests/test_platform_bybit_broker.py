"""Stage 10 — BybitBroker unit + ExecutionEngine→Broker integration (offline).

The Bybit v5 HTTP session is replaced by a fake that records calls and returns
canned v5-shaped responses, so every test runs with no network and no keys.
"""

from __future__ import annotations

import pandas as pd
import pytest

from crypto_signal_bot.platform.execution.bybit_broker import (
    BybitBroker,
    InstrumentInfo,
    build_broker,
)
from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
from crypto_signal_bot.platform.execution.domain import (
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
)
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.intent import TradeIntent

TS = pd.Timestamp("2023-07-01 12:00", tz="UTC")

_INSTRUMENT = {
    "result": {"list": [{
        "symbol": "BTCUSDT",
        "priceFilter": {"tickSize": "0.5"},
        "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001", "minNotionalValue": "5"},
    }]}
}


class FakeSession:
    """Records mutating calls; returns canned Bybit v5 responses."""

    def __init__(self, *, position_size=0.0, place_ret=0, raise_on_place=False):
        self.calls: list[tuple[str, dict]] = []
        self.orders: list[dict] = []
        self.cancels: list[dict] = []
        self._position_size = position_size
        self._place_ret = place_ret
        self._raise_on_place = raise_on_place
        self._oid = 0

    def get_server_time(self):
        self.calls.append(("get_server_time", {}))
        return {"retCode": 0, "result": {"timeSecond": "1700000000"}}

    def get_instruments_info(self, **kw):
        self.calls.append(("get_instruments_info", kw))
        return _INSTRUMENT

    def get_tickers(self, **kw):
        self.calls.append(("get_tickers", kw))
        return {"result": {"list": [{"lastPrice": "100"}]}}

    def place_order(self, **params):
        self.calls.append(("place_order", params))
        if self._raise_on_place:
            raise RuntimeError("network down")
        self.orders.append(params)
        self._oid += 1
        return {"retCode": self._place_ret, "retMsg": "err" if self._place_ret else "OK",
                "result": {"orderId": f"oid-{self._oid}"}}

    def get_positions(self, **kw):
        self.calls.append(("get_positions", kw))
        if self._position_size == 0:
            return {"result": {"list": []}}
        return {"result": {"list": [{
            "symbol": "BTCUSDT", "side": "Buy",
            "size": str(self._position_size), "avgPrice": "100",
        }]}}

    def get_wallet_balance(self, **kw):
        self.calls.append(("get_wallet_balance", kw))
        return {"result": {"list": [{"totalEquity": "10000"}]}}

    def cancel_all_orders(self, **kw):
        self.calls.append(("cancel_all_orders", kw))
        self.cancels.append(kw)
        return {"retCode": 0, "result": {"list": []}}

    def set_leverage(self, **kw):
        self.calls.append(("set_leverage", kw))
        return {"retCode": getattr(self, "_lev_ret", 0), "retMsg": "OK"}


def _cfg(mode=TradingMode.LIVE):
    return BybitConfig(api_key="k", api_secret="s", testnet=True, mode=mode)


def _placed(session, order_type=None):
    orders = session.orders
    if order_type is not None:
        orders = [o for o in orders if o["orderType"] == order_type]
    return orders


# --- config + factory -------------------------------------------------------
def test_config_from_env(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "k")
    monkeypatch.setenv("BYBIT_API_SECRET", "s")
    monkeypatch.setenv("BYBIT_TESTNET", "false")
    monkeypatch.setenv("BYBIT_TRADING_MODE", "live")
    cfg = BybitConfig.from_env()
    assert cfg.api_key == "k" and cfg.api_secret == "s"
    assert cfg.testnet is False and cfg.mode is TradingMode.LIVE and cfg.is_live


def test_config_from_env_prefers_testnet_keys_when_testnet(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "main_k")
    monkeypatch.setenv("BYBIT_API_SECRET", "main_s")
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", "test_k")
    monkeypatch.setenv("BYBIT_TESTNET_API_SECRET", "test_s")
    monkeypatch.setenv("BYBIT_TESTNET", "true")
    cfg = BybitConfig.from_env()
    assert cfg.testnet is True
    assert cfg.api_key == "test_k" and cfg.api_secret == "test_s"


def test_config_from_env_testnet_falls_back_to_base_keys(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "main_k")
    monkeypatch.setenv("BYBIT_API_SECRET", "main_s")
    monkeypatch.delenv("BYBIT_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_TESTNET_API_SECRET", raising=False)
    monkeypatch.setenv("BYBIT_TESTNET", "true")
    cfg = BybitConfig.from_env()
    assert cfg.api_key == "main_k" and cfg.api_secret == "main_s"


def test_config_from_env_mainnet_ignores_testnet_keys(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "main_k")
    monkeypatch.setenv("BYBIT_API_SECRET", "main_s")
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", "test_k")
    monkeypatch.setenv("BYBIT_TESTNET_API_SECRET", "test_s")
    monkeypatch.setenv("BYBIT_TESTNET", "false")
    cfg = BybitConfig.from_env()
    assert cfg.api_key == "main_k" and cfg.api_secret == "main_s"


def test_config_validate_live_requires_credentials():
    with pytest.raises(ValueError):
        BybitConfig(mode=TradingMode.LIVE).validate()


def test_build_broker_maps_modes():
    assert build_broker(BybitConfig(mode=TradingMode.SHADOW)) is None
    paper = build_broker(_cfg(TradingMode.PAPER), session=FakeSession())
    live = build_broker(_cfg(TradingMode.LIVE), session=FakeSession())
    assert isinstance(paper, BybitBroker) and isinstance(live, BybitBroker)


# --- connection + instrument info ------------------------------------------
def test_check_connection_ok_and_failure():
    ok = BybitBroker(_cfg(), session=FakeSession())
    assert ok.check_connection() is True

    class _Boom(FakeSession):
        def get_server_time(self):
            raise RuntimeError("no route")

    assert BybitBroker(_cfg(), session=_Boom()).check_connection() is False


def test_instrument_info_parsed_and_cached():
    s = FakeSession()
    broker = BybitBroker(_cfg(), session=s)
    info = broker.get_instrument_info("BTCUSDT")
    assert info == InstrumentInfo("BTCUSDT", 0.5, 0.001, 0.001, 5.0)
    broker.get_instrument_info("BTCUSDT")  # cached: no second lookup
    assert sum(c[0] == "get_instruments_info" for c in s.calls) == 1


# --- entry translation ------------------------------------------------------
def test_live_entry_translates_notional_to_contracts():
    s = FakeSession()
    broker = BybitBroker(_cfg(), session=s)
    # notional 50 @ price 100 -> 0.5 contracts (floored to qtyStep)
    res = broker.submit(OrderRequest("BTCUSDT", Side.BUY, 50.0, client_id="cid"))
    assert res.status is OrderStatus.FILLED and res.filled_quantity == 0.5
    p = _placed(s, "Market")[0]
    assert p["category"] == "linear" and p["symbol"] == "BTCUSDT"
    assert p["side"] == "Buy" and p["qty"] == "0.5"
    assert p["reduceOnly"] is False and p["orderLinkId"] == "cid"
    assert "price" not in p and "triggerPrice" not in p


def test_paper_entry_simulates_fill_without_network_order():
    s = FakeSession()
    broker = BybitBroker(_cfg(TradingMode.PAPER), session=s)
    res = broker.submit(OrderRequest("BTCUSDT", Side.SELL, 50.0))
    assert res.status is OrderStatus.FILLED
    assert not s.orders  # paper never places a real order
    st = broker.get_portfolio_state()
    assert st.value == 10_000.0
    assert st.notional("BTCUSDT") == pytest.approx(-50.0)  # short, signed notional


def test_min_qty_and_min_notional_rejected():
    broker = BybitBroker(_cfg(TradingMode.PAPER), session=FakeSession())
    tiny = broker.submit(OrderRequest("BTCUSDT", Side.BUY, 0.0001))  # -> 0 contracts
    assert tiny.status is OrderStatus.REJECTED and "min" in tiny.message


# --- reduce-only TP / SL translation ---------------------------------------
def test_reduce_only_tp_and_sl_are_close_only_with_levels():
    s = FakeSession(position_size=0.5)  # an open long to close
    broker = BybitBroker(_cfg(), session=s)

    tp = broker.submit(OrderRequest("BTCUSDT", Side.SELL, 50.0, order_type=OrderType.LIMIT,
                                    price=110.0, reduce_only=True))
    sl = broker.submit(OrderRequest("BTCUSDT", Side.SELL, 50.0, order_type=OrderType.STOP,
                                    price=95.0, reduce_only=True))
    assert tp.status is OrderStatus.PENDING and sl.status is OrderStatus.PENDING

    lp = _placed(s, "Limit")[0]
    assert lp["reduceOnly"] is True and lp["side"] == "Sell"
    assert lp["price"] == "110.0" and lp["qty"] == "0.5"  # sized to the position

    sp = [o for o in s.orders if o.get("triggerPrice")][0]
    assert sp["reduceOnly"] is True and sp["orderType"] == "Market"
    assert sp["triggerPrice"] == "95.0" and sp["triggerDirection"] == 2  # long stop = fall


# --- API error handling -----------------------------------------------------
def test_api_exception_becomes_clean_reject():
    broker = BybitBroker(_cfg(), session=FakeSession(raise_on_place=True))
    res = broker.submit(OrderRequest("BTCUSDT", Side.BUY, 50.0))
    assert res.status is OrderStatus.REJECTED and "network down" in res.message


def test_nonzero_retcode_becomes_reject():
    broker = BybitBroker(_cfg(), session=FakeSession(place_ret=10001))
    res = broker.submit(OrderRequest("BTCUSDT", Side.BUY, 50.0))
    assert res.status is OrderStatus.REJECTED and res.message == "err"


# --- leverage + one-way positionIdx (E2E-critical) --------------------------
def test_orders_carry_one_way_position_idx():
    s = FakeSession()
    BybitBroker(_cfg(), session=s).submit(OrderRequest("BTCUSDT", Side.BUY, 50.0))
    assert _placed(s, "Market")[0]["positionIdx"] == 0  # one-way


def test_set_leverage_ok_and_benign_and_error():
    ok = FakeSession()
    assert BybitBroker(_cfg(), session=ok).set_leverage("BTCUSDT", 3) is True
    assert ("set_leverage", {"category": "linear", "symbol": "BTCUSDT",
                             "buyLeverage": "3", "sellLeverage": "3"}) in ok.calls

    benign = FakeSession(); benign._lev_ret = 110043  # "leverage not modified"
    assert BybitBroker(_cfg(), session=benign).set_leverage("BTCUSDT") is True

    bad = FakeSession(); bad._lev_ret = 10001
    assert BybitBroker(_cfg(), session=bad).set_leverage("BTCUSDT") is False


def test_config_rejects_bad_leverage_and_position_idx():
    with pytest.raises(ValueError):
        BybitConfig(mode=TradingMode.PAPER, leverage=0.5).validate()
    with pytest.raises(ValueError):
        BybitConfig(mode=TradingMode.PAPER, position_idx=3).validate()


# --- live portfolio state ---------------------------------------------------
def test_live_portfolio_state_reads_wallet_and_positions():
    broker = BybitBroker(_cfg(), session=FakeSession(position_size=0.5))
    st = broker.get_portfolio_state()
    assert st.value == 10_000.0
    assert st.notional("BTCUSDT") == pytest.approx(50.0)  # 0.5 * 100, long


# --- ExecutionEngine -> Broker integration ---------------------------------
def _intent():
    return TradeIntent(symbol="BTCUSDT", side=Side.BUY, target_notional=50.0, entry=100.0,
                       take_profit=110.0, stop_loss=95.0, strategy="platform", timestamp=TS)


def test_engine_routes_bracket_through_paper_broker():
    broker = BybitBroker(_cfg(TradingMode.PAPER), session=FakeSession())
    report = ExecutionEngine(broker).execute_bracket_intents([_intent()])
    # entry + TP + SL, entry filled and booked, brackets resting
    assert len(report.orders) == 3
    statuses = [o.status for o in report.orders]
    assert statuses[0] is OrderStatus.FILLED
    assert statuses[1] is OrderStatus.PENDING and statuses[2] is OrderStatus.PENDING
    assert report.resulting_state.notional("BTCUSDT") == pytest.approx(50.0)


def test_engine_routes_bracket_through_live_broker():
    s = FakeSession(position_size=0.5)
    report = ExecutionEngine(BybitBroker(_cfg(), session=s)).execute_bracket_intents([_intent()])
    kinds = [o.request.order_type for o in report.orders]
    assert kinds == [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP]
    # three real orders placed; the two brackets are reduce-only
    assert len(s.orders) == 3
    assert [o["reduceOnly"] for o in s.orders] == [False, True, True]


# --- C3: a signalled position-read error must NOT be read as a flat account -----
class _PosErrSession(FakeSession):
    """get_positions returns a non-zero retCode with an empty list (API error)."""

    def get_positions(self, **kw):
        self.calls.append(("get_positions", kw))
        return {"retCode": 10001, "retMsg": "system busy", "result": {"list": []}}


def test_live_positions_error_raises_instead_of_reporting_flat():
    # Before the fix, a retCode!=0 empty response was silently treated as "no
    # positions". Now the read fails loud so the caller cannot act on a false flat.
    broker = BybitBroker(_cfg(), session=_PosErrSession())
    with pytest.raises(RuntimeError, match="get_positions failed"):
        broker.get_portfolio_state()


def test_retcode_zero_empty_list_is_a_genuine_flat_account():
    # A *successful* empty read (retCode 0 / absent) is still a real flat account.
    broker = BybitBroker(_cfg(), session=FakeSession(position_size=0.0))
    st = broker.get_portfolio_state()
    assert st.positions == {}  # no exception, correctly flat


def test_rebalance_does_not_reopen_when_position_read_errors():
    # The delta-rebalance reads positions FIRST; on a signalled error it must raise
    # (so run_once trips the emergency stop) rather than see a flat book and re-open
    # the full target on top of the real position (the P0-A accumulation class).
    s = _PosErrSession()
    broker = BybitBroker(_cfg(), session=s)
    intent = TradeIntent(symbol="BTCUSDT", side=Side.BUY, target_notional=1000.0,
                         entry=100.0, strategy="platform", timestamp=TS)
    with pytest.raises(RuntimeError, match="get_positions failed"):
        ExecutionEngine(broker).rebalance_bracket_intents([intent], book_symbols={"BTCUSDT"})
    assert not s.orders  # nothing was routed — no re-open on a bad read


# --- C2: prior TP/SL brackets are cancelled before new ones are placed ----------
def test_live_broker_cancel_open_orders_calls_cancel_all():
    s = FakeSession()
    BybitBroker(_cfg(), session=s).cancel_open_orders("BTCUSDT")
    assert s.cancels == [{"category": "linear", "symbol": "BTCUSDT"}]


def test_paper_broker_cancel_open_orders_is_noop():
    s = FakeSession()
    BybitBroker(_cfg(TradingMode.PAPER), session=s).cancel_open_orders("BTCUSDT")
    assert s.cancels == []  # paper has no resting orders


def test_cancel_open_orders_swallows_errors_and_does_not_abort():
    class _Boom(FakeSession):
        def cancel_all_orders(self, **kw):
            raise RuntimeError("venue down")

    # must NOT raise (best-effort hygiene)
    BybitBroker(_cfg(), session=_Boom()).cancel_open_orders("BTCUSDT")


# --- realized PnL is sourced from the closed-PnL feed, not the execution list ----
class _FillsSession(FakeSession):
    """Serves an entry + TP-close execution, plus a closed-PnL record for the close.

    Mirrors the real venue: the execution rows carry closedSize but no realized PnL;
    the closed-PnL feed carries avgEntry/avgExit for the closing order. ``pnl_rows``
    lets a test omit the closed-PnL record to simulate eventual-consistency lag.
    """

    def __init__(self, *, exec_rows, pnl_rows):
        super().__init__()
        self._exec_rows = exec_rows
        self._pnl_rows = pnl_rows

    def get_executions(self, **kw):
        self.calls.append(("get_executions", kw))
        return {"retCode": 0, "result": {"list": self._exec_rows}}

    def get_closed_pnl(self, **kw):
        self.calls.append(("get_closed_pnl", kw))
        return {"retCode": 0, "result": {"list": self._pnl_rows}}


def _exec(execId, side, qty, price, *, closed=0.0, link="", order="o1"):
    return {"execType": "Trade", "execId": execId, "orderId": order, "orderLinkId": link,
            "symbol": "XRPUSDT", "side": side, "execQty": str(qty), "execPrice": str(price),
            "execFee": "0.1", "execTime": "1690000000000", "closedSize": str(closed)}


def test_get_fills_takes_gross_realized_from_closed_pnl():
    # Entry buy @1.1144, TP sell @1.1422 closing 298.9 — a real +8.3 profit that the
    # execution feed reports as zero PnL. The closed-PnL record supplies the prices,
    # and the broker must book the GROSS round-trip (size*(exit-entry)), fees separate.
    s = _FillsSession(
        exec_rows=[
            _exec("e1", "Buy", 298.9, 1.1144, order="entry-o"),
            _exec("e2", "Sell", 298.9, 1.1422, closed=298.9, link="p-XRPUSDT-x-tp", order="tp-o"),
        ],
        pnl_rows=[{"orderId": "tp-o", "side": "Sell", "closedSize": "298.9",
                   "avgEntryPrice": "1.1144", "avgExitPrice": "1.1422", "closedPnl": "8.0"}],
    )
    fills = BybitBroker(_cfg(), session=s).get_fills()
    by_exec = {f.exec_id: f for f in fills}
    assert by_exec["e1"].realized_pnl == pytest.approx(0.0)      # entry realizes nothing
    assert by_exec["e2"].realized_pnl == pytest.approx(298.9 * (1.1422 - 1.1144))
    assert by_exec["e2"].realized_pnl > 0                         # a TP win, not a loss


def test_get_fills_holds_back_close_missing_closed_pnl():
    # If the closed-PnL record has not landed yet, the close is NOT booked at zero
    # (which would freeze a wrong loss via exec_id de-dup): it is skipped for now.
    s = _FillsSession(
        exec_rows=[_exec("e2", "Sell", 298.9, 1.1422, closed=298.9, order="tp-o")],
        pnl_rows=[],
    )
    fills = BybitBroker(_cfg(), session=s).get_fills()
    assert fills == []


def test_get_fills_short_close_realizes_positive_when_exit_below_entry():
    # A Buy close flattens a short: profit when exit < entry -> gross must be positive.
    s = _FillsSession(
        exec_rows=[_exec("e2", "Buy", 100.0, 0.90, closed=100.0, order="sl-o")],
        pnl_rows=[{"orderId": "sl-o", "side": "Buy", "closedSize": "100",
                   "avgEntryPrice": "1.00", "avgExitPrice": "0.90", "closedPnl": "10"}],
    )
    fills = BybitBroker(_cfg(), session=s).get_fills()
    assert fills[0].realized_pnl == pytest.approx(100.0 * (1.00 - 0.90))
