"""Stage 16.3 — AccountingLedger + AccountingStore + persistent realized PnL."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.accounting.ledger import AccountingLedger, AccountingStore
from crypto_signal_bot.platform.execution.domain import Fill, Side


def _fill(exec_id, *, symbol="BTCUSDT", side=Side.SELL, realized=0.0, fee=0.0,
          closed=0.0, link=None, ts="2026-01-02T00:00:00+00:00"):
    return Fill(symbol=symbol, side=side, quantity=10.0, price=100.0, fee=fee,
                timestamp=pd.Timestamp(ts), order_link_id=link, exec_id=exec_id,
                realized_pnl=realized, closed_quantity=closed)


def test_record_books_realized_pnl_fees_and_attribution():
    led = AccountingLedger()
    added = led.record([
        _fill("e0", realized=0.0, closed=0.0, link="platform-BTCUSDT-a-entry"),
        _fill("e1", realized=100.0, fee=0.6, closed=10.0, link="platform-BTCUSDT-a-tp"),
        _fill("e2", realized=-40.0, fee=0.4, closed=10.0, link="platform-BTCUSDT-b-sl"),
    ])
    assert len(added) == 3 and len(led) == 3
    assert np.isclose(led.realized_pnl, 60.0)          # 0 + 100 - 40
    assert np.isclose(led.total_fees, 1.0)
    assert np.isclose(led.net_pnl, 59.0)               # 60 - 1
    by_attr = led.realized_by_attribution()
    assert np.isclose(by_attr["take_profit"], 100.0)
    assert np.isclose(by_attr["stop_loss"], -40.0)
    assert by_attr["entry"] == 0.0


def test_record_dedups_by_exec_id():
    led = AccountingLedger()
    led.record([_fill("e1", realized=100.0, closed=10.0)])
    added = led.record([_fill("e1", realized=100.0, closed=10.0),   # duplicate
                        _fill("e2", realized=25.0, closed=10.0)])    # new
    assert [e.exec_id for e in added] == ["e2"]
    assert len(led) == 2 and np.isclose(led.realized_pnl, 125.0)    # not 225


def test_fills_without_exec_id_are_not_deduped():
    led = AccountingLedger()
    led.record([_fill(None, realized=10.0, closed=10.0)])
    led.record([_fill(None, realized=10.0, closed=10.0)])
    assert len(led) == 2 and np.isclose(led.realized_pnl, 20.0)


def test_realized_by_symbol_and_watermark():
    led = AccountingLedger()
    led.record([
        _fill("e1", symbol="BTCUSDT", realized=100.0, closed=10.0,
              ts="2026-01-02T00:00:00+00:00"),
        _fill("e2", symbol="ETHUSDT", realized=-30.0, closed=10.0,
              ts="2026-01-03T12:00:00+00:00"),
    ])
    by_sym = led.realized_by_symbol()
    assert np.isclose(by_sym["BTCUSDT"], 100.0) and np.isclose(by_sym["ETHUSDT"], -30.0)
    assert led.watermark() == pd.Timestamp("2026-01-03T12:00:00+00:00")


def test_store_roundtrip_persists_realized_pnl_and_dedup_state(tmp_path):
    store = AccountingStore(tmp_path / "sub" / "ledger.json")
    led = AccountingLedger()
    led.record([
        _fill("e1", realized=100.0, fee=0.6, closed=10.0, link="platform-BTCUSDT-a-tp"),
        _fill("e2", realized=-40.0, fee=0.4, closed=10.0, link="platform-BTCUSDT-b-sl"),
    ])
    store.save(led)

    reloaded = store.load()
    assert len(reloaded) == 2
    assert np.isclose(reloaded.realized_pnl, 60.0)
    assert np.isclose(reloaded.net_pnl, 59.0)
    assert reloaded.realized_by_attribution()["take_profit"] == 100.0

    # re-recording the same executions after a restart must not double-count
    added = reloaded.record([_fill("e1", realized=100.0, closed=10.0),
                             _fill("e3", realized=5.0, closed=10.0)])
    assert [e.exec_id for e in added] == ["e3"]
    assert np.isclose(reloaded.realized_pnl, 65.0)     # 60 + 5, e1 skipped


def test_store_load_missing_file_is_empty_ledger(tmp_path):
    led = AccountingStore(tmp_path / "nope.json").load()
    assert len(led) == 0 and led.realized_pnl == 0.0 and led.watermark() is None


# --- end-to-end: real BybitBroker (PAPER) -> get_fills -> ledger -> store ------
class _PriceSession:
    def __init__(self, price=100.0):
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


def test_end_to_end_paper_round_trips_flow_into_persisted_ledger(tmp_path):
    from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
    from crypto_signal_bot.platform.execution.domain import OrderRequest, OrderType

    sess = _PriceSession(100.0)
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER,
                      paper_equity=10_000.0)
    broker = BybitBroker(cfg, session=sess)

    def order(sym, side, notional, reduce_only, kind):
        return OrderRequest(symbol=sym, side=side, quantity=notional,
                            order_type=OrderType.MARKET, reduce_only=reduce_only,
                            client_id=f"platform-{sym}-tok-{kind}")

    broker.submit(order("BTCUSDT", Side.BUY, 1000.0, False, "entry"))   # +10 @100
    broker.submit(order("ETHUSDT", Side.BUY, 1000.0, False, "entry"))   # +10 @100
    sess.price = 130.0
    broker.submit(order("BTCUSDT", Side.SELL, 1000.0, True, "close"))   # +300
    sess.price = 80.0
    broker.submit(order("ETHUSDT", Side.SELL, 1000.0, True, "close"))   # -200

    store = AccountingStore(tmp_path / "ledger.json")
    led = store.load()                      # empty first run
    led.record(broker.get_fills())
    store.save(led)

    reloaded = store.load()
    assert len(reloaded) == 4               # 2 entries + 2 closes
    assert np.isclose(reloaded.realized_pnl, 100.0)     # +300 - 200
    by_attr = reloaded.realized_by_attribution()
    assert np.isclose(by_attr["rebalance"], 100.0) and by_attr["entry"] == 0.0
    by_sym = reloaded.realized_by_symbol()
    assert np.isclose(by_sym["BTCUSDT"], 300.0) and np.isclose(by_sym["ETHUSDT"], -200.0)

    # a second poll of the same fills (restart) does not double-count
    reloaded.record(broker.get_fills())
    assert len(reloaded) == 4 and np.isclose(reloaded.realized_pnl, 100.0)
