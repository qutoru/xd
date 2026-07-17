"""Stage 17.2 — PerformanceReport (known-answer, built only from the ledger)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.accounting.ledger import AccountingLedger
from crypto_signal_bot.platform.accounting.report import PerformanceReport
from crypto_signal_bot.platform.execution.domain import Fill, Side


def _fill(exec_id, *, symbol, realized, fee=0.0, closed=0.0, link=None):
    return Fill(symbol=symbol, side=Side.SELL, quantity=10.0, price=100.0, fee=fee,
                timestamp=pd.Timestamp("2026-01-02T00:00:00+00:00"),
                order_link_id=link, exec_id=exec_id, realized_pnl=realized,
                closed_quantity=closed)


def _ledger():
    led = AccountingLedger()
    led.record([
        _fill("e0", symbol="BTCUSDT", realized=0.0, closed=0.0,
              link="platform-BTCUSDT-a-entry"),
        _fill("e1", symbol="BTCUSDT", realized=100.0, fee=0.6, closed=10.0,
              link="platform-BTCUSDT-a-tp"),
        _fill("e2", symbol="ETHUSDT", realized=-40.0, fee=0.4, closed=10.0,
              link="platform-ETHUSDT-b-sl"),
        _fill("e3", symbol="BTCUSDT", realized=25.0, closed=10.0,
              link="platform-BTCUSDT-c-close"),
        _fill("e4", symbol="SOLUSDT", realized=-10.0, closed=10.0,
              link=None),   # venue-initiated close -> manual
    ])
    return led


def test_report_totals_match_ledger():
    rep = PerformanceReport.from_ledger(_ledger())
    assert np.isclose(rep.realized_pnl, 75.0)      # 0 + 100 - 40 + 25 - 10
    assert np.isclose(rep.fees, 1.0)
    assert np.isclose(rep.net_pnl, 74.0)


def test_report_realized_by_symbol():
    rep = PerformanceReport.from_ledger(_ledger())
    assert np.isclose(rep.realized_by_symbol["BTCUSDT"], 125.0)   # 0 + 100 + 25
    assert np.isclose(rep.realized_by_symbol["ETHUSDT"], -40.0)
    assert np.isclose(rep.realized_by_symbol["SOLUSDT"], -10.0)


def test_report_realized_by_close_reason():
    rep = PerformanceReport.from_ledger(_ledger())
    assert set(rep.realized_by_reason) == {"take_profit", "stop_loss",
                                           "rebalance", "manual"}
    assert np.isclose(rep.realized_by_reason["take_profit"], 100.0)
    assert np.isclose(rep.realized_by_reason["stop_loss"], -40.0)
    assert np.isclose(rep.realized_by_reason["rebalance"], 25.0)
    assert np.isclose(rep.realized_by_reason["manual"], -10.0)


def test_empty_ledger_report_is_all_zero_with_reason_keys_present():
    rep = PerformanceReport.from_ledger(AccountingLedger())
    assert rep.realized_pnl == 0.0 and rep.fees == 0.0 and rep.net_pnl == 0.0
    assert rep.realized_by_symbol == {}
    assert rep.realized_by_reason == {"take_profit": 0.0, "stop_loss": 0.0,
                                      "rebalance": 0.0, "manual": 0.0}
