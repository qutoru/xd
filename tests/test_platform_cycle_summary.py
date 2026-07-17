"""Stage 19 — CycleSummary (observability projection over a PipelineResult).

Known-answer tests feed duck-typed result stubs so the projection/status logic is
pinned exactly; one integration test builds the summary from a real PAPER pipeline
result. The summary computes nothing — it only reshapes already-produced fields.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.execution.reconciler import DiscrepancyKind
from crypto_signal_bot.platform.observability.summary import CycleStatus, CycleSummary
from crypto_signal_bot.platform.risk_control.control import BlockReason


def _result(*, filled=2, rejected=0, traded=1000.0, halted=False, halt_reason=None,
            blocked=0, rec_failed=False, rec_ok=True, rec_kinds=(), rec_msg="",
            perf=None, notify_failed=0, asof="2026-01-02"):
    return NS(
        asof=pd.Timestamp(asof, tz="UTC"),
        execution=NS(n_filled=filled, n_rejected=rejected, traded_notional=traded),
        risk_control=NS(halted=halted, halt_reason=halt_reason, n_blocked=blocked),
        reconciliation=NS(failed=rec_failed, ok=rec_ok, message=rec_msg,
                          kinds=lambda: set(rec_kinds), n_discrepancies=len(rec_kinds)),
        performance=perf,
        shadow=NS(daily_pnl=0.01, cum_pnl=0.05),
        notify=NS(failed=notify_failed),
    )


_PERF = NS(realized_pnl=125.0, fees=1.0, net_pnl=124.0,
           realized_by_reason={"take_profit": 100.0, "stop_loss": -40.0,
                               "rebalance": 65.0, "manual": 0.0})


# --- status derivation -------------------------------------------------------
def test_status_ok_when_clean():
    s = CycleSummary.from_result(_result())
    assert s.status is CycleStatus.OK


def test_status_halted_when_risk_halted():
    s = CycleSummary.from_result(
        _result(halted=True, halt_reason=BlockReason.DAILY_LOSS_LIMIT))
    assert s.status is CycleStatus.HALTED
    assert s.halted and s.halt_reason == "daily_loss_limit"


def test_status_failed_when_reconciliation_failed():
    s = CycleSummary.from_result(_result(rec_failed=True, rec_ok=False, rec_msg="api down"))
    assert s.status is CycleStatus.FAILED
    assert s.reconciliation_failed and s.reconciliation_message == "api down"


def test_failed_takes_precedence_over_halted():
    s = CycleSummary.from_result(
        _result(rec_failed=True, rec_ok=False, rec_msg="boom",
                halted=True, halt_reason=BlockReason.EMERGENCY_STOP))
    assert s.status is CycleStatus.FAILED


def test_status_degraded_on_rejects_blocks_discrepancies_or_notify_fail():
    assert CycleSummary.from_result(_result(rejected=1)).status is CycleStatus.DEGRADED
    assert CycleSummary.from_result(_result(blocked=2)).status is CycleStatus.DEGRADED
    assert CycleSummary.from_result(
        _result(rec_ok=False, rec_kinds=(DiscrepancyKind.SIZE_DECREASED,))
    ).status is CycleStatus.DEGRADED
    assert CycleSummary.from_result(_result(notify_failed=1)).status is CycleStatus.DEGRADED


# --- projection (no recomputation) -------------------------------------------
def test_performance_fields_are_copied_verbatim():
    s = CycleSummary.from_result(_result(perf=_PERF))
    assert s.realized_pnl == 125.0 and s.fees == 1.0 and s.net_pnl == 124.0
    assert s.realized_by_reason == _PERF.realized_by_reason
    assert s.realized_by_reason is not _PERF.realized_by_reason   # defensive copy


def test_accounting_disabled_when_no_performance():
    s = CycleSummary.from_result(_result(perf=None))
    assert s.realized_pnl is None and s.net_pnl is None and s.realized_by_reason == {}
    assert any("accounting: disabled" in ln for ln in s.format_lines())


def test_execution_and_shadow_and_duration_projected():
    s = CycleSummary.from_result(_result(filled=3, rejected=1, traded=555.0), duration_s=1.25)
    assert (s.filled, s.rejected, s.traded_notional) == (3, 1, 555.0)
    assert s.shadow_daily_pnl == 0.01 and s.shadow_cum_pnl == 0.05
    assert s.duration_s == 1.25


def test_as_dict_is_flat_and_json_friendly():
    d = CycleSummary.from_result(_result(perf=_PERF, halted=True,
                                         halt_reason=BlockReason.KILL_SWITCH)).as_dict()
    assert d["status"] == "halted" and d["halt_reason"] == "kill_switch"
    assert d["realized_pnl"] == 125.0 and isinstance(d["discrepancies"], list)
    assert d["realized_by_reason"]["take_profit"] == 100.0


def test_format_lines_render_without_error():
    lines = CycleSummary.from_result(_result(perf=_PERF), duration_s=0.5).format_lines()
    assert lines and lines[0].startswith("cycle 2026-01-02 status=OK")


# --- integration: real PAPER pipeline result ---------------------------------
def test_summary_from_real_pipeline_result(tmp_path):
    from crypto_signal_bot.platform.accounting.ledger import AccountingStore
    from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
    from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
    from crypto_signal_bot.platform.execution.engine import ExecutionEngine
    from crypto_signal_bot.platform.pipeline import DailyPipeline
    from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig
    from crypto_signal_bot.platform.risk.manager import RiskConfig, RiskManager
    from crypto_signal_bot.platform.risk_control.control import (
        ProductionRiskControl, RiskControlConfig,
    )
    from crypto_signal_bot.platform.shadow.runner import ShadowRunner

    syms = [f"C{i}" for i in range(8)]
    idx = pd.date_range(end=pd.Timestamp("2023-05-01", tz="UTC"), periods=15, freq="D")
    rng = np.random.default_rng(7)
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (15, 8)), axis=0),
                          index=idx, columns=syms)
    funding = pd.DataFrame(np.linspace(-1e-4, 1e-4, 8) + rng.normal(0, 1e-6, (15, 8)),
                           index=idx, columns=syms)
    snap = MarketSnapshot(pd.Timestamp("2023-05-01", tz="UTC"), syms, closes,
                          closes.pct_change(), funding)

    class _Prov:
        def snapshot(self, asof, lookback_days):
            return snap

    class _Sess:
        def get_server_time(self):
            return {"retCode": 0, "result": {"timeSecond": "1"}}

        def get_instruments_info(self, **kw):
            return {"result": {"list": [{
                "symbol": kw.get("symbol", ""),
                "priceFilter": {"tickSize": "0.01"},
                "lotSizeFilter": {"qtyStep": "0.0001", "minOrderQty": "0.0001",
                                  "minNotionalValue": "0"}}]}}

        def get_tickers(self, **kw):
            return {"result": {"list": [{"lastPrice": "100"}]}}

    broker = BybitBroker(BybitConfig(api_key="k", api_secret="s",
                                     mode=TradingMode.PAPER), session=_Sess())
    pipe = DailyPipeline(
        _Prov(), signal_names=["alx"], execution_engine=ExecutionEngine(broker),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        risk_manager=RiskManager(RiskConfig(max_position_pct=1.0, max_gross=10.0,
                                            min_notional=1.0)),
        risk_control=ProductionRiskControl(RiskControlConfig()),
        nav=10_000.0, use_brackets=True, execute=True, lookback_days=10,
        accounting_store=AccountingStore(tmp_path / "ledger.json"),
    )
    result = pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))
    summary = CycleSummary.from_result(result, duration_s=0.9)

    assert summary.status in tuple(CycleStatus)
    assert summary.filled == result.execution.n_filled
    assert summary.realized_pnl == result.performance.realized_pnl   # projection matches
    assert summary.duration_s == 0.9
    assert summary.as_dict()["asof"] == "2023-05-01"
