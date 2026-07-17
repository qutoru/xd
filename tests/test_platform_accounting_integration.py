"""Stage 17.1 — AccountingStore/Ledger wired into DailyPipeline.

After reconciliation the pipeline books the broker's fills into a persisted
realized-PnL ledger. Covers: backward compatibility (no store -> unchanged, no
file), the empty-fills no-op, error isolation (a failing get_fills never aborts
the cycle), dedup across cycles, and a PAPER end-to-end where a close realizes PnL
into the persisted ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


class _PaperSession:
    """PAPER fake session with a mutable last price."""

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


_SYMS = [f"C{i}" for i in range(8)]
_IDX = pd.date_range(end=pd.Timestamp("2023-05-01", tz="UTC"), periods=15, freq="D")


def _snapshot(asof, *, flip=False):
    rng = np.random.default_rng(7)
    closes = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.02, (15, 8)), axis=0),
        index=_IDX, columns=_SYMS)
    grad = np.linspace(-1e-4, 1e-4, 8)
    if flip:
        grad = grad[::-1]
    funding = pd.DataFrame(grad + rng.normal(0, 1e-6, (15, 8)),
                           index=_IDX, columns=_SYMS)
    return MarketSnapshot(asof, _SYMS, closes, closes.pct_change(), funding)


class _Prov:
    """Snapshot provider: flips the funding gradient on the second calendar day."""

    def __init__(self, base_day):
        self._base = base_day

    def snapshot(self, asof, lookback_days):
        return _snapshot(asof, flip=(asof.date() != self._base.date()))


def _pipeline(session, *, accounting_store=None, execute=True):
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER)
    broker = BybitBroker(cfg, session=session)
    base = pd.Timestamp("2023-05-01", tz="UTC")
    pipe = DailyPipeline(
        _Prov(base), signal_names=["alx"],
        execution_engine=ExecutionEngine(broker),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        risk_manager=RiskManager(RiskConfig(max_position_pct=1.0, max_gross=10.0,
                                            min_notional=1.0)),
        risk_control=ProductionRiskControl(RiskControlConfig()),
        nav=10_000.0, use_brackets=True, execute=execute, lookback_days=10,
        accounting_store=accounting_store,
    )
    return pipe, broker


# --- backward compatibility --------------------------------------------------
def test_no_store_leaves_pipeline_unchanged_and_writes_nothing(tmp_path):
    pipe, broker = _pipeline(_PaperSession())
    assert pipe._ledger is None                        # accounting disabled
    result = pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))
    assert result.execution is not None                # normal cycle still ran
    assert broker.get_portfolio_state().positions      # a book was opened
    assert not list(tmp_path.iterdir())                # nothing persisted


def test_empty_fills_do_not_create_a_ledger_file(tmp_path):
    # execute=False -> no orders -> no paper fills -> get_fills empty -> no save.
    store = AccountingStore(tmp_path / "ledger.json")
    pipe, _ = _pipeline(_PaperSession(), accounting_store=store, execute=False)
    pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))
    assert len(pipe._ledger) == 0
    assert not (tmp_path / "ledger.json").exists()


def test_failing_get_fills_does_not_abort_the_cycle(tmp_path):
    store = AccountingStore(tmp_path / "ledger.json")
    pipe, broker = _pipeline(_PaperSession(), accounting_store=store)

    def _boom(since=None):
        raise RuntimeError("feed down")

    broker.get_fills = _boom  # accounting read fails
    result = pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))
    assert result.execution is not None                # cycle completed anyway
    assert len(pipe._ledger) == 0


# --- integration: fills flow into the persisted ledger -----------------------
def test_run_once_books_fills_into_persisted_ledger(tmp_path):
    store = AccountingStore(tmp_path / "ledger.json")
    pipe, broker = _pipeline(_PaperSession(), accounting_store=store)
    pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))

    assert (tmp_path / "ledger.json").exists()
    # ledger holds exactly the broker's fills (entries opened this cycle)
    assert [e.exec_id for e in pipe._ledger.entries] == \
           [f.exec_id for f in broker.get_fills()]
    assert len(pipe._ledger) > 0
    reloaded = store.load()
    assert len(reloaded) == len(pipe._ledger)          # persisted faithfully


def test_second_cycle_dedups_and_books_only_new_fills(tmp_path):
    store = AccountingStore(tmp_path / "ledger.json")
    session = _PaperSession(100.0)
    pipe, broker = _pipeline(session, accounting_store=store)

    pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))      # open @100
    session.price = 110.0
    pipe.run_once(pd.Timestamp("2023-05-02", tz="UTC"))      # flip -> closes @110

    all_fills = broker.get_fills()
    # every fill booked exactly once (no dup across cycles, none lost)
    assert [e.exec_id for e in pipe._ledger.entries] == [f.exec_id for f in all_fills]
    assert np.isclose(pipe._ledger.realized_pnl,
                      sum(f.realized_pnl for f in all_fills))


# --- PAPER end-to-end: a close realizes PnL into the ledger -------------------
def test_end_to_end_paper_close_realizes_pnl_into_ledger(tmp_path):
    store = AccountingStore(tmp_path / "ledger.json")
    session = _PaperSession(100.0)
    pipe, broker = _pipeline(session, accounting_store=store)

    pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))      # entries @100
    entries_only = store.load()
    assert len(entries_only) > 0
    assert entries_only.realized_pnl == 0.0                 # opens realize nothing

    session.price = 110.0
    pipe.run_once(pd.Timestamp("2023-05-02", tz="UTC"))      # flips close @110

    reloaded = store.load()
    rebal = [e for e in reloaded.entries if e.attribution == "rebalance"]
    assert rebal, "a flip should book reduce-only rebalance closes"
    assert any(e.realized_pnl != 0.0 for e in rebal)        # realized PnL booked
    assert "rebalance" in reloaded.realized_by_attribution()


# --- Stage 17.2: PerformanceReport on PipelineResult -------------------------
def test_result_has_no_performance_report_without_store():
    pipe, _ = _pipeline(_PaperSession())
    result = pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))
    assert result.performance is None                  # accounting off -> no report


def test_result_performance_report_matches_ledger(tmp_path):
    store = AccountingStore(tmp_path / "ledger.json")
    pipe, _ = _pipeline(_PaperSession(), accounting_store=store)
    result = pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))

    perf = result.performance
    assert perf is not None
    # report is a pure projection of the ledger, no independent bookkeeping
    assert np.isclose(perf.realized_pnl, pipe._ledger.realized_pnl)
    assert np.isclose(perf.net_pnl, pipe._ledger.net_pnl)
    assert set(perf.realized_by_reason) == {"take_profit", "stop_loss",
                                            "rebalance", "manual"}


def test_end_to_end_report_reflects_realized_close(tmp_path):
    store = AccountingStore(tmp_path / "ledger.json")
    session = _PaperSession(100.0)
    pipe, _ = _pipeline(session, accounting_store=store)

    r1 = pipe.run_once(pd.Timestamp("2023-05-01", tz="UTC"))   # opens @100
    assert r1.performance.realized_pnl == 0.0                  # nothing closed yet
    session.price = 110.0
    r2 = pipe.run_once(pd.Timestamp("2023-05-02", tz="UTC"))   # flips close @110

    perf = r2.performance
    # closes were booked as rebalance and DID realize PnL per position (a dollar-
    # neutral book nets ~0 on a parallel move, so assert on the individual closes)
    rebal = [e for e in pipe._ledger.entries if e.attribution == "rebalance"]
    assert rebal and any(e.realized_pnl != 0.0 for e in rebal)
    # the report is a faithful projection of the ledger (no independent bookkeeping)
    assert np.isclose(perf.realized_pnl, pipe._ledger.realized_pnl)
    assert np.isclose(perf.realized_by_reason["rebalance"],
                      pipe._ledger.realized_by_attribution().get("rebalance", 0.0))
