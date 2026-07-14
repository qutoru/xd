"""Stage 11 — Reconciler: expected-vs-actual state classification (offline).

Covers the required scenarios: manual close, partial fill, TP execution, SL
execution, no position, and position size change. Take-profit / stop-loss / manual
closes are indistinguishable from position *state* (they all leave a flat book), so
they are correctly reported structurally as POSITION_CLOSED.
"""

from __future__ import annotations

import pandas as pd

from crypto_signal_bot.platform.execution.domain import PortfolioState, Position
from crypto_signal_bot.platform.execution.reconciler import (
    DiscrepancyKind,
    Reconciler,
    ReconciliationReport,
)

ASOF = pd.Timestamp("2023-07-01", tz="UTC")


def _expected(weights: dict) -> pd.Series:
    return pd.Series(weights, dtype="float64")


def _actual(notionals: dict, value: float = 100.0) -> PortfolioState:
    positions = {s: Position(symbol=s, quantity=q) for s, q in notionals.items()}
    return PortfolioState(value=value, positions=positions)


def _only(report: ReconciliationReport, symbol: str) -> DiscrepancyKind:
    hits = [d for d in report.discrepancies if d.symbol == symbol]
    assert len(hits) == 1, report.discrepancies
    return hits[0].kind


def test_in_sync_has_no_discrepancies():
    rep = Reconciler().reconcile(_expected({"A": 0.5, "B": -0.5}), _actual({"A": 50.0, "B": -50.0}))
    assert rep.ok and rep.n_discrepancies == 0
    assert rep.synced_state.notional("A") == 50.0  # actual is the source of truth


def test_manual_close_detected_as_position_closed():
    rep = Reconciler().reconcile(_expected({"A": 0.5}), _actual({}))
    assert _only(rep, "A") is DiscrepancyKind.POSITION_CLOSED and not rep.ok


def test_take_profit_execution_leaves_flat_book():
    # long expected, position gone (closed at TP) -> structurally POSITION_CLOSED
    rep = Reconciler().reconcile(_expected({"A": 0.5}), _actual({}))
    assert _only(rep, "A") is DiscrepancyKind.POSITION_CLOSED


def test_stop_loss_execution_leaves_flat_book():
    rep = Reconciler().reconcile(_expected({"A": -0.5}), _actual({}))
    assert _only(rep, "A") is DiscrepancyKind.POSITION_CLOSED


def test_absent_position_when_none_expected_is_in_sync():
    rep = Reconciler().reconcile(_expected({"A": 0.0}), _actual({}))
    assert rep.ok


def test_partial_fill_detected_as_size_decreased():
    # targeted 0.5 weight, only half executed
    rep = Reconciler().reconcile(_expected({"A": 0.5}), _actual({"A": 25.0}))
    assert _only(rep, "A") is DiscrepancyKind.SIZE_DECREASED


def test_position_size_increase_detected():
    rep = Reconciler().reconcile(_expected({"A": 0.5}), _actual({"A": 70.0}))
    assert _only(rep, "A") is DiscrepancyKind.SIZE_INCREASED


def test_side_flip_detected():
    rep = Reconciler().reconcile(_expected({"A": 0.5}), _actual({"A": -50.0}))
    assert _only(rep, "A") is DiscrepancyKind.SIDE_FLIPPED


def test_unexpected_position_detected():
    rep = Reconciler().reconcile(_expected({}), _actual({"Z": 40.0}))
    assert _only(rep, "Z") is DiscrepancyKind.UNEXPECTED_POSITION


def test_errored_report_is_failed_and_not_ok():
    rep = ReconciliationReport.errored(ASOF, "broker down")
    assert rep.failed and not rep.ok and rep.message == "broker down"
