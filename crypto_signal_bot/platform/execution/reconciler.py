"""Reconciler — compares expected platform holdings against actual broker state.

Runs after each trading cycle: it takes the target weights the platform *intended*
to hold and the *actual* :class:`PortfolioState` reported by the Broker (for
``BybitBroker`` that is the real Bybit account), classifies any per-symbol
divergence, logs a warning, and returns the actual state as the synchronized
source of truth. It is venue-agnostic — it speaks only Broker abstractions
(PortfolioState) and pandas, never Bybit — so it stays on the execution-core side
of the boundary.

State alone cannot attribute *why* a position closed (take-profit vs stop-loss vs
manual): that needs fill data. Such closures are reported structurally as
POSITION_CLOSED; finer attribution is deliberately left to a future fills-aware step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
from loguru import logger

from crypto_signal_bot.platform.execution.domain import PortfolioState


class DiscrepancyKind(str, Enum):
    """How an actual position diverges from what the platform expected."""

    POSITION_CLOSED = "position_closed"      # expected a position, none held (TP/SL/manual/none)
    UNEXPECTED_POSITION = "unexpected_position"  # holding something we did not target
    SIDE_FLIPPED = "side_flipped"            # long vs short mismatch
    SIZE_DECREASED = "size_decreased"        # partial fill / position shrank
    SIZE_INCREASED = "size_increased"        # holding more than targeted


@dataclass(frozen=True)
class Discrepancy:
    """One symbol's expected-vs-actual mismatch."""

    symbol: str
    kind: DiscrepancyKind
    expected: float  # expected weight
    actual: float    # actual weight


@dataclass(frozen=True)
class ReconciliationReport:
    """Outcome of one reconciliation pass."""

    asof: pd.Timestamp | None
    synced_state: PortfolioState
    discrepancies: tuple[Discrepancy, ...] = ()
    failed: bool = False  # True when actual state could not be obtained
    message: str = ""

    @property
    def ok(self) -> bool:
        """True when the platform and the exchange agree (and nothing errored)."""
        return not self.failed and not self.discrepancies

    @property
    def n_discrepancies(self) -> int:
        return len(self.discrepancies)

    def kinds(self) -> set[DiscrepancyKind]:
        return {d.kind for d in self.discrepancies}

    @classmethod
    def errored(cls, asof, message: str) -> ReconciliationReport:
        """A reconciliation that could not run (e.g. the broker/API failed)."""
        return cls(asof=asof, synced_state=PortfolioState(), failed=True, message=message)


class Reconciler:
    """Classifies divergence between target weights and actual broker holdings."""

    def __init__(self, tol: float = 1e-4) -> None:
        self.tol = tol

    def reconcile(
        self,
        expected_weights: pd.Series,
        actual: PortfolioState,
        *,
        asof: pd.Timestamp | None = None,
    ) -> ReconciliationReport:
        """Compare expected weights vs the broker's actual state; log divergences."""
        actual_w = actual.weights()
        index = expected_weights.index.union(actual_w.index)
        exp = expected_weights.reindex(index).fillna(0.0)
        act = actual_w.reindex(index).fillna(0.0)

        discrepancies: list[Discrepancy] = []
        for symbol in index:
            kind = self._classify(float(exp[symbol]), float(act[symbol]))
            if kind is None:
                continue
            disc = Discrepancy(symbol, kind, float(exp[symbol]), float(act[symbol]))
            discrepancies.append(disc)
            logger.warning(
                "reconcile {}: {} (expected weight {:.4f}, actual {:.4f})",
                symbol, kind.value, disc.expected, disc.actual,
            )
        return ReconciliationReport(
            asof=asof, synced_state=actual, discrepancies=tuple(discrepancies)
        )

    def _classify(self, expected: float, actual: float) -> DiscrepancyKind | None:
        e_zero = abs(expected) <= self.tol
        a_zero = abs(actual) <= self.tol
        if e_zero and a_zero:
            return None
        if a_zero:
            return DiscrepancyKind.POSITION_CLOSED
        if e_zero:
            return DiscrepancyKind.UNEXPECTED_POSITION
        if (expected > 0) != (actual > 0):
            return DiscrepancyKind.SIDE_FLIPPED
        if abs(actual) < abs(expected) * (1.0 - self.tol):
            return DiscrepancyKind.SIZE_DECREASED
        if abs(actual) > abs(expected) * (1.0 + self.tol):
            return DiscrepancyKind.SIZE_INCREASED
        return None
