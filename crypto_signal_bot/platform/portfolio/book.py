"""The TargetBook — the Portfolio Layer's output artifact."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class TargetBook:
    """A dollar-neutral target book of per-symbol weights as of ``asof``.

    ``contributions`` holds each signal's (budget-weighted) score contribution to
    the combined score, for attribution — never required for execution.
    """

    asof: pd.Timestamp
    weights: pd.Series  # index=symbol, target weights
    contributions: dict[str, pd.Series] = field(default_factory=dict)

    @property
    def gross(self) -> float:
        return float(self.weights.abs().sum())

    @property
    def net(self) -> float:
        return float(self.weights.sum())

    @property
    def symbols(self) -> list[str]:
        return list(self.weights.index)
