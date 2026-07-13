"""Contracts between platform layers.

An :class:`AlphaSleeve` is one independent signal producer. It turns a
cross-sectional input panel into (a) a per-name score and (b) a dollar-neutral
target book. The combiner (future) consumes many sleeves through this contract
without knowing their internals — this is the seam that lets new alphas plug in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class AlphaSleeve(Protocol):
    """A single, independent cross-sectional alpha."""

    name: str
    tier: str  # "A" production-weighted, "B" shadow-only, "R" research-only

    def signal(self, funding_panel: pd.DataFrame) -> pd.DataFrame:
        """(time x symbol) cross-sectional score; higher = go long."""
        ...

    def target_book(self, funding_panel: pd.DataFrame) -> pd.Series:
        """Dollar-neutral target weights for the next fill, indexed by symbol."""
        ...
