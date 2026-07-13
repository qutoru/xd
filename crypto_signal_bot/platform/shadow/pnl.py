"""Price PnL of a weight vector against a realized return cross-section."""

from __future__ import annotations

import pandas as pd


def price_pnl(weights: pd.Series, returns_row: pd.Series) -> float:
    """Sum_i w_i * r_i over the book's symbols (missing returns treated as 0)."""
    r = returns_row.reindex(weights.index).fillna(0.0)
    return float((weights * r).sum())
