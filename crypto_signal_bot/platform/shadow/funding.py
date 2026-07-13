"""Funding PnL of a weight vector against a realized funding cross-section.

Convention: positive funding = longs pay shorts, so a long position (w>0) paying
positive funding is a cost -> negative PnL. Hence the sign flip.
"""

from __future__ import annotations

import pandas as pd


def funding_pnl(weights: pd.Series, funding_row: pd.Series) -> float:
    """-Sum_i w_i * funding_i (longs pay funding; shorts receive it)."""
    f = funding_row.reindex(weights.index).fillna(0.0)
    return float(-(weights * f).sum())
