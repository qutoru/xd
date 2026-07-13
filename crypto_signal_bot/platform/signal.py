"""Production cross-sectional funding signal (own implementation).

The ALX score is the negative L-day mean daily funding: persistently rich
funding = crowded longs -> expected relative underperformance, so a high score
(go long) corresponds to persistently *cheap/negative* funding. This is a plain
transform owned by production; it imports nothing from research.
"""

from __future__ import annotations

import pandas as pd


def funding_score(funding_panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """(time x symbol) score = -(rolling ``lookback``-day mean funding).

    Higher score = go long. Ranking is order-based downstream, so no explicit
    cross-sectional demeaning is required (it is order-preserving within a bar).
    """
    return -funding_panel.rolling(lookback).mean()
