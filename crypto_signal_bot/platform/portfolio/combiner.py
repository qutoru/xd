"""Combine several normalized signals into one score, weighted by risk budget.

Alpha-agnostic: works on ``{name: normalized Series}``. Missing symbols in a
given signal are treated as neutral (0). Budgets are normalized to sum(|b|)=1;
if omitted, signals are equal-weighted.
"""

from __future__ import annotations

import pandas as pd


def resolve_budgets(names: list[str], budgets: dict[str, float] | None) -> dict[str, float]:
    """Normalize risk budgets to sum(|b|) = 1 (equal-weight if not given)."""
    if not names:
        return {}
    if budgets is None:
        return {n: 1.0 / len(names) for n in names}
    total = sum(abs(budgets.get(n, 0.0)) for n in names)
    if total == 0:
        return {n: 1.0 / len(names) for n in names}
    return {n: budgets.get(n, 0.0) / total for n in names}


def combine(
    normalized: dict[str, pd.Series],
    budgets: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Budget-weighted sum of normalized signals over the union of symbols.

    Returns the combined score and each signal's weighted contribution.
    """
    names = list(normalized)
    b = resolve_budgets(names, budgets)

    index = pd.Index([])
    for s in normalized.values():
        index = index.union(s.index)

    contributions = {
        n: normalized[n].reindex(index).fillna(0.0) * b[n] for n in names
    }
    combined = pd.Series(0.0, index=index)
    for c in contributions.values():
        combined = combined + c
    return combined, contributions
