"""Portfolio builder — thin orchestrator over the component modules.

Pipeline (each step lives in its own module; the builder only sequences them):

    normalize each signal  -> combine by risk budget -> allocate score->weights
    -> per-name cap -> net-neutralize -> scale gross -> turnover limit -> TargetBook

Alpha-agnostic and execution-agnostic: input is universal ``SignalOutput``,
output is a ``TargetBook``. No knowledge of Bybit, brokers or orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from crypto_signal_bot.platform.portfolio import (
    allocator,
    combiner,
    constraints,
    normalization,
    turnover,
)
from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.signals.base import SignalOutput


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio construction knobs (no strategy parameters)."""

    alloc: str = "topk"  # "topk" | "proportional"
    k_pct: float = 0.30
    gross_target: float = 1.0
    max_weight: float | None = None  # per-name cap, as a fraction of gross
    max_turnover: float | None = None  # one-step sum|Δw| cap


def build_target_book(
    signals: Mapping[str, SignalOutput],
    *,
    budgets: dict[str, float] | None = None,
    prev_book: TargetBook | None = None,
    config: PortfolioConfig = PortfolioConfig(),
) -> TargetBook:
    """Compose the pipeline into a dollar-neutral :class:`TargetBook`."""
    if not signals:
        raise ValueError("no signals supplied to the portfolio builder")

    asof = next(iter(signals.values())).asof

    normalized = {name: normalization.zscore(out.values) for name, out in signals.items()}
    combined, contributions = combiner.combine(normalized, budgets)

    if config.alloc == "proportional":
        weights = allocator.proportional(combined)
    else:
        weights = allocator.dollar_neutral_topk(combined, config.k_pct)

    if config.max_weight is not None:
        weights = constraints.max_position(weights, config.max_weight)
    weights = constraints.enforce_net(weights, 0.0)
    weights = constraints.scale_gross(weights, config.gross_target)

    if config.max_turnover is not None:
        prev = prev_book.weights if prev_book is not None else None
        weights = turnover.limit_turnover(weights, prev, config.max_turnover)

    return TargetBook(asof=asof, weights=weights, contributions=contributions)
