"""RiskManager — sizes portfolio weights into per-symbol target notionals.

The single place that decides *how big* a position is. It converts a dollar-neutral
``TargetBook`` (weights) into signed target notionals using account NAV and a
``RiskConfig`` of exposure limits, so Execution trades the portfolio's risk
position rather than a fixed dollar size.

Independent by construction: imports only pandas and the portfolio TargetBook it
consumes — never Execution, Broker, Bybit, Research or Strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_signal_bot.platform.portfolio.book import TargetBook


@dataclass(frozen=True)
class RiskConfig:
    """Exposure limits and leverage for sizing (all in NAV/notional terms)."""

    gross_target: float = 1.0          # leverage multiple applied to book weights
    max_gross: float = 1.0             # cap on total gross exposure, as a fraction of NAV
    max_position_pct: float = 0.20     # cap per symbol, as a fraction of NAV
    max_symbol_notional: float = float("inf")  # absolute per-symbol notional cap
    min_notional: float = 5.0          # drop positions smaller than this notional


class RiskManager:
    """Maps ``TargetBook`` weights + NAV -> risk-adjusted signed target notionals."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def size(self, book: TargetBook, nav: float) -> pd.Series:
        """Return signed target notional per symbol (min-notional filtered).

        weight x gross_target x NAV, then: cap each name by max_position_pct and
        max_symbol_notional, scale all names down proportionally if total gross
        exceeds max_gross, and finally drop any name below min_notional.
        """
        cfg = self.config
        if nav <= 0:
            return pd.Series(dtype="float64")

        notional = book.weights.astype("float64") * cfg.gross_target * nav

        # Per-symbol caps (whichever binds first): % of NAV and absolute notional.
        per_symbol_cap = min(cfg.max_position_pct * nav, cfg.max_symbol_notional)
        notional = notional.clip(lower=-per_symbol_cap, upper=per_symbol_cap)

        # Total gross cap: scale every position proportionally if exceeded.
        gross = float(notional.abs().sum())
        gross_cap = cfg.max_gross * nav
        if gross_cap > 0 and gross > gross_cap:
            notional = notional * (gross_cap / gross)

        # Drop dust: positions too small to trade produce no intent (no junk TP/SL).
        return notional[notional.abs() >= cfg.min_notional]
