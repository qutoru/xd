"""ALX funding-rank sleeve — an independent Tier-B (shadow) alpha.

Builds the funding-ranked dollar-neutral daily book entirely from production's
own machinery (:mod:`crypto_signal_bot.platform.core`) and its own funding
signal. The net return includes both the price leg (from the book engine) and
the funding cash-flow leg, matching the full economics of the strategy — no
simplification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform import config
from crypto_signal_bot.platform.core.metrics import sharpe
from crypto_signal_bot.platform.core.portfolio import build_book, tranche_weights
from crypto_signal_bot.platform.signal import funding_score


class AlxSleeve:
    """Funding-ranked market-neutral daily sleeve (Tier-B, shadow only)."""

    name = "alx_funding_rank"

    def __init__(
        self,
        *,
        lookback: int = config.ALX_LOOKBACK_DAYS,
        k_pct: float = config.ALX_K_PCT,
        hold: int = config.ALX_HOLD,
        rebalance: int = config.ALX_REBALANCE,
        exec_lag: int = config.ALX_EXEC_LAG,
        tier: str = config.ALX_TIER,
    ) -> None:
        self.lookback = lookback
        self.k_pct = k_pct
        self.hold = hold
        self.rebalance = rebalance
        self.exec_lag = exec_lag
        self.tier = tier

    def signal(self, funding_panel: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional score panel (higher = go long)."""
        return funding_score(funding_panel, self.lookback)

    def target_book(self, funding_panel: pd.DataFrame) -> pd.Series:
        """Dollar-neutral target weights for the next fill (latest signal row)."""
        sig = self.signal(funding_panel)
        weights = tranche_weights(sig.iloc[-1].to_numpy(dtype="float64"), self.k_pct)
        return pd.Series(weights, index=funding_panel.columns, name="target_weight")

    def shadow_replay(
        self,
        returns: pd.DataFrame,
        funding_panel: pd.DataFrame,
        *,
        cost: float = config.COST_MED,
    ) -> dict:
        """Replay the book over history for shadow accounting.

        net = price leg (post-cost) + funding cash-flow leg. Funding paid by
        longs to shorts, so a long position pays funding -> negative sign.
        """
        sig = self.signal(funding_panel)
        res = build_book(
            returns,
            sig,
            hold=self.hold,
            rebalance=self.rebalance,
            k_pct=self.k_pct,
            exec_lag=self.exec_lag,
            cost=cost,
        )
        fund = funding_panel.reindex_like(returns).to_numpy(dtype="float64")
        funding_pnl = -np.nansum(res.weights * np.nan_to_num(fund), axis=1)
        net = res.net + funding_pnl
        return {
            "price_net": res.net,
            "funding_pnl": funding_pnl,
            "net": net,
            "turnover": res.turnover,
            "weights": res.weights,
            "net_sharpe": sharpe(net, config.DAILY_INTERVAL_MIN),
        }
