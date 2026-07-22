"""ALX5 — honest maker-execution cost models on the FROZEN ALX daily book.

Per PREREGISTRATION.md this module ONLY:
  * feeds the frozen panel to the frozen backtester
    ``alpha_library.alx_funding_price_validation.validate.backtest`` (lag=1,
    k=0.30, 7-day funding signal — none of these are touched), and
  * applies the pre-committed cost models to the book's realized
    (gross, turnover) — it never re-implements the strategy or changes a signal
    parameter.

The single object under test is the cost/execution model. gross = price PnL +
funding carry is identical across every cost model; the models differ only in
(a) the per-turnover charge and (b) the fill haircut phi on gross.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx_funding_price_validation.validate import (
    backtest,
    weights_from_signal,
)

# --------------------------------------------------------------------------- #
# LOCKED cost models (round-trip bps on realized turnover) — see §4 of the
# pre-registration. Never adjust after seeing results.
# --------------------------------------------------------------------------- #
MED = 0.00075          # 7.5 bps — descriptive anchor (must reproduce ALX4 ~+1.94)
TAKER = 0.0011         # 11  bps — descriptive (expected weak)
IDEAL_MAKER = 0.0004   # 4   bps — descriptive ceiling (rigged-easy, NOT the gate)

# THE DECISIVE GATE: realistic maker = 4 bps base + 3 bps adverse-selection on
# turnover, AND gross scaled by phi = 0.80 (20% of intended edge missed on
# non-fills).
REAL_MAKER_COST = 0.0007   # 7 bps on turnover
REAL_MAKER_PHI = 0.80      # fill haircut on gross

# Stage-C robustness escalation (harsher fill reality): phi 0.70, +4 bps adverse.
HARSH_MAKER_COST = 0.0008  # 4 base + 4 adverse
HARSH_MAKER_PHI = 0.70

K_PCT = 0.30
LAG = 1


# --------------------------------------------------------------------------- #
# core cost-model arithmetic
# --------------------------------------------------------------------------- #
def apply_cost(price: np.ndarray, funding: np.ndarray, turnover: np.ndarray,
               cost: float, phi: float = 1.0) -> np.ndarray:
    """Net daily PnL under a cost model.

        net = phi * (price + funding) - turnover * cost

    ``phi`` is the non-fill haircut on the intended gross edge (both the price
    leg and the funding carry scale with the fraction of the position actually
    filled). ``cost`` is the round-trip charge per unit of realized turnover.
    """
    gross = np.asarray(price, dtype="float64") + np.asarray(funding, dtype="float64")
    return phi * gross - np.asarray(turnover, dtype="float64") * cost


def book_from_applied(applied: np.ndarray, ret: np.ndarray,
                      fund: np.ndarray) -> dict:
    """(price, funding, turnover) for an arbitrary applied-weight matrix.

    Mirrors the frozen backtester's PnL math exactly (validate.backtest lines
    64-67) so a held book is charged identically to the daily book.
    """
    price = np.nansum(applied * ret, axis=1)
    funding_pnl = -np.nansum(applied * np.nan_to_num(fund), axis=1)
    turn = np.abs(np.diff(applied, axis=0, prepend=0.0)).sum(axis=1)
    return {"price": price, "funding": funding_pnl, "turnover": turn}


# --------------------------------------------------------------------------- #
# frozen daily book (hold = 1) — thin wrapper over the frozen engine
# --------------------------------------------------------------------------- #
def frozen_book(ret: pd.DataFrame, signal: pd.DataFrame, fund: pd.DataFrame,
                *, lag: int = LAG) -> dict:
    """Run the frozen backtester and expose (price, funding, turnover, applied).

    Cost is irrelevant here (we recompute net per model); MED is passed only so
    ``res['net']`` is a ready MED reference identical to ALX4's book.
    """
    res = backtest(ret, signal, funding=fund, lag=lag, cost=MED, k_pct=K_PCT)
    return {"price": res["price"], "funding": res["funding"],
            "turnover": res["turnover"], "applied": res["applied"],
            "net_med": res["net"]}


# --------------------------------------------------------------------------- #
# held book (DESCRIPTIVE ONLY — cost×hold ladder; never gated)
# --------------------------------------------------------------------------- #
def held_applied(signal: pd.DataFrame, ret: pd.DataFrame, *, hold: int = 1,
                 lag: int = LAG, k_pct: float = K_PCT) -> np.ndarray:
    """Applied weights for an ``hold``-day rebalance of the frozen signal.

    Target weights are built each day exactly as the frozen engine does
    (``weights_from_signal``); with ``hold`` > 1 they are only refreshed every
    ``hold``-th day and carried in between (the engine holds target weights flat
    within a period — no intra-hold drift — so this matches its convention).
    ``hold`` = 1 reproduces the frozen daily book's applied weights bit-for-bit.
    """
    f = signal.reindex_like(ret).to_numpy(dtype="float64")
    T, N = f.shape
    target = np.zeros((T, N))
    last = np.zeros(N)
    for t in range(T):
        if t % hold == 0:
            last = weights_from_signal(f[t], k_pct)
        target[t] = last
    applied = target.copy() if lag == 0 else np.zeros_like(target)
    if lag > 0:
        applied[lag:] = target[:-lag]
    return applied
