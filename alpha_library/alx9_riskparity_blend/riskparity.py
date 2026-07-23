"""ALX9 — no-look-ahead risk-parity blend of the frozen funding + momentum sleeves.

Reuses the neutral sleeve mechanics from ALX8 (`momentum` module: frozen signals,
realistic-maker net, random placebo) and adds ONE thing: inverse-vol
(risk-parity) blend weights estimated on trailing, lagged sleeve vols. All params
frozen in PREREGISTRATION.md §4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx8_momentum_sleeve import momentum as mo

VOL_WINDOW = 60      # frozen vol horizon
PHI = mo.PHI
COST = mo.COST


def rp_weights(net_a: np.ndarray, net_b: np.ndarray, *, window: int = VOL_WINDOW):
    """Gross-preserving inverse-vol weights (a'+b'=1), lagged 1 day (no look-ahead).

    Vol estimated on each sleeve's own realistic-maker net over a trailing
    `window`, shifted by 1. Warm-up / undefined vol -> 50/50 fallback.
    """
    va = pd.Series(net_a).rolling(window).std().shift(1).to_numpy()
    vb = pd.Series(net_b).rolling(window).std().shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = 1.0 / va, 1.0 / vb
        tot = a + b
        af, bf = a / tot, b / tot
    bad = ~np.isfinite(af) | ~np.isfinite(bf)
    af = np.where(bad, 0.5, af)
    bf = np.where(bad, 0.5, bf)
    return af, bf


def book_from_applied(applied: np.ndarray, ret_np: np.ndarray, fund_np: np.ndarray,
                      *, phi: float = PHI, cost: float = COST) -> np.ndarray:
    """Realistic-maker net of an arbitrary applied-weight book."""
    price = np.nansum(applied * ret_np, axis=1)
    funding = -np.nansum(applied * np.nan_to_num(fund_np), axis=1)
    turn = np.abs(np.diff(applied, axis=0, prepend=0.0)).sum(axis=1)
    return phi * (price + funding) - turn * cost


def rp_blend_net(applied_a: np.ndarray, applied_b: np.ndarray,
                 net_a: np.ndarray, net_b: np.ndarray,
                 ret_np: np.ndarray, fund_np: np.ndarray, *,
                 window: int = VOL_WINDOW, phi: float = PHI, cost: float = COST):
    """Risk-parity weight-level blend with netting -> realistic-maker net."""
    af, bf = rp_weights(net_a, net_b, window=window)
    comb = af[:, None] * applied_a + bf[:, None] * applied_b
    net = book_from_applied(comb, ret_np, fund_np, phi=phi, cost=cost)
    return net, comb, af, bf
