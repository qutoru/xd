"""ALX8 — momentum sleeve + funding blend on the frozen 40-major panel.

Feeds panels to the FROZEN backtester (`validate.backtest`) unchanged. The two
signals are reused verbatim: funding = -7d-mean (ALX5), momentum = frozen
`relative_strength(logret, 24)` (ALX4 lookback, NOT tuned). New machinery only:
the ALX5 realistic-maker net, the 50/50 weight-level blend with netting, and the
random-sleeve placebo. All params frozen in PREREGISTRATION.md §3/§4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx_funding_price_validation.validate import backtest, weights_from_signal
from alpha_library.alx4_regime_analysis import characterize as ch
from crypto_signal_bot.research.xsection import features as ft

PHI = 0.80
COST = 0.0007        # realistic-maker 7 bps round-trip
K_PCT = 0.30
LAG = 1
MOM_LOOKBACK = 24    # frozen (ALX4), NOT tuned


def momentum_signal(ret: pd.DataFrame) -> pd.DataFrame:
    """Frozen cross-sectional momentum: relative_strength(log1p(ret), 24)."""
    logret = np.log1p(ret)
    return ft.relative_strength(logret, MOM_LOOKBACK)


def sleeve(ret: pd.DataFrame, signal: pd.DataFrame, fund: pd.DataFrame, *, lag: int = LAG):
    """Frozen book for one sleeve (cost applied later). Returns backtest dict + ic."""
    res = backtest(ret, signal, funding=fund, lag=lag, cost=0.0, k_pct=K_PCT)
    res["ic"] = ch.rank_ic_series(signal, ret)
    return res


def realistic_net(res: dict, *, phi: float = PHI, cost: float = COST) -> np.ndarray:
    """ALX5 realistic-maker net for a single sleeve's book."""
    return phi * (res["price"] + res["funding"]) - res["turnover"] * cost


def blend_net(applied_a: np.ndarray, applied_b: np.ndarray, ret_np: np.ndarray,
              fund_np: np.ndarray, *, w: float = 0.5, phi: float = PHI,
              cost: float = COST):
    """50/50 weight-level blend with netting -> realistic-maker net (honest single book)."""
    comb = w * applied_a + (1.0 - w) * applied_b
    price = np.nansum(comb * ret_np, axis=1)
    funding = -np.nansum(comb * np.nan_to_num(fund_np), axis=1)
    turn = np.abs(np.diff(comb, axis=0, prepend=0.0)).sum(axis=1)
    return phi * (price + funding) - turn * cost, comb


def random_applied(ret: pd.DataFrame, seed: int, *, lag: int = LAG) -> np.ndarray:
    """Applied weights of a random cross-sectional signal (placebo), matched
    top/bottom-30% construction, only on valid (non-NaN return) names."""
    r = ret.to_numpy(dtype="float64")
    T, N = r.shape
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal((T, N))
    sig[np.isnan(r)] = np.nan                      # trade only names with a return
    w = np.zeros((T, N))
    for t in range(T):
        w[t] = weights_from_signal(sig[t], K_PCT)
    applied = np.zeros_like(w)
    applied[lag:] = w[:-lag]
    return applied
