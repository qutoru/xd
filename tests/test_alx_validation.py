"""Known-answer unit tests for the ALX independent validation primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx_funding_price_validation.validate import (
    backtest,
    ols_hac,
    weights_from_signal,
)


def test_weights_dollar_neutral_gross_one():
    sig = np.array([0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5])
    w = weights_from_signal(sig, 0.30)  # top/bottom 30% = 3 each
    assert abs(w.sum()) < 1e-12
    assert abs(np.abs(w).sum() - 1.0) < 1e-12
    assert (w[:3] > 0).all() and (w[-3:] < 0).all()


def test_backtest_no_lookahead():
    idx = pd.date_range("2023-01-01", periods=30, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.normal(0, 0.01, (30, 5)), index=idx, columns=list("ABCDE"))
    sig = pd.DataFrame(rng.normal(size=(30, 5)), index=idx, columns=list("ABCDE"))
    base = backtest(ret, sig, lag=1, cost=0.0)["net"]
    ret2 = ret.copy(); ret2.iloc[15:] += 9.0
    pert = backtest(ret2, sig, lag=1, cost=0.0)["net"]
    assert np.allclose(base[:15], pert[:15], atol=1e-12)


def test_sign_reversal_flips_pnl():
    idx = pd.date_range("2023-01-01", periods=40, freq="1D", tz="UTC")
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(rng.normal(0, 0.01, (40, 6)), index=idx, columns=list("ABCDEF"))
    sig = pd.DataFrame(rng.normal(size=(40, 6)), index=idx, columns=list("ABCDEF"))
    a = backtest(ret, sig, lag=1, cost=0.0)["net"]
    b = backtest(ret, -sig, lag=1, cost=0.0)["net"]
    assert np.allclose(a, -b, atol=1e-12)  # gross price PnL is exactly antisymmetric


def test_ols_hac_recovers_known_alpha():
    rng = np.random.default_rng(2)
    n = 4000
    factor = rng.normal(0, 1, n)
    alpha_true = 0.5
    y = alpha_true + 2.0 * factor + rng.normal(0, 0.1, n)
    X = np.column_stack([np.ones(n), factor])
    fit = ols_hac(y, X)
    assert abs(fit["beta"][0] - alpha_true) < 0.02   # intercept
    assert abs(fit["beta"][1] - 2.0) < 0.02          # loading
    assert fit["t"][0] > 5                            # alpha strongly significant
    assert fit["r2"] > 0.99


def test_ols_hac_zero_alpha_insignificant():
    rng = np.random.default_rng(3)
    n = 4000
    factor = rng.normal(0, 1, n)
    y = 1.5 * factor + rng.normal(0, 0.1, n)  # no intercept
    X = np.column_stack([np.ones(n), factor])
    fit = ols_hac(y, X)
    assert abs(fit["beta"][0]) < 0.02
    assert abs(fit["t"][0]) < 3
