"""Known-answer unit tests for the E9 cross-asset lead-lag feature."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.e9.features_leadlag import (
    lead_lag_impulse,
    lead_lag_sustained,
    rolling_beta,
)


def _returns(n=400, m=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    cols = [f"S{i}" for i in range(m)]
    return pd.DataFrame(rng.normal(0, 0.01, size=(n, m)), index=idx, columns=cols)


def test_rolling_beta_recovers_known_constant_beta():
    # Construct followers as exact multiples of the leader (+ tiny noise-free):
    # r_i = k_i * r_L  => OLS beta must equal k_i.
    idx = pd.date_range("2023-01-01", periods=300, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    leader = pd.Series(rng.normal(0, 0.01, size=300), index=idx)
    ks = {"A": 0.5, "B": 1.0, "C": 2.0}
    followers = pd.DataFrame({c: leader * k for c, k in ks.items()})
    beta = rolling_beta(followers, leader, window=168)
    last = beta.iloc[-1]
    for c, k in ks.items():
        assert abs(last[c] - k) < 1e-9


def test_impulse_demeaning_sums_to_zero():
    r = _returns()
    leader = r["S0"]
    followers = r.drop(columns=["S0"])
    feat = lead_lag_impulse(followers, leader, window=168)
    valid = feat.dropna(how="any")
    assert len(valid) > 0
    assert np.allclose(valid.sum(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_sustained_demeaning_sums_to_zero():
    r = _returns(seed=2)
    leader = r["S0"]
    followers = r.drop(columns=["S0"])
    feat = lead_lag_sustained(followers, leader, window=168, agg=4)
    valid = feat.dropna(how="any")
    assert len(valid) > 0
    assert np.allclose(valid.sum(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_feature_has_no_future_lookahead():
    # The feature at bar t must not change if all data strictly after t is altered.
    r = _returns(seed=3)
    leader = r["S0"]
    followers = r.drop(columns=["S0"])
    t = 250
    full = lead_lag_impulse(followers, leader, window=168)

    r2 = r.copy()
    r2.iloc[t + 1:] += 5.0  # arbitrarily corrupt the future
    leader2, followers2 = r2["S0"], r2.drop(columns=["S0"])
    truncated = lead_lag_impulse(followers2, leader2, window=168)

    a, b = full.iloc[t], truncated.iloc[t]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, atol=1e-12)


def test_beta_window_warmup_is_nan():
    r = _returns(seed=4)
    leader = r["S0"]
    followers = r.drop(columns=["S0"])
    beta = rolling_beta(followers, leader, window=168)
    # First window-1 rows have insufficient data -> NaN.
    assert beta.iloc[:167].isna().all().all()
    assert beta.iloc[167:].notna().any().any()
