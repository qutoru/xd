"""Unit tests for the cross-sectional research utilities (known-answer)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.research.xsection import features as ft
from crypto_signal_bot.research.xsection import ic as ic_mod


def _panel(n=200, m=10, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    cols = [f"S{i}" for i in range(m)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, m)), axis=0))
    return pd.DataFrame(prices, index=idx, columns=cols)


def test_residualize_rows_sum_zero():
    r = ft.log_returns(_panel()).dropna()
    resid = ft.residualize(r)
    # Each bar's cross-sectional residuals must sum to ~0 by construction.
    assert np.allclose(resid.sum(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_rank_ic_perfect_and_inverse():
    n, m = 50, 8
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    cols = [f"S{i}" for i in range(m)]
    rng = np.random.default_rng(1)
    feat = pd.DataFrame(rng.normal(size=(n, m)), index=idx, columns=cols)
    # target equals feature -> IC = +1 ; target = -feature -> IC = -1.
    assert ic_mod.rank_ic_series(feat, feat).mean() > 0.999
    assert ic_mod.rank_ic_series(feat, -feat).mean() < -0.999


def test_rank_ic_random_is_near_zero():
    n, m = 400, 12
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    cols = [f"S{i}" for i in range(m)]
    rng = np.random.default_rng(2)
    feat = pd.DataFrame(rng.normal(size=(n, m)), index=idx, columns=cols)
    tgt = pd.DataFrame(rng.normal(size=(n, m)), index=idx, columns=cols)
    assert abs(ic_mod.rank_ic_series(feat, tgt).mean()) < 0.05


def test_newey_west_matches_ols_for_iid():
    rng = np.random.default_rng(3)
    x = rng.normal(0.5, 1.0, size=5000)
    t_nw = ic_mod.newey_west_tstat(x, lag=0)
    t_ols = x.mean() / (x.std(ddof=1) / np.sqrt(x.size))
    assert abs(t_nw - t_ols) / abs(t_ols) < 0.02


def test_newey_west_widens_se_under_autocorrelation():
    # Positively autocorrelated series -> HAC SE larger -> smaller t than iid.
    rng = np.random.default_rng(4)
    e = rng.normal(0, 1, 4000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, x.size):
        x[i] = 0.7 * x[i - 1] + e[i]
    x = x + 0.3
    assert abs(ic_mod.newey_west_tstat(x, lag=20)) < abs(ic_mod.newey_west_tstat(x, lag=0))


def test_funding_dispersion_sign():
    # Highest funding -> most negative feature (fade crowded longs).
    idx = pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC")
    f = pd.DataFrame({"A": [0.01, 0.0, -0.01], "B": [-0.01, 0.0, 0.01]}, index=idx)
    feat = ft.funding_dispersion(f)
    assert (feat["A"] < 0).iloc[0] and (feat["B"] > 0).iloc[0]


def test_forward_target_has_no_current_bar_lookahead():
    r = ft.log_returns(_panel()).dropna()
    fwd = ft.forward_target(r, horizon=4)
    # The last `horizon` rows cannot be resolved -> must be NaN.
    assert fwd.iloc[-4:].isna().all().all()


def test_portfolio_weights_dollar_neutral_gross_one():
    from crypto_signal_bot.research.xsection.portfolio import _tranche_weights
    import numpy as np
    feat = np.array([0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5])
    w = _tranche_weights(feat, 0.30)  # top/bottom 30% = 3 names each side
    assert abs(w.sum()) < 1e-12          # dollar neutral
    assert abs(np.abs(w).sum() - 1.0) < 1e-12  # gross exposure = 1
    assert (w[:3] > 0).all() and (w[-3:] < 0).all()  # long high feature, short low


def test_to_daily_resamples_last_close():
    from crypto_signal_bot.research.xsection.universe import to_daily
    idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")  # 2 UTC days
    panel = pd.DataFrame({"A": np.arange(48.0)}, index=idx)
    d = to_daily(panel)
    assert len(d) == 2                       # two daily bars
    assert d["A"].iloc[0] == 23.0            # last hourly close of day 1
    assert d["A"].iloc[1] == 47.0            # last hourly close of day 2
