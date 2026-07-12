"""Known-answer unit tests for the ALX2 adversarial battery.

These guard the mechanics of the attacks (not the alpha): the point-in-time
funding path has no forward leak, liquidity-aware cost is never cheaper than the
baseline, the deflated-Sharpe formula behaves correctly, and the new T4 factor
(downside beta) recovers a known loading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx2_adversarial_replication import adversarial as adv


def _synthetic(T=400, N=8, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"S{i}" for i in range(N)]
    ret = pd.DataFrame(rng.normal(0, 0.02, (T, N)), index=idx, columns=cols)
    fund = pd.DataFrame(rng.normal(0, 1e-4, (T, N)), index=idx, columns=cols)
    return ret, fund, idx, cols


def test_pit_funding_has_no_forward_leak():
    """Corrupting funding AFTER t0 must not change PnL through t0 (point-in-time)."""
    ret, fund, idx, cols = _synthetic()
    sig = adv._signal_from_funding(fund)
    net = adv.backtest(ret, sig, funding=fund, lag=1, cost=adv.MED)["net"]
    t0 = len(net) // 2
    fund_c = fund.copy()
    fund_c.iloc[t0 + 1:] += 1.0
    net_c = adv.backtest(ret, adv._signal_from_funding(fund_c), funding=fund_c,
                         lag=1, cost=adv.MED)["net"]
    assert np.allclose(net[:t0], net_c[:t0], atol=1e-12)


def test_liquidity_cost_never_cheaper_than_baseline():
    """T5 effective cost per unit turnover must be >= the flat 7.5 bps baseline."""
    ret, fund, idx, cols = _synthetic()
    sig = adv._signal_from_funding(fund)
    adv_daily = pd.Series({c: 1.0e9 for c in cols})  # generous liquidity
    r = adv.t5_liquidity_costs(ret, sig, fund, adv_daily)
    assert r["avg_effective_cost_bps"] >= r["baseline_bps"] - 1e-9


def test_liquidity_cost_penalizes_illiquidity():
    """Lower ADV -> strictly higher effective cost (impact term grows)."""
    ret, fund, idx, cols = _synthetic()
    sig = adv._signal_from_funding(fund)
    liquid = adv.t5_liquidity_costs(ret, sig, fund, pd.Series({c: 1.0e10 for c in cols}))
    illiquid = adv.t5_liquidity_costs(ret, sig, fund, pd.Series({c: 1.0e7 for c in cols}))
    assert illiquid["avg_effective_cost_bps"] > liquid["avg_effective_cost_bps"]


def test_dsr_monotonic_in_sharpe():
    """A stronger, cleaner return stream yields a higher deflated Sharpe."""
    rng = np.random.default_rng(0)
    weak = rng.normal(0.0002, 0.02, 1000)
    strong = rng.normal(0.0020, 0.02, 1000)
    dsr_weak = adv.t6_deflated_sharpe(weak)["dsr"]
    dsr_strong = adv.t6_deflated_sharpe(strong)["dsr"]
    assert 0.0 <= dsr_weak <= 1.0 and 0.0 <= dsr_strong <= 1.0
    assert dsr_strong > dsr_weak


def test_dsr_kills_zero_skill():
    """A near-zero-mean stream must not clear the DSR > 0.95 bar."""
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.02, 1000)
    assert adv.t6_deflated_sharpe(noise)["dsr"] <= 0.95


def test_downside_beta_recovers_known_loading():
    """If r_i = 2*mkt on all days, rolling downside beta ~ 2."""
    T, N = 300, 5
    idx = pd.date_range("2023-01-01", periods=T, freq="D", tz="UTC")
    mkt = pd.Series(np.random.default_rng(7).normal(0, 0.02, T), index=idx)
    ret = pd.DataFrame({f"S{i}": 2.0 * mkt for i in range(N)}, index=idx)
    db = adv._downside_beta(ret, mkt, window=60).iloc[100:]
    assert abs(np.nanmedian(db.to_numpy()) - 2.0) < 0.1
