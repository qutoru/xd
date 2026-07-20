"""ALX4 — known-answer tests for the regime/falsification primitives (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx4_regime_analysis import characterize as ch
from alpha_library.alx4_regime_analysis import forward as fwd


def _idx(T):
    return pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")


def test_rank_ic_detects_perfect_signal():
    # signal(t-1) ranks == ret(t) ranks -> IC ~ +1.
    T, N = 30, 6
    idx = _idx(T)
    base = np.tile(np.arange(N, dtype="float64"), (T, 1))
    ret = pd.DataFrame(base, index=idx, columns=[f"S{i}" for i in range(N)])
    signal = ret.shift(-1)  # signal(t-1) equals ret(t) by construction
    ic = ch.rank_ic_series(signal, ret).dropna()
    assert ic.mean() > 0.95


def test_nw_tstat_sign_and_zero():
    assert ch.nw_tstat(np.ones(200) * 0.01) > 5      # strong positive mean
    assert abs(ch.nw_tstat(np.zeros(200))) == 0.0     # no variation -> 0


def test_shuffle_placebo_detects_real_funding_effect():
    # Construct a world where -funding predicts return: low-funding names win.
    T, N = 260, 6
    idx = _idx(T)
    rng = np.random.default_rng(42)
    f_base = np.linspace(-1, 1, N) * 1e-3            # fixed per-symbol funding level
    fund = pd.DataFrame(f_base + rng.normal(0, 1e-5, (T, N)),
                        index=idx, columns=[f"S{i}" for i in range(N)])
    ret = pd.DataFrame(-f_base * 0.5 + rng.normal(0, 5e-4, (T, N)),
                       index=idx, columns=fund.columns)
    res = ch.shuffle_placebo(ret, fund, n=100, seed=0)
    assert res["pass"] is True
    assert res["real_sharpe"] > res["max"]            # real beats every shuffle
    assert res["median"] < res["real_sharpe"]


def test_shuffle_placebo_has_power_rejects_noise():
    # Pure noise: funding independent of returns -> no real edge; placebo must NOT pass.
    T, N = 260, 6
    idx = _idx(T)
    rng = np.random.default_rng(7)
    fund = pd.DataFrame(rng.normal(0, 1e-3, (T, N)), index=idx, columns=[f"S{i}" for i in range(N)])
    ret = pd.DataFrame(rng.normal(0, 5e-4, (T, N)), index=idx, columns=fund.columns)
    res = ch.shuffle_placebo(ret, fund, n=100, seed=0)
    assert res["pass"] is False                       # test has power (not always-pass)
    assert res["real_sharpe"] <= res["max"]


def test_conditional_table_shape():
    T, N = 120, 5
    idx = _idx(T)
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=idx, columns=[f"S{i}" for i in range(N)])
    signal = pd.DataFrame(rng.normal(0, 1e-3, (T, N)), index=idx, columns=ret.columns)
    net = pd.Series(rng.normal(0, 0.01, T), index=idx)
    ic = pd.Series(rng.normal(0, 0.05, T), index=idx)
    reg = ch.regime_features(ret, signal)
    table = ch.conditional_table(net, ic, reg)
    assert set(table) == {"mkt_vol", "fund_disp", "cs_disp", "trend_60d", "breadth"}
    for buckets in table.values():
        assert set(buckets) == {"low", "mid", "high"}


# ------------------------- forward harness ---------------------------------

def _ledger(dates, net, ic, mkt):
    idx = pd.DatetimeIndex(dates, tz="UTC", name="date")
    return pd.DataFrame({"net": net, "ic": ic, "mkt_ret": mkt}, index=idx)


def test_forward_evaluate_empty_is_observing():
    res = fwd.evaluate(_ledger([], [], [], []))
    assert res["status"] == "OBSERVING" and res["n_days"] == 0


def test_forward_ledger_append_dedup_keeps_first(tmp_path, monkeypatch):
    monkeypatch.setattr(fwd, "LEDGER", tmp_path / "ledger.parquet")
    d = pd.date_range("2026-07-21", periods=3, freq="D", tz="UTC")
    fwd.append_ledger(_ledger(d, [0.01, 0.02, 0.03], [0.0]*3, [0.0]*3))
    # re-append overlapping dates with different values -> must NOT overwrite
    fwd.append_ledger(_ledger(d[1:], [9.0, 9.0], [0.0]*2, [0.0]*2))
    led = fwd.load_ledger()
    assert len(led) == 3
    assert led["net"].tolist() == [0.01, 0.02, 0.03]


def test_forward_fail_on_drawdown():
    d = pd.date_range("2026-07-21", periods=60, freq="D", tz="UTC")
    res = fwd.evaluate(_ledger(d, [-0.01]*60, [0.0]*60, [0.0]*60))  # eq falls >30%
    assert res["status"] == "FAIL" and "drawdown" in res["message"]


def test_forward_fail_on_bull_ic_flip():
    d = pd.date_range("2026-07-21", periods=60, freq="D", tz="UTC")
    res = fwd.evaluate(_ledger(d, [0.0005]*60, [-0.05]*60, [0.01]*60))  # bull, IC<0
    assert res["status"] == "FAIL" and "bull" in res["message"]


def test_forward_observing_before_horizon():
    d = pd.date_range("2026-07-21", periods=100, freq="D", tz="UTC")
    res = fwd.evaluate(_ledger(d, [0.001]*100, [0.03]*100, [0.01]*100))
    assert res["status"] == "OBSERVING" and res["n_days"] == 100


def test_forward_confirmed_when_criteria_met():
    T = 400
    d = pd.date_range("2026-07-21", periods=T, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    net = 0.002 + rng.normal(0, 0.004, T)          # strongly positive Sharpe
    ic = np.where(np.arange(T) < 200, 0.05, 0.02)   # positive, incl. bull half
    mkt = np.where(np.arange(T) < 200, 0.01, -0.01) # bull then bear/flat
    res = fwd.evaluate(_ledger(d, net, ic, mkt))
    assert res["has_bull"] and res["has_bear"]
    assert res["ci_lo"] > 0
    assert res["status"] == "CONFIRMED"


def test_forward_compute_rows_only_post_lock_and_closed():
    idx = pd.date_range("2026-06-01", periods=55, freq="D", tz="UTC")  # >40d warmup
    syms = [f"S{i}" for i in range(5)]
    rng = np.random.default_rng(2)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (55, 5)), axis=0),
                         index=idx, columns=syms)
    dfund = pd.DataFrame(rng.normal(0, 1e-4, (55, 5)), index=idx, columns=syms)
    rows = fwd.compute_forward_rows(close, dfund, now=pd.Timestamp("2026-07-26", tz="UTC"))
    assert (rows.index > fwd.LOCK_DATE).all()            # scores only post-lock
    assert rows.index.max() <= pd.Timestamp("2026-07-25", tz="UTC")  # excludes forming 07-26
