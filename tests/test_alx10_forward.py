"""Known-answer unit tests for the ALX10 forward-paper harness (NOT the alpha).

Guard the protocol plumbing: only post-lock bars are scored, the ledger is
append-only/immutable, the horizon+regime gate returns OBSERVING, and the locked
early-FAIL drawdown trip fires. No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_library.alx10_forward_paper import forward as fwd
from alpha_library.alx3_external_replication import data_sources as ds


def _panel(T=220, N=10, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0),
                         index=idx, columns=cols)
    dfund = pd.DataFrame(rng.normal(0, 1e-4, (T, N)), index=idx, columns=cols)
    return close, dfund, idx


def test_scores_only_post_lock_bars(monkeypatch):
    close, dfund, idx = _panel()
    monkeypatch.setattr(fwd, "LOCK_DATE", idx[120])
    monkeypatch.setattr(fwd, "WARMUP_DAYS", 90)
    rows = fwd.compute_forward_rows(close, dfund, now=idx[-1] + pd.Timedelta(days=1))
    assert len(rows) > 0
    assert (rows.index > idx[120]).all()          # nothing on/before the lock is scored
    assert list(rows.columns) == fwd._COLS


def test_ledger_is_append_only_immutable(monkeypatch, tmp_path):
    monkeypatch.setattr(fwd, "LEDGER", tmp_path / "ledger.parquet")
    d = pd.DatetimeIndex(pd.to_datetime(["2026-08-01", "2026-08-02"]), name="date").tz_localize("UTC")
    first = pd.DataFrame({"net_A": [0.01, 0.02], "net_B": [0.01, 0.02],
                          "ic_A": [0.1, 0.1], "mkt_ret": [0.0, 0.0]}, index=d)
    fwd.append_ledger(first)
    # re-append same dates with DIFFERENT values -> must be ignored (immutable)
    clash = first.copy(); clash["net_A"] = [9.0, 9.0]
    merged = fwd.append_ledger(clash)
    assert merged.loc[d[0], "net_A"] == 0.01
    assert len(merged) == 2


def test_evaluate_observing_before_horizon():
    n = 60
    idx = pd.date_range("2026-07-24", periods=n, freq="D", tz="UTC")
    led = pd.DataFrame({"net_A": np.full(n, 0.001), "net_B": np.full(n, 0.001),
                        "ic_A": np.full(n, 0.02),
                        "mkt_ret": np.linspace(-0.01, 0.01, n)}, index=idx)
    res = fwd.evaluate(led)
    assert res["status"] == "OBSERVING"
    assert res["n_days"] == n


def test_refresh_cache_drops_only_target_venue_files(monkeypatch, tmp_path):
    """refresh_cache removes this venue's kl/fund parquets for the given bases and
    leaves unrelated cache files (onboard, other symbols) untouched."""
    monkeypatch.setattr(ds, "CACHE", tmp_path)
    for name in ("binance_kl_BTCUSDT", "binance_fund_BTCUSDT",
                 "binance_kl_ETHUSDT", "binance_fund_ETHUSDT",
                 "binance_onboard", "binance_kl_ZZZUSDT"):
        (tmp_path / f"{name}.parquet").write_bytes(b"x")
    removed = fwd.refresh_cache(["BTC", "ETH"])
    assert removed == 4
    assert not (tmp_path / "binance_kl_BTCUSDT.parquet").exists()
    assert (tmp_path / "binance_onboard.parquet").exists()      # untouched
    assert (tmp_path / "binance_kl_ZZZUSDT.parquet").exists()   # non-target symbol untouched


def test_evaluate_early_fail_drawdown():
    n = 40
    idx = pd.date_range("2026-07-24", periods=n, freq="D", tz="UTC")
    led = pd.DataFrame({"net_A": np.full(n, -0.05), "net_B": np.full(n, -0.05),
                        "ic_A": np.full(n, 0.0),
                        "mkt_ret": np.zeros(n)}, index=idx)
    res = fwd.evaluate(led)
    assert res["status"] == "FAIL"
    assert "drawdown" in res["message"]
