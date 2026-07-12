"""ALX one-command falsification runner (see PREREGISTRATION.md).

Phase 1 (decisive): A reproducibility, B leakage/implementation, C known-factor
attribution. If Phase 1 is artifact-free it proceeds to Phase 2 (descriptive
robustness only — cannot reject or change anything). Prints all evidence and
writes artifacts to data/research/alx/. Single run, no tuning.

    py alpha_library/alx_funding_price_validation/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from crypto_signal_bot.data.storage import load_parquet
from crypto_signal_bot.research.xsection import features as ft
from crypto_signal_bot.research.xsection import portfolio as pf
from crypto_signal_bot.research.xsection.universe import (
    SURVIVORSHIP_SAFE,
    build_returns_panel,
    to_daily,
)
from alpha_library.alx_funding_price_validation.validate import (
    backtest,
    ols_hac,
    sharpe,
    weights_from_signal,
)

OUT = Path("data/research/alx")
FUNDING_PANEL = Path("data/research/e7/funding_panel.parquet")
MED = 0.00075
AL1_REFERENCE_NET_SHARPE = 1.71  # from AL-1 REPORT (the number we try to destroy)


def load_data():
    panel = build_returns_panel(SURVIVORSHIP_SAFE, "60")
    daily_close = to_daily(panel)
    daily_ret = daily_close.pct_change().dropna(how="all")
    daily_logret = np.log(daily_close / daily_close.shift(1)).dropna(how="all")
    fh = pd.read_parquet(FUNDING_PANEL)
    settle = fh[fh.index.hour.isin((0, 8, 16))]
    daily_fund = settle.resample("1D").sum().dropna(how="all")
    cols = list(daily_ret.columns)
    daily_fund = daily_fund.reindex(index=daily_ret.index, columns=cols)
    signal = (-daily_fund.rolling(7).mean()).reindex(index=daily_ret.index, columns=cols)
    turnover = pd.Series({s: float(np.median(load_parquet(s, "60")["turnover"].to_numpy())) for s in cols})
    return daily_ret, daily_logret, daily_fund, signal, turnover


def factor_books(daily_ret, daily_logret, daily_fund, turnover):
    """Six factor-mimicking books, identical construction, net @Med (price-cost)."""
    mkt = daily_ret.mean(axis=1)
    m1 = mkt.rolling(60).mean()
    var_m = (mkt.pow(2).rolling(60).mean() - m1.pow(2))
    cov = daily_ret.mul(mkt, axis=0).rolling(60).mean().sub(daily_ret.rolling(60).mean().mul(m1, axis=0))
    beta = cov.div(var_m, axis=0)
    facs = {
        "reversal": ft.short_term_reversal(daily_logret, 3),
        "momentum": ft.relative_strength(daily_logret, 24),
        "beta": beta,
        "volatility": daily_ret.rolling(30).std(),
        "size": pd.DataFrame(np.tile(turnover.values, (len(daily_ret), 1)),
                             index=daily_ret.index, columns=daily_ret.columns),
        "raw_funding": -daily_fund,
    }
    return {k: backtest(daily_ret, v, funding=None, lag=1, cost=MED)["net"] for k, v in facs.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    daily_ret, daily_logret, daily_fund, signal, turnover = load_data()
    logger.info("ALX validation | {} symbols x {} days", daily_ret.shape[1], daily_ret.shape[0])

    res = backtest(daily_ret, signal, funding=daily_fund, lag=1, cost=MED)
    net = res["net"]
    idx = daily_ret.index
    alpha_sharpe = sharpe(net)

    # ================= PHASE 1 — VERIFICATION (decisive) =================
    logger.info("=" * 64)
    logger.info("PHASE 1A — REPRODUCIBILITY")
    pf_res = pf.backtest_portfolio(daily_ret, signal, hold=1, rebalance=1, k_pct=0.30,
                                   exec_lag=1, cost=MED)
    shared_net = pf_res.net + res["funding"]  # shared engine (price-cost) + funding leg
    shared_sharpe = sharpe(shared_net)
    logger.info("  independent net Sharpe      = {:+.3f}", alpha_sharpe)
    logger.info("  shared-engine net Sharpe    = {:+.3f}", shared_sharpe)
    logger.info("  AL-1 reference net Sharpe   = {:+.3f}", AL1_REFERENCE_NET_SHARPE)
    reproduced = (np.sign(alpha_sharpe) > 0
                  and abs(alpha_sharpe - AL1_REFERENCE_NET_SHARPE) / AL1_REFERENCE_NET_SHARPE < 0.25
                  and abs(alpha_sharpe - shared_sharpe) < 0.2)
    logger.info("  -> reproduced across 3 paths: {}", reproduced)

    logger.info("PHASE 1B — LEAKAGE / IMPLEMENTATION")
    # future-corruption invariance (no look-ahead)
    t0 = len(net) // 2
    ret_corr = daily_ret.copy(); ret_corr.iloc[t0 + 1:] += 5.0
    net_corr = backtest(ret_corr, signal, funding=daily_fund, lag=1, cost=MED)["net"]
    no_lookahead = np.allclose(net[:t0], net_corr[:t0], atol=1e-12)
    # sign reversal must flip Sharpe
    rev_sharpe = sharpe(backtest(daily_ret, -signal, funding=daily_fund, lag=1, cost=MED)["net"])
    sign_flips = np.sign(rev_sharpe) != np.sign(alpha_sharpe)
    # random-signal placebo
    placebo = np.array([sharpe(backtest(daily_ret,
                        pd.DataFrame(rng.normal(size=daily_ret.shape), index=idx, columns=daily_ret.columns),
                        funding=daily_fund, lag=1, cost=MED)["net"]) for _ in range(200)])
    placebo_pct = float((placebo < alpha_sharpe).mean())
    placebo_clean = alpha_sharpe > np.max(placebo)
    # lag tell (0/1/2) — same-bar leak indicator
    lag_sh = {L: sharpe(backtest(daily_ret, signal, funding=daily_fund, lag=L, cost=MED)["net"]) for L in (0, 1, 2)}
    logger.info("  no-look-ahead invariance: {}", no_lookahead)
    logger.info("  sign-reversal flips Sharpe: {} (reversed={:+.2f})", sign_flips, rev_sharpe)
    logger.info("  placebo: alpha>{:.0%} of 200 random; max placebo={:+.2f}; clean={}",
                placebo_pct, float(placebo.max()), placebo_clean)
    logger.info("  lag Sharpe 0/1/2 = {:+.2f}/{:+.2f}/{:+.2f}", lag_sh[0], lag_sh[1], lag_sh[2])
    leakage_clean = no_lookahead and sign_flips and placebo_clean

    logger.info("PHASE 1C — KNOWN-FACTOR ATTRIBUTION")
    books = factor_books(daily_ret, daily_logret, daily_fund, turnover)
    names = list(books.keys())
    corrs = {k: float(np.corrcoef(net, v)[0, 1]) for k, v in books.items()}
    def _hedged_sharpe(fit_):
        """Sharpe of the factor-hedged position: intercept / std(residual)."""
        return float(fit_["beta"][0] / np.std(fit_["resid"], ddof=1) * np.sqrt(365))

    X = np.column_stack([np.ones(len(net))] + [books[k] for k in names])
    fit = ols_hac(net, X)
    inter_daily, inter_t = fit["beta"][0], fit["t"][0]
    hedged_sharpe = _hedged_sharpe(fit)
    # Supporting: control for the classic anomalies only (exclude funding itself,
    # which is the alpha's own economic variable, not a competing explanation).
    classic = [k for k in names if k != "raw_funding"]
    Xc = np.column_stack([np.ones(len(net))] + [books[k] for k in classic])
    fitc = ols_hac(net, Xc)
    logger.info("  pairwise corr with factors: {}", {k: round(v, 2) for k, v in corrs.items()})
    logger.info("  OLS R^2 (all 6 known factors explain) = {:.1%}", fit["r2"])
    for nm, b, tt in zip(names, fit["beta"][1:], fit["t"][1:]):
        logger.info("    loading {:<12} beta={:+.3f} (t={:+.2f})", nm, b, tt)
    logger.info("  UNEXPLAINED intercept (all 6): {:+.1%}/yr, NW t={:+.2f}, hedged Sharpe={:+.2f}",
                inter_daily * 365, inter_t, hedged_sharpe)
    logger.info("  UNEXPLAINED intercept (classic 5, ex-funding): {:+.1%}/yr, NW t={:+.2f}, hedged Sharpe={:+.2f}",
                fitc["beta"][0] * 365, fitc["t"][0], _hedged_sharpe(fitc))
    resid_sharpe = hedged_sharpe

    # Artifacts
    pd.DataFrame([{"alpha_net_sharpe": alpha_sharpe, "shared_net_sharpe": shared_sharpe,
                   "reproduced": reproduced, "no_lookahead": no_lookahead, "sign_flips": sign_flips,
                   "placebo_pct": placebo_pct, "placebo_clean": placebo_clean,
                   "lag0": lag_sh[0], "lag1": lag_sh[1], "lag2": lag_sh[2],
                   "ols_r2": fit["r2"], "intercept_daily": inter_daily,
                   "intercept_ann": inter_daily * 365, "intercept_t": inter_t,
                   "residual_sharpe": resid_sharpe, **{f"corr_{k}": v for k, v in corrs.items()}}]
                 ).to_csv(OUT / "phase1.csv", index=False)

    artifact_free = reproduced and leakage_clean
    logger.info("=" * 64)
    logger.info("PHASE 1 GATE (artifact-free = reproduced AND leakage-clean): {}", artifact_free)
    if not artifact_free:
        logger.info("PHASE 1 FAIL -> alpha rejected, STOP (no Phase 2).")
        return
    logger.info("Phase 1 artifact-free. Factor-explanation verdict rendered in REPORT.")
    logger.info("Proceeding to PHASE 2 (descriptive robustness; cannot reject/tune).")

    # ================= PHASE 2 — CHARACTERIZATION (descriptive only) =================
    for cname, cval in {"low": 0.0002, "med": MED, "high": 0.0015}.items():
        logger.info("  cost={:>4}: net Sharpe={:+.2f}", cname,
                    sharpe(backtest(daily_ret, signal, funding=daily_fund, lag=1, cost=cval)["net"]))
    logger.info("  exec-lag 1/2/3 net Sharpe: {}",
                {L: round(sharpe(backtest(daily_ret, signal, funding=daily_fund, lag=L, cost=MED)["net"]), 2)
                 for L in (1, 2, 3)})
    folds = np.array_split(np.arange(len(net)), 6)
    wf = [round(sharpe(net[ix]), 2) for ix in folds]
    logger.info("  walk-forward net Sharpe (6 folds): {} ({}/6 > 0)", wf, int(sum(v > 0 for v in wf)))
    net_s = pd.Series(net, index=idx)
    yby = {int(y): round(sharpe(g.to_numpy()), 2) for y, g in net_s.groupby(net_s.index.year)}
    logger.info("  year-by-year net Sharpe: {}", yby)
    mvol = daily_ret.mean(axis=1).rolling(30).std().shift(1).reindex(idx)
    q1, q2 = mvol.quantile([1/3, 2/3])
    lab = pd.Series("mid", index=idx); lab[mvol <= q1] = "low"; lab[mvol >= q2] = "high"
    logger.info("  vol-regime net Sharpe: {}",
                {r: round(sharpe(net_s[lab == r].to_numpy()), 2) for r in ("low", "mid", "high")})
    # subsets
    keep = [c for c in daily_ret.columns if c not in ("BTCUSDT", "ETHUSDT")]
    sh_nobtc = sharpe(backtest(daily_ret[keep], signal[keep], funding=daily_fund[keep], lag=1, cost=MED)["net"])
    top20 = list(turnover.sort_values().index[-20:])
    sh_liq = sharpe(backtest(daily_ret[top20], signal[top20], funding=daily_fund[top20], lag=1, cost=MED)["net"])
    logger.info("  drop BTC&ETH net Sharpe={:+.2f} | top-20 liquid net Sharpe={:+.2f}", sh_nobtc, sh_liq)
    # block bootstrap
    blocks = [net[i:i+20] for i in range(0, len(net) - 20)]
    boot = []
    for _ in range(1000):
        pick = rng.integers(0, len(blocks), size=len(net) // 20 + 1)
        boot.append(sharpe(np.concatenate([blocks[i] for i in pick])[:len(net)]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    logger.info("  block-bootstrap net Sharpe 95% CI: [{:+.2f}, {:+.2f}]", lo, hi)

    pd.DataFrame([{"wf": str(wf), "yby": str(yby), "no_btc_eth": sh_nobtc, "top20_liquid": sh_liq,
                   "boot_lo": lo, "boot_hi": hi}]).to_csv(OUT / "phase2.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(idx, np.cumsum(res["price"]), label="price PnL")
    ax.plot(idx, np.cumsum(res["funding"]), label="funding PnL")
    ax.plot(idx, np.cumsum(net), label="net PnL", color="k")
    ax.axhline(0, color="grey", lw=0.8); ax.legend(); ax.set_title("ALX funding-ranked book PnL")
    fig.tight_layout(); fig.savefig(OUT / "pnl.png", dpi=110); plt.close(fig)
    logger.info("Artifacts in {}", OUT)


if __name__ == "__main__":
    main()
