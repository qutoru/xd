"""E9 one-command runner: cross-asset lead-lag (BTC leader).

Falsification-first. Executes exactly the pre-registered pipeline (see
PREREGISTRATION.md): Stage A cost-anchored dispersion, Stage B rank-IC gates at
h=1 (B1-B4, the 1-bar-gap kill-test is decisive), and — only if a feature clears
Stage B — Stage C survival under costs. Prints a verdict against the pre-committed
PASS/FAIL bar and writes artifacts to data/research/e9/. No parameter tuning.

Reproduce with:
    py experiments/e9/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make both the project root (crypto_signal_bot) and the experiments package importable.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from crypto_signal_bot.config import FEE_RATE
from crypto_signal_bot.research.metrics import bars_per_year
from crypto_signal_bot.research.xsection import features as ft
from crypto_signal_bot.research.xsection import ic as ic_mod
from crypto_signal_bot.research.xsection import portfolio as pf
from crypto_signal_bot.research.xsection.universe import SURVIVORSHIP_SAFE, build_returns_panel
from experiments.e9.features_leadlag import lead_lag_impulse, lead_lag_sustained

OUT = Path("data/research/e9")
INTERVAL = "60"
INTERVAL_MIN = 60
LEADER = "BTCUSDT"

# --- Pre-registered constants (see PREREGISTRATION.md) ---
IR_TARGET = 1.0
TC = 0.5
NW_TSTAT_GATE = 3.0
BINOM_P_GATE = 0.05
PRIMARY_HORIZON = 1
DECAY_HORIZONS = [1, 2, 4, 8, 24]

# Stage C grid (declared for E9).
HOLD = [1, 2, 4, 6, 12, 24]
REBAL = [1, 2, 4]
KPCT = [0.10, 0.20, 0.30]
LAG = [1, 2]
COSTS = {"low": 0.0002, "med": 0.00075, "high": 0.0015}
REP = dict(hold=4, rebalance=1, k_pct=0.20, exec_lag=1)

# Pre-committed PASS bar.
NET_SHARPE_GATE = 1.0
WF_POS_GATE = 4          # of 6 folds
CAPACITY_GATE = 10_000_000.0


def effective_breadth(resid: pd.DataFrame) -> tuple[float, float]:
    """Participation-ratio effective breadth and PC1 variance share."""
    corr = np.nan_to_num(resid.corr().to_numpy(), nan=0.0)
    eig = np.linalg.eigvalsh(corr)
    eig = eig[eig > 1e-10]
    n_eff = float((eig.sum() ** 2) / (eig ** 2).sum())
    pc1 = float(eig.max() / eig.sum())
    return n_eff, pc1


def ic_required(n_eff: float, bpy: float, horizon: int) -> float:
    """B1 threshold from the Fundamental Law: IR = IC * TC * sqrt(BR)."""
    br = n_eff * (bpy / horizon)
    return IR_TARGET / (TC * np.sqrt(br))


def _cs_corr_with_reversal(feat: pd.DataFrame, followers: pd.DataFrame) -> float:
    """Mean per-bar cross-sectional correlation with the E7 reversal feature.

    'Not reversal in disguise' guard: near-zero |corr| shows the lead-lag signal
    is structurally distinct from own-return reversal. Diagnostic only.
    """
    rev = ft.short_term_reversal(followers)
    a, b = feat.align(rev, join="inner")
    corrs = []
    for i in range(len(a)):
        x, y = a.iloc[i].to_numpy(), b.iloc[i].to_numpy()
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() >= 5 and x[m].std() > 0 and y[m].std() > 0:
            corrs.append(np.corrcoef(x[m], y[m])[0, 1])
    return float(np.mean(corrs)) if corrs else float("nan")


def analyse_feature(name: str, feat: pd.DataFrame, followers: pd.DataFrame,
                    ic_req: float, nw_lag: int) -> dict:
    """Full Stage-B diagnostic panel for one feature at the primary horizon."""
    target = ft.forward_target(followers, PRIMARY_HORIZON)
    ic = ic_mod.rank_ic_series(feat, target)
    summ = ic_mod.summarize_ic(ic, nw_lag=nw_lag)
    folds, p_binom = ic_mod.walk_forward_folds(ic)
    regimes = ic_mod.regime_breakdown(ic, followers)

    # B4 kill-test: predict t+2 from the t signal (one extra bar of delay).
    gap_target = ft.forward_target(followers, PRIMARY_HORIZON).shift(-1)
    ic_gap = ic_mod.rank_ic_series(feat, gap_target)
    summ_gap = ic_mod.summarize_ic(ic_gap, nw_lag=nw_lag)

    decay = {h: float(ic_mod.rank_ic_series(feat, ft.forward_target(followers, h)).mean())
             for h in DECAY_HORIZONS}
    rev_corr = _cs_corr_with_reversal(feat, followers)

    b1 = abs(summ["mean_ic"]) >= ic_req
    b2 = abs(summ["nw_tstat"]) >= NW_TSTAT_GATE
    b3 = p_binom < BINOM_P_GATE
    b4 = (abs(summ_gap["mean_ic"]) >= ic_req) and (abs(summ_gap["nw_tstat"]) >= NW_TSTAT_GATE)

    return {"name": name, "ic": ic, "summ": summ, "folds": folds, "p_binom": p_binom,
            "regimes": regimes, "summ_gap": summ_gap, "decay": decay, "rev_corr": rev_corr,
            "ic_req": ic_req, "gates": {"B1": b1, "B2": b2, "B3": b3, "B4": b4}}


def _report_feature(res: dict) -> None:
    s, g = res["summ"], res["gates"]
    logger.info("=== {} ===", res["name"])
    logger.info("  mean_IC={:+.4f} (req |IC|>={:.4f})  IC_IR={:+.3f}  NW_t={:+.2f}  AC1={:+.2f}  n={}",
                s["mean_ic"], res["ic_req"], s["ic_ir"], s["nw_tstat"], s["ac1"], s["n_obs"])
    logger.info("  walk-forward fold mean_IC: {} | binom p={:.3f}",
                [round(v, 4) for v in res["folds"]["mean_ic"]], res["p_binom"])
    logger.info("  regimes: {}", {r["regime"]: round(r["mean_ic"], 4) for _, r in res["regimes"].iterrows()})
    logger.info("  1-bar-GAP IC={:+.4f} NW_t={:+.2f} (B4 kill-test — decisive)",
                res["summ_gap"]["mean_ic"], res["summ_gap"]["nw_tstat"])
    logger.info("  IC decay by horizon: {}", {h: round(v, 4) for h, v in res["decay"].items()})
    logger.info("  cross-sec corr with E7 reversal = {:+.3f} (guard: should be near 0)", res["rev_corr"])
    verdict = "PASS" if all(g.values()) else "FAIL"
    logger.info("  GATES B1={B1} B2={B2} B3={B3} B4={B4} -> {v}", v=verdict, **g)


def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    sd = x.std(ddof=1) if x.size > 1 else 0.0
    return float(x.mean() / sd * np.sqrt(bars_per_year(INTERVAL_MIN))) if sd > 0 else 0.0


def run_grid(simple: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, r, k, lag, (cname, cval) in itertools.product(HOLD, REBAL, KPCT, LAG, COSTS.items()):
        if h < r:
            continue
        res = pf.backtest_portfolio(simple, feat, hold=h, rebalance=r, k_pct=k, exec_lag=lag, cost=cval)
        m = pf.portfolio_metrics(res, INTERVAL_MIN)
        rows.append({"hold": h, "rebalance": r, "k_pct": k, "exec_lag": lag, "cost": cname, **m})
    return pd.DataFrame(rows)


def deep_dive(simple: pd.DataFrame, feat: pd.DataFrame, cost: float, index) -> pd.DataFrame:
    """Walk-forward net Sharpe (6 folds) for the representative config."""
    res = pf.backtest_portfolio(simple, feat, cost=cost, **REP)
    net = pd.Series(res.net, index=index)
    folds = np.array_split(np.arange(len(net)), 6)
    return pd.DataFrame([{"fold": i + 1, "net_sharpe": _sharpe(net.iloc[ix].to_numpy())}
                         for i, ix in enumerate(folds)])


def capacity_estimate(univ_dollar_per_bar: float, turnover_annual: float) -> float:
    tpb = turnover_annual / bars_per_year(INTERVAL_MIN)
    return float("inf") if tpb <= 0 else 0.01 * univ_dollar_per_bar / tpb


def main() -> None:
    from crypto_signal_bot.data.storage import load_parquet

    OUT.mkdir(parents=True, exist_ok=True)
    logger.info("E9 cross-asset lead-lag | leader={} | interval={}m", LEADER, INTERVAL_MIN)

    panel = build_returns_panel(SURVIVORSHIP_SAFE, INTERVAL)
    if LEADER not in panel.columns:
        raise SystemExit(f"Leader {LEADER} missing from panel — cannot run E9.")
    returns = ft.log_returns(panel).dropna(how="all")
    leader = returns[LEADER]
    followers = returns.drop(columns=[LEADER])
    follower_panel = panel.drop(columns=[LEADER])
    logger.info("Followers: {} symbols x {} bars (leader excluded)", followers.shape[1], followers.shape[0])

    bpy = bars_per_year(INTERVAL_MIN)

    # --- Stage A: cost-anchored dispersion (signal-agnostic sanity) ---
    disp = ic_mod.cross_sectional_dispersion(followers).median()
    round_trip = 2 * FEE_RATE
    stage_a = disp >= 2 * round_trip
    logger.info("STAGE A: median CS dispersion={:.4%} vs 2x round-trip={:.4%} -> {}",
                disp, 2 * round_trip, "PASS" if stage_a else "FAIL")

    resid = ft.residualize(followers).dropna(how="all")
    n_eff, pc1 = effective_breadth(resid)
    ic_req = ic_required(n_eff, bpy, PRIMARY_HORIZON)
    logger.info("Effective breadth N_eff={:.1f} (PC1={:.1%}) -> required |IC|>={:.4f} at h={}",
                n_eff, pc1, ic_req, PRIMARY_HORIZON)

    # --- Stage B: the two pre-registered lead-lag features ---
    feats = {
        "lead_lag_impulse": lead_lag_impulse(followers, leader),
        "lead_lag_sustained": lead_lag_sustained(followers, leader),
    }
    nw_lag = max(5, PRIMARY_HORIZON)
    results, rows = [], []
    for name, feat in feats.items():
        r = analyse_feature(name, feat, followers, ic_req, nw_lag)
        _report_feature(r)
        results.append(r)
        r["ic"].to_frame("ic").to_csv(OUT / f"ic_series_{name}.csv")
        s = r["summ"]
        rows.append({"feature": name, "mean_ic": s["mean_ic"], "ic_req": ic_req,
                     "ic_ir": s["ic_ir"], "nw_tstat": s["nw_tstat"], "ac1": s["ac1"],
                     "p_binom": r["p_binom"], "gap_ic": r["summ_gap"]["mean_ic"],
                     "gap_nw_t": r["summ_gap"]["nw_tstat"], "rev_corr": r["rev_corr"],
                     **r["gates"]})
    summary = pd.DataFrame(rows)
    summary.insert(0, "stage_a_pass", stage_a)
    summary.insert(1, "n_eff", round(n_eff, 1))
    summary.to_csv(OUT / "stage_b_summary.csv", index=False)

    # IC distribution + walk-forward plots.
    fig, ax = plt.subplots(figsize=(8, 4))
    for r in results:
        ax.hist(r["ic"].to_numpy(), bins=60, alpha=0.5, label=r["name"])
    ax.axvline(0, color="k", lw=0.8); ax.set_title("E9 per-bar rank-IC distribution")
    ax.set_xlabel("IC"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "ic_distribution.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for r in results:
        ax.plot(r["folds"]["fold"], r["folds"]["mean_ic"], marker="o", label=r["name"])
    ax.axhline(0, color="k", lw=0.8); ax.set_title("E9 walk-forward mean IC per fold")
    ax.set_xlabel("fold"); ax.set_ylabel("mean IC"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "walkforward_ic.png", dpi=110); plt.close(fig)

    passers = [r for r in results if all(r["gates"].values())]
    logger.info("=" * 60)
    if not passers:
        logger.info("E9 STAGE B: neither lead-lag feature passes B1-B4 "
                    "-> H1 FALSIFIED at Stage B. STOP (no Stage C).")
        logger.info("Artifacts in {}", OUT)
        return

    # --- Stage C: survival under costs (only reached if Stage B passed) ---
    chosen = next((r for r in passers if r["name"] == "lead_lag_impulse"), passers[0])
    logger.info("STAGE B PASSED for: {} -> proceeding to Stage C on '{}'",
                [r["name"] for r in passers], chosen["name"])
    feat = feats[chosen["name"]]
    simple = follower_panel.pct_change().dropna(how="all")

    grid = run_grid(simple, feat)
    grid.to_csv(OUT / "stagec_grid.csv", index=False)
    med = grid[grid["cost"] == "med"].sort_values("net_sharpe", ascending=False)
    best_med_net = float(med["net_sharpe"].max())
    n_pos_med = int((med["net_sharpe"] > 0).sum())
    gbest = grid.loc[grid["gross_sharpe"].idxmax()]
    logger.info("Best GROSS Sharpe over grid: {:+.2f} (n gross>0: {}/{})",
                gbest.gross_sharpe, int((grid["gross_sharpe"] > 0).sum()), len(grid))
    logger.info("Net Sharpe>0 @Med: {}/{} configs (best={:+.2f})", n_pos_med, len(med), best_med_net)
    for _, r in med.head(6).iterrows():
        logger.info("  H={:>2} R={} K={:.0%} lag={} | gross={:+.2f} net={:+.2f} turn/yr={:.0f} costDrag={:+.1%}",
                    int(r.hold), int(r.rebalance), r.k_pct, int(r.exec_lag),
                    r.gross_sharpe, r.net_sharpe, r.turnover_annual, r.cost_drag_annual)

    wf = deep_dive(simple, feat, COSTS["med"], simple.index)
    wf.to_csv(OUT / "stagec_walkforward.csv", index=False)
    wf_pos = int((wf["net_sharpe"] > 0).sum())
    logger.info("Walk-forward net Sharpe (rep@med): {} -> {}/6 folds > 0",
                [round(v, 2) for v in wf["net_sharpe"]], wf_pos)

    vols = {s: float(np.median(load_parquet(s, INTERVAL)["turnover"].to_numpy())) for s in follower_panel.columns}
    rep_res = pf.backtest_portfolio(simple, feat, cost=COSTS["med"], **REP)
    rep_m = pf.portfolio_metrics(rep_res, INTERVAL_MIN)
    cap = capacity_estimate(float(sum(vols.values())), rep_m["turnover_annual"])
    logger.info("Capacity estimate (rep config, 1% participation): ~${:,.0f}", cap)

    passed = best_med_net > NET_SHARPE_GATE and wf_pos >= WF_POS_GATE and cap >= CAPACITY_GATE
    logger.info("=" * 60)
    logger.info("E9 PRE-COMMITTED PASS = (best net Sharpe>{:.1f} @Med AND WF>0 in>={}/6 AND cap>=${:,.0f})",
                NET_SHARPE_GATE, WF_POS_GATE, CAPACITY_GATE)
    logger.info("  best_net={:+.2f} | WF>0 folds={}/6 | capacity=${:,.0f} -> {}",
                best_med_net, wf_pos, cap,
                "PASS" if passed else "FAIL -> H1 falsified at Stage C, STOP")
    logger.info("Artifacts in {}", OUT)


if __name__ == "__main__":
    main()
