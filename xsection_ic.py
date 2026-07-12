"""E7 Stage A/B runner: cross-sectional rank-IC diagnostics on BTC-major panel.

Falsification-first. Computes only the pre-registered statistics and checks them
against the revised gates. NO portfolio construction. Emits a console report,
CSV files, and plots. If Stage B gates fail, that is the answer — we stop.

Usage:
    py xsection_ic.py --interval 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from crypto_signal_bot.config import FEE_RATE
from crypto_signal_bot.research.xsection import features as ft
from crypto_signal_bot.research.xsection import ic as ic_mod
from crypto_signal_bot.research.xsection.universe import (
    SURVIVORSHIP_SAFE,
    build_returns_panel,
    to_daily,
)

OUT = Path("data/research/e7")

# Pre-registered constants (see RESEARCH.md E7).
IR_TARGET = 1.0          # target annual information ratio (Sharpe ~1)
TC = 0.5                 # transfer coefficient (Grinold implementation loss)
NW_TSTAT_GATE = 3.0      # Harvey-Liu-Zhu (2016)
BINOM_P_GATE = 0.05
PRIMARY_HORIZON = 1      # gates evaluated at 1-bar forward (pre-committed)
DECAY_HORIZONS = [1, 2, 4, 8, 24]


def effective_breadth(resid: pd.DataFrame) -> tuple[float, float]:
    """Participation-ratio effective breadth and PC1 variance share."""
    corr = resid.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    eig = np.linalg.eigvalsh(corr)
    eig = eig[eig > 1e-10]
    n_eff = float((eig.sum() ** 2) / (eig ** 2).sum())
    pc1 = float(eig.max() / eig.sum())
    return n_eff, pc1


def ic_required(n_eff: float, bars_per_year: float, horizon: int) -> float:
    """B1 threshold derived from IR = IC * TC * sqrt(BR)."""
    br = n_eff * (bars_per_year / horizon)
    return IR_TARGET / (TC * np.sqrt(br))


def analyse_feature(name: str, feat: pd.DataFrame, returns: pd.DataFrame,
                    ic_req: float, nw_lag: int) -> dict:
    """Full Stage-B diagnostic panel for one feature at the primary horizon."""
    target = ft.forward_target(returns, PRIMARY_HORIZON)
    ic = ic_mod.rank_ic_series(feat, target)
    summ = ic_mod.summarize_ic(ic, nw_lag=nw_lag)

    folds, p_binom = ic_mod.walk_forward_folds(ic)
    regimes = ic_mod.regime_breakdown(ic, returns)

    # Microstructure kill-test: skip one bar between decision and target start.
    gap_target = ft.forward_target(returns, PRIMARY_HORIZON).shift(-1)
    ic_gap = ic_mod.rank_ic_series(feat, gap_target)
    summ_gap = ic_mod.summarize_ic(ic_gap, nw_lag=nw_lag)

    # IC decay across horizons (diagnostic only).
    decay = {}
    for h in DECAY_HORIZONS:
        icd = ic_mod.rank_ic_series(feat, ft.forward_target(returns, h))
        decay[h] = float(icd.mean())

    b1 = abs(summ["mean_ic"]) >= ic_req
    b2 = abs(summ["nw_tstat"]) >= NW_TSTAT_GATE
    b3 = p_binom < BINOM_P_GATE
    b4 = (abs(summ_gap["mean_ic"]) >= ic_req) and (abs(summ_gap["nw_tstat"]) >= NW_TSTAT_GATE)

    return {"name": name, "ic": ic, "summ": summ, "folds": folds, "p_binom": p_binom,
            "regimes": regimes, "summ_gap": summ_gap, "decay": decay,
            "ic_req": ic_req, "gates": {"B1": b1, "B2": b2, "B3": b3, "B4": b4}}


def _report(res: dict) -> None:
    s, g = res["summ"], res["gates"]
    logger.info("=== {} ===", res["name"])
    logger.info("  mean_IC={:+.4f} (req |IC|>={:.4f})  IC_IR={:+.3f}  NW_t={:+.2f}  AC1={:+.2f}  n={}",
                s["mean_ic"], res["ic_req"], s["ic_ir"], s["nw_tstat"], s["ac1"], s["n_obs"])
    logger.info("  walk-forward folds mean_IC: {} | binom p={:.3f}",
                [round(v, 4) for v in res["folds"]["mean_ic"]], res["p_binom"])
    logger.info("  regimes: {}", {r["regime"]: round(r["mean_ic"], 4) for _, r in res["regimes"].iterrows()})
    logger.info("  1-bar-gap IC={:+.4f} NW_t={:+.2f} (microstructure kill-test)",
                res["summ_gap"]["mean_ic"], res["summ_gap"]["nw_tstat"])
    logger.info("  IC decay by horizon: {}", {h: round(v, 4) for h, v in res["decay"].items()})
    verdict = "PASS" if all(g.values()) else "FAIL"
    logger.info("  GATES B1={B1} B2={B2} B3={B3} B4={B4} -> {v}", v=verdict, **g)


def _plots(results: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # IC distribution.
    fig, ax = plt.subplots(figsize=(8, 4))
    for r in results:
        ax.hist(r["ic"].to_numpy(), bins=60, alpha=0.5, label=r["name"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Per-bar rank-IC distribution"); ax.set_xlabel("IC"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ic_distribution.png", dpi=110); plt.close(fig)

    # Walk-forward stability.
    fig, ax = plt.subplots(figsize=(8, 4))
    for r in results:
        ax.plot(r["folds"]["fold"], r["folds"]["mean_ic"], marker="o", label=r["name"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Walk-forward mean IC per fold"); ax.set_xlabel("fold"); ax.set_ylabel("mean IC"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "walkforward_ic.png", dpi=110); plt.close(fig)


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--daily", action="store_true",
                    help="Resample the 1h panel to daily bars (E8, low-frequency).")
    args = ap.parse_args()

    if args.daily:
        interval_min = 24 * 60
        OUT = Path("data/research/e8")
        panel = to_daily(build_returns_panel(SURVIVORSHIP_SAFE, "60"))
    else:
        interval_min = args.interval
        panel = build_returns_panel(SURVIVORSHIP_SAFE, str(args.interval))
    bars_per_year = 365 * 24 * 60 / interval_min
    OUT.mkdir(parents=True, exist_ok=True)
    returns = ft.log_returns(panel).dropna(how="all")
    resid = ft.residualize(returns).dropna(how="all")

    # --- Stage A: cost-anchored dispersion ---
    disp = ic_mod.cross_sectional_dispersion(returns).median()
    round_trip = 2 * FEE_RATE
    stage_a = disp >= 2 * round_trip
    logger.info("STAGE A: median CS dispersion={:.4%} vs 2x round-trip cost={:.4%} -> {}",
                disp, 2 * round_trip, "PASS" if stage_a else "FAIL")

    n_eff, pc1 = effective_breadth(resid)
    ic_req = ic_required(n_eff, bars_per_year, PRIMARY_HORIZON)
    logger.info("Effective breadth N_eff={:.1f} (PC1 share={:.1%}) -> required |IC|>={:.4f} at h={}",
                n_eff, pc1, ic_req, PRIMARY_HORIZON)

    # --- Stage B: features ---
    feats = {
        "relative_strength": ft.relative_strength(returns),
        "short_term_reversal": ft.short_term_reversal(returns),
    }
    fpath = Path("data/research/e7/funding_panel.parquet")
    if args.daily:
        logger.info("Daily mode: testing only the classic factors (momentum/reversal)")
    elif fpath.exists():
        funding = pd.read_parquet(fpath)
        feats["funding_dispersion"] = ft.funding_dispersion(funding.reindex_like(returns))
    else:
        logger.warning("No funding panel at {} -> funding_dispersion skipped (data not assembled)", fpath)

    nw_lag = max(5, PRIMARY_HORIZON)
    results = []
    rows = []
    for name, feat in feats.items():
        r = analyse_feature(name, feat, returns, ic_req, nw_lag)
        _report(r)
        results.append(r)
        r["ic"].to_frame("ic").to_csv(OUT / f"ic_series_{name}.csv")
        s = r["summ"]
        rows.append({"feature": name, "mean_ic": s["mean_ic"], "ic_req": ic_req,
                     "ic_ir": s["ic_ir"], "nw_tstat": s["nw_tstat"], "ac1": s["ac1"],
                     "p_binom": r["p_binom"], "gap_ic": r["summ_gap"]["mean_ic"],
                     "gap_nw_t": r["summ_gap"]["nw_tstat"],
                     **{k: v for k, v in r["gates"].items()}})

    summary = pd.DataFrame(rows)
    summary.insert(0, "stage_a_pass", stage_a)
    summary.insert(1, "n_eff", round(n_eff, 1))
    summary.to_csv(OUT / "stage_b_summary.csv", index=False)
    _plots(results)

    any_pass = any(all(r["gates"].values()) for r in results)
    logger.info("=" * 60)
    logger.info("E7 VERDICT: {}", "A GATE PASSED — proceed to Stage C" if any_pass
                else "ALL FEATURES FAIL Stage B — hypothesis falsified, STOP")
    logger.info("Artifacts in {}", OUT)


if __name__ == "__main__":
    main()
