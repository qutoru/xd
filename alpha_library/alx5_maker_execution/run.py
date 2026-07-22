"""ALX5 — maker-execution viability of the FROZEN funding->price book.

One command, stage-gated, prints exactly one verdict (PASS / FAIL). Single run,
no tuning, per PREREGISTRATION.md. The frozen daily book (hold=1, lag=1,
top/bottom 30%, 7-day funding signal) is reused verbatim from ALX4; the ONLY
object under test is the honest-maker cost model (§4). Descriptive cost×hold
ladder is a diagnostic and CANNOT change the verdict.

    py alpha_library/alx5_maker_execution/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from alpha_library.alx_funding_price_validation.validate import sharpe
from alpha_library.alx3_external_replication import replicate as rep
from alpha_library.alx4_regime_analysis import characterize as ch
from alpha_library.alx5_maker_execution import execution as ex

OUT = Path("data/research/alx5")

# ALX4 anchor (frozen): MED net Sharpe on this exact book.
ANCHOR_MED_SHARPE = 1.9412
ANCHOR_TOL = 0.15

# Untouched pre-2023 out-of-sample fold (ALX3 Stage-3 window).
PRE_LO = pd.Timestamp("2021-07-07", tz="UTC")
PRE_HI = pd.Timestamp("2023-07-07", tz="UTC")

# Same 40-major universe ALX4 froze.
BASES = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "LTC", "LINK",
         "BCH", "AVAX", "TRX", "ATOM", "ETC", "XLM", "NEAR", "FIL", "UNI", "AAVE",
         "ICP", "APT", "ARB", "OP", "SAND", "MANA", "AXS", "EOS", "INJ", "RUNE",
         "GRT", "ALGO", "THETA", "EGLD", "CRV", "COMP", "SUSHI", "ZEC", "DASH", "GALA"]


def _folds(net: np.ndarray, k: int = 6) -> list[float]:
    return [sharpe(a) for a in np.array_split(np.asarray(net), k)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    # ---- frozen book (reused verbatim from ALX4; cached, no network) ---------
    close, qvol, dfund, audit = rep.build_daily("binance", BASES)
    ret, fund, signal = ch.build_signal(close, dfund)
    book = ex.frozen_book(ret, signal, fund, lag=ex.LAG)
    price, funding, turn = book["price"], book["funding"], book["turnover"]
    gross = price + funding
    idx = ret.index

    logger.info("=" * 70)
    logger.info("ALX5 — maker execution on FROZEN book  ({}..{}, {} days, {} sym)",
                idx.min().date(), idx.max().date(), len(idx), ret.shape[1])

    # ===================== STAGE A — anchor / machinery (kill-test) ===========
    logger.info("-" * 70)
    net_med = ex.apply_cost(price, funding, turn, ex.MED)
    sh_med = sharpe(net_med)
    ic = ch.rank_ic_series(signal, ret)
    ic_t = ch.nw_tstat(ic.to_numpy(), lag=5)
    ic_mean = float(ic.mean())
    anchor_ok = abs(sh_med - ANCHOR_MED_SHARPE) <= ANCHOR_TOL
    ic_ok = (ic_t >= 3.0) and (ic_mean > 0)
    stageA = bool(anchor_ok and ic_ok)
    logger.info("STAGE A  MED Sharpe={:+.3f} (anchor {:.2f}±{:.2f} -> {})  "
                "rank-IC mean={:+.4f} NW-t(5)={:+.2f} (>=3 -> {})",
                sh_med, ANCHOR_MED_SHARPE, ANCHOR_TOL, "OK" if anchor_ok else "MISMATCH",
                ic_mean, ic_t, "OK" if ic_ok else "FAIL")
    R["stageA"] = {"med_sharpe": sh_med, "anchor_ok": anchor_ok,
                   "ic_mean": ic_mean, "ic_nw_t": ic_t, "ic_ok": ic_ok, "pass": stageA}
    if not stageA:
        _finish(R, "FAIL", "Stage A — broken base (anchor/IC); maker numbers not interpreted")
        return
    logger.info("STAGE A PASS")

    # ===================== STAGE B — net viability @ realistic maker ==========
    logger.info("-" * 70)
    net_real = ex.apply_cost(price, funding, turn, ex.REAL_MAKER_COST, ex.REAL_MAKER_PHI)
    sh_real = sharpe(net_real)
    six = _folds(net_real, 6)
    n_pos = int(sum(v > 0 for v in six))
    net_real_s = pd.Series(net_real, index=idx)
    pre = net_real_s.loc[PRE_LO:PRE_HI].to_numpy()
    sh_pre = sharpe(pre)
    b_sharpe = sh_real > 1.0
    b_folds = n_pos >= 4
    b_pre = sh_pre > 0
    stageB = bool(b_sharpe and b_folds and b_pre)
    logger.info("STAGE B  realistic-maker net Sharpe={:+.3f} (>1 -> {})", sh_real,
                "OK" if b_sharpe else "FAIL")
    logger.info("         6-fold Sharpe {} ({}/6>0 -> {})",
                [round(v, 2) for v in six], n_pos, "OK" if b_folds else "FAIL")
    logger.info("         pre-2023 fold Sharpe={:+.3f} (n={}) (>0 -> {})",
                sh_pre, len(pre), "OK" if b_pre else "FAIL")
    R["stageB"] = {"real_sharpe": sh_real, "six_fold": six, "folds_positive": n_pos,
                   "pre2023_sharpe": sh_pre, "pre2023_n": int(len(pre)),
                   "pass_sharpe": b_sharpe, "pass_folds": b_folds, "pass_pre": b_pre,
                   "pass": stageB}
    if not stageB:
        _finish(R, "FAIL", "Stage B — net edge collapses under honest maker cost (H0 not rejected)")
        return
    logger.info("STAGE B PASS")

    # ===================== STAGE C — maker story is real, not a leak ==========
    logger.info("-" * 70)
    # C1 — lag-0 vs lag-1 tell (same-bar leakage sanity, on cost-free gross)
    book0 = ex.frozen_book(ret, signal, fund, lag=0)
    gross0 = book0["price"] + book0["funding"]
    sh_g0, sh_g1 = sharpe(gross0), sharpe(gross)
    c1 = bool(sh_g0 > sh_g1)   # lag-0 must be conspicuously better; traded lag-1 lower
    logger.info("STAGE C1 lag0 gross Sharpe={:+.3f}  lag1 gross Sharpe={:+.3f}  "
                "(lag0>lag1 -> {})", sh_g0, sh_g1, "OK" if c1 else "FAIL")

    # C2 — execution attribution: maker uplift is ONLY the cost/turnover delta.
    # Same gross across models; reconstruct each net independently and confirm
    # the maker-vs-taker gap equals turnover*(taker-maker_cost) exactly.
    net_taker = ex.apply_cost(price, funding, turn, ex.TAKER)
    net_maker_nofi = ex.apply_cost(price, funding, turn, ex.REAL_MAKER_COST)  # phi=1
    recon_delta = net_maker_nofi - net_taker
    exact_delta = turn * (ex.TAKER - ex.REAL_MAKER_COST)
    ident = bool(np.allclose(recon_delta, exact_delta, atol=1e-15))
    # and the realistic net itself must equal phi*gross - turn*cost bit-for-bit
    recon_real = ex.REAL_MAKER_PHI * gross - turn * ex.REAL_MAKER_COST
    ident_real = bool(np.allclose(net_real, recon_real, atol=1e-18))
    c2 = bool(ident and ident_real)
    logger.info("STAGE C2 maker uplift == cost/turnover delta ({}) & net identity ({}) -> {}",
                ident, ident_real, "OK" if c2 else "FAIL")

    # C3 — adverse-selection escalation (harsher fill reality must stay >0)
    net_harsh = ex.apply_cost(price, funding, turn, ex.HARSH_MAKER_COST, ex.HARSH_MAKER_PHI)
    sh_harsh = sharpe(net_harsh)
    c3 = bool(sh_harsh > 0)
    logger.info("STAGE C3 escalated (phi=0.70,+4bps) net Sharpe={:+.3f} (>0 -> {})",
                sh_harsh, "OK" if c3 else "FAIL")

    stageC = bool(c1 and c2 and c3)
    R["stageC"] = {"lag0_gross_sharpe": sh_g0, "lag1_gross_sharpe": sh_g1, "c1_pass": c1,
                   "attrib_identity": ident, "net_identity": ident_real, "c2_pass": c2,
                   "harsh_sharpe": sh_harsh, "c3_pass": c3, "pass": stageC}

    # ===================== DESCRIPTIVE — cost×hold ladder (NOT gated) =========
    logger.info("-" * 70)
    logger.info("DESCRIPTIVE cost×hold ladder (diagnostic only — cannot change verdict)")
    ret_np = ret.to_numpy(dtype="float64")
    fund_np = fund.reindex_like(ret).to_numpy(dtype="float64")
    ladder: dict = {}
    cost_models = {"taker": (ex.TAKER, 1.0), "MED": (ex.MED, 1.0),
                   "ideal_maker": (ex.IDEAL_MAKER, 1.0),
                   "realistic_maker": (ex.REAL_MAKER_COST, ex.REAL_MAKER_PHI)}
    for hold in (1, 2, 3, 5):
        applied = ex.held_applied(signal, ret, hold=hold)
        bk = ex.book_from_applied(applied, ret_np, fund_np)
        row = {"avg_turnover": float(np.mean(bk["turnover"]))}
        for name, (c, phi) in cost_models.items():
            row[name] = sharpe(ex.apply_cost(bk["price"], bk["funding"],
                                             bk["turnover"], c, phi))
        ladder[f"hold_{hold}"] = row
        logger.info("  hold={} turn={:.3f} | " + " ".join(
            f"{k}={row[k]:+.2f}" for k in cost_models),
            hold, row["avg_turnover"])
    R["descriptive_ladder"] = ladder

    # ===================== VERDICT ===========================================
    verdict = "PASS" if (stageA and stageB and stageC) else "FAIL"
    reason = ("all stages survive — candidate for a separate forward maker-paper protocol"
              if verdict == "PASS"
              else "Stage C kill-test failed — maker story not clean")
    _finish(R, verdict, reason)


def _finish(R: dict, verdict: str, reason: str) -> None:
    R["verdict"] = verdict
    R["reason"] = reason
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    logger.info("=" * 70)
    logger.info("VERDICT: {} — {}", verdict, reason)
    logger.info("Artifacts -> {}", OUT)


if __name__ == "__main__":
    main()
