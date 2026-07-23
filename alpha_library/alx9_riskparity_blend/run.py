"""ALX9 — risk-parity blend: one-command, stage-gated single run.

Tests one claim: does no-look-ahead inverse-vol (risk-parity) weighting make the
proven orthogonal momentum sleeve lift the frozen funding book? Frozen signals
unchanged; only the blend weighting differs from ALX8. Single run, no tuning.
Prints one verdict. See PREREGISTRATION.md.

    py alpha_library/alx9_riskparity_blend/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from loguru import logger

from alpha_library.alx_funding_price_validation.validate import sharpe
from alpha_library.alx3_external_replication import replicate as rep
from alpha_library.alx4_regime_analysis import characterize as ch
from alpha_library.alx8_momentum_sleeve import momentum as mo
from alpha_library.alx9_riskparity_blend import riskparity as rp

OUT = Path("data/research/alx9")

ANCHOR, ANCHOR_TOL, MARGIN = 1.890, 0.15, 0.15
MOM_MIN, CORR_MAX = 0.5, 0.30
N_PLACEBO = 200

BASES = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "LTC", "LINK",
         "BCH", "AVAX", "TRX", "ATOM", "ETC", "XLM", "NEAR", "FIL", "UNI", "AAVE",
         "ICP", "APT", "ARB", "OP", "SAND", "MANA", "AXS", "EOS", "INJ", "RUNE",
         "GRT", "ALGO", "THETA", "EGLD", "CRV", "COMP", "SUSHI", "ZEC", "DASH", "GALA"]


def _folds_ge(exp, base, k=6):
    e, b = np.array_split(exp, k), np.array_split(base, k)
    pairs = [(sharpe(e[i]), sharpe(b[i])) for i in range(k)]
    return sum(se >= sb for se, sb in pairs), pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    close, qvol, dfund, _ = rep.build_daily("binance", BASES)
    ret, fund, fsig = ch.build_signal(close, dfund)
    ret_np = ret.to_numpy(dtype="float64")
    fund_np = fund.reindex_like(ret).to_numpy(dtype="float64")

    fb = mo.sleeve(ret, fsig, fund)
    mb = mo.sleeve(ret, mo.momentum_signal(ret), fund)
    net_f, net_m = mo.realistic_net(fb), mo.realistic_net(mb)
    base_sh, mom_sh = sharpe(net_f), sharpe(net_m)
    corr = float(np.corrcoef(net_f, net_m)[0, 1])

    logger.info("=" * 72)
    logger.info("ALX9 risk-parity blend | funding S={:+.3f} momentum S={:+.3f} corr={:+.3f}",
                base_sh, mom_sh, corr)

    # ===================== STAGE A — anchor + ingredient re-check =============
    logger.info("-" * 72)
    f_ic_t = ch.nw_tstat(fb["ic"].to_numpy(), lag=5)
    anchor_ok = abs(base_sh - ANCHOR) <= ANCHOR_TOL and f_ic_t >= 3.0 and fb["ic"].mean() > 0
    ingredient_ok = (mom_sh > MOM_MIN) and (abs(corr) < CORR_MAX)
    stageA = bool(anchor_ok and ingredient_ok)
    logger.info("STAGE A  funding={:+.3f} (anchor -> {}) IC-t={:+.2f} | momentum={:+.3f}(>0.5) "
                "corr={:+.3f}(|.|<0.3) -> ingredient {}", base_sh,
                "OK" if anchor_ok else "MISMATCH", f_ic_t, mom_sh, corr,
                "OK" if ingredient_ok else "FAIL")
    R["stageA"] = {"base_sharpe": base_sh, "ic_nw_t": f_ic_t, "mom_sharpe": mom_sh,
                   "corr": corr, "pass": stageA}
    if not stageA:
        _finish(R, "FAIL", "Stage A — base or ingredient not intact")
        return
    need = base_sh + MARGIN
    logger.info("STAGE A PASS (base={:+.3f}, decisive bar = base+{:.2f} = {:+.3f})",
                base_sh, MARGIN, need)

    # ===================== STAGE B — risk-parity blend lifts the book =========
    logger.info("-" * 72)
    net_rp, _, af, bf = rp.rp_blend_net(fb["applied"], mb["applied"], net_f, net_m,
                                        ret_np, fund_np)
    rp_sh = sharpe(net_rp)
    b1 = rp_sh >= need
    nfolds, pairs = _folds_ge(net_rp, net_f)
    b2 = nfolds >= 4
    stageB = bool(b1 and b2)
    logger.info("STAGE B1 risk-parity combined Sharpe={:+.3f} (>= {:+.3f} -> {})  "
                "avg weights fund={:.2f}/mom={:.2f}", rp_sh, need,
                "OK" if b1 else "FAIL", float(np.mean(af)), float(np.mean(bf)))
    logger.info("STAGE B2 folds combined>=funding: {}/6 -> {}", nfolds, "OK" if b2 else "FAIL")
    R["stageB"] = {"rp_sharpe": rp_sh, "need": need, "b1": b1, "folds_ge": nfolds,
                   "b2": b2, "avg_w_fund": float(np.mean(af)), "avg_w_mom": float(np.mean(bf)),
                   "pass": stageB}
    if not stageB:
        _finish(R, "FAIL", "Stage B — risk-parity blend does not lift the book by the decisive margin (H0 not rejected)")
        return
    logger.info("STAGE B PASS")

    # ===================== STAGE C — diversification, not leak/averaging ======
    logger.info("-" * 72)
    mb0 = mo.sleeve(ret, mo.momentum_signal(ret), fund, lag=0)
    g0, g1 = sharpe(mb0["price"] + mb0["funding"]), sharpe(mb["price"] + mb["funding"])
    c1 = g0 > g1
    logger.info("STAGE C1 momentum lag0 gross={:+.3f} > lag1 gross={:+.3f} -> {}",
                g0, g1, "OK" if c1 else "FAIL")
    placebo = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        ap = mo.random_applied(ret, seed=i)
        net_r = rp.book_from_applied(ap, ret_np, fund_np)
        nc, _, _, _ = rp.rp_blend_net(fb["applied"], ap, net_f, net_r, ret_np, fund_np)
        placebo[i] = sharpe(nc)
    p975 = float(np.percentile(placebo, 97.5))
    c2 = rp_sh > p975
    logger.info("STAGE C2 real combined={:+.3f} vs random-blend 97.5pct={:+.3f} (median={:+.3f}) -> {}",
                rp_sh, p975, float(np.median(placebo)), "OK" if c2 else "FAIL")
    t0 = len(ret) // 2
    dfund_c = dfund.copy()
    dfund_c.loc[ret.index[t0 + 1]:, :] += 1.0
    ret_c, fund_c, fsig_c = ch.build_signal(close, dfund_c)
    fb_c = mo.sleeve(ret_c, fsig_c, fund_c)
    mb_c = mo.sleeve(ret_c, mo.momentum_signal(ret_c), fund_c)
    net_fc, net_mc = mo.realistic_net(fb_c), mo.realistic_net(mb_c)
    nc_c, _, _, _ = rp.rp_blend_net(fb_c["applied"], mb_c["applied"], net_fc, net_mc,
                                    ret_c.to_numpy("float64"),
                                    fund_c.reindex_like(ret_c).to_numpy("float64"))
    no_leak = bool(np.allclose(net_rp[:t0], nc_c[:t0], atol=1e-12))
    c3 = no_leak
    logger.info("STAGE C3 combined funding no-leak={} -> {}", no_leak, "OK" if c3 else "FAIL")
    stageC = bool(c1 and c2 and c3)
    R["stageC"] = {"mom_lag0_gross": g0, "mom_lag1_gross": g1, "c1": c1,
                   "placebo_975": p975, "placebo_median": float(np.median(placebo)),
                   "c2": c2, "no_leak": no_leak, "c3": c3, "pass": stageC}

    verdict = "PASS" if (stageA and stageB and stageC) else "FAIL"
    reason = ("risk-parity blend lifts the book via real diversification"
              if verdict == "PASS"
              else "the risk-parity blend does not survive the Stage-C kill-tests")
    _finish(R, verdict, reason)


def _finish(R: dict, verdict: str, reason: str) -> None:
    R["verdict"] = verdict
    R["reason"] = reason
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    logger.info("=" * 72)
    logger.info("VERDICT: {} — {}", verdict, reason)
    logger.info("Artifacts -> {}", OUT)


if __name__ == "__main__":
    main()
