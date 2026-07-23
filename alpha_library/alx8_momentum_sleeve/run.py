"""ALX8 — orthogonal momentum sleeve: one-command, stage-gated single run.

Tests one claim: does a 50/50 blend of a frozen momentum sleeve with the frozen
funding book beat the funding book alone? Frozen signals unchanged. Single run,
no tuning. Prints one verdict. See PREREGISTRATION.md.

    py alpha_library/alx8_momentum_sleeve/run.py
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

OUT = Path("data/research/alx8")

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

    fb = mo.sleeve(ret, fsig, fund)                 # funding sleeve
    mb = mo.sleeve(ret, mo.momentum_signal(ret), fund)   # momentum sleeve
    net_f = mo.realistic_net(fb)
    net_m = mo.realistic_net(mb)
    base_sh = sharpe(net_f)

    logger.info("=" * 72)
    logger.info("ALX8 momentum sleeve | {} names, {} days | fund turn={:.3f} mom turn={:.3f}",
                ret.shape[1], len(ret), fb["turnover"].mean(), mb["turnover"].mean())

    # ===================== STAGE A — anchor / machinery =======================
    logger.info("-" * 72)
    f_ic_t = ch.nw_tstat(fb["ic"].to_numpy(), lag=5)
    anchor_ok = abs(base_sh - ANCHOR) <= ANCHOR_TOL
    ic_ok = f_ic_t >= 3.0 and fb["ic"].mean() > 0
    stageA = bool(anchor_ok and ic_ok)
    logger.info("STAGE A  funding net Sharpe={:+.3f} (anchor {:.2f}±{:.2f} -> {})  IC NW-t={:+.2f}",
                base_sh, ANCHOR, ANCHOR_TOL, "OK" if anchor_ok else "MISMATCH", f_ic_t)
    R["stageA"] = {"base_sharpe": base_sh, "ic_nw_t": f_ic_t, "pass": stageA}
    if not stageA:
        _finish(R, "FAIL", "Stage A — funding sleeve does not reproduce the ALX5 anchor")
        return
    need = base_sh + MARGIN
    logger.info("STAGE A PASS (base={:+.3f}, decisive bar = base+{:.2f} = {:+.3f})",
                base_sh, MARGIN, need)

    # ===================== STAGE B — real orthogonal book-lifting alpha =======
    logger.info("-" * 72)
    mom_sh = sharpe(net_m)
    b1 = mom_sh > MOM_MIN
    corr = float(np.corrcoef(net_f, net_m)[0, 1])
    b2 = abs(corr) < CORR_MAX
    net_comb, _ = mo.blend_net(fb["applied"], mb["applied"], ret_np, fund_np)
    comb_sh = sharpe(net_comb)
    b3 = comb_sh >= need
    nfolds, pairs = _folds_ge(net_comb, net_f)
    b4 = nfolds >= 4
    stageB = bool(b1 and b2 and b3 and b4)
    logger.info("STAGE B1 momentum-alone Sharpe={:+.3f} (>{:.1f} -> {})", mom_sh, MOM_MIN,
                "OK" if b1 else "FAIL")
    logger.info("STAGE B2 corr(funding, momentum)={:+.3f} (|.|<{:.2f} -> {})", corr, CORR_MAX,
                "OK" if b2 else "FAIL")
    logger.info("STAGE B3 combined 50/50 Sharpe={:+.3f} (>= {:+.3f} -> {})", comb_sh, need,
                "OK" if b3 else "FAIL")
    logger.info("STAGE B4 folds combined>=funding: {}/6 -> {}", nfolds, "OK" if b4 else "FAIL")
    R["stageB"] = {"mom_sharpe": mom_sh, "b1": b1, "corr": corr, "b2": b2,
                   "combined_sharpe": comb_sh, "need": need, "b3": b3,
                   "folds_ge": nfolds, "b4": b4, "pass": stageB}
    if not stageB:
        _finish(R, "FAIL", "Stage B — momentum sleeve is not a real, orthogonal, book-lifting alpha (H0 not rejected)")
        return
    logger.info("STAGE B PASS")

    # ===================== STAGE C — diversification, not leak/averaging ======
    logger.info("-" * 72)
    mb0 = mo.sleeve(ret, mo.momentum_signal(ret), fund, lag=0)
    g0 = sharpe(mb0["price"] + mb0["funding"])
    g1 = sharpe(mb["price"] + mb["funding"])
    c1 = g0 > g1
    logger.info("STAGE C1 momentum lag0 gross={:+.3f} > lag1 gross={:+.3f} -> {}",
                g0, g1, "OK" if c1 else "FAIL")
    placebo = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        ap = mo.random_applied(ret, seed=i)
        nc, _ = mo.blend_net(fb["applied"], ap, ret_np, fund_np)
        placebo[i] = sharpe(nc)
    p975 = float(np.percentile(placebo, 97.5))
    c2 = comb_sh > p975
    logger.info("STAGE C2 real combined={:+.3f} vs random-blend 97.5pct={:+.3f} (median={:+.3f}) -> {}",
                comb_sh, p975, float(np.median(placebo)), "OK" if c2 else "FAIL")
    t0 = len(ret) // 2
    dfund_c = dfund.copy()
    dfund_c.loc[ret.index[t0 + 1]:, :] += 1.0
    ret_c, fund_c, fsig_c = ch.build_signal(close, dfund_c)
    fb_c = mo.sleeve(ret_c, fsig_c, fund_c)
    mb_c = mo.sleeve(ret_c, mo.momentum_signal(ret_c), fund_c)
    nc_c, _ = mo.blend_net(fb_c["applied"], mb_c["applied"], ret_c.to_numpy("float64"),
                           fund_c.reindex_like(ret_c).to_numpy("float64"))
    no_leak = bool(np.allclose(net_comb[:t0], nc_c[:t0], atol=1e-12))
    c3 = no_leak
    logger.info("STAGE C3 combined funding no-leak={} -> {}", no_leak, "OK" if c3 else "FAIL")
    stageC = bool(c1 and c2 and c3)
    R["stageC"] = {"mom_lag0_gross": g0, "mom_lag1_gross": g1, "c1": c1,
                   "placebo_975": p975, "placebo_median": float(np.median(placebo)),
                   "c2": c2, "no_leak": no_leak, "c3": c3, "pass": stageC}

    verdict = "PASS" if (stageA and stageB and stageC) else "FAIL"
    reason = ("orthogonal momentum sleeve lifts the book via real diversification"
              if verdict == "PASS"
              else "the blend does not survive the Stage-C kill-tests")
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
