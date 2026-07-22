"""ALX6 breadth — one-command, stage-gated single run (see PREREGISTRATION.md).

Tests exactly one claim: does a wider PIT-admitted, liquidity-screened universe
raise the frozen funding->price net Sharpe, surviving honest PIT entry and
illiquidity-scaled maker cost? Frozen signal unchanged; only the name set (and
its locked tiered cost) varies. Single run, no tuning. Prints one verdict.

    py alpha_library/alx6_breadth/run.py
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
from alpha_library.alx3_external_replication import data_sources as ds
from alpha_library.alx4_regime_analysis import characterize as ch
from alpha_library.alx6_breadth import breadth as bd

OUT = Path("data/research/alx6")

ANCHOR = 1.890          # ALX5 realistic-maker baseline (40 names)
ANCHOR_TOL = 0.15
BASELINE = 1.89         # locked Stage-C floor
DECISIVE = 2.10         # locked Stage-B bar
SURV_CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")

# original 40 majors (frozen ALX4 universe)
ORIG40 = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "LTC", "LINK",
          "BCH", "AVAX", "TRX", "ATOM", "ETC", "XLM", "NEAR", "FIL", "UNI", "AAVE",
          "ICP", "APT", "ARB", "OP", "SAND", "MANA", "AXS", "EOS", "INJ", "RUNE",
          "GRT", "ALGO", "THETA", "EGLD", "CRV", "COMP", "SUSHI", "ZEC", "DASH", "GALA"]


def _folds_ge(exp: np.ndarray, base: np.ndarray, k: int = 6) -> tuple[int, list]:
    e, b = np.array_split(exp, k), np.array_split(base, k)
    pairs = [(sharpe(e[i]), sharpe(b[i])) for i in range(k)]
    n = sum(se >= sb for se, sb in pairs)
    return n, pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    # ---- load fetched candidate universe + onboard dates ---------------------
    man = pd.read_csv(OUT / "universe_manifest.csv")
    bases = [s[:-4] for s in man["symbol"]]
    base_to_sym = {b: s for b, s in zip(bases, man["symbol"])}
    onboard = ds.binance_onboard()                       # symbol -> onboard_ms (cached)

    close, qvol, dfund, _ = rep.build_daily("binance", bases)   # cached, no network
    index = close.index

    # ---- PIT admission + liquidity screen (LOCKED rule) ----------------------
    starts = bd.eligibility_start(onboard, base_to_sym, index, list(close.columns))
    close_m, dfund_m = bd.mask_pit(close, dfund, starts)
    adv = bd.window_adv(qvol, close_m)
    admitted = bd.screen(close_m, adv)
    orig = [c for c in ORIG40 if c in admitted]
    new = [c for c in admitted if c not in ORIG40]
    logger.info("=" * 72)
    logger.info("ALX6 breadth | candidates={} admitted N={} (orig40 in={}, new={})",
                len(bases), len(admitted), len(orig), len(new))
    tiers = {"7bps(>=50M)": sum(adv[c] >= 50e6 for c in admitted),
             "12bps(10-50M)": sum(10e6 <= adv[c] < 50e6 for c in admitted),
             "20bps(5-10M)": sum(5e6 <= adv[c] < 10e6 for c in admitted)}
    logger.info("  cost tiers: {}", tiers)
    R["universe"] = {"n_candidates": len(bases), "n_admitted": len(admitted),
                     "n_orig": len(orig), "n_new": len(new), "tiers": tiers,
                     "admitted": admitted}

    exp = bd.tiered_book(close_m, dfund_m, admitted, adv)
    base = bd.tiered_book(close_m, dfund_m, orig, adv)
    exp_sh, base_sh = sharpe(exp["net"]), sharpe(base["net"])

    # ===================== STAGE A — anchor / machinery =======================
    logger.info("-" * 72)
    base_ic_t = ch.nw_tstat(base["ic"].to_numpy(), lag=5)
    anchor_ok = abs(base_sh - ANCHOR) <= ANCHOR_TOL
    ic_ok = base_ic_t >= 3.0 and base["ic"].mean() > 0
    stageA = bool(anchor_ok and ic_ok)
    logger.info("STAGE A  orig40 tiered net Sharpe={:+.3f} (anchor {:.2f}±{:.2f} -> {})  "
                "IC NW-t={:+.2f} (>=3 -> {})", base_sh, ANCHOR, ANCHOR_TOL,
                "OK" if anchor_ok else "MISMATCH", base_ic_t, "OK" if ic_ok else "FAIL")
    R["stageA"] = {"orig40_sharpe": base_sh, "anchor_ok": anchor_ok,
                   "ic_nw_t": base_ic_t, "pass": stageA}
    if not stageA:
        _finish(R, "FAIL", "Stage A — orig40 does not reproduce the ALX5 anchor under the "
                           "tiered cost; expanded numbers off a shifted base are not interpreted")
        return
    logger.info("STAGE A PASS")

    # ===================== STAGE B — real, non-artifact breadth ===============
    logger.info("-" * 72)
    b1 = exp_sh > DECISIVE
    nb = bd.tiered_book(close_m, dfund_m, new, adv)
    nb_sh = sharpe(nb["net"])
    nb_ic_t = ch.nw_tstat(nb["ic"].to_numpy(), lag=5)
    b2 = (nb_sh > 0) and (nb_ic_t >= 2.0) and (nb["ic"].mean() > 0)
    nfolds, pairs = _folds_ge(exp["net"], base["net"])
    b3 = nfolds >= 4
    stageB = bool(b1 and b2 and b3)
    logger.info("STAGE B1 expanded net Sharpe={:+.3f} (>{:.2f} -> {})", exp_sh, DECISIVE,
                "OK" if b1 else "FAIL")
    logger.info("STAGE B2 new-names sub-book Sharpe={:+.3f}  IC NW-t={:+.2f} (>0 & t>=2 -> {})",
                nb_sh, nb_ic_t, "OK" if b2 else "FAIL")
    logger.info("STAGE B3 folds expanded>=baseline: {}/6 -> {}  pairs(exp,base)={}",
                nfolds, "OK" if b3 else "FAIL", [(round(a, 2), round(c, 2)) for a, c in pairs])
    R["stageB"] = {"expanded_sharpe": exp_sh, "b1": b1, "new_sharpe": nb_sh,
                   "new_ic_t": nb_ic_t, "b2": b2, "folds_ge": nfolds, "b3": b3,
                   "pass": stageB}
    if not stageB:
        _finish(R, "FAIL", "Stage B — breadth does not add real, non-artifact edge (H0 not rejected)")
        return
    logger.info("STAGE B PASS")

    # ===================== STAGE C — not survivorship/illiquidity/leak ========
    logger.info("-" * 72)
    # C1 survivorship-tail: drop names onboarded on/after 2023-01-01
    surv = [c for c in admitted if starts[c] - pd.Timedelta(days=bd.BUFFER_DAYS) < SURV_CUTOFF]
    c1_sh = sharpe(bd.tiered_book(close_m, dfund_m, surv, adv)["net"])
    c1 = c1_sh > BASELINE
    logger.info("STAGE C1 survivorship-tail (onboard<2023, N={}) Sharpe={:+.3f} (>{:.2f} -> {})",
                len(surv), c1_sh, BASELINE, "OK" if c1 else "FAIL")
    # C2 illiquidity: drop the $5-10M tier
    liq = [c for c in admitted if adv[c] >= 10e6]
    c2_sh = sharpe(bd.tiered_book(close_m, dfund_m, liq, adv)["net"])
    c2 = c2_sh > BASELINE
    logger.info("STAGE C2 drop $5-10M tier (N={}) Sharpe={:+.3f} (>{:.2f} -> {})",
                len(liq), c2_sh, BASELINE, "OK" if c2 else "FAIL")
    # C3 PIT / leak audit
    t0 = len(index) // 2
    dfund_c = dfund_m.copy()
    dfund_c.loc[index[t0 + 1]:, admitted] += 1.0
    net_c = bd.tiered_book(close_m, dfund_c, admitted, adv)["net"]
    no_leak = bool(np.allclose(exp["net"][:t0], net_c[:t0], atol=1e-12))
    # pre-onboard zero-weight: each name contributes no weight before eligibility
    ap = exp["applied"]
    pos = {c: i for i, c in enumerate(admitted)}
    pit_zero = True
    for c in admitted:
        si = index.searchsorted(starts[c])
        if si > 0 and not np.allclose(ap[:si, pos[c]], 0.0, atol=1e-15):
            pit_zero = False
            break
    c3 = no_leak and pit_zero
    logger.info("STAGE C3 funding no-leak={} & pre-onboard zero-weight={} -> {}",
                no_leak, pit_zero, "OK" if c3 else "FAIL")
    stageC = bool(c1 and c2 and c3)
    R["stageC"] = {"surv_sharpe": c1_sh, "c1": c1, "liq_sharpe": c2_sh, "c2": c2,
                   "no_leak": no_leak, "pit_zero": pit_zero, "c3": c3, "pass": stageC}

    verdict = "PASS" if (stageA and stageB and stageC) else "FAIL"
    reason = ("wider PIT universe raises net Sharpe and survives survivorship/illiquidity/leak "
              "stresses" if verdict == "PASS"
              else "breadth gain does not survive the Stage-C artifact kill-tests")
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
