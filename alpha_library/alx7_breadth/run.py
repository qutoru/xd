"""ALX7 breadth (decoupled anchor) — one-command, stage-gated single run.

Tests one claim: does adding PIT-admitted, liquidity-screened NEW names to the
frozen 40-major book raise net Sharpe? Frozen signal unchanged; majors are the
untouched reference (7 bps), the screen governs only the added names. Single run,
no tuning. Prints one verdict. See PREREGISTRATION.md.

    py alpha_library/alx7_breadth/run.py
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
from alpha_library.alx7_breadth import breadth as bd

OUT = Path("data/research/alx7")

ANCHOR = 1.890
ANCHOR_TOL = 0.15
MARGIN = 0.15                                   # locked decisive lift over base
SURV_CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")

ORIG40 = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "LTC", "LINK",
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

    man = pd.read_csv(Path("data/research/alx6") / "universe_manifest.csv")
    cand_bases = [s[:-4] for s in man["symbol"]]
    bases = sorted(set(cand_bases) | set(ORIG40))          # majors guaranteed present
    base_to_sym = {b: f"{b}USDT" for b in bases}
    onboard = ds.binance_onboard()

    close, qvol, dfund, _ = rep.build_daily("binance", bases)   # cached, no network
    index = close.index
    majors = [c for c in ORIG40 if c in close.columns]
    new_cand = [c for c in close.columns if c not in ORIG40]

    # ---- PIT mask + screen: NEW names only (majors untouched) ----------------
    starts = bd.new_start(onboard, base_to_sym, new_cand)
    onboard_dt = {c: pd.to_datetime(onboard[base_to_sym[c]], unit="ms", utc=True)
                  for c in new_cand}
    close_m, dfund_m = bd.mask_new(close, dfund, starts)        # only new_cand masked
    adv = bd.window_adv(qvol, close_m)
    admitted_new = bd.screen_new(close_m[new_cand], adv)

    cost_map = {c: bd.MAJOR_COST for c in majors}
    cost_map.update({c: bd.new_cost_for_adv(adv[c]) for c in admitted_new})
    expanded = majors + admitted_new

    logger.info("=" * 72)
    logger.info("ALX7 breadth | majors={} new_admitted={} expanded N={}",
                len(majors), len(admitted_new), len(expanded))
    tiers = {"7bps": sum(adv[c] >= 50e6 for c in admitted_new),
             "12bps": sum(10e6 <= adv[c] < 50e6 for c in admitted_new),
             "20bps": sum(5e6 <= adv[c] < 10e6 for c in admitted_new)}
    logger.info("  new-name cost tiers: {}", tiers)
    R["universe"] = {"n_majors": len(majors), "n_new": len(admitted_new),
                     "n_expanded": len(expanded), "tiers": tiers,
                     "new_names": admitted_new}

    base = bd.book(close_m, dfund_m, majors, cost_map)
    exp = bd.book(close_m, dfund_m, expanded, cost_map)
    base_sh, exp_sh = sharpe(base["net"]), sharpe(exp["net"])
    need = base_sh + MARGIN

    # ===================== STAGE A — anchor / machinery =======================
    logger.info("-" * 72)
    base_ic_t = ch.nw_tstat(base["ic"].to_numpy(), lag=5)
    anchor_ok = abs(base_sh - ANCHOR) <= ANCHOR_TOL
    ic_ok = base_ic_t >= 3.0 and base["ic"].mean() > 0
    stageA = bool(anchor_ok and ic_ok)
    logger.info("STAGE A  40-major net Sharpe={:+.3f} (anchor {:.2f}±{:.2f} -> {})  "
                "IC NW-t={:+.2f} (>=3 -> {})", base_sh, ANCHOR, ANCHOR_TOL,
                "OK" if anchor_ok else "MISMATCH", base_ic_t, "OK" if ic_ok else "FAIL")
    R["stageA"] = {"base_sharpe": base_sh, "anchor_ok": anchor_ok,
                   "ic_nw_t": base_ic_t, "pass": stageA}
    if not stageA:
        _finish(R, "FAIL", "Stage A — 40-major reference does not reproduce the ALX5 anchor")
        return
    logger.info("STAGE A PASS  (base={:+.3f}, decisive bar = base+{:.2f} = {:+.3f})",
                base_sh, MARGIN, need)

    # ===================== STAGE B — added names lift the book ================
    logger.info("-" * 72)
    b1 = exp_sh >= need
    nb = bd.book(close_m, dfund_m, admitted_new, cost_map)
    nb_sh = sharpe(nb["net"])
    nb_ic_t = ch.nw_tstat(nb["ic"].to_numpy(), lag=5)
    b2 = (nb_sh > 0) and (nb_ic_t >= 2.0) and (nb["ic"].mean() > 0)
    nfolds, pairs = _folds_ge(exp["net"], base["net"])
    b3 = nfolds >= 4
    stageB = bool(b1 and b2 and b3)
    logger.info("STAGE B1 expanded net Sharpe={:+.3f} (>= {:+.3f} -> {})", exp_sh, need,
                "OK" if b1 else "FAIL")
    logger.info("STAGE B2 new-only sub-book Sharpe={:+.3f}  IC NW-t={:+.2f} (>0 & t>=2 -> {})",
                nb_sh, nb_ic_t, "OK" if b2 else "FAIL")
    logger.info("STAGE B3 folds expanded>=base: {}/6 -> {}  pairs={}",
                nfolds, "OK" if b3 else "FAIL", [(round(a, 2), round(c, 2)) for a, c in pairs])
    R["stageB"] = {"expanded_sharpe": exp_sh, "need": need, "b1": b1,
                   "new_sharpe": nb_sh, "new_ic_t": nb_ic_t, "b2": b2,
                   "folds_ge": nfolds, "b3": b3, "pass": stageB}
    if not stageB:
        _finish(R, "FAIL", "Stage B — added names do not lift the book by the decisive margin (H0 not rejected)")
        return
    logger.info("STAGE B PASS")

    # ===================== STAGE C — not artifact =============================
    logger.info("-" * 72)
    surv_new = [c for c in admitted_new if onboard_dt[c] < SURV_CUTOFF]
    c1_sh = sharpe(bd.book(close_m, dfund_m, majors + surv_new, cost_map)["net"])
    c1 = c1_sh >= need
    logger.info("STAGE C1 drop new onboarded>=2023 (kept new={}) Sharpe={:+.3f} (>= {:+.3f} -> {})",
                len(surv_new), c1_sh, need, "OK" if c1 else "FAIL")
    liq_new = [c for c in admitted_new if adv[c] >= 10e6]
    c2_sh = sharpe(bd.book(close_m, dfund_m, majors + liq_new, cost_map)["net"])
    c2 = c2_sh >= need
    logger.info("STAGE C2 drop $5-10M new tier (kept new={}) Sharpe={:+.3f} (>= {:+.3f} -> {})",
                len(liq_new), c2_sh, need, "OK" if c2 else "FAIL")
    t0 = len(index) // 2
    dfund_c = dfund_m.copy()
    dfund_c.loc[index[t0 + 1]:, expanded] += 1.0
    net_c = bd.book(close_m, dfund_c, expanded, cost_map)["net"]
    no_leak = bool(np.allclose(exp["net"][:t0], net_c[:t0], atol=1e-12))
    ap, pos = exp["applied"], {c: i for i, c in enumerate(expanded)}
    pit_zero = True
    for c in admitted_new:
        si = index.searchsorted(starts[c])
        if si > 0 and not np.allclose(ap[:si, pos[c]], 0.0, atol=1e-15):
            pit_zero = False
            break
    c3 = no_leak and pit_zero
    logger.info("STAGE C3 no-leak={} & pre-onboard zero-weight={} -> {}",
                no_leak, pit_zero, "OK" if c3 else "FAIL")
    stageC = bool(c1 and c2 and c3)
    R["stageC"] = {"surv_sharpe": c1_sh, "c1": c1, "liq_sharpe": c2_sh, "c2": c2,
                   "no_leak": no_leak, "pit_zero": pit_zero, "c3": c3, "pass": stageC}

    verdict = "PASS" if (stageA and stageB and stageC) else "FAIL"
    reason = ("added names lift the frozen book and survive survivorship/illiquidity/leak stresses"
              if verdict == "PASS"
              else "breadth lift does not survive the Stage-C artifact kill-tests")
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
