"""ALX3 one-command independent external replication (see PREREGISTRATION.md).

Reconstructs data from scratch on independent venues and runs the FROZEN ALX book
unchanged across Stages 1-8, then prints exactly one verdict: CONFIRMED /
PLAUSIBLE BUT UNCONFIRMED / REJECTED. Single run, no tuning.

    py alpha_library/alx3_external_replication/run.py
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

from crypto_signal_bot.research.xsection.universe import SURVIVORSHIP_SAFE
from alpha_library.alx_funding_price_validation.validate import backtest, sharpe
from alpha_library.alx3_external_replication import replicate as rp

OUT = Path("data/research/alx3")
MED = rp.MED
DEV_START = pd.Timestamp("2023-07-07", tz="UTC")        # Bybit ALX window start
PRE_START = DEV_START - pd.Timedelta(days=730)          # 2-yr untouched pre-dev period
BASES = [s[:-4] for s in SURVIVORSHIP_SAFE]             # strip 'USDT'


def _stats(net, label, n_trials=100):
    sh = sharpe(net)
    lo, hi = rp.block_bootstrap_ci(net)
    dsr = rp.deflated_sharpe(net, n_trials)
    logger.info("  {:<26} Sharpe={:+.3f}  95%CI=[{:+.2f},{:+.2f}]  DSR={:.3f}  n={}",
                label, sh, lo, hi, dsr, len(net))
    return {"sharpe": sh, "ci_lo": lo, "ci_hi": hi, "dsr": dsr, "n": int(len(net))}


def _net_with_cost(res, cost_array):
    applied = res["applied"]
    ret = res["ret"].to_numpy(dtype="float64")
    fund = res["fund"].reindex_like(res["ret"]).to_numpy(dtype="float64")
    price = np.nansum(applied * ret, axis=1)
    funding = -np.nansum(applied * np.nan_to_num(fund), axis=1)
    return price + funding - cost_array


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}

    # =================== independent reconstruction (Binance) ================
    logger.info("=" * 70)
    logger.info("STAGE 1 — independent data reconstruction (Binance USDT-M primary)")
    close, qvol, dfund, audit = rp.build_daily("binance", BASES)
    logger.info("  off-grid settlement fraction = {:.4%} | {} symbols reconstructed",
                audit["off_grid_frac"], audit["n_symbols"])

    # faithful ALX window (intersection + 98% coverage), like build_returns_panel
    w0, w1 = rp.intersection_window(close)
    idx_full = close.loc[w0:w1].index
    cols = rp.faithful_cols(close, idx_full)
    logger.info("  faithful window {}..{} | {} symbols pass coverage", w0.date(), w1.date(), len(cols))
    res = rp.make_book(close, dfund, cols, idx_full)
    net = res["net"]

    # PIT / no-look-ahead audit on reconstructed funding (funding-side corruption)
    t0 = len(net) // 2
    dfund_c = dfund.copy()
    dfund_c.loc[res["index"][t0 + 1]:] += 1.0
    res_c = rp.make_book(close, dfund_c, cols, idx_full)
    no_leak = np.allclose(net[:t0], res_c["net"][:t0], atol=1e-12)
    logger.info("  funding future-corruption invariance (no look-ahead): {}", no_leak)
    R["stage1"] = {**audit, "window": [str(w0.date()), str(w1.date())],
                   "n_cols": len(cols), "no_funding_leak": bool(no_leak)}

    # =================== STAGE 2 — different exchange (Binance) ==============
    logger.info("=" * 70)
    logger.info("STAGE 2 — frozen strategy on Binance (independent venue)")
    s2 = _stats(net, "Binance full @Med")
    stage2_pass = (s2["sharpe"] > 1.0) and (s2["ci_lo"] > 0)
    R["stage2"] = {**s2, "pass": bool(stage2_pass)}
    logger.info("  STAGE 2 {}", "PASS" if stage2_pass else "FAIL")

    # =================== STAGE 3 — new time period (pre-2023) ================
    logger.info("=" * 70)
    logger.info("STAGE 3 — untouched pre-development period {}..{} (Binance)",
                PRE_START.date(), DEV_START.date())
    idx_pre = close.loc[PRE_START:DEV_START].index
    cols_pre = rp.faithful_cols(close, idx_pre)
    logger.info("  pre-dev cross-section: {} symbols with full coverage", len(cols_pre))
    res_pre = rp.make_book(close, dfund, cols_pre, idx_pre)
    s3 = _stats(res_pre["net"], "Binance pre-2023 @Med")
    stage3_pass = (s3["sharpe"] > 0) and (s3["ci_lo"] > 0)
    R["stage3"] = {**s3, "n_cols": len(cols_pre), "pass": bool(stage3_pass),
                   "window": [str(PRE_START.date()), str(DEV_START.date())]}
    logger.info("  STAGE 3 {}", "PASS" if stage3_pass else "FAIL")

    # =================== STAGE 4 — cross-exchange consistency ================
    logger.info("=" * 70)
    logger.info("STAGE 4 — cross-exchange consistency (OKX indep. short; Bybit control)")
    fexp_bin = rp.factor_exposures(res)
    logger.info("  Binance factor corr: {}", {k: round(v, 2) for k, v in fexp_bin.items()})
    cross = {"binance": {"sharpe": s2["sharpe"], "funding_corr": fexp_bin["raw_funding"]}}

    for ex in ("okx", "bybit"):
        try:
            c2, q2, f2, a2 = rp.build_daily(ex, BASES)
            ww0, ww1 = rp.intersection_window(c2)
            # OKX funding is shallow -> restrict to where funding actually exists
            if ex == "okx":
                fstart = f2.dropna(how="all").index.min()
                ww0 = max(ww0, fstart) if fstart is not None else ww0
            ix = c2.loc[ww0:ww1].index
            cc = rp.faithful_cols(c2, ix)
            if len(cc) < 4 or len(ix) < 40:
                logger.warning("  {}: insufficient panel ({} cols x {} days) — sign-check only",
                               ex, len(cc), len(ix))
            r2 = rp.make_book(c2, f2, cc, ix)
            fx = rp.factor_exposures(r2)
            st = _stats(r2["net"], f"{ex} {ww0.date()}..{ww1.date()}")
            cross[ex] = {"sharpe": st["sharpe"], "ci_lo": st["ci_lo"], "n": st["n"],
                         "n_cols": len(cc), "funding_corr": fx["raw_funding"],
                         "window": [str(ww0.date()), str(ww1.date())]}
        except Exception as exc:
            logger.warning("  {}: failed ({})", ex, exc)
            cross[ex] = {"error": str(exc)}
    R["stage4"] = {"binance_factor_corr": fexp_bin, "cross": cross}

    # =================== STAGE 5 — rolling out-of-sample =====================
    logger.info("=" * 70)
    logger.info("STAGE 5 — rolling out-of-sample stability (Binance, no retrain)")
    net_s = pd.Series(net, index=res["index"])
    folds = {str(y): sharpe(g.to_numpy()) for y, g in net_s.groupby(net_s.index.year) if len(g) > 20}
    six = [sharpe(a) for a in np.array_split(net, 6)]
    n_pos = int(sum(v > 0 for v in six))
    logger.info("  per-year Sharpe: {}", {k: round(v, 2) for k, v in folds.items()})
    logger.info("  6-fold Sharpe: {} ({}/6 > 0)", [round(v, 2) for v in six], n_pos)
    R["stage5"] = {"per_year": folds, "six_fold": six, "folds_positive": n_pos}

    # =================== STAGE 6 — point-in-time universe ====================
    logger.info("=" * 70)
    logger.info("STAGE 6 — point-in-time universe (union panel; only listed names each day)")
    all_cols = list(close.columns)
    res_pit = rp.make_book(close, dfund, all_cols, close.loc[close.index.min():w1].index)
    s6 = _stats(res_pit["net"], "Binance PIT-universe @Med")
    R["stage6"] = {**s6, "n_cols": len(all_cols)}

    # =================== STAGE 7 — realistic execution bounds ================
    logger.info("=" * 70)
    logger.info("STAGE 7 — realistic execution: conservative cost bounds")
    applied = res["applied"]
    dw = np.abs(np.diff(applied, axis=0, prepend=0.0))
    adv = qvol[cols].reindex(res["index"]).median().reindex(cols).to_numpy()
    aum = 1.0e7
    # lower bound (optimistic): flat 2 bps
    net_lo = _net_with_cost(res, dw.sum(axis=1) * 0.0002)
    # upper bound (pessimistic): taker+spread + ADV impact + short borrow
    impact = 0.0010 * (dw * aum) / (0.01 * adv)               # 10bps per 1% ADV
    per_name = (0.00075 + 0.0005) + impact
    short_cost = (np.abs(np.where(applied < 0, applied, 0.0))).sum(axis=1) * 0.0001  # 1bp/day borrow
    net_hi = _net_with_cost(res, (dw * per_name).sum(axis=1) + short_cost)
    sh_lo, sh_hi = sharpe(net_lo), sharpe(net_hi)
    eff_bps = float((((dw * per_name).sum() + short_cost.sum()) / dw.sum()) * 1e4)
    logger.info("  cost band: optimistic(2bps) Sharpe={:+.3f} | Med={:+.3f} | pessimistic Sharpe={:+.3f}",
                sh_lo, s2["sharpe"], sh_hi)
    logger.info("  pessimistic avg effective cost = {:.1f} bps/turnover", eff_bps)
    stage7_pass = sh_hi > 0
    R["stage7"] = {"sharpe_optimistic": sh_lo, "sharpe_med": s2["sharpe"],
                   "sharpe_pessimistic": sh_hi, "eff_bps_pessimistic": eff_bps,
                   "pass": bool(stage7_pass)}
    logger.info("  STAGE 7 {}", "survives (pessimistic > 0)" if stage7_pass else "FAIL")

    # =================== STAGE 8 — economic mechanism (diagnostics) ==========
    logger.info("=" * 70)
    logger.info("STAGE 8 — economic-mechanism diagnostics")
    fund_med = float(np.nanmedian(res["fund"].to_numpy()))
    fund_neg_frac = float(np.nanmean(res["fund"].to_numpy() < 0))
    funding_pnl_share = float(np.nansum(res["funding"]) / np.nansum(net)) if np.nansum(net) else float("nan")
    logger.info("  daily funding: median={:.5f}, %negative-obs={:.1%}, funding-PnL share of net={:.1%}",
                fund_med, fund_neg_frac, funding_pnl_share)
    R["stage8"] = {"funding_median": fund_med, "funding_neg_frac": fund_neg_frac,
                   "funding_pnl_share": funding_pnl_share}

    # =================== VERDICT ============================================
    okx_ok = isinstance(cross.get("okx"), dict) and cross["okx"].get("funding_corr", 0) > 0 \
        and cross["okx"].get("sharpe", -1) > 0
    bybit_ok = isinstance(cross.get("bybit"), dict) and cross["bybit"].get("sharpe", -1) > 1.0
    cross_consistent = (fexp_bin["raw_funding"] > 0) and bybit_ok  # OKX weak/short -> supportive only

    if not (s2["sharpe"] > 0 and s2["ci_lo"] > 0):
        verdict = "C. REJECTED"
    elif stage2_pass and stage3_pass and cross_consistent and stage7_pass:
        verdict = "A. CONFIRMED"
    else:
        verdict = "B. PLAUSIBLE BUT UNCONFIRMED"
    R["verdict"] = verdict
    R["cross_flags"] = {"okx_supportive": bool(okx_ok), "bybit_control_ok": bool(bybit_ok),
                        "binance_funding_corr_pos": bool(fexp_bin["raw_funding"] > 0)}

    with open(OUT / "results.json", "w") as fh:
        json.dump(R, fh, indent=2, default=float)

    logger.info("=" * 70)
    logger.info("VERDICT: {}", verdict)
    logger.info("Artifacts -> {}", OUT)


if __name__ == "__main__":
    main()
