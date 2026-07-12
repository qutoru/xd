"""ALX2 one-command adversarial replication (see PREREGISTRATION.md).

Runs the six FIXED falsification tests T1..T6 in order on the FROZEN ALX
funding->price book, stops at the first test that falsifies, and prints exactly
one verdict: PASS (admit to Alpha Library) or FAIL (name the first failing test).
Single run, no tuning, no parameter/threshold search.

    py alpha_library/alx2_adversarial_replication/run.py
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

from crypto_signal_bot.data.storage import load_parquet
from crypto_signal_bot.research.xsection import features as ft
from crypto_signal_bot.research.xsection.universe import (
    SURVIVORSHIP_SAFE,
    build_returns_panel,
    to_daily,
)
from alpha_library.alx_funding_price_validation.validate import backtest, sharpe
from alpha_library.alx2_adversarial_replication import adversarial as adv

OUT = Path("data/research/alx2")
FUNDING_PANEL = Path("data/research/e7/funding_panel.parquet")
MED = adv.MED


def load_baseline():
    """Object under test, exactly as ALX: reused-panel signal + frozen book."""
    panel = build_returns_panel(SURVIVORSHIP_SAFE, "60")
    daily_close = to_daily(panel)
    daily_ret = daily_close.pct_change().dropna(how="all")
    daily_logret = np.log(daily_close / daily_close.shift(1)).dropna(how="all")

    fh = pd.read_parquet(FUNDING_PANEL)
    settle = fh[fh.index.hour.isin(adv.SETTLE_HOURS)]
    daily_fund = settle.resample("1D").sum().dropna(how="all")
    cols = list(daily_ret.columns)
    daily_fund = daily_fund.reindex(index=daily_ret.index, columns=cols)
    signal = (-daily_fund.rolling(7).mean()).reindex(index=daily_ret.index, columns=cols)

    turnover_perbar = pd.Series(
        {s: float(np.median(load_parquet(s, "60")["turnover"].to_numpy())) for s in cols})
    # ADV = median daily dollar volume (sum of 1h turnover per UTC day), for T5
    adv_daily = {}
    for s in cols:
        df = load_parquet(s, "60")
        ts = pd.Series(df["turnover"].to_numpy(), index=pd.to_datetime(df["datetime"], utc=True))
        adv_daily[s] = float(ts.resample("1D").sum().median())
    adv_daily = pd.Series(adv_daily)
    return daily_ret, daily_logret, daily_fund, signal, turnover_perbar, adv_daily


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (daily_ret, daily_logret, daily_fund, signal,
     turnover_perbar, adv_daily) = load_baseline()
    idx = daily_ret.index

    res = backtest(daily_ret, signal, funding=daily_fund, lag=1, cost=MED)
    net = res["net"]
    base_sharpe = sharpe(net)
    logger.info("ALX2 adversarial | {} symbols x {} days | frozen net Sharpe = {:+.3f}",
                daily_ret.shape[1], daily_ret.shape[0], base_sharpe)

    results: dict[str, dict] = {}
    order = ["T1", "T2", "T3", "T4", "T5", "T6"]
    first_fail = None

    for tag in order:
        logger.info("=" * 64)
        if tag == "T1":
            logger.info("T1 — point-in-time funding reconstruction & alignment audit")
            logger.info("  re-fetching funding for {} symbols ...", daily_ret.shape[1])
            fpit, off_grid_frac, fetched = adv.build_pit_funding(list(daily_ret.columns), idx)
            logger.info("  off-grid settlement fraction = {:.4%} | re-fetched {} symbols",
                        off_grid_frac, len(fetched))
            r = adv.t1_point_in_time(daily_ret, daily_fund, fpit, fetched, base_sharpe)
            r["off_grid_settlement_frac"] = off_grid_frac
            logger.info("  PIT net Sharpe = {:+.3f} (reused {:+.3f}) | panel corr = {:.3f}"
                        " | funding no-leak = {}", r["sharpe_pit"], r["sharpe_reused"],
                        r["panel_corr"], r["no_funding_leak"])
        elif tag == "T2":
            logger.info("T2 — temporal concentration (top-day / crisis dependence)")
            r = adv.t2_temporal_concentration(net, idx, daily_ret)
            logger.info("  drop best 5% days Sharpe = {:+.3f} | ex-crisis-decile Sharpe = {:+.3f}",
                        r["sharpe_drop_top5pct"], r["sharpe_ex_crisis_decile"])
            logger.info("  top-1-day PnL share = {:.1%} | top-5% days PnL share = {:.1%}",
                        r["top1_day_pnl_share"], r["top5pct_pnl_share"])
        elif tag == "T3":
            logger.info("T3 — leave-one-symbol-out concentration")
            r = adv.t3_leave_one_out(daily_ret, signal, daily_fund)
            logger.info("  LOO net Sharpe  min={:+.3f} ({})  median={:+.3f}  max={:+.3f}",
                        r["min_sharpe"], r["worst_drop_symbol"], r["median_sharpe"], r["max_sharpe"])
        elif tag == "T4":
            logger.info("T4 — expanded factor attribution (1d reversal, downside beta, idio vol)")
            r = adv.t4_expanded_attribution(net, daily_ret, daily_logret, daily_fund, turnover_perbar)
            logger.info("  ex-funding: intercept {:+.1%}/yr (NW t={:+.2f}), R^2={:.1%}",
                        r["intercept_ann"], r["intercept_t"], r["r2_ex_funding"])
            logger.info("  with-funding: intercept {:+.1%}/yr (NW t={:+.2f}), R^2={:.1%}",
                        r["intercept_ann_with_funding"], r["intercept_t_with_funding"], r["r2_with_funding"])
            logger.info("  loadings (ex-funding) = {}", r["loadings_ex_funding"])
        elif tag == "T5":
            logger.info("T5 — realistic execution with liquidity-aware costs")
            r = adv.t5_liquidity_costs(daily_ret, signal, daily_fund, adv_daily)
            logger.info("  liquidity-aware net Sharpe = {:+.3f} | avg effective cost = {:.1f} bps"
                        " (baseline {:.1f})", r["sharpe_liq"], r["avg_effective_cost_bps"], r["baseline_bps"])
        else:  # T6
            logger.info("T6 — deflated Sharpe ratio (multiple-testing, N=100 trials)")
            r = adv.t6_deflated_sharpe(net, n_trials=100)
            logger.info("  Sharpe(ann)={:+.3f} vs deflated benchmark SR*={:+.3f} | skew={:+.2f} kurt={:.2f}",
                        r["sharpe_ann"], r["sr_star_ann"], r["skew"], r["kurt"])
            logger.info("  Deflated Sharpe Ratio (DSR) = {:.4f}  (bar > 0.95)", r["dsr"])

        results[tag] = r
        logger.info("  -> {} {}", tag, "SURVIVED" if r["survived"] else "FALSIFIED")
        if not r["survived"]:
            first_fail = tag
            logger.info("STOP RULE: {} falsified the alpha — halting battery.", tag)
            break

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.json", "w") as fh:
        json.dump({"base_sharpe": base_sharpe, "first_fail": first_fail,
                   "results": {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer))
                                        else float(vv)) for kk, vv in v.items()}
                               for k, v in results.items()}}, fh, indent=2, default=float)

    logger.info("=" * 64)
    if first_fail is None:
        logger.success("VERDICT: PASS — alpha survives T1–T6; admitted to the Alpha Library.")
    else:
        logger.error("VERDICT: FAIL — first falsifying test: {}. Investigation stops.", first_fail)
    logger.info("Artifacts -> {}", OUT)


if __name__ == "__main__":
    main()
