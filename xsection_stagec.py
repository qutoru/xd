"""E7 Stage C: does the cross-sectional reversal signal survive execution costs?

Runs the FULL pre-registered robustness grid (not just the best cell), reports
gross/net Sharpe, turnover, cost attribution, and the deep-dive diagnostics
(walk-forward, year-by-year, regime, liquidity subset, capacity). Falsification-
first: if no config clears net Sharpe > 0 at the realistic (Medium) cost, that
is the answer and we stop.

Usage:
    py xsection_stagec.py --interval 60
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

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

OUT = Path("data/research/e7")

# --- Pre-registered grid + cost assumptions (decided before running) ---------
HOLD = [1, 2, 4, 6, 12, 24]
REBAL = [1, 2, 4]
KPCT = [0.10, 0.20, 0.30]
LAG = [1, 2]
COSTS = {"low": 0.0002, "med": 0.00075, "high": 0.0015}  # per unit |Δw|
REVERSAL_LOOKBACK = 3  # fixed, from Stage B — NOT tuned
# Pre-declared representative config for the deep dives (middle of grid).
REP = dict(hold=4, rebalance=1, k_pct=0.20, exec_lag=1)


def _load(interval: str, daily: bool = False):
    panel = build_returns_panel(SURVIVORSHIP_SAFE, "60" if daily else interval)
    if daily:
        panel = to_daily(panel)
    simple = panel.pct_change().dropna(how="all")
    logret = np.log(panel / panel.shift(1)).dropna(how="all")
    feat = ft.short_term_reversal(logret, lookback=REVERSAL_LOOKBACK)
    return panel, simple, feat


def run_grid(simple, feat, interval_min) -> pd.DataFrame:
    rows = []
    for h, r, k, lag, (cname, cval) in itertools.product(HOLD, REBAL, KPCT, LAG, COSTS.items()):
        if h < r:
            continue
        res = pf.backtest_portfolio(simple, feat, hold=h, rebalance=r, k_pct=k,
                                    exec_lag=lag, cost=cval)
        m = pf.portfolio_metrics(res, interval_min)
        rows.append({"hold": h, "rebalance": r, "k_pct": k, "exec_lag": lag,
                     "cost": cname, **m})
    return pd.DataFrame(rows)


def _sharpe(x: np.ndarray, interval_min) -> float:
    from crypto_signal_bot.research.metrics import bars_per_year
    x = np.asarray(x, dtype="float64")
    sd = x.std(ddof=1) if x.size > 1 else 0.0
    return float(x.mean() / sd * np.sqrt(bars_per_year(interval_min))) if sd > 0 else 0.0


def deep_dive(simple, feat, interval_min, cost, index):
    """Walk-forward, year-by-year, and regime net Sharpe for the rep config."""
    res = pf.backtest_portfolio(simple, feat, cost=cost, **REP)
    net = pd.Series(res.net, index=index)

    # Walk-forward (6 folds).
    folds = np.array_split(np.arange(len(net)), 6)
    wf = [{"fold": i + 1, "net_sharpe": _sharpe(net.iloc[ix].to_numpy(), interval_min)}
          for i, ix in enumerate(folds)]

    # Year by year.
    yby = [{"year": int(y), "net_sharpe": _sharpe(g.to_numpy(), interval_min), "bars": len(g)}
           for y, g in net.groupby(net.index.year)]

    # Volatility regime (market-vol terciles).
    mkt_vol = simple.mean(axis=1).rolling(96).std().shift(1).reindex(net.index)
    q1, q2 = mkt_vol.quantile([1 / 3, 2 / 3])
    lab = pd.Series("mid", index=net.index)
    lab[mkt_vol <= q1] = "low_vol"; lab[mkt_vol >= q2] = "high_vol"
    reg = [{"regime": nm, "net_sharpe": _sharpe(net[lab == nm].to_numpy(), interval_min)}
           for nm in ("low_vol", "mid", "high_vol")]
    return pd.DataFrame(wf), pd.DataFrame(yby), pd.DataFrame(reg)


def liquidity_subset(panel, data_interval, interval_min, cost):
    """Q6: re-run the rep config on the top-liquidity half of the universe.

    Liquidity is always ranked from the 1h raw files; ``panel`` is at the target
    frequency (daily or hourly), so the sub-portfolio inherits that frequency.
    """
    vols = {}
    for sym in panel.columns:
        raw = load_parquet(sym, data_interval)
        vols[sym] = float(np.median(raw["turnover"].to_numpy()))  # quote (USDT) volume / 1h bar
    med = pd.Series(vols).sort_values()
    top_half = list(med.index[len(med) // 2:])
    sub = panel[top_half]
    simple = sub.pct_change().dropna(how="all")
    feat = ft.short_term_reversal(np.log(sub / sub.shift(1)).dropna(how="all"), lookback=REVERSAL_LOOKBACK)
    res = pf.backtest_portfolio(simple, feat, cost=cost, **REP)
    return pf.portfolio_metrics(res, interval_min), len(top_half), med


def capacity_estimate(univ_dollar_per_bar: float, turnover_annual: float, interval_min: int) -> float:
    """Q7: coarse AUM capacity at 1% participation of universe dollar volume."""
    from crypto_signal_bot.research.metrics import bars_per_year
    turnover_per_bar = turnover_annual / bars_per_year(interval_min)
    if turnover_per_bar <= 0:
        return float("inf")
    return 0.01 * univ_dollar_per_bar / turnover_per_bar


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--daily", action="store_true",
                    help="Resample to daily bars (E8, low-frequency); grid H/R in days.")
    args = ap.parse_args()
    data_interval = "60"
    if args.daily:
        interval_min = 24 * 60
        OUT = Path("data/research/e8")
    else:
        interval_min = args.interval
        data_interval = str(args.interval)
    OUT.mkdir(parents=True, exist_ok=True)

    panel, simple, feat = _load(str(args.interval), daily=args.daily)
    unit = "days" if args.daily else "bars"
    logger.info("Panel {} symbols x {} {} (grid H/R in {})", panel.shape[1], panel.shape[0], unit, unit)

    grid = run_grid(simple, feat, interval_min)
    grid.to_csv(OUT / "stagec_grid.csv", index=False)

    # Summary: net Sharpe surface at the realistic (Medium) cost.
    med = grid[grid["cost"] == "med"].sort_values("net_sharpe", ascending=False)
    logger.info("=== Net Sharpe @ Medium cost (top 8 of {} configs) ===", len(med))
    for _, r in med.head(8).iterrows():
        logger.info("  H={:>2} R={} K={:.0%} lag={} | gross={:+.2f} net={:+.2f} "
                    "turn/yr={:.0f} costDrag={:+.1%} maxDD={:+.1%}",
                    int(r.hold), int(r.rebalance), r.k_pct, int(r.exec_lag),
                    r.gross_sharpe, r.net_sharpe, r.turnover_annual,
                    r.cost_drag_annual, r.max_drawdown)

    best_med_net = med["net_sharpe"].max()
    n_pos_med = int((med["net_sharpe"] > 0).sum())
    logger.info("Configs with net Sharpe>0 @ Medium: {}/{} (best={:+.2f})",
                n_pos_med, len(med), best_med_net)

    # Does IC even translate to positive GROSS (before any cost)? (research Q1)
    gbest = grid.loc[grid["gross_sharpe"].idxmax()]
    logger.info("Best GROSS Sharpe over whole grid: {:+.2f} at H={} R={} K={:.0%} lag={} "
                "(n gross>0: {}/{})", gbest.gross_sharpe, int(gbest.hold), int(gbest.rebalance),
                gbest.k_pct, int(gbest.exec_lag), int((grid["gross_sharpe"] > 0).sum()), len(grid))

    # Cost sensitivity of the rep config.
    logger.info("=== Cost sensitivity (rep config H=4 R=1 K=20% lag=1) ===")
    for cname, cval in COSTS.items():
        res = pf.backtest_portfolio(simple, feat, cost=cval, **REP)
        m = pf.portfolio_metrics(res, interval_min)
        logger.info("  cost={:>4} ({:.2%}/turn): gross={:+.2f} net={:+.2f}",
                    cname, cval, m["gross_sharpe"], m["net_sharpe"])

    # Deep dives at Medium cost.
    wf, yby, reg = deep_dive(simple, feat, interval_min, COSTS["med"], simple.index)
    logger.info("Walk-forward net Sharpe (rep@med): {}", [round(v, 2) for v in wf["net_sharpe"]])
    logger.info("Year-by-year net Sharpe (rep@med): {}",
                {int(r.year): round(r.net_sharpe, 2) for _, r in yby.iterrows()})
    logger.info("Regime net Sharpe (rep@med): {}",
                {r.regime: round(r.net_sharpe, 2) for _, r in reg.iterrows()})

    # Q6 liquidity subset + Q7 capacity.
    liq_m, n_liq, med_vol = liquidity_subset(panel, data_interval, interval_min, COSTS["med"])
    logger.info("Liquidity subset (top {} names) net Sharpe @med: {:+.2f}", n_liq, liq_m["net_sharpe"])
    # Capacity from the representative config's turnover. Universe dollar volume
    # is per 1h bar; scale to per-day when in daily mode.
    rep_res = pf.backtest_portfolio(simple, feat, cost=COSTS["med"], **REP)
    rep_m = pf.portfolio_metrics(rep_res, interval_min)
    univ_dollar = float(med_vol.sum()) * (24 if args.daily else 1)
    cap = capacity_estimate(univ_dollar, rep_m["turnover_annual"], interval_min)
    logger.info("Capacity estimate (rep config, 1% participation): ~${:,.0f}", cap)

    wf.to_csv(OUT / "stagec_walkforward.csv", index=False)
    yby.to_csv(OUT / "stagec_year_by_year.csv", index=False)
    reg.to_csv(OUT / "stagec_regime.csv", index=False)

    # Plot: net Sharpe vs holding period at each cost (rebalance=1, K=20%, lag=1).
    fig, ax = plt.subplots(figsize=(8, 4))
    for cname, cval in COSTS.items():
        ys = []
        for h in HOLD:
            res = pf.backtest_portfolio(simple, feat, hold=h, rebalance=1, k_pct=0.20,
                                        exec_lag=1, cost=cval)
            ys.append(pf.portfolio_metrics(res, interval_min)["net_sharpe"])
        ax.plot(HOLD, ys, marker="o", label=f"cost={cname}")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("holding period (bars=h)")
    ax.set_ylabel("net Sharpe"); ax.set_title("Net Sharpe vs holding period & cost")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "stagec_holding_cost.png", dpi=110); plt.close(fig)

    logger.info("=" * 60)
    tag = "E8 (daily)" if args.daily else "E7 (1h)"
    if args.daily:
        # Pre-registered E8 pass: >=1 config net Sharpe>=1.0 @ Med AND WF net>0 in >=4/6.
        wf_pos = int((wf["net_sharpe"] > 0).sum())
        passed = best_med_net >= 1.0 and wf_pos >= 4
        logger.info("{} PRE-COMMITTED PASS = (best net Sharpe>=1.0 @Med AND WF>0 in>=4/6). "
                    "best_net={:+.2f}, WF>0 folds={}/6 -> {}", tag, best_med_net, wf_pos,
                    "PASS" if passed else "FAIL -> hypothesis falsified, STOP")
    elif n_pos_med == 0:
        logger.info("{} STAGE C VERDICT: net Sharpe <= 0 for ALL configs at realistic cost "
                    "-> alpha does NOT survive execution. STOP.", tag)
    else:
        logger.info("{} STAGE C: {} configs survive net>0 @ realistic cost (best {:+.2f}).",
                    tag, n_pos_med, best_med_net)
    logger.info("Artifacts in {}", OUT)


if __name__ == "__main__":
    main()
