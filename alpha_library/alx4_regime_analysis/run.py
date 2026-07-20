"""ALX4 one-command characterization + falsification runner (see PREREGISTRATION).

    py alpha_library/alx4_regime_analysis/run.py

Runs the frozen ALX book on the independent Binance reconstruction (cached by
ALX3), then: P1 funding-shuffle placebo (decisive), P2 directionality, P3 factor
independence, and descriptive diagnostics (per-year, rolling, regime table,
concentration). Prints a verdict and writes artifacts to data/research/alx4/.
Single run, no tuning.
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

OUT = Path("data/research/alx4")
BASES = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "LTC", "LINK",
         "BCH", "AVAX", "TRX", "ATOM", "ETC", "XLM", "NEAR", "FIL", "UNI", "AAVE",
         "ICP", "APT", "ARB", "OP", "SAND", "MANA", "AXS", "EOS", "INJ", "RUNE",
         "GRT", "ALGO", "THETA", "EGLD", "CRV", "COMP", "SUSHI", "ZEC", "DASH", "GALA"]


def _yearly(net: pd.Series, ic: pd.Series) -> dict:
    def maxdd(x):
        eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
        return float((eq / pk - 1).min()) if len(x) else 0.0
    out = {}
    for y, g in net.groupby(net.index.year):
        x = g.to_numpy(); icy = ic[ic.index.year == y].dropna()
        out[int(y)] = {"days": len(x), "sharpe": round(sharpe(x), 2),
                       "win": round(float((x > 0).mean()), 3),
                       "maxdd": round(maxdd(x), 3),
                       "mean_ic": round(float(icy.mean()) if len(icy) else float("nan"), 4)}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    close, qvol, dfund, audit = rep.build_daily("binance", BASES)  # cached, no network
    ret, fund, signal = ch.build_signal(close, dfund)
    net = ch.alx_net(ret, signal, fund)
    ic = ch.rank_ic_series(signal, ret)
    full_sh = sharpe(net.to_numpy())
    lo, hi = ch.block_bootstrap_ci(net.to_numpy())

    logger.info("=" * 66)
    logger.info("ALX4 — frozen ALX on INDEPENDENT Binance ({} .. {}, {} days, {} sym)",
                ret.index.min().date(), ret.index.max().date(), len(ret), ret.shape[1])
    logger.info("Full-sample net Sharpe = {:+.3f}  bootstrap 95% CI [{:+.2f}, {:+.2f}]",
                full_sh, lo, hi)

    # ---------- P1 — decisive funding-shuffle placebo ----------
    logger.info("-" * 66)
    logger.info("P1 — FUNDING CROSS-SECTIONAL SHUFFLE PLACEBO (decisive)")
    p1 = ch.shuffle_placebo(ret, fund, n=500, seed=0)
    logger.info("  real Sharpe={:+.2f}  shuffle median={:+.2f}  shuffle 97.5pct={:+.2f}  max={:+.2f}",
                p1["real_sharpe"], p1["median"], p1["pct97_5"], p1["max"])
    logger.info("  real beats {:.1%} of 500 shuffles  ->  P1 {}",
                p1["pct_below_real"], "PASS" if p1["pass"] else "FAIL")

    # ---------- P2 — directionality ----------
    logger.info("-" * 66)
    logger.info("P2 — DIRECTIONALITY (funding->return vs return->funding)")
    p2 = ch.directionality(ret, signal)
    logger.info("  forward  IC={:+.4f} (NW t={:+.2f})   reverse IC={:+.4f} (NW t={:+.2f})",
                p2["fwd_mean_ic"], p2["fwd_nw_t"], p2["rev_mean_ic"], p2["rev_nw_t"])
    p2_ok = p2["fwd_mean_ic"] > 0 and p2["fwd_nw_t"] >= 3

    # ---------- P3 — factor independence ----------
    logger.info("-" * 66)
    logger.info("P3 — FACTOR INDEPENDENCE")
    p3 = ch.factor_independence(ret, fund, qvol, net)
    logger.info("  corr: {}", {k: round(v, 3) for k, v in p3["corrs"].items()})
    logger.info("  multivariate OLS R^2 = {:.1%}  ->  P3 {}",
                p3["r2"], "PASS" if p3["pass"] else "FAIL")

    # ---------- diagnostics ----------
    logger.info("-" * 66)
    logger.info("DIAGNOSTICS (descriptive; not pass/fail)")
    yearly = _yearly(net, ic)
    for y, d in yearly.items():
        logger.info("  {}: Sharpe={:+.2f} win={:.0%} maxDD={:+.0%} IC={:+.4f} (n={})",
                    y, d["sharpe"], d["win"], d["maxdd"], d["mean_ic"], d["days"])
    reg = ch.regime_features(ret, signal)
    cond = ch.conditional_table(net, ic, reg)
    for v, buckets in cond.items():
        logger.info("  regime {:<10}: " + "  ".join(
            f"{L}={buckets[L]['sharpe']:+.2f}(IC{buckets[L]['ic']:+.3f})"
            for L in ["low", "mid", "high"]), v)

    def drop_best(x, p):
        x = np.sort(np.asarray(x)); k = int(len(x) * p)
        return sharpe(x[:-k]) if k > 0 else sharpe(x)
    xnet = net.to_numpy()
    conc = {f"drop_best_{int(p*100)}pct": round(drop_best(xnet, p), 2)
            for p in (0.0, 0.01, 0.02, 0.05, 0.10)}
    logger.info("  top-day removal Sharpe: {}", conc)

    # ---------- verdict ----------
    mechanism_survives = p1["pass"] and p3["pass"]
    logger.info("=" * 66)
    logger.info("P1(shuffle)={}  P2(direction)={}  P3(independence)={}",
                "PASS" if p1["pass"] else "FAIL",
                "OK" if p2_ok else "WEAK",
                "PASS" if p3["pass"] else "FAIL")
    if mechanism_survives:
        logger.info("VERDICT: mechanism SURVIVES cheap falsifiers -> status stays")
        logger.info("  🟡 promising, PENDING the locked forward protocol (NOT graduated).")
    else:
        logger.info("VERDICT: mechanism FALSIFIED -> downgrade toward 🔴 insufficient evidence.")

    # ---------- artifacts ----------
    summary = {
        "window": [str(ret.index.min().date()), str(ret.index.max().date())],
        "n_days": len(ret), "n_symbols": int(ret.shape[1]),
        "full_sharpe": full_sh, "ci": [lo, hi],
        "P1_shuffle": {k: (v if k != "dist" else None) for k, v in p1.items()},
        "P2_direction": p2, "P3_independence": p3,
        "yearly": yearly, "regime": cond, "concentration": conc,
        "verdict": "mechanism_survives" if mechanism_survives else "mechanism_falsified",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.save(OUT / "shuffle_dist.npy", p1["dist"])
    logger.info("Artifacts -> {}", OUT)


if __name__ == "__main__":
    main()
