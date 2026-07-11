"""Research runner: strict validation of a rule hypothesis on BTCUSDT.

Usage:
    py research_cli.py h3 --interval 60
    py research_cli.py h3 --interval 15
"""

from __future__ import annotations

import argparse
import functools

import pandas as pd
from loguru import logger

from crypto_signal_bot.data.storage import load_parquet
from crypto_signal_bot.research import evaluate as ev
from crypto_signal_bot.research import signals as sig

# Hypothesis registry: name -> (signal_fn, default params, perturbation grid).
HYPOTHESES = {
    "h3": (sig.sweep_reversion, {"lookback": 20, "hold": 8},
           {"lookback": [15, 20, 25, 30], "hold": [4, 8, 12, 16]}),
    "h5": (sig.funding_reversion, {"z_window": 96, "z_thresh": 1.5, "hold": 8},
           {"z_thresh": [1.0, 1.5, 2.0], "hold": [4, 8, 12]}),
    "h6": (sig.oi_divergence, {"ret_window": 4, "oi_window": 4, "hold": 8},
           {"ret_window": [2, 4, 8], "hold": [4, 8, 12]}),
}

_PANEL = ["sharpe", "sortino", "calmar", "max_drawdown", "win_rate",
          "profit_factor", "expectancy", "exposure", "turnover_annual", "n_trades"]


def _fmt(m: dict) -> str:
    return (f"sharpe={m['sharpe']:+.2f} sortino={m['sortino']:+.2f} calmar={m['calmar']:+.2f} "
            f"maxDD={m['max_drawdown']:+.1%} win={m['win_rate']:.1%} pf={m['profit_factor']:.2f} "
            f"exp={m['expectancy']:+.4%} expo={m['exposure']:.1%} "
            f"turn={m['turnover_annual']:.0f} trades={m['n_trades']}")


def verdict(splits: dict) -> list[str]:
    """Apply the STEP 6 rejection criteria to the valid/test segments."""
    v, t = splits["valid"], splits["test"]
    fails = []
    if v["sharpe"] < 1.2:
        fails.append(f"valid Sharpe {v['sharpe']:.2f} < 1.2")
    if t["sharpe"] < 1.0:
        fails.append(f"test Sharpe {t['sharpe']:.2f} < 1.0")
    if v["sharpe"] > 0 and abs(v["sharpe"] - t["sharpe"]) / abs(v["sharpe"]) > 0.25:
        fails.append(f"|valid-test| Sharpe gap > 25% ({v['sharpe']:.2f} vs {t['sharpe']:.2f})")
    if t["max_drawdown"] < -0.20:
        fails.append(f"test drawdown {t['max_drawdown']:.1%} > 20%")
    if t["n_trades"] < 200:
        fails.append(f"test trades {t['n_trades']} < 200")
    return fails


def run(name: str, interval: int) -> None:
    signal_fn, params, grid = HYPOTHESES[name]
    fn = functools.partial(signal_fn, **params)

    # H5/H6 need derivatives columns; load the enriched frame when present.
    if name in ("h5", "h6"):
        import pandas as pd
        from crypto_signal_bot.config import RAW_DIR
        df = pd.read_parquet(RAW_DIR / f"BTCUSDT_{interval}_deriv.parquet")
    else:
        df = load_parquet("BTCUSDT", str(interval))
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("=== {} on BTCUSDT {}m — {} bars ({} .. {}) ===",
                name.upper(), interval, len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1])

    # STEP 5: metric panel on 60/20/20.
    splits = ev.evaluate_splits(df, fn, interval)
    for seg in ("train", "valid", "test"):
        logger.info("[{:>5}] {}", seg, _fmt(splits[seg]))

    # STEP 6: verdict.
    fails = verdict(splits)
    if fails:
        logger.warning("VERDICT: REJECT — " + "; ".join(fails))
    else:
        logger.success("VERDICT: PASS gate — proceeding to robustness")

    # Walk-forward.
    wf = ev.walk_forward(df, fn, interval)
    logger.info("Walk-forward folds:\n{}", wf.to_string(index=False))
    logger.info("  WF Sharpe mean={:.2f} median={:.2f} positive={}/{}",
                wf["sharpe"].mean(), wf["sharpe"].median(), int((wf["sharpe"] > 0).sum()), len(wf))

    # STEP 7: robustness.
    pert = ev.perturb_params(df, signal_fn, interval, grid)
    logger.info("Param perturbation Sharpe: min={:.2f} median={:.2f} max={:.2f} (frac>0={:.0%})",
                pert["sharpe"].min(), pert["sharpe"].median(), pert["sharpe"].max(),
                (pert["sharpe"] > 0).mean())
    sens = ev.sensitivity(df, fn(df), interval)
    logger.info("Sensitivity (Sharpe): {}", {k: round(v, 2) for k, v in sens.items()})
    mc = ev.monte_carlo(df, fn(df), interval)
    logger.info("Monte-Carlo Sharpe: p05={p05:.2f} p50={p50:.2f} p95={p95:.2f} frac>0={frac_positive:.0%}".format(**mc))
    reg = ev.regime_split(df, fn, interval)
    logger.info("Regime Sharpe:\n{}", reg.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hypothesis", choices=list(HYPOTHESES))
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    run(args.hypothesis, args.interval)


if __name__ == "__main__":
    main()
