"""ALX4 forward-validation harness (executes the LOCKED forward protocol).

This does NOT invent forward data. It is the instrument that, from the lock date
onward, records the FROZEN ALX book's out-of-sample daily PnL/IC on bars generated
*after* the lock, accumulates them in an append-only ledger, and evaluates the
pre-registered confirm/fail criteria once enough forward history exists. Until
then it honestly reports OBSERVING.

Locked choices (from PREREGISTRATION.md §Forward):
- spec fully frozen (reused via characterize/validate);
- universe rule fixed to coverage-INTERSECTION (removes the union/intersection
  ambiguity: in-sample this was +0.21 vs +1.47);
- metrics + PASS/FAIL + horizon fixed in advance here as constants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha_library.alx_funding_price_validation.validate import sharpe
from alpha_library.alx3_external_replication import replicate as rep
from alpha_library.alx4_regime_analysis import characterize as ch

# ---- LOCKED constants (pre-registered; do not tune) --------------------------
LOCK_DATE = pd.Timestamp("2026-07-20", tz="UTC")   # ALX4 pre-registration lock
MIN_FORWARD_DAYS = 365                              # >= 12 months (prereg: 12-18mo)
COVERAGE = 0.98                                     # intersection-universe rule
WARMUP_DAYS = 40                                    # rolling(7)+lag warmup buffer
FAIL_DRAWDOWN = -0.30                               # worse than any in-sample year (~-17%)
LEDGER = Path("data/research/alx4/forward/ledger.parquet")
BASES = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "LTC", "LINK",
         "BCH", "AVAX", "TRX", "ATOM", "ETC", "XLM", "NEAR", "FIL", "UNI", "AAVE",
         "ICP", "APT", "ARB", "OP", "SAND", "MANA", "AXS", "EOS", "INJ", "RUNE",
         "GRT", "ALGO", "THETA", "EGLD", "CRV", "COMP", "SUSHI", "ZEC", "DASH", "GALA"]


# ----------------------------------------------------------------------- #
# ledger (append-only, dedup on date)
# ----------------------------------------------------------------------- #
def load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_parquet(LEDGER).sort_index()
    return pd.DataFrame(columns=["net", "ic", "mkt_ret"],
                        index=pd.DatetimeIndex([], tz="UTC", name="date"))


def append_ledger(rows: pd.DataFrame) -> pd.DataFrame:
    """Append new forward rows; dedup keeps the FIRST recorded value per date
    (a past forward day, once realized, is immutable — never overwrite it)."""
    cur = load_ledger()
    merged = pd.concat([cur, rows[~rows.index.isin(cur.index)]]).sort_index()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(LEDGER)
    return merged


# ----------------------------------------------------------------------- #
# recording: frozen book on the post-lock, intersection universe
# ----------------------------------------------------------------------- #
def _intersection_cols(close: pd.DataFrame, index) -> list[str]:
    sub = close.reindex(index)
    return [c for c in sub.columns if sub[c].notna().mean() >= COVERAGE]


def compute_forward_rows(close: pd.DataFrame, dfund: pd.DataFrame,
                         *, now: pd.Timestamp) -> pd.DataFrame:
    """Frozen ALX per-day net/IC for fully-closed days strictly AFTER LOCK_DATE."""
    today = pd.Timestamp(now).tz_convert("UTC").normalize()
    last_complete = today - pd.Timedelta(days=1)           # never a forming day
    start = LOCK_DATE - pd.Timedelta(days=WARMUP_DAYS)      # warmup only (not scored)
    idx = close.index[(close.index >= start) & (close.index <= last_complete)]
    if len(idx) < WARMUP_DAYS:
        return pd.DataFrame(columns=["net", "ic", "mkt_ret"])
    cols = _intersection_cols(close, idx)
    if len(cols) < 4:
        return pd.DataFrame(columns=["net", "ic", "mkt_ret"])
    c = close.loc[idx, cols]
    ret, fund, signal = ch.build_signal(c, dfund.loc[idx, cols] if set(cols).issubset(dfund.columns)
                                        else dfund.reindex(index=idx, columns=cols))
    net = ch.alx_net(ret, signal, fund)
    ic = ch.rank_ic_series(signal, ret)
    mkt = ret.mean(axis=1)
    out = pd.DataFrame({"net": net, "ic": ic, "mkt_ret": mkt})
    out = out[out.index > LOCK_DATE].dropna(subset=["net"])   # SCORE only post-lock
    out.index.name = "date"
    return out


def record(*, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fetch latest independent data, compute post-lock rows, append to ledger."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    close, qvol, dfund, audit = rep.build_daily("binance", BASES)  # cache or network
    rows = compute_forward_rows(close, dfund, now=now)
    if rows.empty:
        logger.info("record: no fully-closed post-lock days available yet.")
        return load_ledger()
    return append_ledger(rows)


# ----------------------------------------------------------------------- #
# evaluation: locked confirm/fail criteria
# ----------------------------------------------------------------------- #
def _has_bull_and_bear(mkt: pd.Series) -> tuple[bool, bool]:
    trend = mkt.rolling(30, min_periods=10).mean()
    return bool((trend > 0).any()), bool((trend <= 0).any())


def evaluate(ledger: pd.DataFrame | None = None) -> dict:
    led = load_ledger() if ledger is None else ledger
    n = len(led)
    if n == 0:
        return {"status": "OBSERVING", "n_days": 0,
                "message": "No forward data yet. Begin accumulating post-lock bars."}

    net = led["net"].to_numpy()
    ic = led["ic"].dropna()
    sh = sharpe(net)
    lo, hi = ch.block_bootstrap_ci(net) if n > 25 else (float("nan"), float("nan"))
    ic_t = ch.nw_tstat(ic.to_numpy())
    eq = np.cumprod(1 + net); peak = np.maximum.accumulate(eq)
    maxdd = float((eq / peak - 1).min())
    ssort = np.sort(net); k = max(1, int(n * 0.05))
    top5_share = float(ssort[-k:].sum() / net.sum()) if net.sum() != 0 else float("nan")
    has_bull, has_bear = _has_bull_and_bear(led["mkt_ret"])
    # IC in bull sub-sample
    trend = led["mkt_ret"].rolling(30, min_periods=10).mean()
    bull_ic = float(led["ic"][trend > 0].mean()) if (trend > 0).any() else float("nan")

    metrics = {"n_days": n, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
               "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
               "ic_nw_t": ic_t, "max_drawdown": maxdd, "top5pct_share": top5_share,
               "has_bull": has_bull, "has_bear": has_bear, "bull_ic": bull_ic}

    # ---- locked fail conditions (can fire even before horizon) ----
    if maxdd < FAIL_DRAWDOWN:
        return {"status": "FAIL", "message": f"drawdown {maxdd:.1%} worse than "
                f"locked {FAIL_DRAWDOWN:.0%}", **metrics}
    if has_bull and not np.isnan(bull_ic) and bull_ic < 0:
        return {"status": "FAIL", "message": "IC flipped negative in a bull phase",
                **metrics}

    # ---- horizon gate ----
    if n < MIN_FORWARD_DAYS or not (has_bull and has_bear):
        need = max(0, MIN_FORWARD_DAYS - n)
        return {"status": "OBSERVING", "message":
                f"{n}/{MIN_FORWARD_DAYS} forward days; need >= {need} more and "
                f"both regimes (bull={has_bull}, bear/flat={has_bear}).", **metrics}

    # ---- locked confirm criterion (horizon met) ----
    confirmed = sh > 0 and not np.isnan(lo) and lo > 0 and (np.isnan(bull_ic) or bull_ic > 0)
    status = "CONFIRMED" if confirmed else "FAIL"
    msg = ("forward Sharpe CI excludes 0 and IC positive in bull -> regime alpha"
           if confirmed else "horizon met but CI includes 0 -> not distinguishable from 0")
    return {"status": status, "message": msg, **metrics}
