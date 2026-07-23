"""ALX10 forward-paper harness — executes the LOCKED forward protocol.

Does NOT invent forward data. From LOCK_DATE onward it records, for each
fully-closed post-lock daily bar, the two FROZEN books' out-of-sample net PnL
(Book A = funding-only ALX5; Book B = risk-parity blend ALX9) plus funding IC and
market return, in an append-only immutable ledger, and evaluates the
pre-registered CONFIRM / FAIL / GRADUATE criteria once the horizon + both regimes
are reached. Until then it honestly reports OBSERVING.

Operational note: `record` pulls data via the frozen ALX3 primitives. To ingest
genuinely NEW bars the per-symbol raw cache must be refreshed (the cache returns
stored history as-is); at LOCK there are zero post-lock bars, so OBSERVING is the
correct state. Nothing here is tuned; all constants are pre-registered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha_library.alx_funding_price_validation.validate import sharpe
from alpha_library.alx3_external_replication import replicate as rep
from alpha_library.alx4_regime_analysis import characterize as ch
from alpha_library.alx8_momentum_sleeve import momentum as mo
from alpha_library.alx9_riskparity_blend import riskparity as rp

# ---- LOCKED constants (pre-registered; do not tune) --------------------------
LOCK_DATE = pd.Timestamp("2026-07-23", tz="UTC")
MIN_FORWARD_DAYS = 365
WARMUP_DAYS = 90                 # 60d blend-vol on top of 24d momentum + 7d funding
FAIL_DRAWDOWN = -0.30
FAIL_FILL_RATIO = 0.50           # execution-reality trip (testnet track)
MIN_TESTNET = 60
LEDGER = Path("data/research/alx10/forward/ledger.parquet")
BASES = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "LTC", "LINK",
         "BCH", "AVAX", "TRX", "ATOM", "ETC", "XLM", "NEAR", "FIL", "UNI", "AAVE",
         "ICP", "APT", "ARB", "OP", "SAND", "MANA", "AXS", "EOS", "INJ", "RUNE",
         "GRT", "ALGO", "THETA", "EGLD", "CRV", "COMP", "SUSHI", "ZEC", "DASH", "GALA"]
_COLS = ["net_A", "net_B", "ic_A", "mkt_ret"]


# ----------------------------------------------------------------------- #
# append-only immutable ledger (dedup keeps the FIRST value per date)
# ----------------------------------------------------------------------- #
def load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_parquet(LEDGER).sort_index()
    return pd.DataFrame(columns=_COLS,
                        index=pd.DatetimeIndex([], tz="UTC", name="date"))


def append_ledger(rows: pd.DataFrame) -> pd.DataFrame:
    """Append only dates not already present; a realized forward day is immutable."""
    cur = load_ledger()
    merged = pd.concat([cur, rows[~rows.index.isin(cur.index)]]).sort_index()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(LEDGER)
    return merged


# ----------------------------------------------------------------------- #
# record: both frozen books on post-lock bars
# ----------------------------------------------------------------------- #
def compute_forward_rows(close: pd.DataFrame, dfund: pd.DataFrame,
                         *, now: pd.Timestamp) -> pd.DataFrame:
    """Frozen Book-A/Book-B per-day net (+IC, mkt) for closed days AFTER LOCK_DATE."""
    today = pd.Timestamp(now).tz_convert("UTC").normalize()
    last_complete = today - pd.Timedelta(days=1)
    start = LOCK_DATE - pd.Timedelta(days=WARMUP_DAYS)
    idx = close.index[(close.index >= start) & (close.index <= last_complete)]
    if len(idx) < WARMUP_DAYS:
        return pd.DataFrame(columns=_COLS)
    cols = list(close.columns)
    c = close.loc[idx, cols]
    f = dfund.reindex(index=idx, columns=cols)
    ret, fund, fsig = ch.build_signal(c, f)
    ret_np = ret.to_numpy("float64")
    fund_np = fund.reindex_like(ret).to_numpy("float64")

    fb = mo.sleeve(ret, fsig, fund)                         # Book A ingredients
    mb = mo.sleeve(ret, mo.momentum_signal(ret), fund)      # momentum sleeve
    net_A = mo.realistic_net(fb)
    net_M = mo.realistic_net(mb)
    net_B, _, _, _ = rp.rp_blend_net(fb["applied"], mb["applied"], net_A, net_M,
                                     ret_np, fund_np)        # Book B

    out = pd.DataFrame({"net_A": net_A, "net_B": net_B,
                        "ic_A": ch.rank_ic_series(fsig, ret),
                        "mkt_ret": ret.mean(axis=1)}, index=ret.index)
    out = out[out.index > LOCK_DATE].dropna(subset=["net_A", "net_B"])
    out.index.name = "date"
    return out


def record(*, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    close, qvol, dfund, _ = rep.build_daily("binance", BASES)
    rows = compute_forward_rows(close, dfund, now=now)
    if rows.empty:
        logger.info("record: no fully-closed post-lock days available yet.")
        return load_ledger()
    return append_ledger(rows)


# ----------------------------------------------------------------------- #
# evaluate: locked confirm / fail / graduate criteria
# ----------------------------------------------------------------------- #
def _regimes(mkt: pd.Series) -> tuple[bool, bool]:
    trend = mkt.rolling(30, min_periods=10).mean()
    return bool((trend > 0).any()), bool((trend <= 0).any())


def _maxdd(x: np.ndarray) -> float:
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return float((eq / pk - 1).min()) if len(x) else 0.0


def evaluate(ledger: pd.DataFrame | None = None) -> dict:
    led = load_ledger() if ledger is None else ledger
    n = len(led)
    if n == 0:
        return {"status": "OBSERVING", "n_days": 0,
                "message": "No forward data yet. Accumulating post-lock bars."}

    net_A, net_B = led["net_A"].to_numpy(), led["net_B"].to_numpy()
    ic = led["ic_A"].dropna()
    sh_A, sh_B = sharpe(net_A), sharpe(net_B)
    lo_A, hi_A = ch.block_bootstrap_ci(net_A) if n > 25 else (float("nan"), float("nan"))
    ic_t = ch.nw_tstat(ic.to_numpy()) if len(ic) else 0.0
    dd_A, dd_B = _maxdd(net_A), _maxdd(net_B)
    has_bull, has_bear = _regimes(led["mkt_ret"])
    trend = led["mkt_ret"].rolling(30, min_periods=10).mean()
    bull_ic = float(led["ic_A"][trend > 0].mean()) if (trend > 0).any() else float("nan")
    d = net_B - net_A
    mean_d, t_d = float(d.mean()), ch.nw_tstat(d)

    m = {"n_days": n, "sharpe_A": sh_A, "sharpe_B": sh_B, "ci_lo_A": lo_A, "ci_hi_A": hi_A,
         "ic_mean_A": float(ic.mean()) if len(ic) else float("nan"), "ic_nw_t_A": ic_t,
         "maxdd_A": dd_A, "maxdd_B": dd_B, "has_bull": has_bull, "has_bear": has_bear,
         "bull_ic_A": bull_ic, "paired_mean_BmA": mean_d, "paired_nw_t_BmA": t_d}

    # ---- early-FAIL trips (may fire before horizon) ----
    if dd_A < FAIL_DRAWDOWN:
        return {"status": "FAIL", "message":
                f"Book A drawdown {dd_A:.1%} worse than locked {FAIL_DRAWDOWN:.0%}", **m}
    if has_bull and not np.isnan(bull_ic) and bull_ic < 0:
        return {"status": "FAIL", "message": "Book A IC flipped negative in a bull phase", **m}

    # ---- horizon gate ----
    if n < MIN_FORWARD_DAYS or not (has_bull and has_bear):
        return {"status": "OBSERVING", "message":
                f"{n}/{MIN_FORWARD_DAYS} forward days; need {max(0, MIN_FORWARD_DAYS - n)} "
                f"more and both regimes (bull={has_bull}, bear/flat={has_bear}).", **m}

    # ---- Book A CONFIRM ----
    a_confirm = sh_A > 0 and not np.isnan(lo_A) and lo_A > 0 and ic_t > 0 and \
        (np.isnan(bull_ic) or bull_ic > 0)
    if not a_confirm:
        return {"status": "FAIL_A", "message":
                "horizon met but Book A forward Sharpe CI includes 0 / IC not positive", **m}

    # ---- Book B GRADUATE (only assessed if A confirms) ----
    b_grad = (mean_d > 0) and (t_d >= 2.0) and (dd_B >= dd_A)
    if b_grad:
        return {"status": "GRADUATE_B", "message":
                "Book A confirmed OOS AND blend beats it on paired test -> blend graduates", **m}
    return {"status": "CONFIRM_A", "message":
            "Book A confirmed OOS; blend does NOT beat it on paired test -> funding-only stands", **m}
