"""ALX6 breadth — DATA STEP ONLY: survivorship-aware Binance USDT-M universe.

This is pure data acquisition for the (not-yet-pre-registered) breadth
experiment. It runs NO backtest, computes NO Sharpe, and tests NO hypothesis —
so it may precede the pre-registration. The falsification experiment itself
(book construction, gates) must NOT be written until `PREREGISTRATION.md` is
locked.

What it does:
  1. Pull the full Binance USDT-M PERPETUAL symbol list WITH per-symbol
     onboard (listing) dates — the point-in-time entry anchor that lets the
     experiment admit each name only from its listing, killing entry look-ahead.
  2. Rank candidates by *current* 24h quote (dollar) volume — a cheap
     data-acquisition budget cap, NOT an alpha decision. The pre-registered
     experiment decides the real PIT membership + liquidity screen.
  3. Fetch daily klines + funding history for the top-N candidates, reusing the
     FROZEN ALX3 raw-pull primitives (`alx3_external_replication.data_sources`)
     unchanged (shared venue-level cache under data/research/alx3/cache).
  4. Emit a universe manifest (symbol, base, listing date, 24h volume, history
     span, coverage) under data/research/alx6/.

KNOWN RESIDUAL SURVIVORSHIP LIMIT (documented, not hidden): exchangeInfo and the
24h ticker list only CURRENTLY-listed symbols. Onboard dates make *entry*
point-in-time correct, but fully **delisted** coins are absent — so a
delisting-survivorship bias remains and inflates any breadth backtest. The
pre-registration must treat the raw breadth Sharpe as an OPTIMISTIC ceiling and
gate accordingly (e.g. robustness to dropping the newest / most-illiquid tail).

    py alpha_library/alx6_breadth/fetch_universe.py --max 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from alpha_library.alx3_external_replication import data_sources as ds

OUT = Path("data/research/alx6")
_BINANCE = "https://fapi.binance.com"

# Non-tradeable / degenerate quote symbols to skip (stable-on-stable, indices).
_SKIP = {"USDCUSDT", "USDPUSDT", "TUSDUSDT", "FDUSDUSDT", "BUSDUSDT", "EURUSDT",
         "AEURUSDT", "USD1USDT"}


def binance_24h_quote_volume() -> dict[str, float]:
    """Current 24h quote (USDT) volume per symbol — one call, ranks candidates."""
    data = ds._get(f"{_BINANCE}/fapi/v1/ticker/24hr")
    return {d["symbol"]: float(d["quoteVolume"]) for d in data
            if d["symbol"].endswith("USDT")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=120,
                    help="max candidate symbols to fetch (by current 24h volume)")
    ap.add_argument("--min-vol", type=float, default=1e6,
                    help="drop symbols below this 24h quote volume before ranking")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1) full universe + listing dates (survivorship-aware entry anchor) --
    onboard = ds.binance_onboard()  # {symbol: onboard_ms} for live USDT perps
    logger.info("Binance USDT-M perpetuals currently listed: {}", len(onboard))

    # ---- 2) rank candidates by current 24h dollar volume (budget cap only) ---
    vol = binance_24h_quote_volume()
    cand = [(s, vol.get(s, 0.0)) for s in onboard
            if s not in _SKIP and vol.get(s, 0.0) >= args.min_vol]
    cand.sort(key=lambda x: x[1], reverse=True)
    picked = cand[: args.max]
    logger.info("candidates >= {:.0f} vol: {} | fetching top {}",
                args.min_vol, len(cand), len(picked))

    # ---- 3) fetch klines + funding (reuse FROZEN alx3 primitives, cached) ----
    rows = []
    for i, (sym, qv) in enumerate(picked, 1):
        base = sym[:-4]  # strip 'USDT'
        kl, fu = ds.fetch_symbol("binance", base)
        if kl is None or fu is None:
            logger.warning("  [{}/{}] {}: no data", i, len(picked), sym)
            continue
        onboard_dt = pd.to_datetime(onboard[sym], unit="ms", utc=True)
        rows.append({
            "symbol": sym, "base": base,
            "onboard_date": onboard_dt.date().isoformat(),
            "vol24h_usdt": qv,
            "kl_start": kl["date"].min().date().isoformat(),
            "kl_end": kl["date"].max().date().isoformat(),
            "n_days": int(kl["date"].nunique()),
            "n_funding": int(len(fu)),
        })
        if i % 20 == 0:
            logger.info("  fetched {}/{}", i, len(picked))

    # ---- 4) manifest ---------------------------------------------------------
    man = pd.DataFrame(rows).sort_values("vol24h_usdt", ascending=False)
    man.to_csv(OUT / "universe_manifest.csv", index=False)
    (OUT / "universe_manifest.json").write_text(
        json.dumps({"n_listed": len(onboard), "n_candidates": len(cand),
                    "n_fetched": len(man), "min_vol": args.min_vol,
                    "max": args.max, "symbols": man["symbol"].tolist()},
                   indent=2))
    logger.info("=" * 66)
    logger.info("fetched {} symbols | history spans {} .. {}",
                len(man),
                man["kl_start"].min() if len(man) else None,
                man["kl_end"].max() if len(man) else None)
    logger.info("manifest -> {}", OUT / "universe_manifest.csv")
    logger.info("NOTE: delisted coins are absent (residual survivorship); the "
                "pre-registration must treat raw breadth Sharpe as an optimistic ceiling.")


if __name__ == "__main__":
    main()
