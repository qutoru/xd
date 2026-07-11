"""Reproduce the entire E7 cross-sectional experiment from scratch, one command:

    py reproduce_e7.py            # fetch missing data, then run everything
    py reproduce_e7.py --refetch  # force re-download the universe first

Steps (deterministic given the fetched data snapshot):
    1. Fetch the survivorship-safe universe at 1h / ~3y (only missing symbols,
       unless --refetch).
    2. Build the funding-rate panel.
    3. Stage A/B: cross-sectional rank-IC diagnostics (xsection_ic.py).
    4. Stage C: market-neutral long/short survival grid (xsection_stagec.py).
    5. Unit tests.

Reproducibility note: fetch pulls "the last ~1100 days from now", so absolute
dates shift over time and market data is live — the METHODOLOGY and (given a
fixed data snapshot) the numbers reproduce exactly, but a re-fetch months later
sees a shifted window. Freeze data/raw/*_60.parquet to pin results bit-for-bit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

from crypto_signal_bot.config import RAW_DIR
from crypto_signal_bot.pipeline import fetch_all
from crypto_signal_bot.research.xsection.universe import SURVIVORSHIP_SAFE

INTERVAL = "60"
DAYS = 1100


def _run(cmd: list[str]) -> None:
    logger.info("$ {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true", help="Force re-download the universe.")
    args = ap.parse_args()

    # 1. Fetch (missing symbols only, unless --refetch).
    if args.refetch:
        todo = SURVIVORSHIP_SAFE
    else:
        todo = [s for s in SURVIVORSHIP_SAFE
                if not (RAW_DIR / f"{s}_{INTERVAL}.parquet").exists()]
    if todo:
        logger.info("Fetching {} symbols at {}m / {}d", len(todo), INTERVAL, DAYS)
        fetch_all(todo, interval=INTERVAL, days=DAYS)
    else:
        logger.info("All universe data present — skipping fetch (use --refetch to force)")

    py = sys.executable
    _run([py, "build_funding_panel.py"])          # 2. funding panel
    _run([py, "xsection_ic.py", "--interval", INTERVAL])       # 3. Stage A/B
    _run([py, "xsection_stagec.py", "--interval", INTERVAL])   # 4. Stage C
    _run([py, "-m", "pytest", "tests/test_xsection.py", "-q"]) # 5. tests

    logger.success("E7 fully reproduced. Artifacts in {}", Path("data/research/e7"))


if __name__ == "__main__":
    main()
