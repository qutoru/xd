"""ALX10 forward-paper — one-command record + evaluate of the LOCKED protocol.

    py alpha_library/alx10_forward_paper/run.py            # use cached data
    py alpha_library/alx10_forward_paper/run.py --refresh  # re-fetch fresh bars first

Appends any newly-closed post-lock bars to the append-only ledger, then prints
the current locked status (OBSERVING / FAIL / FAIL_A / CONFIRM_A / GRADUATE_B).
`--refresh` drops the Binance kl/fund cache first so `record` ingests new days
(run it on the daily cadence). No tuning; the criteria live in forward.py as
pre-registered constants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from alpha_library.alx10_forward_paper import forward as fwd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch fresh Binance bars before recording")
    args = ap.parse_args()
    fwd.record(refresh=args.refresh)
    res = fwd.evaluate()
    logger.info("=" * 66)
    logger.info("ALX10 forward status: {}", res["status"])
    logger.info("  {}", res.get("message", ""))
    if res.get("n_days", 0):
        logger.info("  n={} | Book A Sharpe={:+.3f} CI[{:+.2f},{:+.2f}] IC-t={:+.2f} DD={:+.1%}",
                    res["n_days"], res["sharpe_A"], res["ci_lo_A"], res["ci_hi_A"],
                    res["ic_nw_t_A"], res["maxdd_A"])
        logger.info("  Book B Sharpe={:+.3f} DD={:+.1%} | paired B-A mean={:+.6f} NW-t={:+.2f}",
                    res["sharpe_B"], res["maxdd_B"], res["paired_mean_BmA"], res["paired_nw_t_BmA"])


if __name__ == "__main__":
    main()
