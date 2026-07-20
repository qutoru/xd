"""ALX4 forward-validation runner (executes the LOCKED forward protocol).

    py alpha_library/alx4_regime_analysis/forward_run.py

Records any fully-closed post-lock days into the append-only forward ledger, then
evaluates the pre-registered confirm/fail criteria. Intended to be run
periodically (e.g. daily/weekly) — it accumulates genuine out-of-sample evidence
over the locked >=12-18 month horizon. Prints one of OBSERVING / CONFIRMED / FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from alpha_library.alx4_regime_analysis import forward as fwd


def main() -> None:
    logger.info("ALX4 FORWARD — lock={} | horizon>={}d | universe=intersection@{:.0%}",
                fwd.LOCK_DATE.date(), fwd.MIN_FORWARD_DAYS, fwd.COVERAGE)
    try:
        led = fwd.record()
    except Exception as exc:  # network/data unavailable -> evaluate whatever exists
        logger.warning("record step could not fetch new data ({}); evaluating ledger as-is", exc)
        led = fwd.load_ledger()

    res = fwd.evaluate(led)
    logger.info("-" * 60)
    logger.info("forward days accumulated: {}", res.get("n_days", 0))
    if res.get("n_days", 0):
        logger.info("net Sharpe={:+.2f}  CI=[{:+.2f},{:+.2f}]  IC={:+.4f} (t={:+.2f})  maxDD={:+.1%}",
                    res["sharpe"], res["ci_lo"], res["ci_hi"], res["ic_mean"],
                    res["ic_nw_t"], res["max_drawdown"])
        logger.info("regimes seen: bull={} bear/flat={}  bull-IC={:+.4f}  top5%%-share={:.0%}",
                    res["has_bull"], res["has_bear"], res["bull_ic"], res["top5pct_share"])
    logger.info("=" * 60)
    logger.info("STATUS: {} — {}", res["status"], res["message"])
    if res["status"] == "OBSERVING":
        logger.info("Action: keep running this on a schedule until the horizon and "
                    "both regimes are covered. No verdict permitted yet.")


if __name__ == "__main__":
    main()
