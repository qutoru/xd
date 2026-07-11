"""Assemble a (time x symbol) funding-rate panel aligned to the 1h returns grid.

Funding settles every few hours; we forward-fill it onto the hourly index of the
cross-sectional panel. Symbols without funding history are skipped. Output ->
data/research/e7/funding_panel.parquet (used by xsection_ic.py's H-funding test).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_signal_bot.data.derivatives import fetch_funding
from crypto_signal_bot.research.xsection import features as ft
from crypto_signal_bot.research.xsection.universe import SURVIVORSHIP_SAFE, build_returns_panel

OUT = Path("data/research/e7/funding_panel.parquet")


def main() -> None:
    panel = build_returns_panel(SURVIVORSHIP_SAFE, "60")
    returns = ft.log_returns(panel).dropna(how="all")
    index = returns.index

    cols = {}
    for sym in panel.columns:
        try:
            f = fetch_funding(sym, history_days=1100)
        except Exception as exc:  # keep going; funding is optional
            logger.warning("{}: funding fetch failed ({})", sym, exc)
            continue
        if f.empty:
            logger.warning("{}: no funding history", sym)
            continue
        s = pd.Series(f["funding_rate"].to_numpy(),
                      index=pd.to_datetime(f["timestamp"], unit="ms", utc=True))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        cols[sym] = s.reindex(index, method="ffill")

    fp = pd.DataFrame(cols).reindex(index)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fp.to_parquet(OUT)
    logger.success("Funding panel: {} symbols x {} bars -> {}", fp.shape[1], fp.shape[0], OUT)


if __name__ == "__main__":
    main()
