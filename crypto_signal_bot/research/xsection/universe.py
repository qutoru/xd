"""Survivorship-safe (point-in-time approximation) universe + returns panel.

Using today's top-50 list would leak survivorship/lookahead: we would only
test coins that survived and grew. Instead we fix an *ex-ante* list of large,
long-established Bybit USDT perpetuals that were already trading well before the
backtest window opens, and then keep only those with FULL data coverage over
the common window. This biases us conservatively — we exclude the newly-listed
pump-and-dumps, whose inclusion would be the optimistic artifact.

Rebranded/renamed tickers (e.g. MATIC->POL, FTM->S) are deliberately excluded
to avoid broken price continuity.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from crypto_signal_bot.data.storage import load_parquet

# Fixed ex-ante list, chosen before seeing any result. Majors that were liquid
# Bybit linear perps by mid-2023. Coverage is verified from the data, not
# assumed: symbols lacking full history over the common window are dropped.
SURVIVORSHIP_SAFE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT",
    "DOTUSDT", "LTCUSDT", "LINKUSDT", "BCHUSDT", "AVAXUSDT", "TRXUSDT", "ATOMUSDT",
    "ETCUSDT", "XLMUSDT", "NEARUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT", "ICPUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "EOSUSDT",
    "INJUSDT", "RUNEUSDT", "GRTUSDT", "ALGOUSDT", "THETAUSDT", "EGLDUSDT",
    "CRVUSDT", "COMPUSDT", "SUSHIUSDT", "ZECUSDT", "DASHUSDT", "GALAUSDT",
]


def build_returns_panel(
    symbols: list[str],
    interval: str,
    *,
    min_coverage: float = 0.98,
) -> pd.DataFrame:
    """Assemble an aligned close-price panel (index=timestamp, cols=symbols).

    Only symbols whose data covers at least ``min_coverage`` of the common
    window (intersection of all available first/last timestamps) are kept.

    Returns:
        A wide close-price DataFrame indexed by UTC ``datetime``; caller derives
        returns. Symbols with insufficient coverage are logged and dropped.
    """
    series: dict[str, pd.Series] = {}
    spans: list[tuple[int, int]] = []
    for sym in symbols:
        try:
            df = load_parquet(sym, interval).sort_values("timestamp")
        except FileNotFoundError:
            logger.warning("{}: no raw data, skipping", sym)
            continue
        s = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(df["datetime"], utc=True))
        s = s[~s.index.duplicated(keep="last")]
        series[sym] = s
        spans.append((int(df["timestamp"].iloc[0]), int(df["timestamp"].iloc[-1])))

    if not series:
        raise ValueError("No symbols with data found for the panel")

    # Common window = latest start, earliest end (so all symbols overlap).
    start = pd.to_datetime(max(s for s, _ in spans), unit="ms", utc=True)
    end = pd.to_datetime(min(e for _, e in spans), unit="ms", utc=True)
    logger.info("Common window: {} .. {} ({} candidate symbols)", start, end, len(series))

    panel = pd.DataFrame(series).sort_index()
    panel = panel.loc[(panel.index >= start) & (panel.index <= end)]

    n_bars = len(panel)
    keep = [c for c in panel.columns if panel[c].notna().mean() >= min_coverage]
    dropped = sorted(set(panel.columns) - set(keep))
    if dropped:
        logger.warning("Dropped {} symbols below {:.0%} coverage: {}", len(dropped), min_coverage, dropped)
    panel = panel[keep]
    logger.info("Final panel: {} symbols x {} bars", panel.shape[1], n_bars)
    return panel
