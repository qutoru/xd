"""Fetch Bybit derivatives context: open interest and funding rate.

These are perp-specific positioning/sentiment signals (crowded longs, funding
pressure) that plain OHLCV lacks. Both are pulled via public endpoints and
merged onto the 15m kline timeline in :mod:`crypto_signal_bot.data.fetcher`.
"""

from __future__ import annotations

import time

import pandas as pd
from loguru import logger
from pybit.unified_trading import HTTP

from crypto_signal_bot.config import CATEGORY, HISTORY_DAYS

_MS_PER_DAY = 24 * 60 * 60 * 1000
_PAGE_LIMIT = 200


def _session() -> HTTP:
    return HTTP(testnet=False)


def fetch_open_interest(
    symbol: str,
    *,
    interval_time: str = "15min",
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Fetch open-interest history, paging backwards via the cursor.

    Returns:
        DataFrame ``[timestamp, open_interest]`` sorted ascending (may be empty
        if the endpoint returns nothing for the symbol).
    """
    session = _session()
    start_ms = int(time.time() * 1000) - history_days * _MS_PER_DAY
    rows: list[dict] = []
    cursor: str | None = None

    while True:
        resp = session.get_open_interest(
            category=CATEGORY,
            symbol=symbol,
            intervalTime=interval_time,
            limit=_PAGE_LIMIT,
            cursor=cursor,
        )
        result = resp.get("result", {})
        batch = result.get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(r["timestamp"]) for r in batch)
        cursor = result.get("nextPageCursor")
        if not cursor or oldest <= start_ms:
            break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["open_interest"] = df["openInterest"].astype("float64")
    df = df[df["timestamp"] >= start_ms]
    return (
        df[["timestamp", "open_interest"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_funding(
    symbol: str,
    *,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Fetch funding-rate history, paging backwards by end time.

    Funding settles every few hours, so this is sparse relative to 15m bars and
    is forward-filled onto the kline timeline downstream.

    Returns:
        DataFrame ``[timestamp, funding_rate]`` sorted ascending.
    """
    session = _session()
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - history_days * _MS_PER_DAY
    rows: list[dict] = []
    end_ms = now_ms

    while True:
        resp = session.get_funding_rate_history(
            category=CATEGORY,
            symbol=symbol,
            endTime=end_ms,
            limit=_PAGE_LIMIT,
        )
        batch = resp.get("result", {}).get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(r["fundingRateTimestamp"]) for r in batch)
        if oldest <= start_ms or len(batch) < _PAGE_LIMIT:
            break
        end_ms = oldest - 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(rows)
    df["timestamp"] = df["fundingRateTimestamp"].astype("int64")
    df["funding_rate"] = df["fundingRate"].astype("float64")
    df = df[df["timestamp"] >= start_ms]
    return (
        df[["timestamp", "funding_rate"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def merge_derivatives(
    klines: pd.DataFrame,
    symbol: str,
    *,
    interval: str,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Attach ``open_interest`` and ``funding_rate`` columns to a kline frame.

    Open interest is matched to the most recent value at or before each bar;
    funding rate is forward-filled from its (sparse) settlement times. Failures
    are non-fatal: the columns are left as NaN so the pipeline still runs.
    """
    out = klines.sort_values("timestamp").reset_index(drop=True)

    try:
        oi = fetch_open_interest(symbol, interval_time=f"{interval}min", history_days=history_days)
        if not oi.empty:
            out = pd.merge_asof(out, oi, on="timestamp", direction="backward")
        else:
            out["open_interest"] = float("nan")
        logger.info("{}: merged {} open-interest points", symbol, len(oi))
    except Exception as exc:  # keep pipeline alive without OI
        logger.warning("{}: open-interest fetch failed ({}) — leaving NaN", symbol, exc)
        out["open_interest"] = float("nan")

    try:
        funding = fetch_funding(symbol, history_days=history_days)
        if not funding.empty:
            out = pd.merge_asof(out, funding, on="timestamp", direction="backward")
        else:
            out["funding_rate"] = float("nan")
        logger.info("{}: merged {} funding points", symbol, len(funding))
    except Exception as exc:
        logger.warning("{}: funding fetch failed ({}) — leaving NaN", symbol, exc)
        out["funding_rate"] = float("nan")

    return out
