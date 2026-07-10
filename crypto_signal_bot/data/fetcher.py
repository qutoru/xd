"""Fetch OHLCV (kline) data from the Bybit v5 API.

Uses the public ``get_kline`` endpoint (no authentication required) with
backwards pagination to assemble an arbitrary amount of history, and a simple
exponential-backoff retry wrapper around transient network/server errors.
"""

from __future__ import annotations

import time
from typing import Any, List

import pandas as pd
from loguru import logger
from pybit.exceptions import FailedRequestError, InvalidRequestError
from pybit.unified_trading import HTTP

from crypto_signal_bot.config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    CATEGORY,
    HISTORY_DAYS,
    INTERVAL,
    MAX_LIMIT,
    SYMBOL,
)

# Column order returned by Bybit's kline endpoint.
_RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
_FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "turnover"]

_MS_PER_DAY = 24 * 60 * 60 * 1000


def _build_session() -> HTTP:
    """Create a Bybit HTTP session.

    API keys are passed if configured, but the kline endpoint is public and
    works without them.
    """
    return HTTP(
        api_key=BYBIT_API_KEY or None,
        api_secret=BYBIT_API_SECRET or None,
        testnet=False,
    )


def _request_kline_with_retry(
    session: HTTP,
    *,
    symbol: str,
    interval: str,
    end_ms: int,
    limit: int,
    max_retries: int = 5,
    base_backoff: float = 1.0,
) -> List[List[str]]:
    """Request one page of klines, retrying transient failures with backoff.

    Args:
        session: An initialised Bybit HTTP session.
        symbol: Trading symbol, e.g. ``"BTCUSDT"``.
        interval: Kline interval in minutes as a string, e.g. ``"15"``.
        end_ms: Upper bound (inclusive) of the window, in epoch milliseconds.
        limit: Number of candles to request (max 1000 on Bybit).
        max_retries: Maximum number of attempts before giving up.
        base_backoff: Base delay in seconds; grows as ``base * 2**(attempt-1)``.

    Returns:
        The raw ``result.list`` from Bybit (newest candle first).

    Raises:
        RuntimeError: If a non-zero ``retCode`` persists after all retries.
        FailedRequestError: If the network request keeps failing.
        InvalidRequestError: Immediately, on malformed/invalid requests
            (these are not retried since retrying will not help).
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get_kline(
                category=CATEGORY,
                symbol=symbol,
                interval=str(interval),
                end=end_ms,
                limit=limit,
            )
            ret_code = resp.get("retCode")
            if ret_code != 0:
                raise RuntimeError(
                    f"Bybit returned retCode={ret_code} retMsg={resp.get('retMsg')!r}"
                )
            return resp["result"]["list"]
        except InvalidRequestError:
            # Client-side error (bad params) — retrying will not help.
            logger.error("Invalid Bybit request for {} {}", symbol, interval)
            raise
        except (FailedRequestError, RuntimeError, ConnectionError, TimeoutError) as exc:
            if attempt >= max_retries:
                logger.error("Kline request failed after {} attempts: {}", attempt, exc)
                raise
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Kline request failed (attempt {}/{}): {} — retrying in {:.1f}s",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    # Unreachable, but keeps type checkers happy.
    raise RuntimeError("Exhausted retries without returning")


def _to_dataframe(rows: List[List[Any]], start_ms: int) -> pd.DataFrame:
    """Convert raw Bybit rows into a typed, sorted, de-duplicated DataFrame.

    Args:
        rows: Concatenated raw kline rows (each newest-first per page).
        start_ms: Lower bound (epoch ms); candles older than this are dropped.

    Returns:
        DataFrame sorted ascending by time with a UTC ``datetime`` column.
    """
    if not rows:
        return pd.DataFrame(columns=[*_RAW_COLUMNS, "datetime"])

    df = pd.DataFrame(rows, columns=_RAW_COLUMNS)
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in _FLOAT_COLUMNS:
        df[col] = df[col].astype("float64")

    df = df[df["timestamp"] >= start_ms]
    df = (
        df.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_ohlcv(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Download ``history_days`` of OHLCV candles from Bybit.

    Pages backwards from "now" until the requested history is covered or the
    exchange stops returning data.

    Args:
        symbol: Trading symbol, e.g. ``"BTCUSDT"``.
        interval: Kline interval in minutes as a string, e.g. ``"15"``.
        history_days: How many days back to fetch.

    Returns:
        DataFrame with columns ``timestamp, open, high, low, close, volume,
        turnover, datetime`` sorted ascending by time.
    """
    session = _build_session()
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - history_days * _MS_PER_DAY

    rows: List[List[Any]] = []
    end_ms = now_ms
    prev_oldest: int | None = None

    logger.info(
        "Fetching {} {}m klines for the last {} days (category={})",
        symbol,
        interval,
        history_days,
        CATEGORY,
    )

    while True:
        batch = _request_kline_with_retry(
            session,
            symbol=symbol,
            interval=interval,
            end_ms=end_ms,
            limit=MAX_LIMIT,
        )
        if not batch:
            break

        rows.extend(batch)
        # Bybit returns newest-first, so the last element is the oldest.
        oldest_ts = int(batch[-1][0])
        logger.debug("Got {} candles, oldest={}", len(batch), oldest_ts)

        # Stop once we've reached the requested start.
        if oldest_ts <= start_ms:
            break
        # Guard against a page that makes no backwards progress.
        if prev_oldest is not None and oldest_ts >= prev_oldest:
            logger.warning("No pagination progress — stopping early")
            break

        prev_oldest = oldest_ts
        end_ms = oldest_ts - 1
        time.sleep(0.1)  # be gentle with the public endpoint

    df = _to_dataframe(rows, start_ms)
    logger.info("Assembled {} unique candles", len(df))
    return df
