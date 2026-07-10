"""Local persistence of OHLCV data as parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from crypto_signal_bot.config import PROCESSED_DIR, RAW_DIR


def raw_path(symbol: str, interval: str) -> Path:
    """Return the parquet path for a given symbol/interval.

    Args:
        symbol: Trading symbol, e.g. ``"BTCUSDT"``.
        interval: Kline interval in minutes as a string, e.g. ``"15"``.

    Returns:
        Path like ``<repo>/data/raw/BTCUSDT_15.parquet``.
    """
    return RAW_DIR / f"{symbol}_{interval}.parquet"


def save_parquet(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    """Persist an OHLCV DataFrame to parquet, creating directories as needed.

    Args:
        df: The OHLCV DataFrame to store.
        symbol: Trading symbol.
        interval: Kline interval in minutes as a string.

    Returns:
        The path the file was written to.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = raw_path(symbol, interval)
    df.to_parquet(path, index=False)
    logger.info("Saved {} rows to {}", len(df), path)
    return path


def load_parquet(symbol: str, interval: str) -> pd.DataFrame:
    """Load a previously saved OHLCV parquet file.

    Args:
        symbol: Trading symbol.
        interval: Kline interval in minutes as a string.

    Returns:
        The stored DataFrame.

    Raises:
        FileNotFoundError: If no parquet file exists for this symbol/interval.
    """
    path = raw_path(symbol, interval)
    if not path.exists():
        raise FileNotFoundError(f"No data file found at {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded {} rows from {}", len(df), path)
    return df


def processed_path(symbol: str, interval: str) -> Path:
    """Return the parquet path for the processed feature/label dataset.

    Returns:
        Path like ``<repo>/data/processed/BTCUSDT_15.parquet``.
    """
    return PROCESSED_DIR / f"{symbol}_{interval}.parquet"


def save_processed(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    """Persist a processed features+labels DataFrame to parquet.

    Returns:
        The path the file was written to.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = processed_path(symbol, interval)
    df.to_parquet(path, index=False)
    logger.info("Saved {} rows to {}", len(df), path)
    return path


def load_processed(symbol: str, interval: str) -> pd.DataFrame:
    """Load a previously built processed dataset.

    Raises:
        FileNotFoundError: If no processed file exists for this symbol/interval.
    """
    path = processed_path(symbol, interval)
    if not path.exists():
        raise FileNotFoundError(f"No processed dataset found at {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded {} rows from {}", len(df), path)
    return df
