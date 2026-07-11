"""Assemble the model-ready dataset: features + triple-barrier labels.

Pipeline: load raw OHLCV -> compute features -> compute labels -> align, drop
warm-up/unresolved rows -> persist to ``data/processed``.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from crypto_signal_bot.config import (
    HORIZON,
    INTERVAL,
    LABEL_SL_MULT,
    LABEL_TP_MULT,
    SYMBOL,
)
from crypto_signal_bot.data.storage import load_parquet, save_processed
from crypto_signal_bot.features.indicators import build_features, compute_atr
from crypto_signal_bot.features.labeling import triple_barrier_labels

# Non-feature columns carried through for reference / later phases.
_META_COLUMNS = ["timestamp", "datetime", "open", "high", "low", "close", "volume"]
LABEL_COLUMN = "label"


def build_dataset(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    df_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the features+labels dataset from raw OHLCV.

    Args:
        symbol: Trading symbol; also selects the raw file when ``df_raw`` is
            not provided.
        interval: Kline interval in minutes as a string.
        df_raw: Optional in-memory raw OHLCV frame; loaded from disk if omitted.

    Returns:
        A DataFrame with metadata columns, feature columns and an integer
        ``label`` column, with warm-up and unresolved rows removed.
    """
    if df_raw is None:
        df_raw = load_parquet(symbol, interval)

    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
    logger.info("Building dataset from {} raw candles", len(df_raw))

    # Compute ATR once and share it between features and labeling.
    atr = compute_atr(df_raw)
    features = build_features(df_raw)
    labels = triple_barrier_labels(
        df_raw,
        horizon=HORIZON,
        tp_mult=LABEL_TP_MULT,
        sl_mult=LABEL_SL_MULT,
        atr=atr,
    )

    meta = df_raw[[c for c in _META_COLUMNS if c in df_raw.columns]]
    dataset = pd.concat([meta, features], axis=1)
    dataset[LABEL_COLUMN] = labels

    before = len(dataset)
    dataset = dataset.dropna().reset_index(drop=True)
    dataset[LABEL_COLUMN] = dataset[LABEL_COLUMN].astype("int64")
    logger.info(
        "Dropped {} rows with NaN (warm-up/unresolved); {} rows remain",
        before - len(dataset),
        len(dataset),
    )

    _log_label_distribution(dataset[LABEL_COLUMN])
    return dataset


def _log_label_distribution(labels: pd.Series) -> None:
    """Log the class balance so we can sanity-check the labeling."""
    counts = labels.value_counts().sort_index()
    total = len(labels)
    names = {-1: "short", 0: "flat", 1: "long"}
    for value, count in counts.items():
        logger.info(
            "  label {:>5} ({:>2}): {:>6} ({:5.1f}%)",
            names.get(int(value), "?"),
            int(value),
            int(count),
            100.0 * count / total if total else 0.0,
        )


def build_and_save(symbol: str = SYMBOL, interval: str = INTERVAL) -> pd.DataFrame:
    """Build the dataset and persist it to ``data/processed``.

    Returns:
        The built dataset (also written to disk).
    """
    dataset = build_dataset(symbol, interval)
    save_processed(dataset, symbol, interval)
    return dataset
