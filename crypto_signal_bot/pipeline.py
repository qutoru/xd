"""Batch pipeline over the multi-symbol universe.

Runs fetch -> build -> train for every symbol, isolating failures so one bad
(e.g. thinly-traded or newly-listed) symbol never aborts the whole batch.
Symbols with too little history to train are skipped with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from crypto_signal_bot.config import (
    HISTORY_DAYS,
    INTERVAL,
    MIN_ROWS_FOR_TRAINING,
    USE_DERIVATIVES,
)
from crypto_signal_bot.data.derivatives import merge_derivatives
from crypto_signal_bot.data.fetcher import fetch_ohlcv
from crypto_signal_bot.data.storage import load_processed, save_parquet
from crypto_signal_bot.features.dataset import build_and_save
from crypto_signal_bot.model.train import train as train_symbol


@dataclass
class BatchReport:
    """Outcome of a batch stage: which symbols succeeded/failed/were skipped."""

    ok: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def log(self, stage: str) -> None:
        logger.info(
            "[{}] done: {} ok, {} skipped, {} failed",
            stage,
            len(self.ok),
            len(self.skipped),
            len(self.failed),
        )
        if self.failed:
            logger.warning("[{}] failed symbols: {}", stage, ", ".join(self.failed))


def fetch_all(
    symbols: list[str],
    *,
    interval: str = INTERVAL,
    days: int = HISTORY_DAYS,
) -> BatchReport:
    """Fetch and store OHLCV history for every symbol."""
    report = BatchReport()
    for i, symbol in enumerate(symbols, 1):
        logger.info("({}/{}) fetch {}", i, len(symbols), symbol)
        try:
            df = fetch_ohlcv(symbol=symbol, interval=interval, history_days=days)
            if df.empty:
                logger.warning("No candles for {} — skipping", symbol)
                report.skipped.append(symbol)
                continue
            if USE_DERIVATIVES:
                df = merge_derivatives(df, symbol, interval=interval, history_days=days)
            save_parquet(df, symbol, interval)
            report.ok.append(symbol)
        except Exception as exc:  # keep the batch alive
            logger.exception("fetch failed for {}: {}", symbol, exc)
            report.failed.append(symbol)
    report.log("fetch")
    return report


def build_all(symbols: list[str], *, interval: str = INTERVAL) -> BatchReport:
    """Build the features+labels dataset for every symbol."""
    report = BatchReport()
    for i, symbol in enumerate(symbols, 1):
        logger.info("({}/{}) build {}", i, len(symbols), symbol)
        try:
            dataset = build_and_save(symbol, interval)
            if len(dataset) < MIN_ROWS_FOR_TRAINING:
                logger.warning(
                    "{}: only {} rows (< {}) — will be skipped at training",
                    symbol,
                    len(dataset),
                    MIN_ROWS_FOR_TRAINING,
                )
                report.skipped.append(symbol)
            else:
                report.ok.append(symbol)
        except Exception as exc:
            logger.exception("build failed for {}: {}", symbol, exc)
            report.failed.append(symbol)
    report.log("build")
    return report


def train_all(symbols: list[str], *, interval: str = INTERVAL) -> BatchReport:
    """Train a model per symbol, skipping those with too little data."""
    report = BatchReport()
    for i, symbol in enumerate(symbols, 1):
        logger.info("({}/{}) train {}", i, len(symbols), symbol)
        try:
            df = load_processed(symbol, interval)
            if len(df) < MIN_ROWS_FOR_TRAINING:
                logger.warning(
                    "{}: {} rows (< {}) — skipping training",
                    symbol,
                    len(df),
                    MIN_ROWS_FOR_TRAINING,
                )
                report.skipped.append(symbol)
                continue
            train_symbol(symbol, interval)
            report.ok.append(symbol)
        except Exception as exc:
            logger.exception("train failed for {}: {}", symbol, exc)
            report.failed.append(symbol)
    report.log("train")
    return report


def bootstrap_all(symbols: list[str], *, interval: str = INTERVAL) -> None:
    """Run the full fetch -> build -> train pipeline over the universe."""
    logger.info("Bootstrapping pipeline for {} symbols", len(symbols))
    fetch_all(symbols, interval=interval)
    build_all(symbols, interval=interval)
    train_all(symbols, interval=interval)
    logger.success("Bootstrap complete for {} symbols", len(symbols))
