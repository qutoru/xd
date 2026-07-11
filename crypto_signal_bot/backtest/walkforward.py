"""Purged walk-forward cross-validation for the risk-managed strategy.

A single train/valid/test split (see :mod:`crypto_signal_bot.model.splits`)
produces one Sharpe number on one final slice of history — statistically that is
noise, and it is easy to fool yourself with in either direction. Walk-forward CV
instead retrains the model on an *expanding* window and evaluates it on many
consecutive out-of-sample test blocks, yielding a **distribution** of Sharpe
values across regimes. What matters is then the aggregate (mean/median, spread,
and the fraction of positive folds), not any single lucky window.

Each fold is laid out chronologically with embargo gaps so a forward-looking
triple-barrier label never leaks across a boundary::

    [ ........ train (expands) ........ ][emb][ valid ][emb][ test ]
                                                tune thr    OOS eval

The model is refit per fold (no look-ahead: only data before the test block is
ever seen), the confidence threshold is tuned on that fold's valid block, and
the risk-managed event backtest is run on the test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from crypto_signal_bot.config import (
    BACKTEST_DIR,
    EARLY_STOPPING_ROUNDS,
    EMBARGO,
    INTERVAL,
    LGBM_PARAMS,
    MIN_ROWS_FOR_TRAINING,
    SYMBOL,
)
from crypto_signal_bot.backtest.event_engine import run_event_backtest
from crypto_signal_bot.backtest.metrics import event_performance
from crypto_signal_bot.data.storage import load_processed
from crypto_signal_bot.model.metrics import evaluate
from crypto_signal_bot.model.predict import SignalModel
from crypto_signal_bot.model.train import LABEL_COLUMN, feature_columns
from crypto_signal_bot.backtest.runner import _tune_threshold

# Defaults for the walk-forward schedule.
_DEFAULT_SPLITS = 6
# Each test (and valid) block is this fraction of all rows.
_DEFAULT_FOLD_FRAC = 0.10


@dataclass(frozen=True)
class Fold:
    """Positional row ranges for one walk-forward fold (half-open intervals)."""

    train: range
    valid: range
    test: range


def make_folds(
    n_rows: int,
    *,
    n_splits: int = _DEFAULT_SPLITS,
    fold_frac: float = _DEFAULT_FOLD_FRAC,
    embargo: int = EMBARGO,
    min_train: int = MIN_ROWS_FOR_TRAINING,
) -> list[Fold]:
    """Build expanding, purged walk-forward folds tiling the tail of history.

    The last ``n_splits`` blocks of ``fold_size = fold_frac * n_rows`` rows are
    the test blocks; each fold's valid block is the ``fold_size`` rows before
    its test block, and its train block is everything before that. Embargo gaps
    are inserted before valid and before test. Folds whose train block is
    shorter than ``min_train`` are dropped.
    """
    fold_size = int(n_rows * fold_frac)
    if fold_size <= 0:
        raise ValueError("fold_frac too small for the dataset size")

    folds: list[Fold] = []
    for i in range(n_splits):
        # Test blocks are counted from the end: the earliest fold is i=0.
        test_start = n_rows - (n_splits - i) * fold_size
        test_end = test_start + fold_size
        valid_end = test_start - embargo
        valid_start = valid_end - fold_size
        train_end = valid_start - embargo
        if valid_start <= 0 or train_end < min_train:
            continue  # not enough history yet for this fold
        folds.append(
            Fold(
                train=range(0, train_end),
                valid=range(valid_start, valid_end),
                test=range(test_start, test_end),
            )
        )
    return folds


def _fit_fold_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str]) -> SignalModel:
    """Fit a LightGBM model on a fold's train block (early stop on its valid).

    The fitted booster is wrapped in :class:`SignalModel` so the rest of the
    pipeline (``predict_with_conf``) is reused unchanged.
    """
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(
        train_df[features],
        train_df[LABEL_COLUMN],
        eval_set=[(valid_df[features], valid_df[LABEL_COLUMN])],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    # booster_.predict returns columns in clf.classes_ order; SignalModel maps
    # argmax back to the -1/0/1 label space via that same class order.
    return SignalModel(clf.booster_, features, list(clf.classes_))


def run_walkforward(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    *,
    n_splits: int = _DEFAULT_SPLITS,
    fold_frac: float = _DEFAULT_FOLD_FRAC,
) -> dict:
    """Run purged walk-forward CV of the risk-managed strategy for one symbol.

    Returns:
        A summary dict with the aggregate statistics; the per-fold table is also
        saved to ``data/backtests/{symbol}_{interval}_walkforward.parquet``.
    """
    df = load_processed(symbol, interval)
    df = df.sort_values("timestamp").reset_index(drop=True)
    features = feature_columns(df)

    folds = make_folds(len(df), n_splits=n_splits, fold_frac=fold_frac)
    if not folds:
        raise ValueError(
            f"No valid folds for {symbol} {interval}m ({len(df)} rows) — "
            "need more history or a smaller fold_frac/n_splits."
        )
    logger.info(
        "Walk-forward {} {}m: {} folds over {} rows (fold_size~{})",
        symbol,
        interval,
        len(folds),
        len(df),
        int(len(df) * fold_frac),
    )

    rows: list[dict] = []
    for k, fold in enumerate(folds, 1):
        train_df = df.iloc[fold.train.start : fold.train.stop].reset_index(drop=True)
        valid_df = df.iloc[fold.valid.start : fold.valid.stop].reset_index(drop=True)
        test_df = df.iloc[fold.test.start : fold.test.stop].reset_index(drop=True)

        logger.info(
            "[fold {}/{}] train={} valid={} test={} | test {} .. {}",
            k,
            len(folds),
            len(train_df),
            len(valid_df),
            len(test_df),
            test_df["datetime"].iloc[0],
            test_df["datetime"].iloc[-1],
        )

        model = _fit_fold_model(train_df, valid_df, features)
        valid_sig, valid_conf = model.predict_with_conf(valid_df)
        test_sig, test_conf = model.predict_with_conf(test_df)

        threshold = _tune_threshold(valid_df, valid_sig, valid_conf)
        result = run_event_backtest(test_df, test_sig, test_conf, threshold=threshold)
        stats = event_performance(result.bars, result.trades)
        macro_f1 = evaluate(test_df[LABEL_COLUMN].to_numpy(), test_sig)["macro_f1"]

        rows.append(
            {
                "fold": k,
                "test_start": test_df["datetime"].iloc[0],
                "test_end": test_df["datetime"].iloc[-1],
                "threshold": threshold,
                "macro_f1": macro_f1,
                "sharpe": stats["sharpe"],
                "total_return": stats["total_return"],
                "n_trades": stats["n_trades"],
                "win_rate": stats["win_rate"],
            }
        )
        logger.info(
            "[fold {}/{}] thr={:.2f} macro_f1={:.3f} sharpe={:.2f} "
            "return={:+.2%} trades={}",
            k,
            len(folds),
            threshold,
            macro_f1,
            stats["sharpe"],
            stats["total_return"],
            stats["n_trades"],
        )

    table = pd.DataFrame(rows)
    summary = _summarize(table)
    _save(table, symbol, interval)
    return summary


def _summarize(table: pd.DataFrame) -> dict:
    """Aggregate the per-fold table into a robustness summary."""
    # Only fold with >=1 trade carry Sharpe signal; keep all for transparency.
    sharpe = table["sharpe"].to_numpy(dtype="float64")
    traded = table[table["n_trades"] > 0]
    return {
        "n_folds": int(len(table)),
        "mean_sharpe": float(np.mean(sharpe)),
        "median_sharpe": float(np.median(sharpe)),
        "std_sharpe": float(np.std(sharpe)),
        "frac_positive": float(np.mean(sharpe > 0)),
        "mean_macro_f1": float(table["macro_f1"].mean()),
        "mean_return": float(table["total_return"].mean()),
        "total_trades": int(table["n_trades"].sum()),
        "folds_with_trades": int(len(traded)),
    }


def log_walkforward(summary: dict) -> None:
    """Log the walk-forward robustness summary."""
    logger.info("Walk-forward summary over {} folds:", summary["n_folds"])
    logger.info(
        "  Sharpe: mean={:.2f} median={:.2f} std={:.2f} | positive folds={:.0%}",
        summary["mean_sharpe"],
        summary["median_sharpe"],
        summary["std_sharpe"],
        summary["frac_positive"],
    )
    logger.info(
        "  mean macro_f1={:.3f} | mean return/fold={:+.2%} | trades={} ({} folds traded)",
        summary["mean_macro_f1"],
        summary["mean_return"],
        summary["total_trades"],
        summary["folds_with_trades"],
    )


def _save(table: pd.DataFrame, symbol: str, interval: str) -> None:
    """Persist the per-fold walk-forward table for later inspection."""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKTEST_DIR / f"{symbol}_{interval}_walkforward.parquet"
    table.to_parquet(path, index=False)
    logger.info("Saved {} walk-forward folds -> {}", len(table), path)
