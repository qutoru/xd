"""Train the LightGBM signal classifier and report baseline metrics.

Pipeline: load the processed dataset -> chronological purged split -> fit
LightGBM (multiclass, early stopping on validation) -> evaluate on validation
and test -> persist the booster and a JSON metadata sidecar.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from loguru import logger

from crypto_signal_bot.config import (
    CLASS_ORDER,
    EARLY_STOPPING_ROUNDS,
    INTERVAL,
    LGBM_PARAMS,
    MODELS_DIR,
    SYMBOL,
)
from crypto_signal_bot.data.storage import load_processed
from crypto_signal_bot.model.metrics import evaluate, log_report
from crypto_signal_bot.model.splits import time_split

# Columns that are metadata or the target, i.e. never model inputs.
_NON_FEATURE = {
    "timestamp",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "label",
}
LABEL_COLUMN = "label"


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered feature columns of a processed dataset."""
    return [c for c in df.columns if c not in _NON_FEATURE]


def model_path(symbol: str, interval: str) -> Path:
    """Path to the saved LightGBM booster text file."""
    return MODELS_DIR / f"{symbol}_{interval}.txt"


def meta_path(symbol: str, interval: str) -> Path:
    """Path to the model metadata JSON sidecar."""
    return MODELS_DIR / f"{symbol}_{interval}_meta.json"


def train(symbol: str = SYMBOL, interval: str = INTERVAL) -> dict:
    """Train and persist the signal model.

    Returns:
        The metadata dict (also written next to the model), including
        validation and test metrics.
    """
    df = load_processed(symbol, interval)
    df = df.sort_values("timestamp").reset_index(drop=True)

    features = feature_columns(df)
    logger.info("Training on {} rows, {} features", len(df), len(features))

    splits = time_split(len(df))
    logger.info(
        "Split (purged): train={} valid={} test={}",
        len(splits.train),
        len(splits.valid),
        len(splits.test),
    )

    x = df[features]
    y = df[LABEL_COLUMN]
    x_train, y_train = x.iloc[splits.train], y.iloc[splits.train]
    x_valid, y_valid = x.iloc[splits.valid], y.iloc[splits.valid]
    x_test, y_test = x.iloc[splits.test], y.iloc[splits.test]

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    logger.info("Best iteration: {}", clf.best_iteration_)

    valid_metrics = evaluate(y_valid.to_numpy(), clf.predict(x_valid))
    test_metrics = evaluate(y_test.to_numpy(), clf.predict(x_test))
    log_report("valid", valid_metrics)
    log_report("test", test_metrics)

    meta = {
        "symbol": symbol,
        "interval": interval,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "class_order": list(CLASS_ORDER),
        "params": LGBM_PARAMS,
        "best_iteration": clf.best_iteration_,
        "n_train": len(splits.train),
        "n_valid": len(splits.valid),
        "n_test": len(splits.test),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    _save(clf, meta, symbol, interval)
    return meta


def _save(clf: "lgb.LGBMClassifier", meta: dict, symbol: str, interval: str) -> None:
    """Persist the booster and its metadata sidecar."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mpath = model_path(symbol, interval)
    # Save the best iteration only.
    clf.booster_.save_model(str(mpath), num_iteration=clf.best_iteration_)
    with meta_path(symbol, interval).open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Saved model -> {} (+ meta json)", mpath)
