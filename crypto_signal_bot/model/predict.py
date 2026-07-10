"""Load a trained model and turn feature rows into signal predictions.

Reused by the backtest (Phase 4) and later the live loop (Phase 7). The booster
is saved by :mod:`crypto_signal_bot.model.train`; its class column order is
recorded in the metadata sidecar so we can map probabilities back to the
``-1/0/1`` signal space.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from crypto_signal_bot.config import INTERVAL, SYMBOL
from crypto_signal_bot.model.train import meta_path, model_path


class SignalModel:
    """A loaded LightGBM booster plus its feature list and class order."""

    def __init__(self, booster: lgb.Booster, features: list[str], class_order: list[int]):
        self.booster = booster
        self.features = features
        self.class_order = np.asarray(class_order)

    @classmethod
    def load(cls, symbol: str = SYMBOL, interval: str = INTERVAL) -> "SignalModel":
        """Load the persisted booster and metadata for a symbol/interval."""
        mpath = model_path(symbol, interval)
        if not mpath.exists():
            raise FileNotFoundError(f"No trained model at {mpath} — run `train` first")
        booster = lgb.Booster(model_file=str(mpath))
        with meta_path(symbol, interval).open(encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(booster, meta["features"], meta["class_order"])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities (rows x classes) for a feature frame."""
        return self.booster.predict(df[self.features])

    def predict_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Return the predicted ``-1/0/1`` signal for each row."""
        proba = self.predict_proba(df)
        return self.class_order[proba.argmax(axis=1)]

    def predict_with_conf(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(signals, confidence)`` where confidence is the top proba.

        Used by the risk-managed backtest to gate entries on model confidence.
        """
        proba = self.predict_proba(df)
        idx = proba.argmax(axis=1)
        return self.class_order[idx], proba[np.arange(len(idx)), idx]
