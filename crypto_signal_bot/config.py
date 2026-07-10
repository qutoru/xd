"""Project configuration.

Holds filesystem paths, Bybit credentials (loaded from ``.env``) and the
base market parameters used across the bot. Risk-management parameters are
intentionally absent at this phase.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if present. Never hardcode secrets.
load_dotenv()

# --- Filesystem paths -------------------------------------------------------
# BASE_DIR is the repository root (parent of the crypto_signal_bot package).
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
BACKTEST_DIR: Path = DATA_DIR / "backtests"
MODELS_DIR: Path = BASE_DIR / "models"

# --- Bybit credentials (optional for public market data) --------------------
BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")

# --- Market parameters ------------------------------------------------------
SYMBOL: str = "BTCUSDT"
# Bybit kline interval in minutes as a string ("1", "5", "15", "60", ...).
INTERVAL: str = "15"
# How many days of history to download.
HISTORY_DAYS: int = 180

# --- Bybit API constants ----------------------------------------------------
# "linear" = USDT-margined perpetual futures.
CATEGORY: str = "linear"
# Maximum number of candles Bybit returns per kline request.
MAX_LIMIT: int = 1000

# --- Phase 2: labeling parameters -------------------------------------------
# Triple-barrier labeling (Lopez de Prado). For each bar we place symmetric
# horizontal barriers at close +/- ATR_MULT * ATR and a vertical (time) barrier
# HORIZON bars ahead. The class is decided by whichever barrier is touched
# first (up -> long=1, down -> short=-1, time/none -> flat=0).
HORIZON: int = 8  # prediction horizon in bars (8 * 15m = 2h)
ATR_PERIOD: int = 14  # ATR lookback used to size the barriers
ATR_MULT: float = 1.5  # barrier distance as a multiple of ATR

# Integer label encoding used across training/inference.
LABEL_SHORT: int = -1
LABEL_FLAT: int = 0
LABEL_LONG: int = 1

# Ordered classes for the multiclass model. The list index is the LightGBM
# class id (0,1,2); the value is our -1/0/1 signal label.
CLASS_ORDER: tuple[int, ...] = (LABEL_SHORT, LABEL_FLAT, LABEL_LONG)
CLASS_NAMES: dict[int, str] = {LABEL_SHORT: "short", LABEL_FLAT: "flat", LABEL_LONG: "long"}

# --- Phase 3: training parameters -------------------------------------------
RANDOM_SEED: int = 42
# Chronological split (no shuffle). Test is the remainder (0.15).
TRAIN_FRAC: float = 0.70
VALID_FRAC: float = 0.15
# Purge this many bars from the tail of each earlier split so a training label,
# which looks HORIZON bars into the future, cannot peek into the next split.
EMBARGO: int = HORIZON

# LightGBM hyperparameters (sensible MVP defaults; tune later).
LGBM_PARAMS: dict = {
    "objective": "multiclass",
    "num_class": len(CLASS_ORDER),
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS: int = 100

# --- Phase 4: backtest parameters -------------------------------------------
# Bybit linear-perp taker fee (0.055%). The per-bar signal flips often, so we
# assume taker fills. Charged on each unit of position change (turnover).
FEE_RATE: float = 0.00055
# 15m bars in a year: 4 * 24 * 365. Used to annualize return/Sharpe.
BARS_PER_YEAR: int = 4 * 24 * 365

# --- Phase 5: risk management + backtest v2 ---------------------------------
# One position at a time; exit on stop-loss / take-profit / time barrier.
# SL/TP are sized in ATR units (same volatility scale as the labels).
SL_ATR_MULT: float = 1.5
TP_ATR_MULT: float = 1.5
# Risk-based position sizing: risk this fraction of equity per trade, given the
# ATR stop distance. Notional is capped at MAX_LEVERAGE x equity.
RISK_PER_TRADE: float = 0.01
MAX_LEVERAGE: float = 5.0
# Only enter when the model's top-class probability clears a threshold. The
# threshold is tuned on the validation segment (grid below) and then applied
# out-of-sample on test.
PROB_THRESHOLD_GRID: tuple[float, ...] = (0.34, 0.40, 0.45, 0.50, 0.55, 0.60)
