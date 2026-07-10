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
