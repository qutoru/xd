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
