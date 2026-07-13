"""Platform configuration — paths and neutral defaults only.

Kept separate from the ML bot's ``config.py``. No strategy-specific parameters
live here: each signal plugin owns its own parameters. Only neutral filesystem
paths and generic cost/cadence defaults are defined.
"""

from __future__ import annotations

from pathlib import Path

from crypto_signal_bot.config import DATA_DIR  # neutral base path only

# --- Cadence & cost bands (generic; not strategy-specific) ------------------
DAILY_INTERVAL_MIN: int = 24 * 60
COST_LOW: float = 0.0002
COST_MED: float = 0.00075
COST_HIGH: float = 0.0015

# --- Platform data layout (isolated from ML bot data dirs) ------------------
PLATFORM_DIR: Path = DATA_DIR / "platform"
BOOKS_DIR: Path = PLATFORM_DIR / "books"
UNIVERSE_SNAP_DIR: Path = PLATFORM_DIR / "universe"
FUNDING_PANEL_PATH: Path = PLATFORM_DIR / "funding_panel.parquet"
SHADOW_NAV_PATH: Path = PLATFORM_DIR / "shadow_nav.parquet"
