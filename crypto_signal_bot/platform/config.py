"""Platform configuration — kept separate from the ML bot's ``config.py``.

The ALX strategy specification below is the frozen research *formula* re-stated
as plain text/constants. It is NOT imported from ``alpha_library`` — production
owns its own independent implementation. Only neutral filesystem paths are
borrowed from the shared project config.
"""

from __future__ import annotations

from pathlib import Path

from crypto_signal_bot.config import DATA_DIR  # neutral base path only

# --- ALX sleeve spec (research FORMULA re-expressed; NOT an import) ----------
# Concept: funding-ranked dollar-neutral daily book — long top / short bottom
# K% by the negative L-day mean daily funding, hold 1 day, execute at lag 1 day.
ALX_LOOKBACK_DAYS: int = 7
ALX_K_PCT: float = 0.30
ALX_HOLD: int = 1
ALX_REBALANCE: int = 1
ALX_EXEC_LAG: int = 1
ALX_TIER: str = "B"  # Tier-B = shadow only, zero weight into any real book

# --- Cadence & cost bands ---------------------------------------------------
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
