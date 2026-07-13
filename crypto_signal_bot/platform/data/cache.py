"""Simple parquet panel cache for the platform data layer.

Keyed (time x symbol) panels on disk with mtime-based freshness. Used by the
Bybit providers to avoid re-fetching the same day repeatedly; the daily runner
appends one fresh day and reuses the rest.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


class ParquetCache:
    """A tiny keyed parquet store under a root directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.parquet"

    def load(self, key: str) -> pd.DataFrame | None:
        """Return the cached panel, or ``None`` if absent."""
        p = self.path(key)
        return pd.read_parquet(p) if p.exists() else None

    def save(self, key: str, df: pd.DataFrame) -> Path:
        """Persist a panel and return its path."""
        p = self.path(key)
        df.to_parquet(p)
        return p

    def age_seconds(self, key: str) -> float | None:
        """Seconds since the cache entry was last written (``None`` if absent)."""
        p = self.path(key)
        return time.time() - p.stat().st_mtime if p.exists() else None

    def is_fresh(self, key: str, max_age_seconds: float) -> bool:
        """True if the entry exists and is younger than ``max_age_seconds``."""
        age = self.age_seconds(key)
        return age is not None and age <= max_age_seconds
