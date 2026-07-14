"""StaticUniverseProvider — a fixed, explicit symbol set.

Implements the same :class:`UniverseProvider` contract as the live Bybit provider,
so it drops into ``DailySnapshotProvider`` unchanged. Its purpose is to pin the
traded universe to a small, explicit list (e.g. majors that exist on *both* Bybit
mainnet and testnet), so market data (pulled from mainnet for history) and order
routing (on testnet) operate on a compatible symbol set — without any Data->
Execution dependency.
"""

from __future__ import annotations

import pandas as pd


class StaticUniverseProvider:
    """Returns a fixed symbol list regardless of ``asof``."""

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = list(symbols)

    def universe(self, asof: pd.Timestamp) -> list[str]:  # asof unused: fixed set
        return list(self._symbols)
