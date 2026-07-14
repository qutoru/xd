"""Bybit execution configuration — isolated from Telegram and platform config.

Only Bybit/venue + trading-mode settings live here (req: keep exchange config in
its own module). Credentials are read from the environment; nothing is hardcoded.
This module is venue-config only: it imports no exchange SDK and no platform data
layer, so the execution core boundary guard stays green.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class TradingMode(str, Enum):
    """How execution is routed.

    - SHADOW: no exchange at all — virtual accounting only (no broker).
    - PAPER:  full Bybit translation/precision, fills simulated locally (no
      real orders, no mutation of the live account).
    - LIVE:   real orders placed on Bybit USDT perpetual futures.
    """

    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BybitConfig:
    """Bybit connection + trading-mode settings (USDT perpetuals only)."""

    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    mode: TradingMode = TradingMode.SHADOW
    category: str = "linear"  # USDT perpetual futures
    recv_window: int = 5000
    paper_equity: float = 10_000.0  # simulated NAV used in PAPER mode

    @classmethod
    def from_env(cls) -> BybitConfig:
        """Build from ``BYBIT_*`` environment variables (safe defaults)."""
        return cls(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=_env_bool("BYBIT_TESTNET", True),
            mode=TradingMode(os.getenv("BYBIT_TRADING_MODE", "shadow").strip().lower()),
        )

    @property
    def is_live(self) -> bool:
        return self.mode is TradingMode.LIVE

    @property
    def needs_exchange(self) -> bool:
        """True for PAPER/LIVE (a Bybit session is required), False for SHADOW."""
        return self.mode in (TradingMode.PAPER, TradingMode.LIVE)

    def validate(self) -> None:
        """Fail fast on an unsafe/incoherent configuration."""
        if self.category != "linear":
            raise ValueError("BybitConfig supports only linear (USDT perpetual) futures")
        if self.is_live and not (self.api_key and self.api_secret):
            raise ValueError("LIVE mode requires BYBIT_API_KEY and BYBIT_API_SECRET")
