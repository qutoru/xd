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
    leverage: float = 1.0  # per-symbol leverage the broker sets before entries
    position_idx: int = 0  # 0 = one-way mode (hedge mode uses 1/2 — unsupported)

    @classmethod
    def from_env(cls) -> BybitConfig:
        """Build from ``BYBIT_*`` environment variables (safe defaults).

        When ``BYBIT_TESTNET`` is true, testnet credentials are read from
        ``BYBIT_TESTNET_API_KEY`` / ``BYBIT_TESTNET_API_SECRET`` if set, so
        testnet and mainnet keys can coexist and the venue is switched by the
        single ``BYBIT_TESTNET`` flag. Testnet keys fall back to the base
        ``BYBIT_API_KEY`` / ``BYBIT_API_SECRET`` when the testnet slots are empty.
        """
        testnet = _env_bool("BYBIT_TESTNET", True)
        base_key = os.getenv("BYBIT_API_KEY", "")
        base_secret = os.getenv("BYBIT_API_SECRET", "")
        if testnet:
            api_key = os.getenv("BYBIT_TESTNET_API_KEY") or base_key
            api_secret = os.getenv("BYBIT_TESTNET_API_SECRET") or base_secret
        else:
            api_key = base_key
            api_secret = base_secret
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            mode=TradingMode(os.getenv("BYBIT_TRADING_MODE", "shadow").strip().lower()),
            leverage=float(os.getenv("BYBIT_LEVERAGE", "1")),
            position_idx=int(os.getenv("BYBIT_POSITION_IDX", "0")),
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
        if self.leverage < 1.0:
            raise ValueError("leverage must be >= 1.0")
        if self.position_idx not in (0, 1, 2):
            raise ValueError("position_idx must be 0 (one-way) or 1/2 (hedge)")
