"""Discover and persist the traded symbol universe.

The bot trades the ``TOP_N_SYMBOLS`` most liquid linear USDT perpetuals, ranked
by 24h turnover from Bybit's public ``get_tickers`` endpoint. The resolved list
is cached to ``UNIVERSE_PATH`` so every stage (fetch/build/train/live) operates
on exactly the same set.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from loguru import logger
from pybit.unified_trading import HTTP

from crypto_signal_bot.config import CATEGORY, TOP_N_SYMBOLS, UNIVERSE_PATH


def discover_top_symbols(
    n: int = TOP_N_SYMBOLS,
    category: str = CATEGORY,
) -> list[str]:
    """Return the top-``n`` linear USDT perpetuals by 24h turnover.

    Args:
        n: How many symbols to keep.
        category: Bybit product category ("linear").

    Returns:
        Symbol strings (e.g. ``"BTCUSDT"``) ordered by descending turnover.
    """
    session = HTTP(testnet=False)
    resp = session.get_tickers(category=category)
    rows = resp["result"]["list"]

    # Keep USDT-quoted perpetuals only; sort by 24h turnover (USD volume).
    usdt = [r for r in rows if r.get("symbol", "").endswith("USDT")]
    usdt.sort(key=lambda r: float(r.get("turnover24h") or 0.0), reverse=True)

    symbols = [r["symbol"] for r in usdt[:n]]
    logger.info("Discovered {} symbols (top of {} USDT perps)", len(symbols), len(usdt))
    return symbols


def save_universe(symbols: list[str]) -> None:
    """Persist the universe list (with a timestamp) to ``UNIVERSE_PATH``."""
    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
    }
    with UNIVERSE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved universe of {} symbols -> {}", len(symbols), UNIVERSE_PATH)


def load_universe() -> list[str]:
    """Load the cached universe; raise if it has not been created yet."""
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(
            f"No universe at {UNIVERSE_PATH} — run `py main.py symbols` first"
        )
    with UNIVERSE_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["symbols"]


def refresh_universe(n: int = TOP_N_SYMBOLS) -> list[str]:
    """Discover the current top-``n`` symbols and cache them."""
    symbols = discover_top_symbols(n)
    save_universe(symbols)
    return symbols


def get_universe() -> list[str]:
    """Return the cached universe, discovering and caching it if absent."""
    try:
        return load_universe()
    except FileNotFoundError:
        logger.warning("Universe not found — discovering it now")
        return refresh_universe()
