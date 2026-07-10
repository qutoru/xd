"""Live loop: emit a signal shortly after each bar closes (Phase 7).

Wakes just after every ``interval``-minute boundary (plus a small buffer so the
exchange has published the closed candle), generates a signal, de-duplicates on
bar time (so a data lag never sends the same bar twice) and notifies Telegram.
Each iteration is wrapped so a transient fetch/API error is logged but does not
kill the loop. Ctrl+C stops it cleanly.

Runs one signal at a time — the process holds no positions and executes no
orders; it only broadcasts signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from crypto_signal_bot.config import (
    INTERVAL,
    LIVE_BAR_BUFFER_SEC,
    LIVE_NOTIFY_NO_TRADE,
    SYMBOL,
)
from crypto_signal_bot.live.signal import generate_signal
from crypto_signal_bot.live.telegram import send_telegram


@dataclass
class _LoopState:
    last_bar_time: pd.Timestamp | None = None


def _seconds_to_next_bar(interval_min: int, buffer_sec: int) -> float:
    """Seconds from now until ``buffer_sec`` past the next bar boundary."""
    interval_sec = interval_min * 60
    now = time.time()
    next_close = (int(now // interval_sec) + 1) * interval_sec
    return next_close + buffer_sec - now


def run_once(
    state: _LoopState,
    *,
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    notify_no_trade: bool = LIVE_NOTIFY_NO_TRADE,
    dry_run: bool = False,
) -> bool:
    """Generate and (maybe) send one signal, de-duplicating on bar time.

    Returns:
        True if a fresh bar was processed, False if it was a duplicate.
    """
    sig = generate_signal(symbol, interval)

    if state.last_bar_time is not None and sig.bar_time <= state.last_bar_time:
        logger.warning(
            "Bar {} already processed (data not updated yet) — skipping",
            sig.bar_time,
        )
        return False
    state.last_bar_time = sig.bar_time

    text = sig.format()
    if sig.is_trade or notify_no_trade:
        send_telegram(text, dry_run=dry_run)
    else:
        logger.info("No-trade bar {} — logged, not notified:\n{}", sig.bar_time, text)
    return True


def run_live(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    *,
    notify_no_trade: bool = LIVE_NOTIFY_NO_TRADE,
    dry_run: bool = False,
    once: bool = False,
) -> None:
    """Run the live signalling loop (or a single iteration if ``once``)."""
    state = _LoopState()

    if once:
        run_once(
            state,
            symbol=symbol,
            interval=interval,
            notify_no_trade=notify_no_trade,
            dry_run=dry_run,
        )
        return

    logger.info("Live loop started for {} {}m (Ctrl+C to stop)", symbol, interval)
    try:
        while True:
            wait = _seconds_to_next_bar(int(interval), LIVE_BAR_BUFFER_SEC)
            logger.info("Sleeping {:.0f}s until next bar close", wait)
            time.sleep(max(1.0, wait))
            try:
                run_once(
                    state,
                    symbol=symbol,
                    interval=interval,
                    notify_no_trade=notify_no_trade,
                    dry_run=dry_run,
                )
            except Exception as exc:  # keep the loop alive on transient errors
                logger.exception("Live iteration failed, continuing: {}", exc)
    except KeyboardInterrupt:
        logger.info("Live loop stopped by user")
