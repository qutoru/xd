"""Live loop over the multi-symbol universe (Phase 7 + multi-symbol).

Wakes just after every ``interval``-minute boundary (plus a small buffer so the
exchange has published the closed candle), then for **each symbol** in the
universe generates a signal, de-duplicates on bar time and notifies Telegram.
Per-symbol errors are isolated so one bad symbol never stops the sweep, and one
bad iteration never kills the loop. Models are loaded once and cached.

Runs signals only — the process holds no positions and executes no orders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from crypto_signal_bot.config import (
    INTERVAL,
    LIVE_BAR_BUFFER_SEC,
    LIVE_NOTIFY_NO_TRADE,
)
from crypto_signal_bot.data.universe import get_universe
from crypto_signal_bot.live.signal import generate_signal
from crypto_signal_bot.live.telegram import send_telegram
from crypto_signal_bot.model.predict import SignalModel
from crypto_signal_bot.model.train import model_path


@dataclass
class _LoopState:
    # Last processed bar time per symbol (de-duplication).
    last_bar_time: dict[str, pd.Timestamp] = field(default_factory=dict)
    # Cache of loaded models per symbol.
    models: dict[str, SignalModel] = field(default_factory=dict)


def _seconds_to_next_bar(interval_min: int, buffer_sec: int) -> float:
    """Seconds from now until ``buffer_sec`` past the next bar boundary."""
    interval_sec = interval_min * 60
    now = time.time()
    next_close = (int(now // interval_sec) + 1) * interval_sec
    return next_close + buffer_sec - now


def _get_model(state: _LoopState, symbol: str, interval: str) -> SignalModel:
    """Return a cached model for ``symbol``, loading it on first use."""
    model = state.models.get(symbol)
    if model is None:
        model = SignalModel.load(symbol, interval)
        state.models[symbol] = model
    return model


def _process_symbol(
    state: _LoopState,
    symbol: str,
    *,
    interval: str,
    notify_no_trade: bool,
    dry_run: bool,
) -> bool:
    """Generate and maybe send one symbol's signal; return True if fresh."""
    model = _get_model(state, symbol, interval)
    sig = generate_signal(symbol, interval, model=model)

    last = state.last_bar_time.get(symbol)
    if last is not None and sig.bar_time <= last:
        logger.debug("{}: bar {} already processed — skipping", symbol, sig.bar_time)
        return False
    state.last_bar_time[symbol] = sig.bar_time

    if sig.is_trade:
        send_telegram(sig.format(), dry_run=dry_run)
    elif notify_no_trade:
        send_telegram(sig.format(), dry_run=dry_run)
    else:
        logger.info("{}: no-trade bar {} (logged only)", symbol, sig.bar_time)
    return True


def run_sweep(
    state: _LoopState,
    symbols: list[str],
    *,
    interval: str = INTERVAL,
    notify_no_trade: bool = LIVE_NOTIFY_NO_TRADE,
    dry_run: bool = False,
) -> None:
    """Run one sweep over all symbols, isolating per-symbol failures."""
    for symbol in symbols:
        try:
            _process_symbol(
                state,
                symbol,
                interval=interval,
                notify_no_trade=notify_no_trade,
                dry_run=dry_run,
            )
        except Exception as exc:  # isolate a bad symbol
            logger.exception("{}: signal failed, continuing: {}", symbol, exc)
    logger.info("Sweep complete over {} symbols", len(symbols))


def run_live(
    symbols: list[str] | None = None,
    *,
    interval: str = INTERVAL,
    notify_no_trade: bool = LIVE_NOTIFY_NO_TRADE,
    dry_run: bool = False,
    once: bool = False,
) -> None:
    """Run the live signalling loop over the universe (or a single sweep)."""
    if symbols is None:
        symbols = get_universe()

    # Only trade symbols that actually have a trained model (others were skipped
    # at training, e.g. too little history) — avoids retrying them every sweep.
    tradable = [s for s in symbols if model_path(s, interval).exists()]
    missing = sorted(set(symbols) - set(tradable))
    if missing:
        logger.warning("Skipping {} symbols without a model: {}", len(missing), missing)

    state = _LoopState()
    logger.info("Live universe: {} tradable symbols", len(tradable))
    symbols = tradable

    if once:
        run_sweep(
            state,
            symbols,
            interval=interval,
            notify_no_trade=notify_no_trade,
            dry_run=dry_run,
        )
        return

    logger.info("Live loop started for {} symbols (Ctrl+C to stop)", len(symbols))
    try:
        while True:
            wait = _seconds_to_next_bar(int(interval), LIVE_BAR_BUFFER_SEC)
            logger.info("Sleeping {:.0f}s until next bar close", wait)
            time.sleep(max(1.0, wait))
            try:
                run_sweep(
                    state,
                    symbols,
                    interval=interval,
                    notify_no_trade=notify_no_trade,
                    dry_run=dry_run,
                )
            except Exception as exc:  # keep the loop alive
                logger.exception("Live sweep failed, continuing: {}", exc)
    except KeyboardInterrupt:
        logger.info("Live loop stopped by user")
