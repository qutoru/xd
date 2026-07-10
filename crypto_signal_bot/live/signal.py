"""Generate a trading signal for the latest closed bar.

Pulls recent candles, computes features, runs the trained model on the most
recent **closed** bar (the still-forming bar is dropped to avoid acting on
incomplete data), gates the direction on model confidence, and derives ATR-based
stop-loss / take-profit levels — the same levels the backtest trades. This is a
signal only; nothing is executed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from crypto_signal_bot.config import (
    HORIZON,
    INTERVAL,
    LIVE_LOOKBACK_DAYS,
    LIVE_PROB_THRESHOLD,
    SL_ATR_MULT,
    SYMBOL,
    TP_ATR_MULT,
)
from crypto_signal_bot.data.fetcher import fetch_ohlcv
from crypto_signal_bot.features.indicators import build_features
from crypto_signal_bot.model.predict import SignalModel

_DIR_TEXT = {1: "LONG", -1: "SHORT", 0: "NO-TRADE"}


@dataclass(frozen=True)
class Signal:
    """A single point-in-time trading signal."""

    symbol: str
    interval: str
    bar_time: pd.Timestamp
    direction: int  # gated by confidence: -1/0/1
    raw_direction: int  # model argmax before the confidence gate
    confidence: float
    threshold: float
    entry: float
    stop_loss: float
    take_profit: float
    horizon: int

    @property
    def is_trade(self) -> bool:
        return self.direction != 0

    def format(self) -> str:
        """Render a human-readable Telegram message."""
        head = f"📊 {self.symbol} {self.interval}m — {_DIR_TEXT[self.direction]}"
        ts = self.bar_time.strftime("%Y-%m-%d %H:%M UTC")
        conf = f"confidence {self.confidence:.0%} (gate {self.threshold:.0%})"

        if not self.is_trade:
            reason = (
                "below confidence gate"
                if self.raw_direction != 0
                else "model says flat"
            )
            return (
                f"{head}\n"
                f"Bar close: {ts}\n"
                f"Model lean: {_DIR_TEXT[self.raw_direction]}, {conf}\n"
                f"→ No trade ({reason}).\n"
                f"⚠️ Signal only — not financial advice."
            )

        sl_pct = (self.stop_loss / self.entry - 1.0) * (1 if self.direction == 1 else -1)
        tp_pct = (self.take_profit / self.entry - 1.0) * (1 if self.direction == 1 else -1)
        return (
            f"{head}\n"
            f"Bar close: {ts}\n"
            f"{conf}\n"
            f"Entry: {self.entry:,.2f}\n"
            f"Stop-loss: {self.stop_loss:,.2f} ({sl_pct:+.2%})\n"
            f"Take-profit: {self.take_profit:,.2f} ({tp_pct:+.2%})\n"
            f"Horizon: {self.horizon} bars (~{self.horizon * int(self.interval) / 60:.0f}h)\n"
            f"⚠️ Signal only — no auto-execution. Not financial advice."
        )


def _drop_forming_bar(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Drop the most recent bar if its interval has not fully elapsed."""
    if df.empty:
        return df
    interval_ms = int(interval) * 60 * 1000
    now_ms = int(time.time() * 1000)
    if now_ms < int(df["timestamp"].iloc[-1]) + interval_ms:
        return df.iloc[:-1]
    return df


def generate_signal(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    threshold: float = LIVE_PROB_THRESHOLD,
    model: SignalModel | None = None,
) -> Signal:
    """Produce a :class:`Signal` for the latest closed bar.

    Args:
        model: Optional pre-loaded model (avoids re-reading it from disk when
            iterating many symbols in the live loop).

    Raises:
        RuntimeError: If not enough history is available to compute features.
    """
    raw = fetch_ohlcv(symbol=symbol, interval=interval, history_days=LIVE_LOOKBACK_DAYS)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw = _drop_forming_bar(raw, interval)

    features = build_features(raw)
    last_feat = features.iloc[[-1]]
    if last_feat.isna().any(axis=1).iloc[0]:
        raise RuntimeError(
            "Latest feature row has NaNs — increase LIVE_LOOKBACK_DAYS for warm-up"
        )

    if model is None:
        model = SignalModel.load(symbol, interval)
    signals, confs = model.predict_with_conf(last_feat)
    raw_direction = int(signals[0])
    confidence = float(confs[0])
    direction = raw_direction if confidence >= threshold else 0

    last = raw.iloc[-1]
    atr = float(features["atr"].iloc[-1])
    entry = float(last["close"])
    # SL/TP are only meaningful for an actual trade direction.
    d = direction if direction != 0 else raw_direction or 1
    stop_loss = entry - d * SL_ATR_MULT * atr
    take_profit = entry + d * TP_ATR_MULT * atr

    sig = Signal(
        symbol=symbol,
        interval=interval,
        bar_time=pd.Timestamp(last["datetime"]),
        direction=direction,
        raw_direction=raw_direction,
        confidence=confidence,
        threshold=threshold,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        horizon=HORIZON,
    )
    logger.info(
        "Signal @ {}: dir={} conf={:.1%} entry={:.2f}",
        sig.bar_time,
        _DIR_TEXT[direction],
        confidence,
        entry,
    )
    return sig
