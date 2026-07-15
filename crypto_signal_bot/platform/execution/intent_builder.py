"""Build TradeIntents from a TargetBook and market data.

Bridges the portfolio decision (weights) to strategy-facing TradeIntents:
- entry  = latest close of each traded symbol,
- TP/SL  = entry moved by a multiple of the symbol's realized return volatility
           (not a fixed percentage; no ATR/OHLC needed, no Production Core dup),
- confidence = combined signal magnitude, max-normalized to [0, 1]. This is a
           temporary signal-strength proxy, NOT a probability of trade success;
           it will be replaced by a model-quality confidence (win rate / class
           probability / ensemble) without changing this interface.

Takes plain price/return frames (not a MarketSnapshot) so it imports no data
layer — keeping the execution package free of Bybit/data dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.intent import TradeIntent
from crypto_signal_bot.platform.portfolio.book import TargetBook


@dataclass(frozen=True)
class IntentParams:
    """Risk-level sizing knobs (volatility multiples, not fixed percentages)."""

    vol_window: int = 20
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    notional_per_name: float = 1.0


def _confidence_source(book: TargetBook, active: pd.Series) -> pd.Series:
    """Per-symbol signal magnitude backing the confidence proxy.

    Combined score if available, else |weight|. Magnitude only — a stand-in for a
    future model-quality confidence, not a success probability.
    """
    if book.contributions:
        combined: pd.Series | None = None
        for s in book.contributions.values():
            combined = s if combined is None else combined.add(s, fill_value=0.0)
        return combined.reindex(active.index).abs()
    return active.abs()


def build_trade_intents(
    book: TargetBook,
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    params: IntentParams = IntentParams(),
    notionals: pd.Series | None = None,
) -> list[TradeIntent]:
    """Turn a dollar-neutral TargetBook into per-symbol TradeIntents.

    ``notionals`` (signed target notional per symbol, e.g. from RiskManager) sets
    the position size and side when supplied — symbols absent from it get no
    intent. IntentBuilder never computes risk itself; without ``notionals`` it
    falls back to the flat ``params.notional_per_name``.
    """
    active = book.weights[book.weights != 0.0]
    if active.empty:
        return []

    last_close = closes.iloc[-1]
    vol = returns.tail(params.vol_window).std()
    conf_src = _confidence_source(book, active)
    cmax = float(conf_src.max()) if len(conf_src) else 0.0

    intents: list[TradeIntent] = []
    for symbol, weight in active.items():
        entry = float(last_close.get(symbol, float("nan")))
        v = float(vol.get(symbol, float("nan")))
        if not (entry > 0) or not (v >= 0):
            continue

        if notionals is not None:
            if symbol not in notionals.index:
                continue  # sized out (below min_notional) -> no intent
            signed = float(notionals[symbol])
            if signed == 0.0:
                continue
            side = Side.BUY if signed > 0 else Side.SELL
            target_notional = abs(signed)
        else:
            side = Side.BUY if weight > 0 else Side.SELL
            target_notional = params.notional_per_name

        sign = 1.0 if side is Side.BUY else -1.0
        conf = float(conf_src.get(symbol, 0.0) / cmax) if cmax > 0 else 0.0
        intents.append(
            TradeIntent(
                symbol=symbol,
                side=side,
                target_notional=target_notional,
                entry=entry,
                take_profit=entry * (1.0 + sign * params.tp_mult * v),
                stop_loss=entry * (1.0 - sign * params.sl_mult * v),
                confidence=conf,
                strategy="platform",
                timestamp=book.asof,
            )
        )
    return intents
