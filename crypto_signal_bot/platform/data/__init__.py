"""Platform data layer — alpha-agnostic cross-sectional market data.

Provides the daily inputs every signal needs (tradable universe, aligned close
panel, daily funding panel) behind small provider interfaces so the source can
be swapped (Bybit today, a backtest replay or a test double tomorrow). Concrete
Bybit providers are thin wrappers over the existing production data layer
(:mod:`crypto_signal_bot.data`); no research code is imported.
"""
