"""Autonomous production trading platform (multi-alpha, cross-sectional).

This package is deliberately self-contained: it imports NOTHING from
``alpha_library`` / ``experiments`` / any frozen research alpha, and nothing from
``crypto_signal_bot.research``. The neutral market-neutral machinery it needs is
vendored under :mod:`crypto_signal_bot.platform.core` so that production and
frozen research can evolve independently. The only link to research is
conceptual: the ALX *formula* is re-expressed as plain constants in
:mod:`crypto_signal_bot.platform.config` — never imported.
"""
