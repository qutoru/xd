"""Accounting layer — realized-PnL bookkeeping for live/paper trading.

Venue-agnostic: it consumes execution-domain ``Fill`` objects (from any Broker's
``get_fills``) and never speaks to an exchange, a signal or a portfolio. Stage 16
builds it in three pieces: fill attribution, then a persisted realized-PnL ledger.
"""
