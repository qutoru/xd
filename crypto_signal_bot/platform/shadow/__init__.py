"""Shadow Layer — virtual (no-trade) daily accounting of a TargetBook.

Given a ready TargetBook and market data, computes how the book would have
behaved if traded, WITHOUT opening any position. Fully alpha-agnostic (no ALX,
funding-rank, momentum or ML) and execution-agnostic (no Bybit, orders, broker
or API). Each responsibility — price PnL, funding PnL, costs, per-day replay,
stateful daily loop — is its own small module.
"""
