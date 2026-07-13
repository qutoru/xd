"""Portfolio Layer — the only place that decides the book.

Consumes universal signal outputs (``TargetScores`` / ``TargetWeights``) and
produces a :class:`TargetBook`. It is alpha-agnostic (knows nothing about ALX,
momentum, funding, ML, OI) and execution-agnostic (knows nothing about Bybit,
brokers, orders). Each responsibility — combining, normalization, allocation,
constraints, turnover — is its own small module; the builder only orchestrates.
"""
