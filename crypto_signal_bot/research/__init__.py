"""Systematic alpha research harness.

Deterministic, config-driven, no-lookahead backtesting of rule-based signal
hypotheses, with the full quant metric panel (Sharpe/Sortino/Calmar/...),
chronological 60/20/20 evaluation and purged walk-forward. Kept separate from
the production ML pipeline so hypotheses can be tested in isolation.
"""
