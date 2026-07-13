"""Signal Layer — a strategy-agnostic plugin system.

The core (:mod:`base`, :mod:`registry`) knows nothing about any specific alpha.
Each strategy lives in its own subpackage (``signals/<name>/signal.py``) and
registers itself as a :class:`SignalProvider`. Providers return only
``TargetScores`` or ``TargetWeights`` — never portfolios, risk or normalization.
"""
