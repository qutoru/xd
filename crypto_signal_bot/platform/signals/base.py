"""Signal Layer core — the strategy-agnostic plugin API.

Knows nothing about Bybit, ALX, funding, momentum, ML or portfolio construction.
A :class:`SignalProvider` maps a :class:`MarketSnapshot` (plus optional config and
state) to exactly one of two standardized outputs: :class:`TargetScores` or
:class:`TargetWeights`. It must NOT normalize, risk-adjust, control turnover, or
impose dollar-neutrality — that is the Portfolio Layer's responsibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping, Union

import pandas as pd

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot


class SignalKind(str, Enum):
    """The two — and only two — permitted output formats."""

    SCORES = "scores"
    WEIGHTS = "weights"


def _validate(values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError("signal output must be a pandas Series indexed by symbol")
    if values.index.hasnans or not values.index.is_unique:
        raise ValueError("signal index must be unique, non-null symbols")


@dataclass(frozen=True)
class TargetScores:
    """Cross-sectional preference; higher = more long. Un-normalized."""

    values: pd.Series
    asof: pd.Timestamp
    kind: ClassVar[SignalKind] = SignalKind.SCORES

    def __post_init__(self) -> None:
        _validate(self.values)


@dataclass(frozen=True)
class TargetWeights:
    """Explicit per-name target weights proposed by the signal (still raw)."""

    values: pd.Series
    asof: pd.Timestamp
    kind: ClassVar[SignalKind] = SignalKind.WEIGHTS

    def __post_init__(self) -> None:
        _validate(self.values)


SignalOutput = Union[TargetScores, TargetWeights]


class SignalProvider(ABC):
    """Abstract plugin: ``MarketSnapshot`` (+config/state) -> ``SignalOutput``."""

    name: ClassVar[str] = ""

    @abstractmethod
    def generate(
        self,
        snapshot: MarketSnapshot,
        config: Mapping[str, Any] | None = None,
        state: Any | None = None,
    ) -> SignalOutput:
        """Produce the target scores/weights for ``snapshot.asof``."""
        ...
