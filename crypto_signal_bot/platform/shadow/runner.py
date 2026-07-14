"""Stateful daily Shadow Runner — accumulates virtual PnL, opens no positions.

Feed it a ready TargetBook and the day's MarketSnapshot; it marks the book,
carries forward cumulative PnL and the previous book (for turnover), and records
history. It knows nothing about Bybit, execution, orders or brokers — only books
and market data. Book construction (signals -> portfolio) happens upstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.shadow.replay import replay_day
from crypto_signal_bot.platform.shadow.report import ShadowParams, ShadowReport


class Recommendation(Protocol):
    """Anything serializable the runner can log — keeps shadow execution-agnostic.

    Duck-typed so the shadow layer never imports TradeIntent (or any execution
    entity); it just records objects that know how to serialize themselves.
    """

    def to_dict(self) -> dict: ...


class ShadowRunner:
    """Runs virtual daily accounting over a stream of target books."""

    def __init__(self, params: ShadowParams | None = None) -> None:
        self.params = params or ShadowParams()
        self._prev_book: TargetBook | None = None
        self._cum_pnl: float = 0.0
        self._history: list[ShadowReport] = []
        self._intents: list[list[Recommendation]] = []

    def step(
        self,
        book: TargetBook,
        snapshot: MarketSnapshot,
        intents: list[Recommendation] | None = None,
    ) -> ShadowReport:
        """Account for one day; store the day's intents and return the report."""
        report = replay_day(
            book, self._prev_book, snapshot, self.params, cum_pnl_prev=self._cum_pnl
        )
        self._cum_pnl = report.cum_pnl
        self._prev_book = book
        self._history.append(report)
        self._intents.append(list(intents) if intents else [])
        return report

    @property
    def cum_pnl(self) -> float:
        return self._cum_pnl

    def history(self) -> pd.DataFrame:
        """Full accounting history as a DataFrame (empty if no steps yet)."""
        return pd.DataFrame([r.as_row() for r in self._history])

    def intents_frame(self) -> pd.DataFrame:
        """All recorded recommendations flattened, tagged with their replay date."""
        rows: list[dict] = []
        for report, intents in zip(self._history, self._intents):
            for intent in intents:
                row = intent.to_dict()
                row["asof"] = report.asof
                rows.append(row)
        return pd.DataFrame(rows)

    def save_history(self, path: str | Path) -> Path:
        """Persist the PnL history to parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.history().to_parquet(path)
        return path

    def save_intents(self, path: str | Path) -> Path:
        """Persist the recorded TradeIntents to parquet (recommendation log)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.intents_frame().to_parquet(path)
        return path
