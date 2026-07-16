"""Minimal file-backed persistence for the stateful production risk controls.

Only the two pieces of risk state that must survive a process restart live here:
the emergency-stop latch and the day-start NAV baseline for the daily-loss limit.
It is a single small JSON file — no database, no framework — read at the start of
a cycle and written when the state changes.

Independent by construction: imports only stdlib (json/dataclasses/pathlib). It
knows nothing about brokers, execution, portfolio, strategy or Telegram.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RiskState:
    """Persisted risk state carried across separate program runs."""

    emergency_stopped: bool = False
    day: str | None = None            # ISO date the anchor belongs to
    day_start_nav: float | None = None  # NAV at the first run of that day


class RiskStateStore:
    """Load/save :class:`RiskState` as one small JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RiskState:
        """Return the persisted state, or a fresh default if unreadable/absent."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return RiskState()
        return RiskState(
            emergency_stopped=bool(data.get("emergency_stopped", False)),
            day=data.get("day"),
            day_start_nav=data.get("day_start_nav"),
        )

    def save(self, state: RiskState) -> None:
        """Persist ``state`` (creating the parent directory if needed)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "emergency_stopped": state.emergency_stopped,
                    "day": state.day,
                    "day_start_nav": state.day_start_nav,
                }
            ),
            encoding="utf-8",
        )
