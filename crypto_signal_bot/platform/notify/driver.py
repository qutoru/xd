"""In-process signal driver — self-drives ALX generation + paced delivery.

Starting the Telegram bot (``main.py bot``) is meant to be the whole flow: no
separate manual ``trade`` run. This driver, ticked once per long-poll iteration by
the listener, does two time-based jobs:

* **Cycle** — every ``cycle_interval_s`` it calls ``signal_source()`` (one ALX
  pipeline run + quality filter) and enqueues the signals that passed the check.
  It runs immediately on the first tick, so signals start flowing right after the
  bot starts.
* **Release** — at most one queued signal every ``release_interval_s`` (default 60s
  = one per minute) is handed to ``deliver(intent)``, which fans it out to the
  owner (buttoned) and to subscribers (plain). The first queued signal is released
  immediately; the rest drip out one per minute.

Both collaborators are injected as plain callables, so the pacing/queue logic is
unit-tested with fakes — no pipeline, no network, no Telegram. Time is injected via
``now_fn`` for the same reason. All work is best-effort: a failing cycle or
delivery is logged and never propagates into the listener's poll loop.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

from loguru import logger


class SignalDriver:
    """Periodically generate ALX signals and release them one per minute."""

    def __init__(
        self,
        *,
        signal_source: Callable[[], Sequence],
        deliver: Callable[[object], None],
        cycle_interval_s: float,
        release_interval_s: float = 60.0,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._signal_source = signal_source
        self._deliver = deliver
        self._cycle_interval = float(cycle_interval_s)
        self._release_interval = float(release_interval_s)
        self._now = now_fn
        self._queue: list = []
        self._last_cycle: float | None = None
        self._last_release: float | None = None

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def tick(self) -> None:
        """Advance one step: refresh candidates if due, then release at most one."""
        now = self._now()
        if self._last_cycle is None or (now - self._last_cycle) >= self._cycle_interval:
            self._refresh(now)
        if self._queue and (
            self._last_release is None
            or (now - self._last_release) >= self._release_interval
        ):
            self._release(now)

    def _refresh(self, now: float) -> None:
        """Run one signal cycle and enqueue signals not already queued (by symbol)."""
        self._last_cycle = now  # set first: a failing source still respects the interval
        try:
            candidates = list(self._signal_source())
        except Exception as exc:  # a bad cycle must not break the poll loop
            logger.error("Signal driver: cycle failed: {}", exc)
            return
        queued = {getattr(i, "symbol", None) for i in self._queue}
        added = 0
        for intent in candidates:
            if getattr(intent, "symbol", None) in queued:
                continue  # already waiting to be released; don't duplicate
            self._queue.append(intent)
            queued.add(getattr(intent, "symbol", None))
            added += 1
        logger.info(
            "Signal driver: cycle produced {} candidate(s), {} newly queued (queue={})",
            len(candidates), added, len(self._queue),
        )

    def _release(self, now: float) -> None:
        """Deliver the next queued signal (FIFO). Failure is logged, not retried."""
        intent = self._queue.pop(0)
        self._last_release = now
        try:
            self._deliver(intent)
        except Exception as exc:  # delivery is best-effort; move on to the next tick
            logger.error(
                "Signal driver: delivery failed for {}: {}",
                getattr(intent, "symbol", "?"), exc,
            )
