"""SignalDriver: scheduled ALX cycles + one-per-minute paced release.

Pure timing/queue logic with fake collaborators — no pipeline, no Telegram, no
network. A mutable clock drives ``tick`` so the cycle interval and the 1/min
release cadence are asserted deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

from crypto_signal_bot.platform.notify.driver import SignalDriver


@dataclass
class _Sig:
    symbol: str


class _Clock:
    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


def _driver(clock, source, delivered, *, cycle=100.0, release=60.0):
    return SignalDriver(
        signal_source=source,
        deliver=lambda i: delivered.append(i),
        cycle_interval_s=cycle,
        release_interval_s=release,
        now_fn=clock,
    )


def test_first_tick_runs_cycle_and_releases_one_immediately():
    clock = _Clock(0.0)
    delivered: list = []
    d = _driver(clock, lambda: [_Sig("A"), _Sig("B"), _Sig("C")], delivered)

    d.tick()

    assert [s.symbol for s in delivered] == ["A"]  # only one released
    assert d.queue_size == 2


def test_releases_one_per_minute_fifo():
    clock = _Clock(0.0)
    delivered: list = []
    # Large cycle so only the release cadence (not a re-cycle) is exercised here.
    d = _driver(clock, lambda: [_Sig("A"), _Sig("B"), _Sig("C")], delivered, cycle=1e9)

    d.tick()                       # t=0 -> A
    clock.t = 30.0; d.tick()       # too soon, nothing
    assert [s.symbol for s in delivered] == ["A"]
    clock.t = 60.0; d.tick()       # -> B
    clock.t = 120.0; d.tick()      # -> C
    assert [s.symbol for s in delivered] == ["A", "B", "C"]
    assert d.queue_size == 0


def test_cycle_not_rerun_before_interval():
    clock = _Clock(0.0)
    calls = {"n": 0}

    def source():
        calls["n"] += 1
        return [_Sig("A")]

    d = _driver(clock, source, [], cycle=100.0)
    d.tick()                       # cycle #1 at t=0
    clock.t = 60.0; d.tick()       # within interval -> no new cycle
    assert calls["n"] == 1
    clock.t = 100.0; d.tick()      # interval elapsed -> cycle #2
    assert calls["n"] == 2


def test_refresh_dedupes_already_queued_symbols():
    clock = _Clock(0.0)
    delivered: list = []
    # Huge release interval so the second tick only refreshes (no release to muddy
    # the queue), isolating the dedup: B (still queued) must not be duplicated.
    d = _driver(clock, lambda: [_Sig("A"), _Sig("B")], delivered, cycle=100.0, release=1e9)

    d.tick()                       # t=0: queue [A,B], release A -> queue [B]
    clock.t = 100.0; d.tick()      # cycle #2 returns A,B again; B already queued
    assert sorted(s.symbol for s in d._queue) == ["A", "B"]  # A re-added, B not duped
    assert d.queue_size == 2


def test_failing_source_does_not_raise_and_respects_interval():
    clock = _Clock(0.0)
    calls = {"n": 0}

    def bad_source():
        calls["n"] += 1
        raise RuntimeError("boom")

    d = _driver(clock, bad_source, [], cycle=100.0)
    d.tick()                       # cycle attempted, swallowed
    clock.t = 50.0; d.tick()       # within interval -> not retried
    assert calls["n"] == 1


def test_failing_delivery_advances_and_drops_signal():
    clock = _Clock(0.0)

    def boom(_intent):
        raise RuntimeError("send failed")

    d = SignalDriver(
        signal_source=lambda: [_Sig("A"), _Sig("B")],
        deliver=boom, cycle_interval_s=100.0, release_interval_s=60.0, now_fn=clock,
    )
    d.tick()                       # A pops, delivery raises but is swallowed
    assert d.queue_size == 1       # A dropped, B still queued
    clock.t = 60.0; d.tick()       # B pops next minute
    assert d.queue_size == 0
