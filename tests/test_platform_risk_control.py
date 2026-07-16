"""Stage 15 — ProductionRiskControl: the final gate on new entries (offline)."""

from __future__ import annotations

import ast
import pathlib

import pandas as pd

from crypto_signal_bot.platform.risk_control.control import (
    BlockReason,
    ProductionRiskControl,
    RiskControlConfig,
    RiskControlDecision,
)


def _series(d: dict) -> pd.Series:
    return pd.Series(d, dtype="float64")


# --- no-op default -----------------------------------------------------------
def test_default_config_is_a_noop_gate():
    rc = ProductionRiskControl()
    d = rc.evaluate(_series({"A": 1e9, "B": -1e9}), None, nav=1.0)
    assert isinstance(d, RiskControlDecision)
    assert d.ok and set(d.allowed) == {"A", "B"} and d.n_blocked == 0


# --- 1. Kill Switch ----------------------------------------------------------
def test_kill_switch_blocks_all_new_entries():
    rc = ProductionRiskControl(RiskControlConfig(kill_switch=True))
    d = rc.evaluate(_series({"A": 1000.0, "B": -500.0}), None, nav=10_000.0)
    assert d.halted and d.halt_reason is BlockReason.KILL_SWITCH
    assert d.allowed == () and d.n_blocked == 2
    assert all(r is BlockReason.KILL_SWITCH for _, r in d.blocked)


# --- 2. Daily Loss Limit -----------------------------------------------------
def test_daily_loss_limit_halts_once_loss_exceeds_threshold():
    rc = ProductionRiskControl(RiskControlConfig(daily_loss_limit=0.05))
    # day-start 10_000, now 9_400 -> 6% loss >= 5% -> halt
    d = rc.evaluate(_series({"A": 100.0}), None, nav=9_400.0, day_start_nav=10_000.0)
    assert d.halted and d.halt_reason is BlockReason.DAILY_LOSS_LIMIT


def test_daily_loss_limit_allows_when_loss_below_threshold():
    rc = ProductionRiskControl(RiskControlConfig(daily_loss_limit=0.05))
    # 4% loss < 5% -> new entries still allowed
    d = rc.evaluate(_series({"A": 100.0}), None, nav=9_600.0, day_start_nav=10_000.0)
    assert not d.halted and d.is_allowed("A")


# --- 3. Max Open Positions ---------------------------------------------------
def test_max_open_positions_caps_new_entries_counting_held():
    rc = ProductionRiskControl(RiskControlConfig(max_open_positions=2))
    positions = _series({"X": 500.0})  # already holding one
    d = rc.evaluate(_series({"A": 100.0, "B": 100.0, "C": 100.0}), positions, nav=10_000.0)
    assert d.is_allowed("A")  # 1 held + 1 new = 2 (ok)
    assert d.reason_for("B") is BlockReason.MAX_OPEN_POSITIONS
    assert d.reason_for("C") is BlockReason.MAX_OPEN_POSITIONS


# --- 4. Max Exposure ---------------------------------------------------------
def test_max_exposure_blocks_entries_that_would_exceed_gross():
    rc = ProductionRiskControl(RiskControlConfig(max_exposure=0.10))  # cap = 1000
    positions = _series({"X": 600.0})  # 600 gross already used
    d = rc.evaluate(_series({"A": 300.0, "B": 300.0}), positions, nav=10_000.0)
    assert d.is_allowed("A")  # 600 + 300 = 900 <= 1000
    assert d.reason_for("B") is BlockReason.MAX_EXPOSURE  # 900 + 300 = 1200 > 1000


# --- 5. Max Position Size (blocks even if RiskManager errs) -------------------
def test_max_position_size_blocks_oversized_order():
    rc = ProductionRiskControl(RiskControlConfig(max_position_size=0.05))  # cap = 500
    d = rc.evaluate(_series({"A": 400.0, "B": 800.0}), None, nav=10_000.0)
    assert d.is_allowed("A")
    assert d.reason_for("B") is BlockReason.MAX_POSITION_SIZE


def test_max_position_size_applies_to_existing_positions_too():
    # Even a re-affirm of an already-open symbol is blocked if oversized.
    rc = ProductionRiskControl(RiskControlConfig(max_position_size=0.05))  # cap = 500
    positions = _series({"X": 400.0})
    d = rc.evaluate(_series({"X": 800.0}), positions, nav=10_000.0)
    assert d.reason_for("X") is BlockReason.MAX_POSITION_SIZE


# --- 6. Emergency Stop -------------------------------------------------------
def test_emergency_stop_latches_and_resets():
    rc = ProductionRiskControl()
    assert not rc.emergency_stopped
    assert rc.evaluate(_series({"A": 100.0}), None, nav=10_000.0).is_allowed("A")

    rc.trip_emergency_stop()
    assert rc.emergency_stopped
    d = rc.evaluate(_series({"A": 100.0}), None, nav=10_000.0)
    assert d.halted and d.halt_reason is BlockReason.EMERGENCY_STOP

    rc.reset_emergency_stop()
    assert rc.evaluate(_series({"A": 100.0}), None, nav=10_000.0).is_allowed("A")


# --- existing-position management continues ----------------------------------
def test_reaffirming_an_open_position_is_allowed_under_count_and_exposure_caps():
    # max_open_positions=1 and a tiny exposure cap would block any *new* entry, but
    # re-affirming an already-open symbol adds no new position -> allowed.
    rc = ProductionRiskControl(RiskControlConfig(max_open_positions=1, max_exposure=0.01))
    positions = _series({"X": 5_000.0})
    d = rc.evaluate(_series({"X": 6_000.0}), positions, nav=10_000.0)
    assert d.is_allowed("X")


def test_halt_refuses_entries_without_touching_positions_argument():
    # A halt blocks proposed entries; the positions input is never mutated/closed.
    rc = ProductionRiskControl(RiskControlConfig(kill_switch=True))
    positions = _series({"X": 500.0})
    d = rc.evaluate(_series({"A": 100.0}), positions, nav=10_000.0)
    assert d.halted and not d.is_allowed("A")
    assert positions.to_dict() == {"X": 500.0}  # untouched


# --- P1: configurable from env ----------------------------------------------
def test_risk_control_config_from_env_reads_all_limits(monkeypatch):
    monkeypatch.setenv("RISK_KILL_SWITCH", "true")
    monkeypatch.setenv("RISK_DAILY_LOSS_LIMIT", "0.05")
    monkeypatch.setenv("RISK_MAX_OPEN_POSITIONS", "3")
    monkeypatch.setenv("RISK_MAX_EXPOSURE", "0.5")
    monkeypatch.setenv("RISK_MAX_POSITION_SIZE", "0.1")
    cfg = RiskControlConfig.from_env()
    assert cfg.kill_switch is True
    assert cfg.daily_loss_limit == 0.05
    assert cfg.max_open_positions == 3
    assert cfg.max_exposure == 0.5
    assert cfg.max_position_size == 0.1


def test_risk_control_config_from_env_unset_is_the_noop_default(monkeypatch):
    for k in ("RISK_KILL_SWITCH", "RISK_DAILY_LOSS_LIMIT", "RISK_MAX_OPEN_POSITIONS",
              "RISK_MAX_EXPOSURE", "RISK_MAX_POSITION_SIZE"):
        monkeypatch.delenv(k, raising=False)
    assert RiskControlConfig.from_env() == RiskControlConfig()


# --- P2: persistence store ---------------------------------------------------
def test_risk_state_store_roundtrip_and_missing_file(tmp_path):
    from crypto_signal_bot.platform.risk_control.state import RiskState, RiskStateStore

    store = RiskStateStore(tmp_path / "risk.json")
    assert store.load() == RiskState()  # missing file -> fresh default
    store.save(RiskState(emergency_stopped=True, day="2023-05-01", day_start_nav=10_000.0))
    got = store.load()
    assert got.emergency_stopped is True
    assert got.day == "2023-05-01" and got.day_start_nav == 10_000.0


# --- independence (AST/boundary) ---------------------------------------------
def test_risk_control_is_independent_of_execution_and_venue():
    rc_dir = (
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_signal_bot" / "platform" / "risk_control"
    )
    forbidden = (
        "execution", "broker", "bybit", "pybit", "crypto_signal_bot.research",
        "signals", "portfolio", "strategy", "telegram", "notify", "shadow",
        "alpha_library", "experiments",
    )
    for py in rc_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        mods: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods.append(node.module or "")
        bad = [m for m in mods if any(f in m for f in forbidden)]
        assert not bad, f"{py.name} imports forbidden: {bad}"
