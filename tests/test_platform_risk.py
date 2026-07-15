"""Stage 14 — RiskManager: weight -> risk-sized notional, exposure limits (offline)."""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.risk.manager import RiskConfig, RiskManager

TS = pd.Timestamp("2023-07-01", tz="UTC")


def _book(weights: dict) -> TargetBook:
    return TargetBook(asof=TS, weights=pd.Series(weights, dtype="float64"))


def test_scaling_by_nav():
    rm = RiskManager(RiskConfig(gross_target=1.0, max_gross=10.0,
                                max_position_pct=1.0, min_notional=0.0))
    book = _book({"A": 0.5, "B": -0.5})
    n10 = rm.size(book, 10_000)
    n20 = rm.size(book, 20_000)
    assert np.isclose(n10["A"], 5_000) and np.isclose(n10["B"], -5_000)
    assert np.isclose(n20["A"], 10_000)  # doubling NAV doubles notional


def test_scaling_by_gross_target():
    book = _book({"A": 0.5, "B": -0.5})
    cfg = dict(max_gross=10.0, max_position_pct=1.0, min_notional=0.0)
    n1 = RiskManager(RiskConfig(gross_target=1.0, **cfg)).size(book, 10_000)
    n2 = RiskManager(RiskConfig(gross_target=2.0, **cfg)).size(book, 10_000)
    assert np.isclose(n2["A"], 2 * n1["A"])  # gross_target is a leverage multiple


def test_max_position_pct_caps_each_name():
    rm = RiskManager(RiskConfig(max_position_pct=0.10, max_gross=10.0, min_notional=0.0))
    n = rm.size(_book({"A": 0.5, "B": -0.5}), 10_000)  # raw 5000 -> cap 0.10*10000
    assert np.isclose(abs(n["A"]), 1_000) and np.isclose(abs(n["B"]), 1_000)


def test_max_symbol_notional_caps_absolute():
    rm = RiskManager(RiskConfig(max_symbol_notional=500.0, max_position_pct=1.0,
                                max_gross=10.0, min_notional=0.0))
    n = rm.size(_book({"A": 0.5, "B": -0.5}), 10_000)  # raw 5000 -> abs cap 500
    assert np.isclose(abs(n["A"]), 500) and np.isclose(abs(n["B"]), 500)


def test_max_gross_scales_all_positions_proportionally():
    rm = RiskManager(RiskConfig(max_gross=0.5, max_position_pct=1.0, min_notional=0.0))
    n = rm.size(_book({"A": 0.5, "B": -0.5}), 10_000)  # raw gross 10000 -> cap 5000
    assert np.isclose(n.abs().sum(), 5_000)
    assert np.isclose(n["A"], 2_500) and np.isclose(n["B"], -2_500)  # proportional


def test_min_notional_drops_dust_positions():
    rm = RiskManager(RiskConfig(max_position_pct=1.0, max_gross=10.0, min_notional=2_000.0))
    n = rm.size(_book({"A": 0.9, "B": -0.1}), 10_000)  # A=9000 kept, B=1000 dropped
    assert list(n.index) == ["A"] and np.isclose(n["A"], 9_000)


def test_nonpositive_nav_yields_no_positions():
    assert RiskManager().size(_book({"A": 0.5, "B": -0.5}), 0.0).empty


def test_risk_manager_is_independent_of_execution_and_venue():
    risk_dir = pathlib.Path(__file__).resolve().parents[1] / "crypto_signal_bot" / "platform" / "risk"
    forbidden = ("execution", "broker", "bybit", "pybit", "crypto_signal_bot.research",
                 "signals", "alpha_library", "experiments")
    for py in risk_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods.append(node.module or "")
        bad = [m for m in mods if any(f in m for f in forbidden)]
        assert not bad, f"{py.name} imports forbidden: {bad}"
