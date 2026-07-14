"""Stage 5 — Execution Layer: domain, engine, fake broker, layer purity (offline)."""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.execution.domain import (
    ExecutionReport,
    OrderStatus,
    PortfolioState,
    Position,
    Side,
)
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.fake_broker import InMemoryBroker
from crypto_signal_bot.platform.portfolio.book import TargetBook

ASOF = pd.Timestamp("2023-04-01", tz="UTC")


def _book(weights):
    return TargetBook(asof=ASOF, weights=pd.Series(weights, dtype="float64"))


def test_portfolio_state_weights():
    st = PortfolioState(value=100.0, positions={"A": Position("A", 50.0), "B": Position("B", -50.0)})
    w = st.weights()
    assert np.isclose(w["A"], 0.5) and np.isclose(w["B"], -0.5)
    assert st.notional("A") == 50.0 and st.notional("Z") == 0.0


def test_engine_plans_orders_from_flat():
    broker = InMemoryBroker(value=100.0)
    eng = ExecutionEngine(broker)
    reqs = eng.plan(_book({"A": 0.5, "B": -0.5}), broker.get_portfolio_state())
    by_sym = {r.symbol: r for r in reqs}
    assert by_sym["A"].side == Side.BUY and np.isclose(by_sym["A"].quantity, 50.0)
    assert by_sym["B"].side == Side.SELL and np.isclose(by_sym["B"].quantity, 50.0)
    assert by_sym["A"].target_weight == 0.5


def test_engine_executes_and_updates_state():
    broker = InMemoryBroker(value=100.0)
    eng = ExecutionEngine(broker)
    report = eng.execute(_book({"A": 0.5, "B": -0.5}))
    assert isinstance(report, ExecutionReport)
    assert report.n_filled == 2 and report.n_rejected == 0
    assert np.isclose(report.traded_notional, 100.0)
    w = report.resulting_state.weights()
    assert np.isclose(w["A"], 0.5) and np.isclose(w["B"], -0.5)


def test_engine_no_orders_when_already_on_target():
    broker = InMemoryBroker(value=100.0)
    eng = ExecutionEngine(broker)
    eng.execute(_book({"A": 0.5, "B": -0.5}))
    second = eng.execute(_book({"A": 0.5, "B": -0.5}))
    assert second.orders == []  # zero delta -> no trades


def test_engine_rebalances_delta_only():
    broker = InMemoryBroker(value=100.0)
    eng = ExecutionEngine(broker)
    eng.execute(_book({"A": 0.5, "B": -0.5}))
    report = eng.execute(_book({"A": 0.25, "B": -0.25}))  # halve both legs
    q = {o.request.symbol: (o.request.side, o.request.quantity) for o in report.orders}
    assert q["A"][0] == Side.SELL and np.isclose(q["A"][1], 25.0)
    assert q["B"][0] == Side.BUY and np.isclose(q["B"][1], 25.0)


def test_broker_reject_path():
    broker = InMemoryBroker(value=100.0, reject={"B"})
    eng = ExecutionEngine(broker)
    report = eng.execute(_book({"A": 0.5, "B": -0.5}))
    statuses = {o.request.symbol: o.status for o in report.orders}
    assert statuses["A"] == OrderStatus.FILLED
    assert statuses["B"] == OrderStatus.REJECTED


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    return mods


def test_execution_core_is_broker_and_alpha_agnostic():
    root = pathlib.Path(__file__).resolve().parents[1]
    exec_dir = root / "crypto_signal_bot" / "platform" / "execution"
    # Core files must not touch exchanges/alphas/research. bybit_broker is exempt
    # (it is the adapter) but must still avoid research/alpha.
    core_forbidden = ("bybit", "pybit", "signals.alx", "signals.registry",
                      "platform.data", "crypto_signal_bot.research", "alpha_library",
                      "experiments", "funding", "momentum")
    for py in exec_dir.glob("*.py"):
        mods = _imports(py)
        if py.name == "bybit_broker.py":
            assert not any("research" in m or "alpha_library" in m or "experiments" in m for m in mods)
            continue
        bad = [m for m in mods if any(f in m for f in core_forbidden)]
        assert not bad, f"{py.name} imports forbidden modules: {bad}"


def test_broker_side_does_not_know_portfolio():
    root = pathlib.Path(__file__).resolve().parents[1]
    exec_dir = root / "crypto_signal_bot" / "platform" / "execution"
    for name in ("domain.py", "broker.py", "fake_broker.py"):
        mods = _imports(exec_dir / name)
        assert not any("portfolio.book" in m or "signals" in m for m in mods), name
