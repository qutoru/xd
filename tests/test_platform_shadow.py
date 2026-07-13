"""Stage 4 — Shadow Layer: components, replay, runner, layer purity (no network)."""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.shadow import costs
from crypto_signal_bot.platform.shadow.funding import funding_pnl
from crypto_signal_bot.platform.shadow.pnl import price_pnl
from crypto_signal_bot.platform.shadow.replay import replay_day
from crypto_signal_bot.platform.shadow.report import ShadowParams, ShadowReport
from crypto_signal_bot.platform.shadow.runner import ShadowRunner

SYMS = ["A", "B", "C", "D"]
ASOF = pd.Timestamp("2023-03-01", tz="UTC")


def _book(weights, asof=ASOF):
    return TargetBook(asof=asof, weights=pd.Series(weights, index=SYMS, dtype="float64"))


def _snapshot(ret_last, fund_last, asof=ASOF, n=3):
    idx = pd.date_range(end=asof, periods=n, freq="D")
    closes = pd.DataFrame(100.0, index=idx, columns=SYMS)
    returns = pd.DataFrame(0.0, index=idx, columns=SYMS)
    returns.iloc[-1] = pd.Series(ret_last, index=SYMS)
    funding = pd.DataFrame(0.0, index=idx, columns=SYMS)
    funding.iloc[-1] = pd.Series(fund_last, index=SYMS)
    return MarketSnapshot(asof, SYMS, closes, returns, funding)


def test_cost_functions_linear():
    assert costs.fees(2.0, 0.001) == 0.002
    assert costs.slippage(2.0, 0.0005) == 0.001
    assert costs.cost_of_capital(4.0, 0.0001) == 0.0004


def test_price_and_funding_pnl_signs():
    w = pd.Series([0.5, -0.5], index=["A", "B"])
    assert np.isclose(price_pnl(w, pd.Series([0.10, -0.10], index=["A", "B"])), 0.10)
    # long A pays positive funding -> negative funding PnL
    assert np.isclose(funding_pnl(w, pd.Series([0.001, 0.0], index=["A", "B"])), -0.0005)


def test_replay_day_known_answer():
    book = _book([0.5, -0.5, 0.0, 0.0])
    snap = _snapshot(ret_last=[0.10, -0.10, 0, 0], fund_last=[0.001, 0.0, 0, 0])
    params = ShadowParams(fee_rate=0.001, slippage_rate=0.0, cost_of_capital=0.0)
    rep = replay_day(book, None, snap, params)
    # price = 0.5*0.10 + -0.5*-0.10 = 0.10 ; funding = -(0.5*0.001) = -0.0005
    # turnover from flat = |0.5|+|0.5| = 1.0 ; fees = 1.0*0.001 = 0.001
    assert np.isclose(rep.price_pnl, 0.10)
    assert np.isclose(rep.funding_pnl, -0.0005)
    assert np.isclose(rep.turnover, 1.0)
    assert np.isclose(rep.fees, 0.001)
    assert np.isclose(rep.daily_pnl, 0.10 - 0.0005 - 0.001)
    assert np.isclose(rep.gross, 1.0) and np.isclose(rep.net, 0.0)
    assert rep.n_long == 1 and rep.n_short == 1


def test_replay_turnover_vs_prev_book():
    prev = _book([0.5, -0.5, 0.0, 0.0])
    curr = _book([-0.5, 0.5, 0.0, 0.0])  # full flip -> turnover 2.0
    snap = _snapshot([0, 0, 0, 0], [0, 0, 0, 0])
    rep = replay_day(curr, prev, snap, ShadowParams(fee_rate=0.0, slippage_rate=0.0))
    assert np.isclose(rep.turnover, 2.0)


def test_runner_accumulates_and_history():
    runner = ShadowRunner(ShadowParams(fee_rate=0.0, slippage_rate=0.0))
    snap = _snapshot([0.02, -0.02, 0, 0], [0, 0, 0, 0])
    r1 = runner.step(_book([0.5, -0.5, 0, 0]), snap)
    r2 = runner.step(_book([0.5, -0.5, 0, 0]), snap)
    assert isinstance(r1, ShadowReport)
    assert np.isclose(r2.cum_pnl, r1.daily_pnl + r2.daily_pnl)
    assert np.isclose(runner.cum_pnl, r2.cum_pnl)
    hist = runner.history()
    assert len(hist) == 2
    for col in ["daily_pnl", "cum_pnl", "price_pnl", "funding_pnl", "fees",
                "slippage", "turnover", "gross", "net", "long_exposure", "short_exposure"]:
        assert col in hist.columns


def test_runner_save_roundtrip(tmp_path):
    runner = ShadowRunner()
    runner.step(_book([0.5, -0.5, 0, 0]), _snapshot([0.01, 0, 0, 0], [0, 0, 0, 0]))
    p = runner.save_history(tmp_path / "nav.parquet")
    assert p.exists()
    assert len(pd.read_parquet(p)) == 1


def test_shadow_layer_is_execution_and_alpha_agnostic():
    root = pathlib.Path(__file__).resolve().parents[1]
    shadow_dir = root / "crypto_signal_bot" / "platform" / "shadow"
    forbidden = ("bybit", "execution", "broker", "order", "signals.alx",
                 "signals.registry", "crypto_signal_bot.research", "alpha_library",
                 "experiments")
    offenders = {}
    for py in shadow_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods.append(node.module or "")
        bad = [m for m in mods if any(f in m for f in forbidden)]
        if bad:
            offenders[py.name] = bad
    assert not offenders, offenders
