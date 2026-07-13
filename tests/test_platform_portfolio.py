"""Stage 3 — Portfolio Layer: components + end-to-end + layer purity."""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from crypto_signal_bot.platform.portfolio import (
    allocator,
    combiner,
    constraints,
    normalization,
    turnover,
)
from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig, build_target_book
from crypto_signal_bot.platform.signals.base import TargetScores

ASOF = pd.Timestamp("2023-02-01", tz="UTC")
SYMS = ["A", "B", "C", "D", "E", "F"]


def _scores(vals):
    return TargetScores(pd.Series(vals, index=SYMS, dtype="float64"), ASOF)


def test_zscore_and_degenerate():
    z = normalization.zscore(pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"]))
    assert np.isclose(z.mean(), 0.0) and np.isclose(z.std(ddof=0), 1.0)
    flat = normalization.zscore(pd.Series([5.0, 5.0], index=["A", "B"]))
    assert (flat == 0.0).all()


def test_combiner_budget_weighting_and_union():
    a = pd.Series([1.0, -1.0], index=["A", "B"])
    b = pd.Series([2.0, 2.0], index=["B", "C"])
    combined, contrib = combiner.combine({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
    # union of symbols, missing treated as 0, budgets normalized to sum|b|=1 (0.5/0.5)
    assert set(combined.index) == {"A", "B", "C"}
    assert np.isclose(combined["A"], 0.5 * 1.0)
    assert np.isclose(combined["B"], 0.5 * -1.0 + 0.5 * 2.0)
    assert set(contrib) == {"a", "b"}


def test_allocator_dollar_neutral():
    w = allocator.dollar_neutral_topk(pd.Series(range(6), index=SYMS, dtype="float64"), 0.34)
    assert np.isclose(w.sum(), 0.0)  # net 0
    assert np.isclose(w.abs().sum(), 1.0)  # gross 1
    assert w["F"] > 0 and w["A"] < 0  # long highest, short lowest


def test_constraints_cap_gross_net():
    w = pd.Series([0.4, 0.1, -0.1, -0.4], index=["A", "B", "C", "D"])
    capped = constraints.max_position(w, 0.25)
    assert capped.abs().max() <= 0.25 + 1e-9
    assert np.isclose(capped.sum(), 0.0)  # stays neutral
    scaled = constraints.scale_gross(capped, 2.0)
    assert np.isclose(scaled.abs().sum(), 2.0)


def test_turnover_cap():
    prev = pd.Series([0.5, -0.5], index=["A", "B"])
    target = pd.Series([-0.5, 0.5], index=["A", "B"])  # full flip: turnover 2.0
    limited = turnover.limit_turnover(target, prev, 0.5)
    assert np.isclose((limited - prev).abs().sum(), 0.5)
    # under the cap: unchanged
    assert turnover.limit_turnover(target, prev, 10.0).equals(target)


def test_build_end_to_end_dollar_neutral_with_limits():
    s1 = _scores([3, 2, 1, -1, -2, -3])
    s2 = _scores([1, 1, 1, -1, -1, -1])
    cfg = PortfolioConfig(alloc="topk", k_pct=0.34, gross_target=2.0,
                          max_weight=0.6, max_turnover=None)
    book = build_target_book({"alx": s1, "mom": s2}, budgets={"alx": 0.7, "mom": 0.3}, config=cfg)
    assert isinstance(book, TargetBook)
    assert np.isclose(book.net, 0.0, atol=1e-9)  # dollar-neutral
    assert np.isclose(book.gross, 2.0, atol=1e-9)  # gross target
    assert book.weights.abs().max() <= 0.6 * 2.0 + 1e-9  # per-name cap (fraction of gross)
    assert set(book.contributions) == {"alx", "mom"}
    assert book.asof == ASOF


def test_build_respects_turnover_vs_prev():
    prev = build_target_book({"a": _scores([3, 2, 1, -1, -2, -3])},
                             config=PortfolioConfig(k_pct=0.34, gross_target=1.0))
    nxt = build_target_book({"a": _scores([-3, -2, -1, 1, 2, 3])},  # full reversal
                            prev_book=prev,
                            config=PortfolioConfig(k_pct=0.34, gross_target=1.0, max_turnover=0.5))
    assert (nxt.weights - prev.weights).abs().sum() <= 0.5 + 1e-9


def test_empty_signals_raises():
    with pytest.raises(ValueError):
        build_target_book({})


def test_portfolio_layer_is_alpha_and_execution_agnostic():
    """Portfolio must not import data sources, the registry, any alpha, or execution."""
    pkg = pathlib.Path(build_target_book.__module__.replace(".", "/")).parent
    root = pathlib.Path(__file__).resolve().parents[1]
    portfolio_dir = root / "crypto_signal_bot" / "platform" / "portfolio"
    forbidden = ("platform.data", "signals.registry", "signals.alx", "bybit",
                 "execution", "broker", "crypto_signal_bot.research", "alpha_library")
    offenders = {}
    for py in portfolio_dir.rglob("*.py"):
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
