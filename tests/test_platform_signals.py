"""Stage 2 — signal layer: plugin API + registry + ALX plugin (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.signals import registry
from crypto_signal_bot.platform.signals.base import (
    SignalKind,
    SignalProvider,
    TargetScores,
    TargetWeights,
)


def _snapshot(n=20, syms=("A", "B", "C", "D")):
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    closes = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.02, (n, len(syms))), axis=0),
        index=idx, columns=list(syms),
    )
    funding = pd.DataFrame(rng.normal(0, 1e-4, (n, len(syms))), index=idx, columns=list(syms))
    return MarketSnapshot(idx[-1], list(syms), closes, closes.pct_change(), funding)


def test_output_types_carry_kind():
    s = pd.Series([1.0, 2.0], index=["A", "B"])
    t = pd.Timestamp("2023-01-01", tz="UTC")
    assert TargetScores(s, t).kind is SignalKind.SCORES
    assert TargetWeights(s, t).kind is SignalKind.WEIGHTS


def test_output_rejects_bad_index():
    with pytest.raises(ValueError):
        TargetScores(pd.Series([1.0, 2.0], index=["A", "A"]), pd.Timestamp("2023-01-01", tz="UTC"))


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        SignalProvider()  # abstract generate()


def test_registry_register_list_load():
    @registry.register_signal("dummy_test")
    class Dummy(SignalProvider):
        def generate(self, snapshot, config=None, state=None):
            return TargetScores(pd.Series(0.0, index=snapshot.symbols), snapshot.asof)

    try:
        assert "dummy_test" in registry.list_signals()
        inst = registry.load_signal("dummy_test")
        assert isinstance(inst, SignalProvider) and inst.name == "dummy_test"
    finally:
        registry._REGISTRY.pop("dummy_test", None)


def test_duplicate_registration_raises():
    @registry.register_signal("dup_x")
    class A(SignalProvider):
        def generate(self, snapshot, config=None, state=None):
            return TargetScores(pd.Series(0.0, index=snapshot.symbols), snapshot.asof)

    try:
        with pytest.raises(ValueError):
            @registry.register_signal("dup_x")
            class B(SignalProvider):
                def generate(self, snapshot, config=None, state=None):
                    return TargetScores(pd.Series(0.0, index=snapshot.symbols), snapshot.asof)
    finally:
        registry._REGISTRY.pop("dup_x", None)


def test_alx_is_discovered_as_plugin():
    # Core knows ALX only via discovery — no import of the plugin here.
    assert "alx" in registry.list_signals()


def test_alx_plugin_returns_pure_scores():
    prov = registry.load_signal("alx")
    snap = _snapshot()
    out = prov.generate(snap, {"lookback": 7})
    assert isinstance(out, TargetScores)
    assert list(out.values.index) == snap.symbols
    expected = (-snap.funding.rolling(7).mean()).iloc[-1]
    assert np.allclose(out.values.to_numpy(), expected.to_numpy())
    # Signal layer does NOT dollar-neutralize/normalize: scores need not sum to 0.
    assert not np.isclose(out.values.sum(), 0.0)
