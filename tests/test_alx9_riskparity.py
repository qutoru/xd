"""Known-answer unit tests for ALX9 risk-parity blend mechanics (NOT the alpha).

Guard the weighting plumbing: gross-preserving normalization, 50/50 warm-up,
inverse-vol direction (lower vol -> higher weight), and no-look-ahead (weights at
t use only data through t-1). No network.
"""

from __future__ import annotations

import numpy as np

from alpha_library.alx9_riskparity_blend import riskparity as rp


def _streams(T=400, seed=1):
    rng = np.random.default_rng(seed)
    net_a = rng.normal(0, 0.002, T)     # low vol sleeve
    net_b = rng.normal(0, 0.010, T)     # high vol sleeve
    return net_a, net_b


def test_weights_gross_preserving():
    a, b = _streams()
    af, bf = rp.rp_weights(a, b)
    assert np.allclose(af + bf, 1.0, atol=1e-12)


def test_warmup_is_fifty_fifty():
    a, b = _streams()
    af, bf = rp.rp_weights(a, b, window=60)
    # first `window` points have no lagged 60d vol -> default 0.5/0.5
    assert np.allclose(af[:60], 0.5, atol=1e-12)
    assert np.allclose(bf[:60], 0.5, atol=1e-12)


def test_inverse_vol_direction():
    """The lower-vol sleeve receives the higher weight after warm-up."""
    a, b = _streams()
    af, bf = rp.rp_weights(a, b)
    assert np.mean(af[80:]) > np.mean(bf[80:])   # a is ~5x lower vol -> heavier


def test_no_look_ahead():
    """Weights at t depend only on data through t-1: perturbing the LAST net
    value must not change any earlier weight."""
    a, b = _streams()
    af0, _ = rp.rp_weights(a, b)
    a2 = a.copy(); a2[-1] += 100.0
    af1, _ = rp.rp_weights(a2, b)
    assert np.allclose(af0[:-1], af1[:-1], atol=1e-12)


def test_book_from_applied_matches_formula():
    rng = np.random.default_rng(3)
    T, N = 50, 6
    applied = rng.normal(0, 0.1, (T, N))
    ret = rng.normal(0, 0.02, (T, N))
    fund = rng.normal(0, 1e-4, (T, N))
    got = rp.book_from_applied(applied, ret, fund)
    price = np.nansum(applied * ret, axis=1)
    funding = -np.nansum(applied * np.nan_to_num(fund), axis=1)
    turn = np.abs(np.diff(applied, axis=0, prepend=0.0)).sum(axis=1)
    want = rp.PHI * (price + funding) - turn * rp.COST
    assert np.allclose(got, want, atol=1e-15)
