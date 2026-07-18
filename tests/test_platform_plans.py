"""Subscription tier model: label -> Tier mapping and per-tier entitlements."""

from __future__ import annotations

from crypto_signal_bot.platform.notify.plans import (
    ENTITLEMENTS,
    Tier,
    entitlements_of,
    tier_of,
)


def test_tier_of_maps_labels_case_and_space_insensitively():
    assert tier_of("START") == Tier.START
    assert tier_of(" pro ") == Tier.PRO
    assert tier_of("Vip") == Tier.VIP
    assert tier_of("вип") == Tier.VIP  # RU spelling accepted


def test_tier_of_unknown_or_empty_is_none():
    assert tier_of(None) is None
    assert tier_of("") is None
    assert tier_of("platinum") is None


def test_tiers_are_ordered_cheapest_to_richest():
    assert Tier.START < Tier.PRO < Tier.VIP


def test_entitlements_gate_features_by_tier():
    assert ENTITLEMENTS[Tier.START].max_pairs == 3
    assert ENTITLEMENTS[Tier.PRO].max_pairs is None
    assert ENTITLEMENTS[Tier.VIP].max_pairs is None
    # early-access strategies: VIP only
    assert ENTITLEMENTS[Tier.VIP].beta_strategies is True
    assert ENTITLEMENTS[Tier.PRO].beta_strategies is False
    assert ENTITLEMENTS[Tier.START].beta_strategies is False
    # trade statistics: PRO and up
    assert ENTITLEMENTS[Tier.PRO].stats is True
    assert ENTITLEMENTS[Tier.START].stats is False
    # personal risk: VIP only
    assert ENTITLEMENTS[Tier.VIP].custom_risk is True
    assert ENTITLEMENTS[Tier.PRO].custom_risk is False
    # priority delivery: richer tier served first (lower value)
    assert ENTITLEMENTS[Tier.VIP].priority < ENTITLEMENTS[Tier.PRO].priority
    assert ENTITLEMENTS[Tier.PRO].priority < ENTITLEMENTS[Tier.START].priority


def test_entitlements_of_resolves_or_returns_none():
    assert entitlements_of("vip") is ENTITLEMENTS[Tier.VIP]
    assert entitlements_of("unknown") is None
