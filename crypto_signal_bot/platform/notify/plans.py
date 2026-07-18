"""Subscription tiers and their signal-delivery entitlements.

The subscription store keeps a free-text ``plan`` label per chat (granted by the
admin via ``/grant``); this module turns that label into an ordered
:class:`Tier` and the concrete :class:`Entitlements` the fan-out broadcaster
enforces. Keeping the ladder and the capability table in one place means adding a
plan or changing a limit is a one-line edit here, not a change scattered across
the delivery code.

Independent by construction: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Tier(IntEnum):
    """Subscription tiers, ordered cheapest -> richest (so ``<`` compares rank)."""

    START = 1
    PRO = 2
    VIP = 3


@dataclass(frozen=True)
class Entitlements:
    """What a tier is allowed to receive/use."""

    max_pairs: int | None      # cap on distinct symbols per broadcast (None = all)
    stats: bool                # /stats (trade statistics) available
    beta_strategies: bool      # receives signals from early-access ("beta") strategies
    custom_risk: bool          # /risk (personal risk settings) available
    priority: int              # lower = delivered earlier ("priority delivery")


# The single source of truth for tier capabilities. Mirrors the /subscribe copy:
#   START — up to 3 pairs, real-time signals with TP/SL (base).
#   PRO   — all pairs, priority delivery, trade statistics.
#   VIP   — everything in PRO + early-access strategies + personal risk settings.
ENTITLEMENTS: dict[Tier, Entitlements] = {
    Tier.START: Entitlements(
        max_pairs=3, stats=False, beta_strategies=False, custom_risk=False, priority=2
    ),
    Tier.PRO: Entitlements(
        max_pairs=None, stats=True, beta_strategies=False, custom_risk=False, priority=1
    ),
    Tier.VIP: Entitlements(
        max_pairs=None, stats=True, beta_strategies=True, custom_risk=True, priority=0
    ),
}

# Accepted plan labels (case/space-insensitive), EN + RU spellings.
_ALIASES: dict[str, Tier] = {
    "start": Tier.START, "старт": Tier.START,
    "pro": Tier.PRO, "про": Tier.PRO,
    "vip": Tier.VIP, "вип": Tier.VIP,
}


def tier_of(plan: str | None) -> Tier | None:
    """Map a stored ``plan`` label to a :class:`Tier` (case/space-insensitive).

    Returns ``None`` for an empty or unrecognised label, so callers treat an
    unknown plan explicitly (e.g. fall back to the base tier) rather than
    silently granting the richest access.
    """
    if not plan:
        return None
    return _ALIASES.get(plan.strip().lower())


def entitlements_of(plan: str | None) -> Entitlements | None:
    """Entitlements for a plan label, or ``None`` if the label is unrecognised."""
    tier = tier_of(plan)
    return ENTITLEMENTS[tier] if tier is not None else None


__all__ = ["Tier", "Entitlements", "ENTITLEMENTS", "tier_of", "entitlements_of"]
