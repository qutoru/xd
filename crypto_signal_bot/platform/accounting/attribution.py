"""Attribute a fill to the reason it changed the position.

Pure function over an execution-domain :class:`Fill`; no broker, no venue. The
reason is read from the originating order's ``order_link_id`` (the platform's own
``client_id``), whose trailing segment encodes the order's role:

* ``-tp``     -> a reduce-only take-profit bracket   -> TAKE_PROFIT
* ``-sl``     -> a reduce-only stop-loss bracket      -> STOP_LOSS
* ``-close``  -> a delta-rebalance flatten             -> REBALANCE
* ``-reduce`` -> a delta-rebalance size-down           -> REBALANCE

A fill that only opens or increases a position (``closed_quantity ~ 0``) is an
ENTRY. A *closing* fill that carries no recognised platform tag — an empty or
foreign ``order_link_id``, or a venue-initiated close such as a liquidation — is
MANUAL. This fulfils the fills-aware attribution the Reconciler deferred.
"""

from __future__ import annotations

from enum import Enum

from crypto_signal_bot.platform.execution.domain import Fill

_TOL = 1e-9


class Attribution(str, Enum):
    """Why a fill changed the position."""

    ENTRY = "entry"              # opened or increased a position
    TAKE_PROFIT = "take_profit"  # reduce-only TP bracket fired
    STOP_LOSS = "stop_loss"      # reduce-only SL bracket fired
    REBALANCE = "rebalance"      # platform delta-rebalance close/reduce
    MANUAL = "manual"            # closed with no recognised platform tag


_CLOSE_SUFFIX: dict[str, Attribution] = {
    "tp": Attribution.TAKE_PROFIT,
    "sl": Attribution.STOP_LOSS,
    "close": Attribution.REBALANCE,
    "reduce": Attribution.REBALANCE,
}


def classify(fill: Fill) -> Attribution:
    """Return the :class:`Attribution` for one fill (see module docstring)."""
    if fill.closed_quantity <= _TOL:
        return Attribution.ENTRY
    link = fill.order_link_id or ""
    suffix = link.rsplit("-", 1)[-1] if link else ""
    return _CLOSE_SUFFIX.get(suffix, Attribution.MANUAL)
