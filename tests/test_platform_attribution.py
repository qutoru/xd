"""Stage 16.2 — fill attribution (TP / SL / Rebalance / Manual / Entry)."""

from __future__ import annotations

import pytest

from crypto_signal_bot.platform.accounting.attribution import Attribution, classify
from crypto_signal_bot.platform.execution.domain import Fill, Side


def _fill(link, *, closed):
    return Fill(symbol="BTCUSDT", side=Side.SELL, quantity=10.0, price=100.0,
                order_link_id=link, closed_quantity=closed)


def test_opening_fill_is_entry_regardless_of_tag():
    assert classify(_fill("platform-BTCUSDT-a-entry", closed=0.0)) is Attribution.ENTRY
    # even a tp-tagged order that did not close anything is not a closure
    assert classify(_fill("platform-BTCUSDT-a-tp", closed=0.0)) is Attribution.ENTRY


@pytest.mark.parametrize("suffix,expected", [
    ("tp", Attribution.TAKE_PROFIT),
    ("sl", Attribution.STOP_LOSS),
    ("close", Attribution.REBALANCE),
    ("reduce", Attribution.REBALANCE),
])
def test_closing_fill_attributed_from_client_id_suffix(suffix, expected):
    assert classify(_fill(f"platform-BTCUSDT-tok-{suffix}", closed=10.0)) is expected


def test_closing_fill_without_platform_tag_is_manual():
    assert classify(_fill(None, closed=10.0)) is Attribution.MANUAL       # liquidation
    assert classify(_fill("", closed=10.0)) is Attribution.MANUAL
    assert classify(_fill("someExternalOrder", closed=10.0)) is Attribution.MANUAL
    # a closing fill tagged as an entry is anomalous -> not a known close reason
    assert classify(_fill("platform-BTCUSDT-a-entry", closed=10.0)) is Attribution.MANUAL
