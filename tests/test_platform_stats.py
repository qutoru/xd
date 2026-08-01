"""TradeStats: realized-PnL projection and localized /stats rendering (offline)."""

from __future__ import annotations

from crypto_signal_bot.platform.accounting.ledger import AccountingLedger, LedgerEntry
from crypto_signal_bot.platform.notify.i18n import t
from crypto_signal_bot.platform.notify.stats import TradeStats, format_stats


def _entry(symbol, pnl, fee=0.0, exec_id="x", attribution="take_profit"):
    return LedgerEntry(
        exec_id=exec_id, timestamp="2026-07-01T00:00:00+00:00", symbol=symbol,
        side="sell", quantity=1.0, price=100.0, fee=fee, realized_pnl=pnl,
        attribution=attribution,
    )


def test_from_ledger_counts_wins_losses_and_extremes():
    ledger = AccountingLedger([
        _entry("BTCUSDT", 30.0, fee=0.5, exec_id="1"),
        _entry("ETHUSDT", -10.0, fee=0.3, exec_id="2", attribution="stop_loss"),
        _entry("SOLUSDT", 5.0, fee=0.2, exec_id="3"),
    ])
    stats = TradeStats.from_ledger(ledger)
    assert stats.n_closed == 3
    assert stats.wins == 2 and stats.losses == 1
    assert stats.win_rate == 2 / 3
    assert round(stats.realized_pnl, 2) == 25.0
    assert round(stats.fees, 2) == 1.0
    assert round(stats.net_pnl, 2) == 24.0
    assert stats.best_trade == ("BTCUSDT", 30.0)
    assert stats.worst_trade == ("ETHUSDT", -10.0)


def test_from_ledger_ignores_opening_fills_with_zero_pnl():
    ledger = AccountingLedger([
        _entry("BTCUSDT", 0.0, exec_id="open", attribution="entry"),
        _entry("BTCUSDT", 12.0, exec_id="close"),
    ])
    stats = TradeStats.from_ledger(ledger)
    assert stats.n_closed == 1 and stats.wins == 1


def test_partial_fills_counted_as_one_trade():
    # One stop-loss settled in 5 partial executions at the same instant must count
    # as ONE losing trade with the summed PnL — not five losses.
    ts = "2026-07-31T11:29:54.628000+00:00"
    entries = [
        LedgerEntry(exec_id=f"p{i}", timestamp=ts, symbol="SOLUSDT", side="sell",
                    quantity=1.0, price=100.0, fee=0.1, realized_pnl=-2.0,
                    attribution="stop_loss")
        for i in range(5)
    ]
    # A genuinely separate ADA take-profit an hour later stays its own trade.
    entries.append(LedgerEntry(
        exec_id="ada", timestamp="2026-07-31T12:30:00+00:00", symbol="ADAUSDT",
        side="sell", quantity=1.0, price=1.0, fee=0.0, realized_pnl=7.0,
        attribution="take_profit"))
    stats = TradeStats.from_ledger(AccountingLedger(entries))
    assert stats.n_closed == 2          # one SOL loss + one ADA win, not 6
    assert stats.wins == 1 and stats.losses == 1
    assert round(stats.realized_pnl, 2) == -3.0   # 5*-2.0 + 7.0 (exact sums preserved)


def test_separate_episodes_same_symbol_counted_apart():
    # Two SOL stop-losses far apart in time are two trades, not one.
    entries = [
        LedgerEntry(exec_id="a", timestamp="2026-07-31T10:00:00+00:00", symbol="SOLUSDT",
                    side="sell", quantity=1.0, price=100.0, fee=0.0, realized_pnl=-3.0,
                    attribution="stop_loss"),
        LedgerEntry(exec_id="b", timestamp="2026-07-31T15:00:00+00:00", symbol="SOLUSDT",
                    side="sell", quantity=1.0, price=100.0, fee=0.0, realized_pnl=-4.0,
                    attribution="stop_loss"),
    ]
    stats = TradeStats.from_ledger(AccountingLedger(entries))
    assert stats.n_closed == 2 and stats.losses == 2


def test_best_worst_are_per_trade_not_per_symbol():
    # ADA wins twice in separate trades (+3 then +4); per-SYMBOL best would be +7,
    # but the best single TRADE is +4. Worst is the lone BTC loss.
    entries = [
        LedgerEntry(exec_id="a1", timestamp="2026-07-31T10:00:00+00:00", symbol="ADAUSDT",
                    side="sell", quantity=1.0, price=1.0, fee=0.0, realized_pnl=3.0,
                    attribution="take_profit"),
        LedgerEntry(exec_id="a2", timestamp="2026-07-31T14:00:00+00:00", symbol="ADAUSDT",
                    side="sell", quantity=1.0, price=1.0, fee=0.0, realized_pnl=4.0,
                    attribution="take_profit"),
        LedgerEntry(exec_id="b1", timestamp="2026-07-31T11:00:00+00:00", symbol="BTCUSDT",
                    side="sell", quantity=1.0, price=100.0, fee=0.0, realized_pnl=-5.0,
                    attribution="stop_loss"),
    ]
    stats = TradeStats.from_ledger(AccountingLedger(entries))
    assert stats.n_closed == 3
    assert stats.best_trade == ("ADAUSDT", 4.0)     # single best trade, not the +7 symbol total
    assert stats.worst_trade == ("BTCUSDT", -5.0)


def test_format_empty_ledger_uses_empty_message():
    stats = TradeStats.from_ledger(AccountingLedger())
    assert format_stats(stats, "en") == t("en", "stats_empty")
    assert format_stats(stats, "ru") == t("ru", "stats_empty")


def test_format_non_empty_contains_header_and_figures():
    ledger = AccountingLedger([
        _entry("BTCUSDT", 30.0, exec_id="1"),
        _entry("ETHUSDT", -10.0, exec_id="2", attribution="stop_loss"),
    ])
    text = format_stats(TradeStats.from_ledger(ledger), "en")
    assert t("en", "stats_header") in text
    assert "50%" in text            # 1 win of 2 closed
    assert "BTCUSDT" in text and "ETHUSDT" in text
    assert "+20.00" in text         # net PnL (no fees)
