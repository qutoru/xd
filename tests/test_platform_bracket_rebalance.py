"""P0-A fix — delta-based bracket rebalance (no daily position accumulation).

Proves the execution layer rebalances to the *target* position instead of
re-opening the full target every cycle: repeated same-target runs do not double
the position, a smaller target reduces it, a zero/absent target closes it, and a
symbol that leaves the book is auto-closed — while a risk-control-blocked symbol
still wanted by the book is left untouched. Offline: PAPER BybitBroker + a fake
Bybit session (last price 100, no min-notional floor), so 10% of NAV 10_000 is a
clean 1000 notional (10 contracts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker
from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
from crypto_signal_bot.platform.execution.domain import Side
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.execution.intent import TradeIntent


class _FakeSession:
    def get_server_time(self):
        return {"retCode": 0, "result": {"timeSecond": "1"}}

    def get_instruments_info(self, **kw):
        return {"result": {"list": [{
            "symbol": kw.get("symbol", ""),
            "priceFilter": {"tickSize": "0.01"},
            "lotSizeFilter": {"qtyStep": "0.0001", "minOrderQty": "0.0001",
                              "minNotionalValue": "0"},
        }]}}

    def get_tickers(self, **kw):
        return {"result": {"list": [{"lastPrice": "100"}]}}


def _engine():
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER,
                      paper_equity=10_000.0)
    return ExecutionEngine(BybitBroker(cfg, session=_FakeSession()))


def _intent(symbol: str, pct: float, day: str) -> TradeIntent:
    signed = pct / 100.0 * 10_000.0  # % of NAV -> signed notional
    return TradeIntent(
        symbol=symbol,
        side=Side.BUY if signed >= 0 else Side.SELL,
        target_notional=abs(signed),
        entry=100.0, take_profit=120.0, stop_loss=90.0,
        strategy="platform", timestamp=pd.Timestamp(day, tz="UTC"),
    )


def _held(engine: ExecutionEngine, symbol: str) -> float:
    p = engine.broker.get_portfolio_state().positions.get(symbol)
    return 0.0 if p is None else p.quantity


# 1. Day1 10%, Day2 10% -> ONE position of 10%, no doubling.
def test_repeated_same_target_does_not_accumulate():
    eng = _engine()
    eng.rebalance_bracket_intents([_intent("BTCUSDT", 10, "2026-01-01")],
                                  book_symbols={"BTCUSDT"})
    assert np.isclose(_held(eng, "BTCUSDT"), 1000.0)
    eng.rebalance_bracket_intents([_intent("BTCUSDT", 10, "2026-01-02")],
                                  book_symbols={"BTCUSDT"})
    assert np.isclose(_held(eng, "BTCUSDT"), 1000.0)  # NOT 2000


# 2. Day1 10%, Day2 5% -> reduce-only decrease, final position 5%.
def test_smaller_target_reduces_position():
    eng = _engine()
    eng.rebalance_bracket_intents([_intent("BTCUSDT", 10, "2026-01-01")],
                                  book_symbols={"BTCUSDT"})
    eng.rebalance_bracket_intents([_intent("BTCUSDT", 5, "2026-01-02")],
                                  book_symbols={"BTCUSDT"})
    assert np.isclose(_held(eng, "BTCUSDT"), 500.0)


# 3. Day1 10%, Day2 0% -> full close.
def test_zero_target_closes_position():
    eng = _engine()
    eng.rebalance_bracket_intents([_intent("BTCUSDT", 10, "2026-01-01")],
                                  book_symbols={"BTCUSDT"})
    eng.rebalance_bracket_intents([], book_symbols=set())  # 0% == no longer wanted
    assert np.isclose(_held(eng, "BTCUSDT"), 0.0)


# 4. Day1 BTC present, Day2 BTC gone from book -> auto close; others untouched.
def test_symbol_leaving_book_is_auto_closed():
    eng = _engine()
    eng.rebalance_bracket_intents(
        [_intent("BTCUSDT", 10, "2026-01-01"), _intent("ETHUSDT", 10, "2026-01-01")],
        book_symbols={"BTCUSDT", "ETHUSDT"},
    )
    assert np.isclose(_held(eng, "BTCUSDT"), 1000.0)
    assert np.isclose(_held(eng, "ETHUSDT"), 1000.0)
    eng.rebalance_bracket_intents([_intent("ETHUSDT", 10, "2026-01-02")],
                                  book_symbols={"ETHUSDT"})
    assert np.isclose(_held(eng, "BTCUSDT"), 0.0)      # auto-closed
    assert np.isclose(_held(eng, "ETHUSDT"), 1000.0)   # untouched


# 4b. A held symbol still IN the book but refused by risk control is NOT closed
#     (risk control never closes existing positions).
def test_blocked_but_still_in_book_symbol_is_not_closed():
    eng = _engine()
    eng.rebalance_bracket_intents([_intent("BTCUSDT", 10, "2026-01-01")],
                                  book_symbols={"BTCUSDT"})
    eng.rebalance_bracket_intents([], book_symbols={"BTCUSDT"})  # wanted, but blocked
    assert np.isclose(_held(eng, "BTCUSDT"), 1000.0)   # left untouched, NOT closed


# 4c. Side flip long -> short rebalances to the new signed target (no doubling).
def test_side_flip_rebalances_to_new_side():
    eng = _engine()
    eng.rebalance_bracket_intents([_intent("BTCUSDT", 10, "2026-01-01")],
                                  book_symbols={"BTCUSDT"})   # +1000
    eng.rebalance_bracket_intents([_intent("BTCUSDT", -5, "2026-01-02")],
                                  book_symbols={"BTCUSDT"})   # target -500
    assert np.isclose(_held(eng, "BTCUSDT"), -500.0)


# 5. Two consecutive full pipeline runs (same book) do not accumulate.
def test_two_pipeline_runs_do_not_accumulate():
    from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
    from crypto_signal_bot.platform.pipeline import DailyPipeline
    from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig
    from crypto_signal_bot.platform.risk.manager import RiskConfig, RiskManager
    from crypto_signal_bot.platform.risk_control.control import (
        ProductionRiskControl, RiskControlConfig,
    )
    from crypto_signal_bot.platform.shadow.runner import ShadowRunner

    syms = [f"C{i}" for i in range(8)]
    idx = pd.date_range(end=pd.Timestamp("2023-05-01", tz="UTC"), periods=15, freq="D")
    rng = np.random.default_rng(7)
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (15, 8)), axis=0),
                          index=idx, columns=syms)
    funding = pd.DataFrame(np.linspace(-1e-4, 1e-4, 8) + rng.normal(0, 1e-5, (15, 8)),
                           index=idx, columns=syms)
    snap = MarketSnapshot(pd.Timestamp("2023-05-01", tz="UTC"), syms, closes,
                          closes.pct_change(), funding)

    class _Prov:
        def snapshot(self, asof, lookback_days):
            return snap

    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER)
    broker = BybitBroker(cfg, session=_FakeSession())
    pipe = DailyPipeline(
        _Prov(), signal_names=["alx"],
        execution_engine=ExecutionEngine(broker),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        risk_manager=RiskManager(RiskConfig(max_position_pct=1.0, max_gross=10.0,
                                            min_notional=1.0)),
        risk_control=ProductionRiskControl(RiskControlConfig()),
        nav=10_000.0, use_brackets=True, execute=True, lookback_days=10,
    )
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    pipe.run_once(asof)
    held1 = {s: p.quantity for s, p in broker.get_portfolio_state().positions.items()}
    assert held1  # opened a book

    pipe.run_once(asof + pd.Timedelta(days=1))  # identical snapshot -> identical target
    held2 = {s: p.quantity for s, p in broker.get_portfolio_state().positions.items()}

    assert set(held1) == set(held2)
    gross1 = sum(abs(v) for v in held1.values())
    gross2 = sum(abs(v) for v in held2.values())
    # No accumulation: a repeated identical target leaves gross ~unchanged (a
    # doubling bug would make gross2 ~= 2 * gross1).
    assert gross2 <= gross1 * 1.01 + 1.0, (gross1, gross2)


# C1: on a real venue that de-dups orderLinkId across cancellation, a same-day
# re-run must NOT be left without TP/SL. The rebalance cancels the prior brackets
# then re-places them; if the bracket id were the (deterministic) date-stamp, the
# venue would reject the re-placement as a duplicate -> naked position. Fresh
# bracket ids keep the re-placement accepted.
class _DedupLiveSession:
    """LIVE Bybit fake: retains every orderLinkId (even after cancel) and rejects
    a re-used one, and tracks positions so a same-day re-run reads back its fill."""

    def __init__(self):
        self.used_ids: set[str] = set()
        self.pos: dict[str, tuple[float, float]] = {}  # symbol -> (signed_qty, price)

    def get_server_time(self):
        return {"retCode": 0, "result": {"timeSecond": "1"}}

    def get_instruments_info(self, **kw):
        return {"result": {"list": [{
            "symbol": kw.get("symbol", ""),
            "priceFilter": {"tickSize": "0.01"},
            "lotSizeFilter": {"qtyStep": "0.0001", "minOrderQty": "0.0001",
                              "minNotionalValue": "0"},
        }]}}

    def get_tickers(self, **kw):
        return {"result": {"list": [{"lastPrice": "100"}]}}

    def get_wallet_balance(self, **kw):
        return {"result": {"list": [{"totalEquity": "10000"}]}}

    def get_positions(self, **kw):
        rows = [{"symbol": s, "side": "Buy" if q > 0 else "Sell",
                 "size": str(abs(q)), "avgPrice": str(p)}
                for s, (q, p) in self.pos.items() if q != 0]
        return {"retCode": 0, "result": {"list": rows}}

    def cancel_all_orders(self, **kw):
        # NB: does NOT release the retained orderLinkIds (pessimistic venue).
        return {"retCode": 0, "result": {"list": []}}

    def place_order(self, **params):
        link = params.get("orderLinkId", "")
        if link in self.used_ids:
            return {"retCode": 110072, "retMsg": "orderLinkId exist", "result": {}}
        self.used_ids.add(link)
        # A market entry (not reduce-only) fills and moves the tracked position.
        if params.get("orderType") == "Market" and not params.get("reduceOnly"):
            qty = float(params["qty"]) * (1 if params["side"] == "Buy" else -1)
            cur, _ = self.pos.get(params["symbol"], (0.0, 100.0))
            self.pos[params["symbol"]] = (cur + qty, 100.0)
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": link or "oid"}}


def test_same_day_rerun_does_not_leave_position_without_tp_sl_on_dedup_venue():
    from crypto_signal_bot.platform.execution.bybit_config import BybitConfig
    from crypto_signal_bot.platform.execution.domain import OrderStatus

    cfg = BybitConfig(api_key="k", api_secret="s", testnet=True, mode=TradingMode.LIVE)
    session = _DedupLiveSession()
    eng = ExecutionEngine(BybitBroker(cfg, session=session))

    r1 = eng.rebalance_bracket_intents([_intent("BTCUSDT", 10, "2026-01-02")],
                                       book_symbols={"BTCUSDT"})
    # entry filled + both brackets accepted on the first run
    assert all(o.status is not OrderStatus.REJECTED for o in r1.orders)
    assert np.isclose(session.pos["BTCUSDT"][0], 10.0)  # 1000 / 100

    # Same day, same target -> delta ~0 -> only the brackets are re-placed after the
    # cancel. On the old date-stamped id these would be duplicate-rejected (naked).
    r2 = eng.rebalance_bracket_intents([_intent("BTCUSDT", 10, "2026-01-02")],
                                       book_symbols={"BTCUSDT"})
    brackets = [o for o in r2.orders if o.request.reduce_only]
    assert brackets, "re-run must re-place the TP/SL bracket"
    assert all(o.status is not OrderStatus.REJECTED for o in brackets), \
        "brackets rejected as duplicates -> position left with no stop-loss"
    assert np.isclose(session.pos["BTCUSDT"][0], 10.0)  # no accumulation


# C2: stale brackets are cancelled before new ones are (re)placed; untouched
# (risk-control-blocked but still held) symbols keep theirs.
def test_rebalance_cancels_stale_brackets_before_placing_new():
    from crypto_signal_bot.platform.execution.broker import Broker
    from crypto_signal_bot.platform.execution.domain import (
        OrderResult, OrderStatus, PortfolioState, Position,
    )

    class _RecBroker(Broker):
        def __init__(self, held):
            self.log = []
            self._held = dict(held)
        def get_portfolio_state(self):
            return PortfolioState(value=10_000.0, positions={
                s: Position(symbol=s, quantity=q) for s, q in self._held.items()})
        def cancel_open_orders(self, symbol):
            self.log.append(("cancel", symbol))
        def submit(self, req):
            self.log.append(("submit", req.symbol, req.client_id.rsplit("-", 1)[1]))
            return OrderResult(request=req, status=OrderStatus.FILLED,
                               filled_quantity=req.quantity)

    # BTC: re-affirmed same target (delta~0 -> only TP/SL re-placed).
    # ETH: still wanted (in book) but refused by risk control (not in intents).
    # SOL: gone from the book entirely (departed).
    broker = _RecBroker({"BTCUSDT": 1000.0, "ETHUSDT": 1000.0, "SOLUSDT": 800.0})
    ExecutionEngine(broker).rebalance_bracket_intents(
        [_intent("BTCUSDT", 10, "2026-01-02")],
        book_symbols={"BTCUSDT", "ETHUSDT"},
    )
    log = broker.log

    def _first_submit(sym):
        idx = [i for i, e in enumerate(log) if e[0] == "submit" and e[1] == sym]
        return min(idx) if idx else None

    # BTC touched: cancelled, and the cancel precedes BTC's new orders.
    assert ("cancel", "BTCUSDT") in log
    assert log.index(("cancel", "BTCUSDT")) < _first_submit("BTCUSDT")
    # SOL departed: closed AND its stale brackets cancelled (cancel before close).
    assert ("cancel", "SOLUSDT") in log
    assert log.index(("cancel", "SOLUSDT")) < _first_submit("SOLUSDT")
    # ETH blocked-but-held: untouched -> NOT cancelled, NOT traded (keeps brackets).
    assert ("cancel", "ETHUSDT") not in log
    assert _first_submit("ETHUSDT") is None
