"""Stage 18 — realized-PnL-driven daily-loss for RiskControl.

Known-answer tests drive the pure control with a plain realized-PnL number; the
integration/e2e tests drive the whole pipeline (Ledger -> today's realized PnL ->
RiskControl halt), and confirm the feature flag and accounting-off fall back to the
unchanged NAV-based logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_signal_bot.platform.accounting.ledger import AccountingLedger, AccountingStore
from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker
from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
from crypto_signal_bot.platform.execution.domain import Fill, Side
from crypto_signal_bot.platform.execution.engine import ExecutionEngine
from crypto_signal_bot.platform.pipeline import DailyPipeline
from crypto_signal_bot.platform.portfolio.builder import PortfolioConfig
from crypto_signal_bot.platform.risk.manager import RiskConfig, RiskManager
from crypto_signal_bot.platform.risk_control.control import (
    BlockReason, ProductionRiskControl, RiskControlConfig,
)
from crypto_signal_bot.platform.risk_control.state import RiskState, RiskStateStore
from crypto_signal_bot.platform.shadow.runner import ShadowRunner


# =============================================================================
# Known-answer — the pure control, fed only a realized-PnL scalar
# =============================================================================
def _control(**cfg):
    base = dict(daily_loss_limit=0.02, use_realized_daily_loss=True)
    base.update(cfg)
    return ProductionRiskControl(RiskControlConfig(**base))

_PROPOSED = pd.Series({"BTCUSDT": 1000.0})
_NAV = 10_000.0            # limit 0.02 * 10_000 = 200 loss to halt


def _eval(control, *, realized, nav=_NAV, day_start_nav=_NAV):
    return control.evaluate(_PROPOSED, None, nav,
                            day_start_nav=day_start_nav, realized_daily_pnl=realized)


def test_realized_profit_does_not_halt():
    d = _eval(_control(), realized=+500.0)
    assert not d.halted and d.is_allowed("BTCUSDT")


def test_realized_loss_below_limit_does_not_halt():
    d = _eval(_control(), realized=-199.0)      # loss 199 < 200
    assert not d.halted


def test_realized_loss_at_limit_halts():
    d = _eval(_control(), realized=-200.0)      # loss 200 == limit
    assert d.halted and d.halt_reason is BlockReason.DAILY_LOSS_LIMIT


def test_realized_loss_over_limit_halts():
    d = _eval(_control(), realized=-250.0)
    assert d.halted and d.halt_reason is BlockReason.DAILY_LOSS_LIMIT


def test_accounting_off_falls_back_to_nav_logic():
    # flag on but no realized value supplied -> NAV delta governs the halt
    ctrl = _control()
    assert not _eval(ctrl, realized=None, nav=_NAV).halted            # no NAV loss
    d = _eval(ctrl, realized=None, nav=9_700.0)                       # NAV loss 300
    assert d.halted and d.halt_reason is BlockReason.DAILY_LOSS_LIMIT


def test_flag_off_ignores_realized_and_uses_nav():
    ctrl = _control(use_realized_daily_loss=False)
    # a huge realized loss is ignored; NAV is flat -> no halt
    assert not _eval(ctrl, realized=-5_000.0, nav=_NAV).halted
    # and the NAV delta still governs
    assert _eval(ctrl, realized=+5_000.0, nav=9_700.0).halted


# =============================================================================
# Pipeline plumbing shared by the integration / e2e tests
# =============================================================================
_SYMS = [f"C{i}" for i in range(8)]
_IDX = pd.date_range(end=pd.Timestamp("2023-05-01", tz="UTC"), periods=15, freq="D")


def _snapshot(asof, *, flip=False):
    rng = np.random.default_rng(7)
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (15, 8)), axis=0),
                          index=_IDX, columns=_SYMS)
    grad = np.linspace(-1e-4, 1e-4, 8)
    if flip:
        grad = grad[::-1]
    funding = pd.DataFrame(grad + rng.normal(0, 1e-6, (15, 8)), index=_IDX, columns=_SYMS)
    return MarketSnapshot(asof, _SYMS, closes, closes.pct_change(), funding)


class _CountProv:
    """Returns snapshot A on the first call, then flipped-funding B (same asof)."""

    def __init__(self):
        self.calls = 0

    def snapshot(self, asof, lookback_days):
        flip = self.calls > 0
        self.calls += 1
        return _snapshot(asof, flip=flip)


class _MultiPriceSession:
    """PAPER session with a per-symbol last price (default 100)."""

    def __init__(self):
        self.prices: dict[str, float] = {}

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
        px = self.prices.get(kw.get("symbol", ""), 100.0)
        return {"result": {"list": [{"lastPrice": str(px)}]}}


def _pipeline(session, *, accounting_store=None, risk_state=None,
              use_realized=False, daily_loss_limit=0.02):
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER)
    broker = BybitBroker(cfg, session=session)
    pipe = DailyPipeline(
        _CountProv(), signal_names=["alx"],
        execution_engine=ExecutionEngine(broker),
        shadow_runner=ShadowRunner(),
        portfolio_config=PortfolioConfig(k_pct=0.30, gross_target=1.0),
        risk_manager=RiskManager(RiskConfig(max_position_pct=1.0, max_gross=10.0,
                                            min_notional=1.0)),
        risk_control=ProductionRiskControl(RiskControlConfig(
            daily_loss_limit=daily_loss_limit, use_realized_daily_loss=use_realized)),
        risk_state=risk_state,
        nav=10_000.0, use_brackets=True, execute=True, lookback_days=10,
        accounting_store=accounting_store,
    )
    return pipe, broker


# =============================================================================
# Integration — a persisted day-start anchor above the current ledger => a
# realized loss today => halt (exercises pipeline -> ledger -> control)
# =============================================================================
def test_integration_realized_daily_loss_halts_via_pipeline(tmp_path):
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    # day started with +1000 cumulative realized; ledger now at 0 => today -1000.
    rss = RiskStateStore(tmp_path / "risk.json")
    rss.save(RiskState(day=asof.date().isoformat(), day_start_nav=10_000.0,
                       day_start_realized_pnl=1000.0))
    store = AccountingStore(tmp_path / "ledger.json")   # empty ledger -> cum 0

    pipe, _ = _pipeline(_MultiPriceSession(), accounting_store=store,
                        risk_state=rss, use_realized=True)
    result = pipe.run_once(asof)
    assert result.risk_control.halted
    assert result.risk_control.halt_reason is BlockReason.DAILY_LOSS_LIMIT


def test_integration_flag_off_does_not_halt_on_realized_loss(tmp_path):
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    rss = RiskStateStore(tmp_path / "risk.json")
    rss.save(RiskState(day=asof.date().isoformat(), day_start_nav=10_000.0,
                       day_start_realized_pnl=1000.0))
    store = AccountingStore(tmp_path / "ledger.json")

    pipe, _ = _pipeline(_MultiPriceSession(), accounting_store=store,
                        risk_state=rss, use_realized=False)   # flag OFF
    result = pipe.run_once(asof)
    assert not result.risk_control.halted                     # NAV flat -> no halt


def test_integration_accounting_off_uses_nav_logic(tmp_path):
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    rss = RiskStateStore(tmp_path / "risk.json")
    rss.save(RiskState(day=asof.date().isoformat(), day_start_nav=10_000.0,
                       day_start_realized_pnl=1000.0))
    # no accounting_store -> realized_daily_pnl is None -> NAV fallback even with flag
    pipe, _ = _pipeline(_MultiPriceSession(), accounting_store=None,
                        risk_state=rss, use_realized=True)
    result = pipe.run_once(asof)
    assert not result.risk_control.halted


# =============================================================================
# Full e2e — real PAPER trading realizes a loss that halts a later same-day run
# =============================================================================
def test_e2e_realized_loss_from_trading_halts_next_cycle(tmp_path):
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    session = _MultiPriceSession()
    store = AccountingStore(tmp_path / "ledger.json")
    pipe, broker = _pipeline(session, accounting_store=store, use_realized=True,
                             daily_loss_limit=0.02)

    # cycle 1: open the book at 100 (day-start realized anchor = 0)
    r1 = pipe.run_once(asof)
    assert not r1.risk_control.halted
    held = broker.get_portfolio_state().positions
    assert held
    # price every open position to a guaranteed loss on close: longs down, shorts up
    for sym, pos in held.items():
        session.prices[sym] = 80.0 if pos.quantity > 0 else 120.0

    # cycle 2 (same day): funding flips -> positions close at a loss (booked after)
    r2 = pipe.run_once(asof)
    assert not r2.risk_control.halted                 # loss not realized yet at gate
    assert pipe._ledger.realized_pnl < 0              # ...but now it is booked

    # cycle 3 (same day): gate sees today's realized loss -> halt
    r3 = pipe.run_once(asof)
    assert r3.risk_control.halted
    assert r3.risk_control.halt_reason is BlockReason.DAILY_LOSS_LIMIT


# =============================================================================
# Backward compatibility — a same-day state persisted before Stage 18 has no
# realized baseline (day_start_realized_pnl is None). The anchor must backfill it
# from the current cumulative, NOT treat the whole lifetime realized PnL as
# today's loss (which the old ``... or 0.0`` fallback did).
# =============================================================================
def test_pre_stage18_state_backfills_realized_anchor(tmp_path):
    asof = pd.Timestamp("2023-05-01", tz="UTC")
    rss = RiskStateStore(tmp_path / "risk.json")
    # Pre-Stage-18 record: NAV baseline present, realized baseline absent.
    rss.save(RiskState(day=asof.date().isoformat(), day_start_nav=10_000.0))
    pipe, _ = _pipeline(_MultiPriceSession(), risk_state=rss, use_realized=True)

    state = rss.load()
    assert state.day_start_realized_pnl is None            # pre-Stage-18 shape

    # Lifetime cumulative realized is a large negative; it must be adopted as the
    # day-start baseline (today's delta = 0), not read as a -5000 loss booked today.
    day_start_nav, day_start_realized = pipe._risk_day_anchor(
        asof, 10_000.0, -5000.0, state)
    assert day_start_nav == 10_000.0
    assert day_start_realized == -5000.0

    # ...and the backfill is persisted so later same-day runs stay consistent.
    assert rss.load().day_start_realized_pnl == -5000.0
