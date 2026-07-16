"""Stage 12 — production trade runner: preflight, config gating, symbol pinning.

All offline: exchange modes use an injected fake session, and the abort paths
return before any network snapshot/order.
"""

from __future__ import annotations

import pandas as pd

from crypto_signal_bot.app.trade_runner import parse_symbols, preflight, run_trade
from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode


class _Broker:
    def __init__(self, connected: bool):
        self._connected = connected
    def check_connection(self) -> bool:
        return self._connected


class _DeadSession:
    """A session whose connectivity ping fails (check_connection -> False)."""
    def get_server_time(self):
        raise RuntimeError("no route to host")


def _env(monkeypatch, **kw):
    for k in ("BYBIT_API_KEY", "BYBIT_API_SECRET", "BYBIT_TESTNET",
              "BYBIT_TRADING_MODE", "BYBIT_SYMBOLS",
              "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    for k, v in kw.items():
        monkeypatch.setenv(k, v)


# --- symbol parsing ---------------------------------------------------------
def test_parse_symbols_splits_trims_uppercases():
    assert parse_symbols(" btcusdt, ethusdt ,, ") == ["BTCUSDT", "ETHUSDT"]
    assert parse_symbols("") == [] and parse_symbols(None) == []


# --- preflight --------------------------------------------------------------
def test_preflight_live_requires_key_and_secret():
    assert "BYBIT_API_KEY" in preflight(
        BybitConfig(mode=TradingMode.LIVE, api_secret="s"), _Broker(True))
    assert "BYBIT_API_SECRET" in preflight(
        BybitConfig(mode=TradingMode.LIVE, api_key="k"), _Broker(True))


def test_preflight_reports_connection_failure():
    cfg = BybitConfig(mode=TradingMode.PAPER, api_key="k", api_secret="s")
    assert "cannot reach Bybit" in preflight(cfg, _Broker(False))


def test_preflight_ok_returns_none():
    cfg = BybitConfig(mode=TradingMode.LIVE, api_key="k", api_secret="s")
    assert preflight(cfg, _Broker(True)) is None


# --- run_trade abort paths (never reach a network cycle) --------------------
def test_run_trade_aborts_on_live_without_credentials(monkeypatch):
    _env(monkeypatch, BYBIT_TRADING_MODE="live")  # no keys
    assert run_trade(signals=["alx"], asof=pd.Timestamp("2024-01-02", tz="UTC")) == 2


def test_run_trade_aborts_on_preflight_connection_failure(monkeypatch):
    _env(monkeypatch, BYBIT_TRADING_MODE="paper", BYBIT_API_KEY="k",
         BYBIT_API_SECRET="s", BYBIT_SYMBOLS="BTCUSDT")
    rc = run_trade(signals=["alx"], asof=pd.Timestamp("2024-01-02", tz="UTC"),
                   session=_DeadSession())
    assert rc == 1  # aborted at pre-flight, before the trading cycle


# --- P1: production entrypoint configures RiskControl from env ---------------
def _clear_risk_env(monkeypatch):
    for k in ("RISK_KILL_SWITCH", "RISK_DAILY_LOSS_LIMIT", "RISK_MAX_OPEN_POSITIONS",
              "RISK_MAX_EXPOSURE", "RISK_MAX_POSITION_SIZE", "RISK_STATE_PATH"):
        monkeypatch.delenv(k, raising=False)


def test_build_trade_pipeline_paper_gets_env_risk_config_and_persistence(monkeypatch):
    from crypto_signal_bot.app.trade_runner import build_trade_pipeline

    _env(monkeypatch)
    _clear_risk_env(monkeypatch)
    monkeypatch.setenv("RISK_KILL_SWITCH", "true")
    monkeypatch.setenv("RISK_MAX_EXPOSURE", "0.5")
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.PAPER)
    pipe = build_trade_pipeline(cfg, signals=["alx"], symbols=["BTCUSDT"], notifier=None)
    assert pipe.risk_control.config.kill_switch is True          # not the default no-op
    assert pipe.risk_control.config.max_exposure == 0.5
    assert pipe.risk_state is not None                            # persistence wired


def test_build_trade_pipeline_live_gets_env_risk_config(monkeypatch):
    from crypto_signal_bot.app.trade_runner import build_trade_pipeline

    _env(monkeypatch)
    _clear_risk_env(monkeypatch)
    monkeypatch.setenv("RISK_MAX_POSITION_SIZE", "0.1")
    cfg = BybitConfig(api_key="k", api_secret="s", mode=TradingMode.LIVE)
    pipe = build_trade_pipeline(cfg, signals=["alx"], symbols=["BTCUSDT"], notifier=None)
    assert pipe.risk_control.config.max_position_size == 0.1
