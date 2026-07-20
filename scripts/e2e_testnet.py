"""Bybit TESTNET end-to-end lifecycle harness (Gate 2) — with scoring.

Drives the FULL execution cycle against Bybit testnet using the production
BybitBroker + Reconciler (no mocks), scoring each step. This is the stage where
real bugs surface: reduce_only, qty/tick rounding, minNotional, leverage,
hedge/one-way (positionIdx), retCode!=0, and state reconciliation after restart.

SAFETY: refuses to run unless BYBIT_TESTNET is truthy and mode is LIVE. It trades
only tiny notionals and always tries to flatten + cancel in a finally block.

Run (PowerShell):
    $env:BYBIT_API_KEY="<testnet key>"; $env:BYBIT_API_SECRET="<testnet secret>"
    $env:BYBIT_TESTNET="1"; $env:BYBIT_TRADING_MODE="live"; $env:BYBIT_LEVERAGE="2"
    py scripts/e2e_testnet.py
Optional: $env:E2E_SYMBOL="BTCUSDT"  $env:E2E_NOTIONAL="20"
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load credentials/flags from the repo-root .env so this harness can be run with
# a single command (it doesn't import crypto_signal_bot.config, which is the
# usual load_dotenv entry point).
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pandas as pd
from loguru import logger

from crypto_signal_bot.platform.execution.bybit_broker import BybitBroker, build_broker
from crypto_signal_bot.platform.execution.bybit_config import BybitConfig, TradingMode
from crypto_signal_bot.platform.execution.domain import OrderRequest, OrderStatus, OrderType, Side
from crypto_signal_bot.platform.execution.reconciler import Reconciler

SYMBOL = os.getenv("E2E_SYMBOL", "BTCUSDT")
NOTIONAL = float(os.getenv("E2E_NOTIONAL", "20"))
_RESULTS: list[tuple[str, str, str]] = []  # (step, status, detail)


def _record(step: str, ok: bool | None, detail: str = "") -> bool | None:
    status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    _RESULTS.append((step, status, detail))
    logger.info("[{}] {} — {}", status, step, detail)
    return ok


def _session(broker: BybitBroker):
    return broker._get_session()  # harness-only access to the raw v5 session


def run(broker: BybitBroker) -> None:
    cfg = broker.config

    # 1. connect
    _record("1_connect", broker.check_connection(), f"testnet={cfg.testnet}")

    # 2. download instruments
    try:
        info = broker.get_instrument_info(SYMBOL)
        ok = all(v > 0 for v in (info.tick_size, info.qty_step, info.min_qty)) and info.min_notional >= 0
        _record("2_instruments", ok, f"tick={info.tick_size} step={info.qty_step} "
                f"minQty={info.min_qty} minNotional={info.min_notional}")
    except Exception as exc:
        _record("2_instruments", False, str(exc)); return

    # 3. rounding + minNotional (validated against the REAL filters)
    q = broker._floor_to_step(NOTIONAL / broker._last_price(SYMBOL), info.qty_step)
    tick_ok = abs((broker._round_to_tick(info.tick_size * 3.3, info.tick_size) / info.tick_size)
                  - round(info.tick_size * 3.3 / info.tick_size)) < 1e-9
    step_ok = abs((q / info.qty_step) - round(q / info.qty_step)) < 1e-6
    sub_min = broker.submit(OrderRequest(SYMBOL, Side.BUY, info.min_notional * 0.01))
    _record("3_rounding_minNotional",
            step_ok and tick_ok and sub_min.status is OrderStatus.REJECTED,
            f"qty={q} step_ok={step_ok} tick_ok={tick_ok} submin_rejected={sub_min.status.name}")

    # 4. hedge/one-way detection
    try:
        pos = _session(broker).get_positions(category=cfg.category, symbol=SYMBOL)
        idxs = {int(r.get("positionIdx", 0)) for r in pos.get("result", {}).get("list", [])}
        one_way = (idxs <= {0}) and cfg.position_idx == 0
        _record("4_account_mode", one_way, f"positionIdx seen={idxs or '{0}'} cfg={cfg.position_idx} "
                f"({'one-way' if one_way else 'HEDGE or mismatch — fix account/config'})")
    except Exception as exc:
        _record("4_account_mode", None, f"could not read: {exc}")

    # 5. leverage
    lev_ok = broker.set_leverage(SYMBOL, cfg.leverage)
    try:
        pos = _session(broker).get_positions(category=cfg.category, symbol=SYMBOL)
        lev_now = next((r.get("leverage") for r in pos.get("result", {}).get("list", [])), None)
        _record("5_leverage", lev_ok, f"set={cfg.leverage} venue={lev_now}")
    except Exception as exc:
        _record("5_leverage", lev_ok, f"set returned {lev_ok}; verify read failed: {exc}")

    # 6. submit market entry
    since = pd.Timestamp.now(tz="UTC")
    entry = broker.submit(OrderRequest(SYMBOL, Side.BUY, NOTIONAL, order_type=OrderType.MARKET,
                                       client_id=f"e2e-entry-{int(time.time())}"))
    _record("6_entry_market", entry.status is OrderStatus.FILLED,
            f"status={entry.status.name} qty={entry.filled_quantity} msg={entry.message}")

    # 7. fills — partial vs full
    time.sleep(2)
    fills = broker.get_fills(since=since)
    filled = sum(f.quantity for f in fills)
    requested = entry.filled_quantity or 0.0
    kind = "full" if filled >= requested - info.qty_step else "PARTIAL"
    _record("7_fills", len(fills) > 0, f"{len(fills)} fill(s), qty={filled} ({kind}); "
            f"avg={fills[0].price if fills else 'n/a'}")

    # 8/9. reduce-only TP (LIMIT far up) + SL (STOP far down) resting
    px = broker._last_price(SYMBOL)
    tp = broker.submit(OrderRequest(SYMBOL, Side.SELL, NOTIONAL, order_type=OrderType.LIMIT,
                                    price=px * 1.20, reduce_only=True, client_id=f"e2e-tp-{int(time.time())}"))
    sl = broker.submit(OrderRequest(SYMBOL, Side.SELL, NOTIONAL, order_type=OrderType.STOP,
                                    price=px * 0.80, reduce_only=True, client_id=f"e2e-sl-{int(time.time())}"))
    _record("8_tp_resting", tp.status is OrderStatus.PENDING, f"status={tp.status.name} msg={tp.message}")
    _record("9_sl_resting", sl.status is OrderStatus.PENDING, f"status={sl.status.name} msg={sl.message}")
    open_before = _session(broker).get_open_orders(category=cfg.category, symbol=SYMBOL)
    n_open = len(open_before.get("result", {}).get("list", []))

    # 10. cancel resting brackets
    broker.cancel_open_orders(SYMBOL)
    time.sleep(1)
    open_after = _session(broker).get_open_orders(category=cfg.category, symbol=SYMBOL)
    n_after = len(open_after.get("result", {}).get("list", []))
    _record("10_cancel", n_open >= 2 and n_after == 0, f"open before={n_open} after={n_after}")

    # 11. reduce-only position close
    held = broker._position_base_qty(SYMBOL)
    close = broker.submit(OrderRequest(SYMBOL, Side.SELL, NOTIONAL, order_type=OrderType.MARKET,
                                       reduce_only=True, client_id=f"e2e-close-{int(time.time())}"))
    time.sleep(2)
    flat = abs(broker._position_base_qty(SYMBOL)) < info.qty_step
    _record("11_position_close", close.status is OrderStatus.FILLED and flat,
            f"held={held} close={close.status.name} now_flat={flat}")

    # 12. restart + reconciliation (fresh broker/session)
    fresh = build_broker(cfg)  # brand-new session — reads true venue state
    state = fresh.get_portfolio_state()
    rep = Reconciler().reconcile(pd.Series(dtype="float64"), state)  # expected flat
    _record("12_restart_reconcile", rep.ok,
            f"nav={state.value} positions={len(state.positions)} discrepancies={rep.n_discrepancies}")

    # 13. retCode!=0 handling (deliberately invalid symbol -> clean REJECTED, no crash)
    bad = broker.submit(OrderRequest("NOTAREALSYMBOLUSDT", Side.BUY, NOTIONAL))
    _record("13_retcode_handling", bad.status is OrderStatus.REJECTED, f"msg={bad.message[:60]}")

    # 14. reduce-only safety when flat (must not open a new position)
    ro_flat = broker.submit(OrderRequest(SYMBOL, Side.SELL, NOTIONAL, order_type=OrderType.MARKET,
                                          reduce_only=True, client_id=f"e2e-roflat-{int(time.time())}"))
    time.sleep(1)
    still_flat = abs(broker._position_base_qty(SYMBOL)) < info.qty_step
    _record("14_reduce_only_when_flat", still_flat,
            f"status={ro_flat.status.name} still_flat={still_flat}")


def cleanup(broker: BybitBroker) -> None:
    try:
        broker.cancel_open_orders(SYMBOL)
        if abs(broker._position_base_qty(SYMBOL)) > 0:
            broker.submit(OrderRequest(SYMBOL, Side.SELL, NOTIONAL * 5, order_type=OrderType.MARKET,
                                       reduce_only=True, client_id=f"e2e-cleanup-{int(time.time())}"))
        logger.info("cleanup: cancelled orders and attempted flatten")
    except Exception as exc:
        logger.warning("cleanup failed: {}", exc)


def main() -> int:
    cfg = BybitConfig.from_env()
    if not cfg.testnet:
        logger.error("REFUSING TO RUN: BYBIT_TESTNET must be truthy for this harness."); return 2
    if cfg.mode is not TradingMode.LIVE:
        logger.error("Set BYBIT_TRADING_MODE=live (testnet). Current mode={}", cfg.mode.value); return 2
    if not (cfg.api_key and cfg.api_secret):
        logger.error("Set BYBIT_API_KEY / BYBIT_API_SECRET (testnet)."); return 2

    broker = build_broker(cfg)
    logger.info("E2E testnet start — symbol={} notional={} leverage={} positionIdx={}",
                SYMBOL, NOTIONAL, cfg.leverage, cfg.position_idx)
    try:
        run(broker)
    except Exception as exc:
        logger.exception("E2E aborted with an exception: {}", exc)
        _record("ABORTED", False, str(exc))
    finally:
        cleanup(broker)

    # scorecard
    scored = [r for r in _RESULTS if r[1] != "SKIP"]
    passed = sum(1 for r in scored if r[1] == "PASS")
    logger.info("=" * 60)
    for step, status, detail in _RESULTS:
        logger.info("  {:<24} {}", step, status)
    logger.info("=" * 60)
    logger.info("SCORE: {}/{} passed ({} skipped)",
                passed, len(scored), len(_RESULTS) - len(scored))
    verdict = "E2E GREEN — execution plumbing verified" if passed == len(scored) else \
              "E2E has FAILURES — do not proceed to real capital until fixed"
    logger.info("VERDICT: {}", verdict)
    return 0 if passed == len(scored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
