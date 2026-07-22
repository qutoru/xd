"""ALX3 — independent data reconstruction from raw exchange APIs.

Fetches OHLCV (daily), funding history (settlement timestamps), quote/dollar
volume, and listing/onboard dates directly from Binance USDT-M Futures, OKX
USDT-SWAP, and Bybit linear — using **none** of the project's existing parquet,
funding_panel, or prior artifacts. Raw pulls are cached under
``data/research/alx3/cache`` (kept separate from the Bybit research data).

All timestamps are UTC epoch-ms. Funding sign convention on every venue: positive
rate = longs pay shorts (verified in Stage-1 audit).
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
from loguru import logger

CACHE = Path("data/research/alx3/cache")
_UA = {"User-Agent": "Mozilla/5.0 (research)"}
_MS_DAY = 86_400_000


def _get(url: str, tries: int = 4, pause: float = 0.25):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as exc:  # transient network / rate limit
            last = exc
            time.sleep(pause * (2 ** i))
    raise RuntimeError(f"GET failed after {tries}: {url} :: {last}")


def _cache_path(kind: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / kind


def _cached(kind: str, builder):
    p = _cache_path(kind + ".parquet")
    if p.exists():
        return pd.read_parquet(p)
    df = builder()
    if df is not None and not df.empty:
        df.to_parquet(p)
    return df


# ======================================================================= #
# symbol mapping
# ======================================================================= #
def native_symbol(exchange: str, base: str) -> str:
    if exchange == "okx":
        return f"{base}-USDT-SWAP"
    return f"{base}USDT"  # binance, bybit


# ======================================================================= #
# BINANCE USDT-M Futures
# ======================================================================= #
_BINANCE = "https://fapi.binance.com"


def binance_klines_daily(symbol: str) -> pd.DataFrame:
    def build():
        rows, start = [], 1_500_000_000_000  # ~2017-07, before any listing
        while True:
            url = (f"{_BINANCE}/fapi/v1/klines?symbol={symbol}&interval=1d"
                   f"&startTime={start}&limit=1500")
            batch = _get(url)
            if not batch:
                break
            rows.extend(batch)
            last = batch[-1][0]
            if len(batch) < 1500:
                break
            start = last + 1
            time.sleep(0.12)
        if not rows:
            return pd.DataFrame(columns=["date", "close", "qvol"])
        df = pd.DataFrame(rows).iloc[:, [0, 4, 7]]
        df.columns = ["open_ms", "close", "qvol"]
        df["date"] = pd.to_datetime(df["open_ms"].astype("int64"), unit="ms", utc=True).dt.normalize()
        df["close"] = df["close"].astype("float64")
        df["qvol"] = df["qvol"].astype("float64")
        return df[["date", "close", "qvol"]].drop_duplicates("date")
    return _cached(f"binance_kl_{symbol}", build)


def binance_funding(symbol: str) -> pd.DataFrame:
    def build():
        rows, start = [], 1_500_000_000_000
        while True:
            url = (f"{_BINANCE}/fapi/v1/fundingRate?symbol={symbol}"
                   f"&startTime={start}&limit=1000")
            batch = _get(url)
            if not batch:
                break
            rows.extend(batch)
            last = batch[-1]["fundingTime"]
            if len(batch) < 1000:
                break
            start = last + 1
            time.sleep(0.12)
        if not rows:
            return pd.DataFrame(columns=["ts", "rate"])
        df = pd.DataFrame(rows)
        df["ts"] = df["fundingTime"].astype("int64")
        df["rate"] = df["fundingRate"].astype("float64")
        return df[["ts", "rate"]].drop_duplicates("ts").sort_values("ts")
    return _cached(f"binance_fund_{symbol}", build)


def binance_onboard() -> dict:
    def build():
        info = _get(f"{_BINANCE}/fapi/v1/exchangeInfo")
        rows = [{"symbol": s["symbol"], "onboard_ms": int(s["onboardDate"])}
                for s in info["symbols"]
                if s.get("contractType") == "PERPETUAL" and s["symbol"].endswith("USDT")]
        return pd.DataFrame(rows)
    df = _cached("binance_onboard", build)
    return dict(zip(df["symbol"], df["onboard_ms"]))


# ======================================================================= #
# OKX USDT-SWAP
# ======================================================================= #
_OKX = "https://www.okx.com"


def okx_klines_daily(inst: str) -> pd.DataFrame:
    def build():
        rows, after = [], int(time.time() * 1000)
        while True:
            url = (f"{_OKX}/api/v5/market/history-candles?instId={inst}"
                   f"&bar=1Dutc&after={after}&limit=100")
            data = _get(url).get("data", [])
            if not data:
                break
            rows.extend(data)
            after = int(data[-1][0])  # oldest ts in batch (desc order)
            if len(data) < 100:
                break
            time.sleep(0.15)
        if not rows:
            return pd.DataFrame(columns=["date", "close", "qvol"])
        df = pd.DataFrame(rows).iloc[:, [0, 4, 7]]
        df.columns = ["ts", "close", "qvol"]
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True).dt.normalize()
        df["close"] = df["close"].astype("float64")
        df["qvol"] = df["qvol"].astype("float64")
        return df[["date", "close", "qvol"]].drop_duplicates("date").sort_values("date")
    return _cached(f"okx_kl_{inst}", build)


def okx_funding(inst: str) -> pd.DataFrame:
    def build():
        rows, after = [], int(time.time() * 1000)
        while True:
            url = (f"{_OKX}/api/v5/public/funding-rate-history?instId={inst}"
                   f"&after={after}&limit=100")
            data = _get(url).get("data", [])
            if not data:
                break
            rows.extend(data)
            after = int(data[-1]["fundingTime"])
            if len(data) < 100:
                break
            time.sleep(0.15)
        if not rows:
            return pd.DataFrame(columns=["ts", "rate"])
        df = pd.DataFrame(rows)
        df["ts"] = df["fundingTime"].astype("int64")
        df["rate"] = df["realizedRate"].astype("float64")
        return df[["ts", "rate"]].drop_duplicates("ts").sort_values("ts")
    return _cached(f"okx_fund_{inst}", build)


def okx_listing() -> dict:
    def build():
        data = _get(f"{_OKX}/api/v5/public/instruments?instType=SWAP").get("data", [])
        rows = [{"inst": d["instId"], "list_ms": int(d["listTime"])}
                for d in data if d["instId"].endswith("-USDT-SWAP") and d.get("listTime")]
        return pd.DataFrame(rows)
    df = _cached("okx_listing", build)
    return dict(zip(df["inst"], df["list_ms"]))


# ======================================================================= #
# BYBIT linear (control — fresh re-fetch, no old parquet)
# ======================================================================= #
_BYBIT = "https://api.bybit.com"


def bybit_klines_daily(symbol: str) -> pd.DataFrame:
    def build():
        rows, end = [], int(time.time() * 1000)
        while True:
            url = (f"{_BYBIT}/v5/market/kline?category=linear&symbol={symbol}"
                   f"&interval=D&end={end}&limit=1000")
            lst = _get(url).get("result", {}).get("list", [])
            if not lst:
                break
            rows.extend(lst)
            oldest = int(lst[-1][0])  # desc
            if len(lst) < 1000:
                break
            end = oldest - 1
            time.sleep(0.12)
        if not rows:
            return pd.DataFrame(columns=["date", "close", "qvol"])
        df = pd.DataFrame(rows).iloc[:, [0, 4, 6]]
        df.columns = ["ts", "close", "qvol"]
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True).dt.normalize()
        df["close"] = df["close"].astype("float64")
        df["qvol"] = df["qvol"].astype("float64")
        return df[["date", "close", "qvol"]].drop_duplicates("date").sort_values("date")
    return _cached(f"bybit_kl_{symbol}", build)


def bybit_funding(symbol: str) -> pd.DataFrame:
    def build():
        rows, end = [], int(time.time() * 1000)
        while True:
            url = (f"{_BYBIT}/v5/market/funding/history?category=linear&symbol={symbol}"
                   f"&endTime={end}&limit=200")
            lst = _get(url).get("result", {}).get("list", [])
            if not lst:
                break
            rows.extend(lst)
            oldest = int(lst[-1]["fundingRateTimestamp"])
            if len(lst) < 200:
                break
            end = oldest - 1
            time.sleep(0.12)
        if not rows:
            return pd.DataFrame(columns=["ts", "rate"])
        df = pd.DataFrame(rows)
        df["ts"] = df["fundingRateTimestamp"].astype("int64")
        df["rate"] = df["fundingRate"].astype("float64")
        return df[["ts", "rate"]].drop_duplicates("ts").sort_values("ts")
    return _cached(f"bybit_fund_{symbol}", build)


# ======================================================================= #
# unified dispatch
# ======================================================================= #
def fetch_symbol(exchange: str, base: str):
    """Return (klines_df[date,close,qvol], funding_df[ts,rate]) or (None, None)."""
    sym = native_symbol(exchange, base)
    try:
        if exchange == "binance":
            kl, fu = binance_klines_daily(sym), binance_funding(sym)
        elif exchange == "okx":
            kl, fu = okx_klines_daily(sym), okx_funding(sym)
        elif exchange == "bybit":
            kl, fu = bybit_klines_daily(sym), bybit_funding(sym)
        else:
            raise ValueError(exchange)
    except Exception as exc:
        logger.warning("{} {}: fetch failed ({})", exchange, sym, exc)
        return None, None
    if kl is None or kl.empty or fu is None or fu.empty:
        return None, None
    return kl, fu
