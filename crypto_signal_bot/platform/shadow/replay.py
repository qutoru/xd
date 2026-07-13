"""Single-day virtual replay: TargetBook + market data -> ShadowReport.

Pure function, alpha-agnostic and execution-agnostic. Marks the given book
against the most recent realized return/funding cross-section in the snapshot and
charges turnover-based costs against the previous book.
"""

from __future__ import annotations

import pandas as pd

from crypto_signal_bot.platform.data.snapshot import MarketSnapshot
from crypto_signal_bot.platform.portfolio.book import TargetBook
from crypto_signal_bot.platform.shadow import costs
from crypto_signal_bot.platform.shadow.funding import funding_pnl
from crypto_signal_bot.platform.shadow.pnl import price_pnl
from crypto_signal_bot.platform.shadow.report import ShadowParams, ShadowReport


def _turnover(weights: pd.Series, prev_weights: pd.Series | None) -> float:
    prev = prev_weights if prev_weights is not None else pd.Series(dtype="float64")
    idx = weights.index.union(prev.index)
    w = weights.reindex(idx).fillna(0.0)
    p = prev.reindex(idx).fillna(0.0)
    return float((w - p).abs().sum())


def replay_day(
    book: TargetBook,
    prev_book: TargetBook | None,
    snapshot: MarketSnapshot,
    params: ShadowParams = ShadowParams(),
    *,
    cum_pnl_prev: float = 0.0,
) -> ShadowReport:
    """Compute one day's :class:`ShadowReport` for ``book`` (no positions opened)."""
    w = book.weights
    returns_row = snapshot.returns.iloc[-1]
    funding_row = snapshot.funding.iloc[-1]
    prev_w = prev_book.weights if prev_book is not None else None

    turnover = _turnover(w, prev_w)
    price = price_pnl(w, returns_row)
    funding = funding_pnl(w, funding_row)
    gross = float(w.abs().sum())
    fee = costs.fees(turnover, params.fee_rate)
    slip = costs.slippage(turnover, params.slippage_rate)
    coc = costs.cost_of_capital(gross, params.cost_of_capital)

    daily = price + funding - fee - slip - coc
    return ShadowReport(
        asof=book.asof,
        daily_pnl=daily,
        cum_pnl=cum_pnl_prev + daily,
        price_pnl=price,
        funding_pnl=funding,
        fees=fee,
        slippage=slip,
        cost_of_capital=coc,
        turnover=turnover,
        gross=gross,
        net=float(w.sum()),
        long_exposure=float(w[w > 0].sum()),
        short_exposure=float(w[w < 0].sum()),
        n_long=int((w > 0).sum()),
        n_short=int((w < 0).sum()),
        portfolio_value=params.portfolio_value,
    )
