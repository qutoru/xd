"""ALX2 — the six fixed adversarial tests (see PREREGISTRATION.md).

The object under test is the FROZEN ALX funding->price book, reused verbatim via
``alpha_library.alx_funding_price_validation.validate``. Nothing here modifies the
signal, lookback, ranking, construction, holding period, execution, baseline
costs, or universe. Tests only apply *strictly harsher* assumptions as attacks.

Battery (fixed, no additions): T1 point-in-time funding audit, T2 temporal
concentration, T3 leave-one-symbol-out, T4 expanded factor attribution, T5
liquidity-aware costs, T6 deflated Sharpe. Each returns a dict with a boolean
``survived`` and the evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm

from crypto_signal_bot.data.derivatives import fetch_funding
from crypto_signal_bot.research.xsection import features as ft
from alpha_library.alx_funding_price_validation.validate import (
    backtest,
    ols_hac,
    sharpe,
    weights_from_signal,
)

MED = 0.00075
K_PCT = 0.30
SETTLE_HOURS = (0, 8, 16)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _signal_from_funding(daily_fund: pd.DataFrame) -> pd.DataFrame:
    """The FROZEN signal: -(trailing 7-day mean of daily realized funding)."""
    return -daily_fund.rolling(7).mean()


def _applied_weights(signal: pd.DataFrame, ret: pd.DataFrame, lag: int = 1) -> np.ndarray:
    """Reproduce the frozen backtester's executed weights (lagged), for per-name
    trade-size accounting in T5. Identical rule to validate.weights_from_signal."""
    f = signal.reindex_like(ret).to_numpy(dtype="float64")
    T, N = f.shape
    w = np.zeros((T, N))
    for t in range(T):
        w[t] = weights_from_signal(f[t], K_PCT)
    applied = np.zeros_like(w)
    applied[lag:] = w[:-lag]
    return applied


# --------------------------------------------------------------------------- #
# T1 — point-in-time funding reconstruction & alignment audit
# --------------------------------------------------------------------------- #
def build_pit_funding(cols, daily_index, *, history_days: int = 1200):
    """Independently re-fetch funding and reconstruct the daily realized-funding
    panel STRICTLY point-in-time: each day = sum of that day's {0,8,16} UTC
    settlements, known only at/after settlement. No forward-fill from the future.

    Returns (daily_fund_pit, off_grid_frac, fetched_cols).
    """
    settle_series = {}
    off_grid = 0
    total = 0
    fetched = []
    for sym in cols:
        try:
            f = fetch_funding(sym, history_days=history_days)
        except Exception as exc:  # network / symbol issue -> drop from PIT set
            logger.warning("T1: {} funding re-fetch failed ({})", sym, exc)
            continue
        if f.empty:
            logger.warning("T1: {} no funding history", sym)
            continue
        ts = pd.to_datetime(f["timestamp"], unit="ms", utc=True)
        s = pd.Series(f["funding_rate"].to_numpy(), index=ts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        total += len(s)
        off_grid += int((~s.index.hour.isin(SETTLE_HOURS)).sum())
        # keep only on-grid settlements, then sum per UTC day (point-in-time)
        s = s[s.index.hour.isin(SETTLE_HOURS)]
        settle_series[sym] = s
        fetched.append(sym)

    # daily realized funding = sum of a day's settlements (backward-only)
    daily = {sym: s.resample("1D").sum() for sym, s in settle_series.items()}
    panel = pd.DataFrame(daily)
    panel.index = panel.index.tz_convert("UTC") if panel.index.tz else panel.index
    daily_fund_pit = panel.reindex(index=daily_index, columns=list(cols))
    off_grid_frac = off_grid / total if total else 0.0
    return daily_fund_pit, off_grid_frac, fetched


def t1_point_in_time(daily_ret, daily_fund_reused, daily_fund_pit, fetched_cols,
                     alpha_sharpe_reused):
    """Audit alignment and re-run the frozen strategy on the freshly reconstructed
    point-in-time funding panel. Includes a funding-side future-corruption test
    (the ALX return-corruption test could not catch a funding-construction leak)."""
    cols = [c for c in fetched_cols if c in daily_ret.columns]
    ret = daily_ret[cols]
    fpit = daily_fund_pit[cols]

    sig_pit = _signal_from_funding(fpit)
    res_pit = backtest(ret, sig_pit, funding=fpit, lag=1, cost=MED)
    net_pit = res_pit["net"]
    sh_pit = sharpe(net_pit)

    # funding-side no-look-ahead: corrupt settlements AFTER t0, PnL through t0 fixed
    t0 = len(net_pit) // 2
    fpit_corr = fpit.copy()
    fpit_corr.iloc[t0 + 1:] += 1.0
    sig_corr = _signal_from_funding(fpit_corr)
    net_corr = backtest(ret, sig_corr, funding=fpit_corr, lag=1, cost=MED)["net"]
    no_funding_leak = np.allclose(net_pit[:t0], net_corr[:t0], atol=1e-12)

    # agreement with the reused E7 panel (columns present in both)
    common = [c for c in cols if c in daily_fund_reused.columns]
    a = daily_fund_reused[common].to_numpy().ravel()
    b = fpit[common].to_numpy().ravel()
    m = np.isfinite(a) & np.isfinite(b)
    corr = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")

    survived = (no_funding_leak
                and np.sign(sh_pit) == np.sign(alpha_sharpe_reused)
                and sh_pit > 1.0)
    return {"survived": survived, "sharpe_pit": sh_pit, "sharpe_reused": alpha_sharpe_reused,
            "no_funding_leak": no_funding_leak, "panel_corr": corr,
            "n_symbols": len(cols)}


# --------------------------------------------------------------------------- #
# T2 — temporal concentration (top-day / crisis dependence)
# --------------------------------------------------------------------------- #
def t2_temporal_concentration(net, idx, daily_ret):
    net = np.asarray(net)
    total = net.sum()
    order = np.argsort(net)[::-1]
    top1_share = float(net[order[0]] / total) if total > 0 else float("nan")
    k5 = max(1, int(len(net) * 0.05))
    top5_share = float(net[order[:k5]].sum() / total) if total > 0 else float("nan")

    # (a) drop the best 5% PnL days
    thr = np.quantile(net, 0.95)
    sh_drop = sharpe(net[net < thr])

    # (b) exclude the highest-volatility decile of days (crisis proxy)
    mvol = daily_ret.mean(axis=1).rolling(30).std().shift(1).reindex(idx)
    hi = mvol.quantile(0.90)
    keep = (mvol < hi).to_numpy()
    sh_nocrisis = sharpe(net[keep])

    survived = (sh_drop > 0) and (sh_nocrisis > 0)
    return {"survived": survived, "sharpe_drop_top5pct": sh_drop,
            "sharpe_ex_crisis_decile": sh_nocrisis,
            "top1_day_pnl_share": top1_share, "top5pct_pnl_share": top5_share}


# --------------------------------------------------------------------------- #
# T3 — leave-one-symbol-out concentration
# --------------------------------------------------------------------------- #
def t3_leave_one_out(daily_ret, signal, daily_fund):
    cols = list(daily_ret.columns)
    out = {}
    for s in cols:
        keep = [c for c in cols if c != s]
        net = backtest(daily_ret[keep], signal[keep], funding=daily_fund[keep],
                       lag=1, cost=MED)["net"]
        out[s] = sharpe(net)
    vals = np.array(list(out.values()))
    worst = min(out, key=out.get)
    survived = float(vals.min()) > 1.0
    return {"survived": survived, "min_sharpe": float(vals.min()),
            "median_sharpe": float(np.median(vals)), "max_sharpe": float(vals.max()),
            "worst_drop_symbol": worst, "n": len(cols)}


# --------------------------------------------------------------------------- #
# T4 — expanded factor attribution
# --------------------------------------------------------------------------- #
def _rolling_beta(daily_ret, mkt, window=60):
    m1 = mkt.rolling(window).mean()
    var_m = mkt.pow(2).rolling(window).mean() - m1.pow(2)
    cov = (daily_ret.mul(mkt, axis=0).rolling(window).mean()
           - daily_ret.rolling(window).mean().mul(m1, axis=0))
    return cov.div(var_m, axis=0)


def _downside_beta(daily_ret, mkt, window=60):
    """Rolling beta estimated on down-market days only (mkt < 0)."""
    d = (mkt < 0).astype("float64")
    n = d.rolling(window).sum()
    sm = (mkt * d).rolling(window).sum()
    smm = (mkt * mkt * d).rolling(window).sum()
    var = smm - sm.pow(2).div(n)
    sr = daily_ret.mul(d, axis=0).rolling(window).sum()
    srm = daily_ret.mul(mkt * d, axis=0).rolling(window).sum()
    cov = srm - sr.mul(sm, axis=0).div(n, axis=0)
    return cov.div(var, axis=0)


def t4_expanded_attribution(net, daily_ret, daily_logret, daily_fund, turnover_perbar):
    mkt = daily_ret.mean(axis=1)
    beta = _rolling_beta(daily_ret, mkt)
    resid = daily_ret.sub(beta.mul(mkt, axis=0))

    classic = {
        "reversal": ft.short_term_reversal(daily_logret, 3),
        "momentum": ft.relative_strength(daily_logret, 24),
        "beta": beta,
        "volatility": daily_ret.rolling(30).std(),
        "size": pd.DataFrame(np.tile(turnover_perbar.values, (len(daily_ret), 1)),
                             index=daily_ret.index, columns=daily_ret.columns),
    }
    new = {
        "rev_1d": ft.short_term_reversal(daily_logret, 1),
        "downside_beta": _downside_beta(daily_ret, mkt),
        "idio_vol": resid.rolling(30).std(),
    }
    funding_only = {"raw_funding": -daily_fund}

    def book(v):
        return backtest(daily_ret, v, funding=None, lag=1, cost=MED)["net"]

    def fit_set(feat_map):
        names = list(feat_map)
        X = np.column_stack([np.ones(len(net))] + [book(feat_map[k]) for k in names])
        f = ols_hac(net, X)
        return names, f

    # primary: classic anomalies + 3 new controls (ex own-funding)
    ex_names, ex_fit = fit_set({**classic, **new})
    # reported alongside: + funding
    all_names, all_fit = fit_set({**classic, **new, **funding_only})

    inter_t = float(ex_fit["t"][0])
    inter_ann = float(ex_fit["beta"][0] * 365)
    survived = (inter_t >= 2.0) and (inter_ann > 0)
    loadings = {n: (round(float(b), 3), round(float(t), 2))
                for n, b, t in zip(ex_names, ex_fit["beta"][1:], ex_fit["t"][1:])}
    return {"survived": survived, "intercept_t": inter_t, "intercept_ann": inter_ann,
            "r2_ex_funding": float(ex_fit["r2"]),
            "intercept_t_with_funding": float(all_fit["t"][0]),
            "intercept_ann_with_funding": float(all_fit["beta"][0] * 365),
            "r2_with_funding": float(all_fit["r2"]), "loadings_ex_funding": loadings}


# --------------------------------------------------------------------------- #
# T5 — realistic execution with liquidity-aware transaction costs
# --------------------------------------------------------------------------- #
def t5_liquidity_costs(daily_ret, signal, daily_fund, adv_daily,
                       *, aum=1.0e7, taker=0.00075, spread=0.0005,
                       impact_bps_per_1pct=0.0010):
    """Strictly harsher, liquidity-aware costs. Per name/rebalance on |Δw_i|:
    (taker + spread) linear, plus impact = impact_bps_per_1pct per 1% of ADV
    traded (trade_notional_i / (1% × ADV_i)). Never cheaper than the 7.5bps base."""
    cols = list(daily_ret.columns)
    ret = daily_ret[cols].to_numpy(dtype="float64")
    fund = daily_fund[cols].to_numpy(dtype="float64")
    applied = _applied_weights(signal[cols], daily_ret[cols], lag=1)

    dw = np.abs(np.diff(applied, axis=0, prepend=0.0))            # |Δw_i| per day
    adv = adv_daily.reindex(cols).to_numpy(dtype="float64")       # $ daily volume
    participation = (dw * aum) / (0.01 * adv)                     # units of 1% ADV
    impact_rate = impact_bps_per_1pct * participation            # fractional, per name
    cost_frac = (taker + spread) + impact_rate                   # >= base everywhere
    cost = (dw * cost_frac).sum(axis=1)

    price = np.nansum(applied * ret, axis=1)
    funding_pnl = -np.nansum(applied * np.nan_to_num(fund), axis=1)
    net = price + funding_pnl - cost
    sh = sharpe(net)
    eff_bps = float((cost.sum() / dw.sum()) * 1e4) if dw.sum() > 0 else float("nan")
    survived = sh > 1.0
    return {"survived": survived, "sharpe_liq": sh,
            "avg_effective_cost_bps": eff_bps, "baseline_bps": taker * 1e4}


# --------------------------------------------------------------------------- #
# T6 — deflated Sharpe ratio / multiple-testing adjustment
# --------------------------------------------------------------------------- #
def t6_deflated_sharpe(net, *, n_trials=100):
    x = np.asarray(net, dtype="float64")
    T = x.size
    mu, sd = x.mean(), x.std(ddof=1)
    sr = mu / sd                                    # per-observation Sharpe
    z = (x - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())                   # Pearson (normal = 3)

    gamma = 0.5772156649015329
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sqrt_v0 = 1.0 / math.sqrt(T - 1)                # null sampling std of SR (per-obs)
    sr_star = sqrt_v0 * ((1.0 - gamma) * z1 + gamma * z2)

    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    dsr = float(norm.cdf((sr - sr_star) * math.sqrt(T - 1) / denom))

    bpy = math.sqrt(365.0)
    survived = dsr > 0.95
    return {"survived": survived, "dsr": dsr, "n_trials": n_trials,
            "sharpe_ann": sr * bpy, "sr_star_ann": sr_star * bpy,
            "skew": skew, "kurt": kurt, "T": int(T)}
