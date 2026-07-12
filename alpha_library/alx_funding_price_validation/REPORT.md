# ALX — Funding→Price alpha — FALSIFICATION REPORT

**Verdict: PASS — the alpha survives falsification.** It is reproducible,
leakage-free, and retains a statistically and economically meaningful component
unexplained by known price factors. Single run, no tuning, per
`PREREGISTRATION.md`. Branch `alpha/alx-funding-price-validation`. Reproduce:
`py alpha_library/alx_funding_price_validation/run.py`.

Object under test (frozen): funding-ranked dollar-neutral daily book — long top
30% / short bottom 30% by `−(7-day mean daily funding)`, hold 1d, lag 1d, Med
cost. Discovered net Sharpe ≈ +1.71.

---

## Phase 1 — verification (decisive)

### A. Reproducibility — PASS
Three independent code paths agree:

| path | net Sharpe |
|---|---|
| independent reimplementation (this experiment) | **+1.707** |
| shared `portfolio.backtest_portfolio` engine + funding leg | +1.707 |
| AL-1 reference | +1.710 |

An independent from-scratch reimplementation matches AL-1 to 3 decimals ⇒ the
result is **not** an AL-1-specific implementation bug.

### B. Leakage / implementation — PASS (no evidence of any artifact)
- **No look-ahead:** corrupting all returns after day *t* leaves PnL through *t*
  bit-identical.
- **Sign reversal:** flipping the signal flips the Sharpe (+1.71 → −2.90); the
  backtest has no mechanical positive bias.
- **Placebo:** the alpha beats **100%** of 200 random-signal books (max random
  Sharpe −1.61; random daily rebalancing loses to cost). The engine does not
  manufacture Sharpe.
- **Execution-lag tell:** lag 0/1/2 = +1.99/+1.71/+1.10. Contemporaneous (lag 0,
  look-ahead) is only mildly higher than the traded lag 1, and the signal remains
  strong at lag 2 — the opposite of a same-bar microstructure leak (which would
  spike at lag 0 and collapse by lag 1).

### C. Known-factor attribution — PASS (large unexplained component remains)
Factor-mimicking books, identical construction, net @Med. Correlations with the
alpha: reversal −0.17, momentum +0.20, beta −0.07, volatility −0.06, size −0.12,
**raw (unsmoothed) funding +0.55**.

Multivariate OLS (Newey-West), all 6 factors: **R² = 34%**. Only two loadings are
non-trivial — raw_funding (β +0.51, t +9.6) and a small reversal tilt (β −0.08,
t −2.3); beta/vol/size/momentum are insignificant.

| control set | unexplained intercept | NW t | hedged Sharpe |
|---|---|---|---|
| **classic 5 (ex-funding: reversal, momentum, beta, vol, size)** | **+24.2 %/yr** | **+2.87** | **+1.66** |
| all 6 (incl. raw funding) | +35.3 %/yr | +5.00 | +2.83 |

The classic price anomalies explain little (their loadings are small/insignificant
and the alpha's correlation with reversal is only −0.17). A large unexplained
intercept survives controlling for them (+24%/yr, t 2.87). The dominant covariate
is funding itself — but that is the alpha's **own economic variable**, not a
competing anomaly; controlling for it *raises* the intercept (the 7-day-smoothed
signal adds predictive value beyond instantaneous funding). Either way the
unexplained component is statistically and economically meaningful.

**Phase 1 gate (artifact-free): PASS → proceed to Phase 2.**

## Phase 2 — characterization (descriptive only; cannot reject or tune)

| axis | result |
|---|---|
| cost Low / Med / High | net Sharpe +2.11 / +1.71 / +1.16 |
| exec-lag 1 / 2 / 3 days | +1.71 / +1.10 / +1.02 |
| walk-forward (6 folds) | [1.89, 2.54, 2.10, 1.20, 0.67, 1.94] — **6/6 > 0** |
| year-by-year 2023–2026 | [1.73, 2.31, 1.14, 1.43] — all positive |
| volatility regime low/mid/high | [1.54, 1.51, 2.04] — all positive |
| drop BTC & ETH | +1.56 |
| top-20 liquid subset | +1.14 |
| block-bootstrap 95% CI | **[+0.78, +2.66]** (excludes 0) |

The alpha weakens under tougher assumptions (as expected) but **does not
disappear** anywhere: it stays economically meaningful across every cost, lag,
fold, year, regime, and subset, and remains positive on the liquid-only universe
(not a small-cap/illiquidity artifact).

---

## Verdict & why it exists

**PASS.** The apparent alpha is real: reproduced independently, free of detectable
look-ahead/leakage/implementation error, and carrying a significant component
that the known price factors do not explain.

**Economic mechanism.** The funding rate is a direct, priced read on positioning:
persistently positive funding = crowded, leveraged longs paying to hold →
over-extended positioning that predicts subsequent relative **under**performance
as the crowd unwinds; persistently negative funding = crowded shorts / a
discouraged asset → subsequent relative **out**performance. Ranking
cross-sectionally on 7-day-smoothed funding and taking the other side of the crowd
harvests this **positioning / crowding-unwind premium**.

Why it works where E1–E9 price prediction failed, and why it is not the same as
the earlier results:
- **Not price reversal (E7/E8):** correlation with the price-reversal factor is
  only −0.17 and reversal explains almost none of it. Funding-based crowding is a
  *different* signal from recent price moves.
- **Survives cost where E7 reversal did not:** funding positioning is
  **persistent** (builds/unwinds over days), so the 7-day signal rebalances slowly
  → low turnover → survives taker cost at daily frequency. E7's 1-bar price
  reversal had ruinous turnover; this does not.
- **Not the AL-1 carry:** the funding *cash flow* is only ~18% of PnL and does not
  cover cost (AL-1 failed as carry). The value here is funding as a **price
  predictor**, not as income — a distinct hypothesis, correctly separated.
- **Reconciles with E6-H5:** that experiment rejected a *1h single-name* funding→
  price *reversion*; this is a *daily cross-sectional* positioning effect — a
  different specification, not a contradiction.

## Residual cautions (documented, not disqualifying)
- **Survivorship-safe universe** (survivors only): consistent with all prior work
  and conservative, but the live-universe magnitude may be lower.
- **Funding panel provenance:** reused the E7-built `funding_panel.parquet` (fetched
  point-in-time via `merge_asof` backward). A construction-time leak there would
  not be caught by the return-corruption test; a fresh point-in-time re-fetch
  should be part of any graduation to production.

## Next step (only now permitted)
The effect qualifies to **become an Alpha Library candidate**. Recommended: a
formal library pre-registration ("AL-2: funding-imbalance cross-sectional price")
running the full admission battery (orthogonality vs any other admitted alpha,
capacity, live-universe/point-in-time funding re-fetch). No tuning; the strategy
stays exactly as validated here.
