# ALX3 — Independent External Replication of the Funding→Price alpha — PRE-REGISTRATION (LOCKED)

**Status:** LOCKED on 2026-07-13, before writing any ALX3 replication code or
seeing any ALX3 result. Branch `alpha/alx3-external-replication`. This document is
committed **by itself** as the lock; code and `REPORT.md` follow later. Single
run, no tuning. **The prior is that ALX is FALSE until it reproduces on fully
independent data.** The goal is to *destroy* the effect on new data; survival is
the strongest possible evidence it is real.

The decisive criterion of this whole study: **independent external replication on
a different exchange AND a new time period is stronger evidence than any further
test on the original dataset.** Absence of external replication will NOT be
compensated with extra internal stress tests.

---

## 1. Object under test — FROZEN, used EXACTLY as registered in ALX

The ALX funding-ranked dollar-neutral daily book, reused **verbatim** through the
frozen backtester `alpha_library.alx_funding_price_validation.validate`
(`backtest`, `weights_from_signal`, `sharpe`, `ols_hac`). Any change to the
strategy invalidates the experiment. Frozen and NOT modifiable:

- **Signal:** `−(trailing 7-day mean of daily realized funding)`; daily realized
  funding = sum of settlements at hours {0, 8, 16} UTC.
- **Lookback:** 7 days. **Ranking:** long top 30% / short bottom 30%.
- **Construction:** dollar-neutral, equal-weight, gross 1. **Hold:** 1 day.
  **Rebalance:** 1 day. **Execution lag:** 1 day.
- **Cost baseline:** Med = 7.5 bps per unit turnover. **Net = price + funding −
  cost.**
- **Universe:** the same fixed ALX major base-coin set (`SURVIVORSHIP_SAFE`),
  mapped to each exchange's native perpetual symbol. The universe is **not
  changed within an exchange**; only the exchange (data source) changes.

No parameter, lookback, holding period, execution, construction, ranking, or
in-exchange universe may change. No parameter search, no data mining, one run.

## 2. Stage 1 — Independent data reconstruction (from scratch)

Rebuild the entire dataset **from raw exchange APIs**, using **none** of the old
Bybit parquet, the existing `funding_panel.parquet`, or any prior artifact. Fetch
independently, per exchange, per symbol:

- OHLCV daily bars (native 1d klines close = daily last close, faithful to ALX's
  `to_daily`), and quote (dollar) volume for ADV.
- Full funding-rate history with settlement timestamps.
- Listing / onboard dates (for the point-in-time universe, Stage 6).

**Audits (documented in REPORT):** (a) funding **sign** convention — positive
funding = longs pay shorts on every venue (verified, so the ALX `−funding` signal
means the same everywhere); (b) **timezone** — all timestamps UTC epoch-ms;
(c) **settlement time** — settlements fall on {0,8,16} UTC (report off-grid
fraction); (d) **aggregation** — daily realized funding = sum of a day's
settlements; (e) **point-in-time correctness** — funding at day `t` uses only
settlements ≤ `t`; a funding-side future-corruption test must leave PnL through
`t` bit-identical. If the new reconstruction differs from the old Bybit one, the
difference is explained (a different exchange is a *different sample*, so absolute
numbers are expected to differ; the economic effect is what must recur).

## 3. Exchanges (data sources)

1. **Binance USDT-M Futures — PRIMARY independent venue.** Funding & klines back
   to 2019, broad USDT-perp cross-section.
2. **OKX USDT-SWAP — SECOND independent venue.**
3. **Bybit (fresh re-fetch) — CONTROL replication only** (must reproduce ALX
   ≈ +1.71; confirms the pipeline, not independence).
4. **Deribit — declared UNSUITABLE:** it lists only a handful of perpetuals (no
   broad USDT-perp cross-section), so a 40-name cross-sectional strategy cannot be
   hosted there. This is recorded as a **data-insufficiency**, not approximated.

## 4. Stages 2–8 (pre-committed)

- **Stage 2 — different exchange.** Run the frozen strategy unchanged on Binance
  (and OKX). PASS if the effect reproduces; FAIL if the Sharpe vanishes or is
  statistically indistinguishable from zero.
- **Stage 3 — new time period (most important).** Binance history begins ~2019,
  well before the Bybit development window opens (~2023). The Binance segment
  **strictly before the Bybit window start** is a period **never used in any
  experiment** → run the frozen strategy there, no recalibration. This is genuine
  out-of-sample **in time**. (See §6: a *forward*, post-development period does
  not exist because development runs to the present date — stated as a limitation,
  not approximated.)
- **Stage 4 — cross-exchange consistency.** Compare Sharpe, turnover, factor
  exposures, funding distribution across Binance / OKX / Bybit-control. The test
  is the *same economic effect*, not identical numbers.
- **Stage 5 — rolling out-of-sample.** Expanding/rolling walk-forward (per-period,
  no retraining — the strategy has no fitted parameters) to check stability over
  time on the independent data.
- **Stage 6 — point-in-time universe.** At each date include only symbols already
  listed then (onboard date ≤ date, real data present); no future listings, no
  survivorship padding.
- **Stage 7 — realistic execution.** Conservative **lower and upper cost bounds**
  incorporating spread, slippage, market impact, ADV, and short-side/borrow cost.
  Report the net Sharpe band, not a point estimate.
- **Stage 8 — economic mechanism.** If the alpha survives, explain *why* negative
  funding today predicts relative outperformance tomorrow, which risk the premium
  compensates, and why it is not instantly arbitraged. If no convincing mechanism
  exists, say so honestly.

## 5. Pre-committed statistical bars (external standards, fixed now)

- **"Distinguishable from zero"** = block-bootstrap 95% CI of net Sharpe excludes
  0 (block length 20 days). This defines Stage-2/3 FAIL ("indistinguishable from
  zero").
- **Admission bar** = net Sharpe @Med **> 1.0** (same bar as the E9/ALX line),
  applied to the Binance full-period result.
- **Sign consistency** = the effect has the same sign (positive net Sharpe from
  the `−funding` signal) on every independent venue and period.
- No other numeric cutoff is introduced after seeing results.

## 6. Missing-data honesty clause (binding)

If a check cannot be run for lack of data, it is **not** replaced with an
approximation or an assumption. State explicitly: which data are missing, why no
final conclusion is possible without them, and what must be collected. Known gaps
declared **now**:

- **Forward (post-development) out-of-sample does not exist:** the development
  window runs to the present date, so there is no untouched *future* period yet.
  Stage 3 therefore uses the untouched *pre-2023* Binance history instead
  (earlier, not later). The absence of a forward hold-out is a documented
  limitation that caps certainty.
- **Deribit** cross-section is unavailable (unsuitable venue, §3).
- **Survivorship of delisted names:** a fully live delisted-inclusive universe is
  not reconstructable from these APIs beyond onboard dates; residual survivorship
  is documented, not corrected away.

Missing external replication is never compensated by additional internal
stress-tests on the original Bybit dataset.

## 7. Final verdict — exactly one (pre-committed mapping)

- **A. CONFIRMED** — Binance reproduces (Stage 2 PASS: net Sharpe > 1.0, CI
  excludes 0, correct sign) **AND** the new-time pre-2023 segment reproduces
  (Stage 3 PASS: positive, correct sign, CI excludes 0) **AND** cross-exchange
  consistent (Binance & OKX same sign, both CI exclude 0) **AND** survives the
  conservative upper-bound realistic costs (Stage 7: net Sharpe still > 0). Becomes
  an Alpha Library candidate, with the forward-OOS gap noted.
- **B. PLAUSIBLE BUT UNCONFIRMED** — the effect looks real on Binance but is not
  fully corroborated (a new-time, cross-exchange, or realistic-cost leg fails or a
  key datum is missing). Not recommended for production.
- **C. REJECTED** — the effect vanishes on independent data (Sharpe ≤ 0 or CI
  includes 0 on Binance) or is explained by an artifact.

## 8. Deliverables

`alpha_library/alx3_external_replication/` — `data_sources.py` (independent
Binance/OKX/Bybit fetchers; klines, funding, listing; no reuse of old artifacts),
`replicate.py` (build per-exchange panels; run frozen backtester; Stages 1–8),
`run.py` (one command; runs all stages; prints the single verdict),
`PREREGISTRATION.md` (this file, committed first), `REPORT.md` (after the run);
`tests/test_alx3_external.py` (known-answer: funding sign, UTC alignment,
settlement aggregation, PIT no-leak); artifacts under `data/research/alx3/`
(newly fetched raw caches allowed there, clearly separate from the Bybit data).
One command: `py alpha_library/alx3_external_replication/run.py`. Single code
commit **only after** the run is complete.
