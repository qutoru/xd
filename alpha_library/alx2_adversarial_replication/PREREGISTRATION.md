# ALX2 — Adversarial Replication of the Funding→Price alpha — PRE-REGISTRATION (LOCKED)

**Status:** LOCKED on 2026-07-12, before writing any ALX2 code or seeing any
ALX2 result. Branch `alpha/alx2-adversarial-replication` (from the ALX branch,
because the object under test is the *exact* ALX-validated code — which does not
exist on `main`). This document is committed **by itself** as the lock; code and
`REPORT.md` follow in a later commit. Nothing below may change after observing
results. Single run, no tuning. Assume the alpha is FALSE until it survives.

**This is the FINAL adversarial replication.** The objective is to decide whether
the alpha survives a hostile audit — not to open an endless sequence of new
tests. The battery is **fixed to exactly six tests (T1–T6)**. No additional
falsification test may be invented after seeing intermediate results.

---

## 1. Object under test — FROZEN (never modified)

The funding-ranked dollar-neutral daily book exactly as validated in
`alpha_library/alx_funding_price_validation/` (net Sharpe ≈ **+1.71 @ Med**).
Reused **verbatim** via `validate.backtest` / `weights_from_signal`:

- **Universe:** `SURVIVORSHIP_SAFE`, coverage-filtered (~39 majors), daily bars
  (1h close → daily last).
- **Signal:** `−(trailing 7-day mean of daily realized funding)`; daily realized
  funding = sum of settlements at hours {0, 8, 16} UTC.
- **Construction:** long top 30% / short bottom 30% by signal, dollar-neutral,
  equal-weight, gross 1, hold 1 day, rebalance 1 day, exec_lag 1 day.
- **Cost:** Med = 7.5 bps (0.00075) per unit turnover. **Net = price + funding −
  cost.**

**Frozen — NOT modifiable by any test:** signal, lookback (7d), ranking (top/bottom
30%), portfolio construction (dollar-neutral equal-weight gross 1), holding
period (1d), execution (lag 1d), costs (Med 7.5 bps baseline), universe.
Tests may apply *strictly harsher* assumptions (e.g. T5's liquidity-aware costs
that are never cheaper than 7.5 bps) as an attack; they may never make the
strategy cheaper, smarter, or otherwise improved. **No optimization, no parameter
search, no threshold search, no Sharpe improvement.** Every test is purely
falsification.

## 2. The fixed adversarial battery (exactly these six — no more may be added)

### T1 — Point-in-time funding reconstruction & alignment audit
The validated result reused the E7-built `funding_panel.parquet` (forward-filled
onto the 1h grid); its provenance was the one documented caution. T1 attacks it:
independently **re-fetch** funding history from the exchange and reconstruct the
daily realized-funding panel **strictly point-in-time** — each day's realized
funding is the sum of settlements whose settlement timestamp falls in that UTC
day at {0,8,16}, known only at/after settlement; the 7-day signal at day `t` uses
only funding settled through `t`; execution at `t+1`. Audit: (a) no future
settlement leaks into any day, (b) settlement hours are exactly {0,8,16},
(c) the reconstruction is not forward-filled from the future. Re-run the frozen
strategy on the freshly reconstructed panel.
**Falsifies if:** the alignment audit reveals any look-ahead/forward-leak, OR the
point-in-time net Sharpe @Med flips sign vs the reused panel or falls to ≤ 1.0.

### T2 — Temporal concentration (top-day / crisis dependence)
Is the Sharpe carried by a handful of days or one crisis window? Compute daily
net PnL and: (a) drop the best 5% of PnL days and recompute Sharpe; (b) exclude
the highest-market-volatility decile of days (crisis proxy) and recompute.
Report the PnL share of the single best day and best 5% of days as diagnostics.
**Falsifies if:** after dropping the best 5% of PnL days net Sharpe ≤ 0, OR
excluding the top-volatility decile net Sharpe ≤ 0.

### T3 — Leave-one-symbol-out concentration
Re-run the frozen strategy `N` times, each time removing exactly one symbol from
the universe (weights renormalize within the frozen rule). Report the full
distribution of net Sharpe @Med.
**Falsifies if:** the minimum leave-one-out net Sharpe ≤ 1.0 (i.e. removing some
single name collapses the alpha below the admission bar → dangerous concentration).

### T4 — Expanded factor attribution
Add three controls not in the ALX battery, each a factor-mimicking book built
with the **identical** construction: (1) 1-day reversal
(`short_term_reversal`, lookback 1); (2) downside beta (rolling beta of the
symbol on the market estimated on down-market days only); (3) idiosyncratic
volatility (rolling std of market-residual returns). Multivariate Newey-West OLS
of the frozen net return on the classic anomaly set {reversal, momentum, beta,
volatility, size} **plus these three** (excluding the alpha's own funding
variable, which is not a competing anomaly). Report R², all loadings, and the
unexplained intercept with its NW t. The with-funding variant is reported too.
**Falsifies if:** the unexplained intercept's Newey-West t < 2.0, OR the
annualized intercept is ≤ 0 (a new factor absorbs the alpha).

### T5 — Realistic execution with liquidity-aware transaction costs
Replace the flat 7.5 bps only with a **strictly harsher, liquidity-aware** cost
model (an attack, not a modification of the strategy). Pre-committed model, per
symbol `i`, per rebalance, on turnover `|Δw_i|`, at a fixed notional AUM =
**$10M**: `cost_i = 7.5 bps (taker) + 5 bps (spread buffer) + 10 bps ×
(trade_notional_i / (1% × ADV_i))`, where `trade_notional_i = |Δw_i| × AUM` and
`ADV_i` = median daily dollar volume (`turnover`) of symbol `i`. This is ≥ the
baseline everywhere and penalizes trading illiquid names.
**Falsifies if:** net Sharpe under this liquidity-aware model ≤ 1.0.

### T6 — Deflated Sharpe Ratio / multiple-testing adjustment
Account for selection across the whole research program. Compute the Deflated
Sharpe Ratio (Bailey & López de Prado 2014) on the frozen daily net returns:
`PSR(SR*) = Φ( (SR̂ − SR*)·√(T−1) / √(1 − skew·SR̂ + ((kurt−1)/4)·SR̂²) )`,
with SR in per-observation units, `T` daily observations, and `skew`/`kurt` of
the daily net returns. The deflated benchmark is the expected maximum Sharpe of
**N = 100** zero-skill trials of this sample length (a deliberately harsh trial
count covering the program's E1–E9 + AL-1 + config searches):
`SR* = √(V₀)·[ (1−γ)·Z_{1−1/N} + γ·Z_{1−1/(N·e)} ]`, `γ = 0.5772`,
`V₀ = 1/(T−1)` = the null sampling variance of the per-observation Sharpe.
**Falsifies if:** DSR ≤ 0.95.

## 3. Survivorship bias — LIMITATION ONLY (not a pass/fail test)
A true point-in-time *live* universe including delisted assets is **not**
available (we have only survivors). Survivorship bias is therefore documented as
a **limitation**, not scored as T7 and not a gate. The `SURVIVORSHIP_SAFE`
construction is conservative (excludes newly-listed pumps) but cannot recover
delisted names; the live-universe magnitude may be lower. This is recorded in the
REPORT's Limitations section and does not affect the verdict.

## 4. Pre-committed falsification bars are EXTERNAL standards
- **Sharpe > 1.0** = the Alpha-Library admission bar (same bar used in the E9
  pre-registration), not fit to any ALX2 result.
- **NW t ≥ 2.0** and **DSR > 0.95** are field-standard significance bars.
- **Sharpe > 0** for concentration robustness.
All bars are fixed here, before running any of T1–T6.

## 5. Stopping rule (pre-committed)
Run T1 → T2 → T3 → T4 → T5 → T6 **in order**. Stop at the first test that
falsifies. If the strategy survives all of T1–T6, **the investigation ends** and
the alpha graduates from *candidate* to **accepted Alpha Library component**. No
further adversarial test may be invented afterward **unless**: new data become
available, a concrete implementation bug is discovered, or an external
replication contradicts the results.

## 6. Final verdict — exactly one of two (no ambiguous conclusion)
- **PASS:** the alpha survives the entire adversarial battery T1–T6 and is
  admitted to the Alpha Library.
- **FAIL:** name the **first** test that falsified the alpha, and stop.

## 7. Deliverables
`alpha_library/alx2_adversarial_replication/` — `adversarial.py` (the six tests,
reusing the frozen ALX backtester), `run.py` (one command; runs T1–T6 in order,
stops at first failure, prints the single verdict), `PREREGISTRATION.md` (this
file, committed first as the lock), `REPORT.md` (after the single run);
`tests/test_alx2_adversarial.py` (known-answer unit tests: point-in-time panel
has no forward leak, liquidity cost ≥ baseline, DSR formula on a known input);
artifacts under `data/research/alx2/`. Reuses only the frozen ALX code and
neutral shared infrastructure. One command: `py alpha_library/alx2_adversarial_replication/run.py`.
Single code commit **only after** the run is complete.
