# ALX — Funding→Price alpha — FALSIFICATION PRE-REGISTRATION (LOCKED)

**Status:** LOCKED 2026-07-12, before code. Branch `alpha/alx-funding-price-validation`
(from `main`, which does NOT contain AL-1 code — the reimplementation is
independent by construction). Single run, no tuning. **Goal: determine whether the
alpha actually exists — not to prove it works, not to improve it.** Assume the
result is FALSE until proven otherwise.

## Object under test (frozen exactly as discovered; never modified)
The funding-ranked dollar-neutral daily book from AL-1, whose net Sharpe ≈ **+1.71
@ Med cost** was an unexpected byproduct (~82% of cumulative PnL from price):
- Universe: 39 survivorship-safe majors, daily bars (1h close → daily last).
- Signal: `−(trailing 7-day mean of daily realized funding)`; daily realized
  funding = sum of settlements at hours {0,8,16} UTC.
- Construction: long top 30% / short bottom 30% by signal, dollar-neutral,
  equal-weight, gross 1, hold 1 day, rebalance 1 day, exec_lag 1 day.

No parameter, feature, holding period, construction, or threshold may change.

## The experiment answers exactly three questions (Phase 1 — verification, decisive)

**A. Reproducibility.** Does an *independent* reimplementation (own backtester,
not AL-1's `carry.py`) reproduce the result? Triangulated against the shared
`portfolio.backtest_portfolio` engine and against AL-1's reported 1.71. All three
should agree in sign and magnitude.

**B. Implementation error / data leakage.** Is the result an artifact? Checks:
future-corruption invariance (no look-ahead), signal sign-reversal (Sharpe must
flip), random-signal placebo (the backtest must not manufacture Sharpe),
timestamp/alignment audit (return, funding, signal, execution all on the correct
day), and a lag-0-vs-lag-1 tell for same-bar leakage. Any evidence of leakage or a
bug ⇒ FAIL.

**C. Explanation by a known factor.** Is the return *almost entirely* explained by
an already-tested factor? Attribution against six factor-mimicking books built
with the identical construction: short-term reversal, momentum, market beta,
volatility, size, and raw (unsmoothed) funding level. Evidence: pairwise
correlations and a multivariate OLS reporting R², factor loadings, the intercept
(unexplained alpha) with its Newey-West t, and the unexplained-residual Sharpe.
If the return is almost entirely explained (intercept economically negligible
and/or statistically indistinguishable from zero, or a single factor accounts for
nearly all of it) ⇒ FAIL.

**Everything else is supporting evidence, not an acceptance test.** No pre-registered
numeric cutoffs (no t ≥ 3, no Sharpe ≥ 0.5, no correlation ≥ 0.7, no bootstrap-CI
rule). The verdict on A/B/C is a documented, honest judgment.

**If Phase 1 fails, stop immediately** — reject the alpha, do not run Phase 2.

## Phase 2 — characterization (descriptive only; runs ONLY if Phase 1 passes)
Purely to describe how the alpha behaves; it CANNOT reject the alpha, change the
strategy, or influence any parameter, and no optimization is permitted. Reported
axes: transaction costs (Low/Med/High), execution lag (1/2/3), walk-forward folds,
volatility regimes, year-by-year, drop BTC & ETH, top-liquidity subset, block
bootstrap. **A lower Sharpe under tougher assumptions is expected and is NOT a
failure.** The only question these inform: does the alpha *disappear completely*,
or does it *remain economically meaningful*?

## Single PASS criterion (pre-committed)
**The independently implemented strategy PASSES iff it reproduces the original
result (A), survives the leakage/implementation checks (B), and retains a
statistically and economically meaningful *unexplained* component after
controlling for known factors (C).** Otherwise it is REJECTED.

If it passes, the REPORT will explain WHY it exists economically before any
further work is proposed. Every failed hypothesis is documented regardless.

## Deliverables
`alpha_library/alx_funding_price_validation/` — `validate.py` (independent impl +
Phase 1 + Phase 2), `run.py` (one command; gates Phase 2 on Phase 1),
`PREREGISTRATION.md`, `REPORT.md`; `tests/test_alx_validation.py`; artifacts
`data/research/alx/`; single commit on the branch. Reuses only neutral shared
infrastructure (universe loader, metrics, Newey-West, factor features) — not
AL-1's carry code.
