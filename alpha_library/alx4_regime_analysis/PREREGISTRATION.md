# ALX4 — Regime characterization + causal falsification + forward protocol — PRE-REGISTRATION (LOCKED)

**Status:** LOCKED before analysis code. Branch `alpha/alx4-regime-analysis` (off
`main`). Single run, no tuning. Object under test is the **frozen** ALX book, used
**verbatim** through `alpha_library.alx_funding_price_validation.validate.backtest`
(`-7d-mean funding`, top/bottom 30%, hold 1d, lag 1d, Med 7.5 bps). No parameter,
lookback, universe rule, ranking, threshold, holding, or cost may change.

## What this study IS and IS NOT
- **IS:** (1) a *characterization* of how ALX behaves across time and market
  regimes; (2) two *new, decisive falsification tests of the mechanism* that were
  NOT run before (funding cross-sectional shuffle placebo; funding→return
  direction); (3) a *locked forward-validation protocol* for future data.
- **IS NOT:** a re-run, re-parameterisation, or reinterpretation of the locked
  **ALX3 REJECTED** verdict. ALX3's binary outcome stands. This study cannot
  graduate ALX to the Alpha Library — only genuinely forward data (per §Forward)
  can. It can only answer: *does the effect's existence/mechanism survive cheap,
  decisive falsifiers, or is it an artifact?*

## Hypotheses
- **H1 (existence/mechanism):** ALX reflects a genuine funding→price
  *crowding/leverage-unwind* effect — persistently rich (crowded-long) funding
  predicts relative underperformance — that is (a) destroyed when the
  funding↔symbol linkage is broken, and (b) statistically independent of known
  price factors and of the funding *cash-flow* (carry).
- **H0 (null):** ALX is a statistical artifact — a random/known-factor
  repackaging or an engine bias — whose apparent edge does **not** depend on the
  specific funding-to-symbol assignment.

Start from the assumption H0 is true.

## Decisive falsifiers (NEW tests, PASS/FAIL committed in advance)

**P1 — Funding cross-sectional shuffle placebo (decisive).**
Each day, randomly permute the funding row across symbols (destroying the
funding↔symbol linkage) and rebuild the *frozen* book. Repeat N=500 shuffles
(seed=0). If ALX's edge is real and funding-driven, shuffling must destroy it.
- **PASS (H0 rejected)** iff: real net Sharpe **>** the 97.5th percentile of the
  shuffle distribution AND the shuffle-distribution **median |Sharpe| < 0.5**
  (i.e. shuffled books have no systematic edge).
- **FAIL (artifact, H1 rejected)** iff the real Sharpe lies **inside** the shuffle
  distribution (≤ 97.5th pct), i.e. a random funding→symbol map does as well.

**P2 — Directionality (funding→return, not return→funding).**
Pooled panel test of the frozen signal's predictive direction:
mean daily rank-IC of `signal(t-1) → ret(t)` (forward) vs the reverse-map
`ret(t-1) → funding-signal(t)`. H1 predicts the forward IC is the material one.
- **Diagnostic gate:** forward |mean IC| must be positive with Newey-West
  t ≥ 3 over the full history. Reverse mapping is reported for context. (Labelled
  diagnostic-leaning: a reverse-only relationship would be a red flag.)

**P3 — Factor independence (reproduce, decisive on repackaging).**
Correlation of ALX net with identically-built momentum(24)/reversal(3)/beta/
size(SMB)/volatility/market/funding-carry books; multivariate OLS R².
- **PASS** iff no single known-factor |corr| ≥ 0.7 AND OLS R² < 0.5 (a materially
  unexplained component remains). **FAIL** if a known factor (or carry) explains
  ≥ that share.

**Stop rule:** P1 is decisive for *existence of a funding-specific effect*. If P1
FAILS, H1 is falsified — report the artifact and stop weighting the mechanism.

## Diagnostics (descriptive only — cannot pass/fail, cannot tune)
These quantities were already observed in prior director-level analysis and are
reported here for the record, NOT as acceptance tests:
- Per-year Sharpe / CAGR / win-rate / maxDD / mean-IC (independent Binance).
- Rolling-180d net Sharpe and rank-IC.
- Regime-conditional net Sharpe & IC by **pre-defined, t-1-lagged** variables:
  market realized vol, funding-score dispersion, cross-sec return dispersion,
  trailing trend (bull/bear), breadth. (Pre-defined from the mechanism; NOT
  scanned/optimized. Reported as a table; NOT used to build a trading gate here.)
- Top-day removal sensitivity (drop best 1/2/5/10% days) and Gini/contribution.

Interpretation rule (locked): if no single pre-defined regime variable *gates*
the edge (all buckets same sign), we will NOT claim a working regime classifier.

## Data
Independent Binance USDT-M reconstruction already cached under
`data/research/alx3/cache` (built by ALX3 from raw REST, no project parquet).
40 ex-ante majors, daily, 2019-09 … 2026-07. To observe multi-year dynamics the
coverage-*intersection* universe rule is necessarily relaxed to **per-day
available names** (union + NaN-masking); this is the ONLY deviation, it touches
universe *coverage handling* (unavoidable over 7 years), and all **signal**
parameters stay frozen. Both the union number and ALX3's frozen-intersection
number are reported so the sensitivity is explicit.

## Forward-validation protocol (LOCKED for future data)
The only path that can upgrade ALX's status. Pre-committed now:
- **Data:** only bars generated **after** this lock date (2026-07-20). Binance +
  Bybit; OKX as its funding archive lengthens.
- **Spec:** fully frozen. Universe rule fixed to **coverage-intersection** (to
  remove the union/intersection ambiguity: in-sample this was +0.21 vs +1.47).
- **Metrics fixed in advance:** net Sharpe + block-bootstrap 95% CI; daily
  rank-IC + NW t; conditional Sharpe by market-vol tercile; maxDD; top-5%-day
  PnL share.
- **Confirm (→ upgrade to "regime alpha"):** forward net Sharpe > 0 with 95% CI
  **excluding 0** over a window spanning ≥1 bull and ≥1 bear/flat phase, AND IC
  positive in the bull/high-vol sub-sample.
- **Fail (→ abandon):** full-window CI includes 0; OR IC flips sign in a bull
  phase; OR a drawdown exceeds the historical worst without recovery.
- **Horizon:** ≥ 12–18 months (≥1 regime cycle; episodic concentration means the
  *effective* sample ≪ the day count).

## Single verdict (pre-committed, this study)
This characterization study returns one of:
- **Mechanism survives** — P1 PASS, P3 PASS, P2 diagnostic consistent → the
  funding-specific, factor-independent effect is corroborated; status stays
  **🟡 promising, pending the locked forward protocol** (NOT graduated).
- **Mechanism falsified** — P1 FAIL (edge survives shuffle) OR P3 FAIL (a known
  factor/carry explains it) → downgrade toward **🔴 insufficient evidence**.

No tuning. Every result — including a falsification — is recorded in REPORT.md.

## Deliverables
`alpha_library/alx4_regime_analysis/` — `characterize.py`, `run.py` (one command,
prints verdict), `PREREGISTRATION.md`, `REPORT.md`; `tests/test_alx4_regime.py`
(known-answer); artifacts `data/research/alx4/`. Reuses only the frozen ALX
`validate.backtest` + neutral shared `research.xsection.features` and
`alx3_external_replication` data reconstruction. Single commit when complete.
