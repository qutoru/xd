# ALX5 — Maker-execution / low-turnover viability of the funding→price book — PREREGISTRATION

**STATUS: LOCKED 2026-07-22 (owner sign-off).** Committed before any ALX5
experiment code exists and before any ALX5 backtest is run. Single run, no
tuning, per `CLAUDE.md`. From this point the spec in §3–6 is frozen; observing a
result does not reopen it.

Branch: `alpha/alx5-maker-execution` (created off the current tip, not `main`,
because the frozen infra this experiment *reuses* — ALX1
`alx_funding_price_validation/validate.py` and ALX3
`alx3_external_replication/{data_sources,replicate}.py` — does not exist on
`main`; `alpha_library/` is empty there. All *new* ALX5 code is isolated under
`alpha_library/alx5_maker_execution/`; no frozen alpha/parameters are edited.)

---

## 0. Motivation (what we already know — the prior)

A **descriptive** decile-lift diagnostic (scratchpad, not a locked result) on the
ALX3 independent Binance reconstruction established two things, and this
pre-registration is written *with that knowledge*, so its gate is set to be
**hard**, not to rubber-stamp a foregone conclusion:

1. **Selectivity-by-strength is FALSE.** Grouping name-days by funding-signal
   *magnitude* `|score|` shows **no** monotone strength→return relation
   (Spearman rho = −0.18, p = 0.63); the strongest decile has ~0 gross edge, the
   9th decile is the worst. "Take only the strongest" does not concentrate edge.
   → We do **not** pursue a strength gate. That lever is dead.
2. **The book is execution-dependent.** The signed L/S spread (long
   most-negative-funding, short most-positive) is **+6.9 bps/day gross**; under a
   full-round-trip-per-name charge it is **+2.9 bps/day after 4 bps maker** but
   **−4.1 bps/day after 11 bps taker**. The frozen portfolio survives 7.5 bps at
   MED only because signal persistence keeps turnover low.

So the genuinely open, non-trivial question is **not** "does maker beat 1.0"
(maker is cheaper than the MED at which ALX4 already showed +1.94 — that bar is
rigged-easy and we explicitly reject it). The open question is:

## 1. Hypothesis

**H1 (economic claim):** The funding→price cross-sectional edge is tradeable
under *honest maker execution* — i.e. the **frozen daily book** survives a
**realistic** post-only cost model that charges maker fees **plus
adverse-selection plus non-fill haircut**, remaining net-profitable and
fold-stable. The taker-executed book, by contrast, is expected to be
marginal-to-negative; the maker route is what makes it real.

**H0 (null — assume TRUE until rejected):** The apparent profitability is an
artifact of optimistic execution assumptions. Once post-only fills are charged
their real adverse-selection and non-fill penalty, the net edge collapses to
≤ the decisive bar (net Sharpe ≤ 1.0 **or** it is not fold-stable). Cheap
execution, not a durable premium, was carrying it.

**One hypothesis only** — the maker-**execution** viability of the *frozen* ALX
signal on its native daily book. We do **not** re-tune the signal, universe, k%,
lag, **or hold**. (Turnover reduction via a longer hold is a *separate* lever;
per "one experiment = one hypothesis" it is **not** bundled here — it is reported
descriptively in §4 and, if it looks live, becomes its own future experiment
`ALX6`. Bundling it would make a FAIL ambiguous between execution and hold.)

## 2. Universe / data

- ALX3 independent **Binance USDT-M** daily reconstruction (cached under
  `data/research/alx3/cache`), 40 majors, faithful intersection+98%-coverage
  window (identical to ALX3 Stage 2 / ALX4). No project parquet.
- Untouched **pre-2023** period (ALX3 Stage 3 window) used as the out-of-sample
  fold check.

## 3. Frozen signal (unchanged — reused verbatim)

`signal = -funding.rolling(7).mean()`, dollar-neutral equal-weight top/bottom
**30%**, execution **lag = 1 day**. Funding carry accrues to the held leg exactly
as in `validate.backtest`. **None of these are touched.**

## 4. The ONE thing under test: the execution/cost model (pre-committed)

The tested object is the **frozen daily book** (§3, hold = 1 day, lag = 1,
top/bottom 30% — *unchanged*). The only variable is the **cost model** applied to
its realized turnover:

| model | round-trip | role |
|---|---|---|
| taker | 11 bps | descriptive (expected to be weak — that's the point) |
| MED | 7.5 bps | descriptive anchor (must reproduce ALX4's ~+1.94) |
| ideal maker | 4 bps | descriptive ceiling (NOT the gate — rigged-easy) |
| **realistic maker** | **4 bps base + 3 bps adverse-selection = 7 bps on turnover, AND gross return scaled by φ = 0.80 (20% of intended edge missed on non-fills)** | **← THE DECISIVE GATE** |

The realistic-maker model is deliberately harsh — and this harshness is the whole
point, because we already know from ALX4 that at cheap cost the book looks good;
the open question is whether it survives *fill reality*. Post-only orders are
adversely selected (they fill when the trade is already going against you: +3 bps)
and miss a share of fast-moving winners entirely (φ = 0.80 haircut on gross).
Without the φ haircut, "maker 7 bps" is merely cheaper than MED and would pass
trivially — a rigged gate; φ is what makes survival genuinely uncertain. These
numbers are **locked now**; no adjustment after seeing results.

**Turnover / hold — DESCRIPTIVE ONLY (not gated, not the hypothesis).** A hold
ladder {1, 2, 3, 5} × the cost models above is reported purely to characterize
the turnover→net-cost tradeoff. It **cannot** change the ALX5 verdict and no
"best hold" may be selected from it. If a longer hold looks live, it is promoted
to a *separate* pre-registered experiment (ALX6), not folded in here.

## 5. Statistical gates — objective PASS/FAIL committed IN ADVANCE

Staged gauntlet with a hard stop between stages (research-ladder pattern). All
gated metrics are computed under the **realistic-maker** model on the **frozen
daily book** unless stated.

- **Stage A — machinery/anchor sanity (kill-test).** Reproduce the frozen daily
  book's net Sharpe at **MED** and confirm it matches ALX4 (~+1.94, tolerance
  ±0.15) — verifies the reused engine and panel are intact and not silently
  altered. Also confirm the book's rank-IC Newey-West t (lag 5) **≥ 3.0**,
  positive sign (the signal itself is present). If the anchor does not reproduce
  or IC t < 3, **STOP → FAIL** (something is broken; do not interpret maker
  numbers off a broken base).

- **Stage B — net viability under honest maker cost (decisive).** On the full
  faithful window: **net Sharpe > 1.0** under realistic-maker; **AND ≥ 4 of 6**
  equal walk-forward folds net-positive; **AND** the untouched **pre-2023** fold
  net-positive. Any one fails ⇒ **STOP → FAIL**.

- **Stage C — the maker story is real, not a leak (kill-tests).**
  1. **lag-0 vs lag-1 tell:** lag-0 must be conspicuously better (same-bar
     leakage sanity); the traded lag-1 result must not depend on look-ahead.
  2. **execution attribution:** the taker version *may* be ≤ 0 (expected); but
     the maker uplift must be explained **entirely by the cost/turnover delta**,
     verified by reconstructing net = gross − turnover·cost with the same gross.
     If maker "edge" appears from anywhere other than lower cost ⇒ FAIL.
  3. **adverse-selection escalation (robustness):** re-run with a **harsher**
     φ = 0.70 and +4 bps adverse; net Sharpe must stay **> 0** (not necessarily
     > 1). If a mild worsening of the fill assumption flips it negative, the
     maker advantage is too fragile to be real ⇒ FAIL.

**Objective verdict rule (locked):**
- **PASS (→ candidate for a forward maker-paper protocol):** Stage A t ≥ 3 **and**
  Stage B all-true **and** all three Stage-C kill-tests survive.
- **FAIL (→ H0 not rejected):** any stage fails. Recorded as a full-value result.

**No real money and no production graduation from this experiment.** A PASS only
authorizes drafting a *separate*, forward, maker-paper pre-registration (the plan's
Lever-3 step), whose positive out-of-sample window is the sole path toward sizing.

## 6. Single run, no tuning (binding)

The procedure in §4–5 is run **once** on the frozen daily book (hold = 1,
unchanged). φ = 0.80, +3 bps adverse, the gate thresholds, and the window are all
fixed above. Descriptive ladders (cost×hold) are diagnostics, never knobs.
Observing any result ends the freedom to change the spec; a failure *is* the
recorded outcome.

## 7. Deliverables

`run.py` (one command, stage-gated, prints the single verdict) +
known-answer unit tests under `tests/test_alx5_maker.py` (turnover-vs-hold
identity, cost-model monotonicity, φ-haircut arithmetic) + artifacts under
`data/research/alx5/` + `REPORT.md` (honest verdict) + `RESEARCH.md` update +
one squashed commit **after** completion.
