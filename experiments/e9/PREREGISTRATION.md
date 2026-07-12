# E9 — Cross-asset lead-lag — PRE-REGISTRATION (locked before any code)

**Status:** LOCKED on 2026-07-12, before writing experiment code or seeing any
E9 result. Nothing below may be changed after observing results. If the
hypothesis fails, that is the recorded outcome — no re-parameterisation, no
"one more variant".

E7 and E8 are CLOSED and IMMUTABLE. E9 reuses only *infrastructure*
(`crypto_signal_bot.research.xsection.{universe,ic,features.forward_target,
features.residualize,portfolio}`, `research.metrics`) — never their alpha,
parameters, thresholds, or conclusions. All new E9 code lives under
`experiments/e9/`.

---

## 1. Hypothesis (H1)

Cross-asset information diffuses with a measurable delay: the past return of a
leading crypto asset (BTC) predicts the **market-neutral** forward return of
other assets, and this prediction still holds after a one-bar execution delay.

Because a common "BTC up → everything up" move is removed by market-neutral
residualisation, the tradable claim is specifically about **differential**
diffusion: high-leader-beta followers under-react to a leader impulse at bar `t`
and catch up (in the cross-sectional residual sense) at bar `t+1`.

## 2. Null hypothesis (H0)

After enforcing a one-bar execution gap and realistic transaction costs, no
economically meaningful cross-asset lead-lag relationship survives
out-of-sample: the beta-scaled leader impulse has no cross-sectional predictive
power for follower residual returns beyond bar `t` (IC indistinguishable from
zero, or present only at the immediate next tick and gone with one more bar of
delay — i.e. untradeable microstructure), and no long/short construction clears
the Stage C pass bar.

## 3. Economic mechanism (why the edge *could* exist)

Gradual information diffusion (Hou 2007 lead-lag in equities; Lo-MacKinlay 1990
cross-autocorrelation; documented for crypto in several 2020–2022 studies). BTC
is the undisputed price-discovery hub of crypto; information and flow hit BTC
first and propagate to alts as slower participants and cross-market arbitrageurs
react. If propagation takes longer than one bar, the leader's realised move
carries information about followers' *next-bar* residual return. The
cross-sectional loading (beta to the leader) determines who lags and by how
much, so a beta-scaled leader impulse is the natural cross-sectional signal.

**This is a genuinely different mechanism from E1–E8** — not single-asset
autocorrelation (E1–E5), not same-bar cross-sectional ranking of an asset's own
past (E7/E8 reversal). See §11 for the explicit "not reversal in disguise" guard.

## 4. What would falsify H1

Any one of:
- Stage B: the beta-scaled leader impulse produces IC not distinguishable from
  zero (fails B1/B2), or its sign is unstable across folds (fails B3).
- **Decisively — Stage B4 (the lead-lag kill-test):** the IC exists at the
  immediate next bar but does **not** survive one additional bar of execution
  delay. That would mean the effect is stale-price / non-synchronous-trading
  microstructure, not tradable diffusion — the same wall E7 hit.
- Stage C: even if Stage B passes, no configuration clears net Sharpe > 1 with
  ≥ 4/6 walk-forward folds positive at realistic cost, or capacity is
  unacceptable.

## 5. Universe & leader (pre-registered, not tunable)

- **Follower universe:** `SURVIVORSHIP_SAFE` (the fixed ex-ante major list in
  `universe.py`, coverage-filtered from data) **minus the leader**. BTC is the
  leader and is excluded from both the tradable cross-section and the
  market-mean used for residualisation, so the leader cannot leak into its own
  target.
- **Leader:** `BTCUSDT` — a **single, fixed** leader chosen a priori as crypto's
  price-discovery hub. We deliberately test exactly one leader specification to
  avoid multiple-testing across leader sets. **Leader set will NOT be changed
  after results** (per restriction).

## 6. Timeframe & data

- **Interval:** 1h (primary and only). Rationale: E7 showed 1h has enough
  cross-sectional breadth (N_eff ≈ 26) and observation count (~26k bars over
  3y) to power the t ≥ 3 gate; lead-lag/diffusion is a short-horizon effect that
  decays within days (Hou 2007), so daily (E8-style) is the wrong frequency and
  is not tested.
- **Data:** existing Bybit 1h parquet for the survivorship-safe majors. **No new
  datasets.** ~26k bars, 2023–2026.

## 7. Feature definition (the lead-lag signal — locked)

Let `r_i(t)` be per-bar log returns of follower `i`, `r_L(t)` the leader (BTC)
per-bar log return.

- **Leader beta (loading):** `beta_i(t)` = rolling OLS slope of `r_i` on
  contemporaneous `r_L` over a trailing window `W = 168` bars (7 days — a
  standard market-beta window; **not tuned**). Uses only data up to and
  including bar `t`.
- **LL1 — impulse (primary):**
  `f1_i(t) = beta_i(t) · r_L(t)`, then cross-sectionally demeaned across
  followers at each `t`. Reading: high-beta followers, right after a leader up-move,
  are predicted to have positive next-bar residual return (they catch up).
- **LL2 — sustained (secondary):**
  `f2_i(t) = beta_i(t) · [r_L(t) + r_L(t−1) + r_L(t−2) + r_L(t−3)]` (4-bar
  cumulative leader move), cross-sectionally demeaned. Tests whether diffusion is
  a multi-bar rather than single-bar phenomenon.

Exactly **two** features, both fixed now. No other feature forms, windows, or
leader-return horizons will be tried. Both are computable at the close of bar
`t` (no lookahead); the beta window and demeaning use only information through `t`.

## 8. Target

Market-neutral forward return via the existing `features.forward_target`
(residualised sum of returns over `t+1 .. t+h`), computed over the **follower**
panel (leader excluded from the mean). **Primary horizon h = 1** (diffusion is
short-horizon; the tradable claim is next-bar). Horizons {1,2,4,8,24} are
reported for IC-decay **diagnostics only** — the gates are evaluated at h = 1.

## 9. Execution delay

- **Structural:** the feature at decision-close `t` is traded at `t+1`
  (`exec_lag = 1`) — this one-bar gap is intrinsic to the hypothesis and to H0.
- **Kill-test (B4):** shift the target one further bar (predict `t+2` from the
  `t` signal). If IC does not survive this, the effect is microstructure, not
  diffusion → falsified. This is the decisive gate for E9.

## 10. Portfolio construction (Stage C)

Reuse `portfolio.backtest_portfolio` verbatim: cross-sectional rank each
rebalance, long top-K% / short bottom-K% of followers, dollar-neutral,
equal-weight, gross = 1, overlapping Jegadeesh-Titman tranches. No leverage, no
vol targeting, no sizing. Feature ranked = LL1 (primary); LL2 reported alongside.

**Pre-registered Stage C grid (declared for E9; values match the harness grid
for comparability, locked now):**
- HOLD ∈ {1, 2, 4, 6, 12, 24} bars
- REBAL ∈ {1, 2, 4} bars (REBAL ≤ HOLD)
- K ∈ {10%, 20%, 30%}
- exec_lag ∈ {1, 2}
- Representative deep-dive config: HOLD=4, REBAL=1, K=20%, exec_lag=1.

## 11. Cost assumptions (per unit of turnover |Δw|)

- Low = 2 bps (0.0002), **Med = 7.5 bps (0.00075, realistic taker)**,
  High = 15 bps (0.0015). Decisive evaluation at **Med**.
- Stage A round-trip anchor uses `FEE_RATE = 0.00055`.

## 12. Statistical procedure & gates

**Stage A — cost-anchored dispersion (signal-agnostic sanity):**
median per-bar cross-sectional dispersion of follower residual returns ≥ 2 ×
round-trip cost (2 × 2·FEE_RATE). (Expected to pass — same panel family as E7;
Stage A is not the discriminating gate.)

**Stage B — rank-IC diagnostics at h = 1 (hard gates, well-powered at 1h):**
- **B1:** `|mean IC| ≥ IR_target / (TC · √BR)`, with `IR_target = 1`, `TC = 0.5`,
  `BR = N_eff · (bars_per_year / h)`. (Fundamental Law; same formula as E7.)
- **B2:** Newey-West IC t-stat ≥ 3 (`nw_lag = max(5, h)`). (Harvey-Liu-Zhu 2016.)
- **B3:** binomial p(sign of IC consistent across 6 chronological folds) < 0.05.
- **B4 (decisive):** the 1-bar-gap IC still satisfies B1 **and** B2.
- Reported diagnostics (not gates): IC-IR, IC autocorrelation, per-fold IC,
  volatility-regime IC breakdown, IC decay across {1,2,4,8,24}, and — as the
  explicit **"not reversal in disguise"** guard — the cross-sectional correlation
  between each LL feature and the E7 `short_term_reversal` feature, plus the LL
  IC **sign** (diffusion predicts the catch-up/positive sign; the reversal sign
  would be the opposite). LL features use the leader's return and a beta loading;
  they do not call `short_term_reversal`.

**Stop rule:** if **neither** LL feature passes all of B1–B4, H1 is falsified at
Stage B — STOP, do not proceed to Stage C, do not add features or tweak windows.

**Stage C — survival under costs (only if Stage B passes):**
full grid at all three costs; walk-forward (6 folds), year-by-year, regime, and
capacity deep-dives at Med cost, exactly as the harness does.

## 13. Objective PASS / FAIL (pre-committed)

**E9 PASSES only if ALL hold:**
1. At least one LL feature passes **all** Stage B gates B1–B4.
2. Stage C: **best net Sharpe > 1.0 at Med cost.**
3. Stage C: net Sharpe **> 0 in ≥ 4 of 6 walk-forward folds** (Med, rep config).
4. **Capacity ≥ $10M** at 1% participation of universe dollar volume.

Otherwise **FAIL → H0 not rejected → record and STOP.** No threshold will be
changed after observing results.

## 14. Deliverables

- `experiments/e9/` — `features_leadlag.py` (isolated E9 alpha), `run.py`
  (one-command Stage A→B→[C], writes artifacts + prints verdict).
- `experiments/e9/REPORT.md` — written after the single run.
- `tests/test_e9_leadlag.py` — known-answer unit tests for the LL feature
  (constant-beta recovery, demeaning sums to zero, no-lookahead).
- Artifacts under `data/research/e9/`.
- RESEARCH.md updated with the E9 block.
- One-command reproducibility: `py experiments/e9/run.py`.
- Single git commit **only after** the experiment is complete.

## 15. Skeptic's stance

The prior is against H1: lead-lag in *liquid* majors is largely arbitraged, and
the most likely positive-looking result is a stale-price artifact that B4 is
designed to kill. A clean falsification (especially "IC at t+1 but dead at t+2")
is a full-value outcome: it would let us conclude, jointly with E1–E8, that
directional predictability in liquid crypto majors — from an asset's own past
*or* from the leader's past — lives only in the intraday microstructure band a
taker cannot capture.
