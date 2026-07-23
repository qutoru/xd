# ALX9 — Risk-parity blend: does no-look-ahead vol-weighting make the momentum sleeve lift the book? — PREREGISTRATION

**STATUS: LOCKED 2026-07-23 (owner sign-off).** Committed before any ALX9
experiment/book code exists and before any ALX9 backtest is run. Single run, no
tuning, per `CLAUDE.md`. From this point the spec in §3–6 is frozen; observing a
result does not reopen it. **Start from the assumption the blend does NOT clear
the bar (risk-parity weights by vol, not Sharpe, so it may not tilt enough toward
the stronger funding sleeve).**

Branch: `alpha/alx9-riskparity-blend` (off the current tip, not `main`, for the
same reason as ALX5–8: the frozen infra it reuses — ALX1 `validate.backtest`, ALX3
`data_sources`/`replicate`, ALX4 `characterize`, ALX5 `execution`, and the ALX8
neutral sleeve mechanics `alx8_momentum_sleeve.momentum` — does not exist on
`main`). All *new* ALX9 code is isolated under
`alpha_library/alx9_riskparity_blend/`; no frozen alpha/parameters edited.

---

## 0. Why ALX9, and its honest prior

ALX8 established, on a valid base, that the **momentum sleeve is a real alpha**
(net Sharpe +0.79) and is **orthogonal** to the funding book (corr −0.09), but the
**locked 50/50 gross blend** lifted the book only to +1.896 vs the +2.04 bar —
because equal *gross* under-weights the stronger funding sleeve. ALX8's REPORT
recorded the fix: the +0.15 bar is motivated by `√(S₁²+S₂²)`, which assumes a
*risk-optimal* weighting, so the blend must use a principled weighting, not equal
gross.

ALX9 tests the **one** principled, **no-look-ahead** weighting that is clearly not
post-hoc tuning: **risk parity (inverse trailing realized vol).** It needs only
volatilities (robust), not return forecasts (which would be look-ahead-prone /
tuning). **Honest caveat, stated before the run:** risk parity weights by vol, not
by Sharpe, so it does **not** deliberately tilt toward the higher-Sharpe funding
sleeve; it may land near or below the 50/50 result. A Sharpe-optimal tilt (the
in-sample tangency ~2.12 diagnostic) is explicitly **not** used, because
estimating it out-of-sample is a separate, much stronger claim. So a FAIL here is
a genuine, informative possibility — it would mean the proven second alpha does
**not** lift the book under honest, non-tuned weighting.

## 1. Hypothesis

**H1 (economic claim):** Blending the frozen momentum sleeve with the frozen
funding book by **no-look-ahead risk parity** (each sleeve weighted ∝ 1/its
trailing lagged realized vol) yields a combined realistic-maker net Sharpe
**meaningfully above the funding book alone (+1.89)** — the diversification
benefit of the proven orthogonal sleeve is realized once the sleeves are
risk-balanced rather than gross-balanced.

**H0 (null — assume TRUE):** The risk-parity blend does **not** meaningfully beat
+1.89, because inverse-vol weighting does not tilt toward the higher-Sharpe
funding sleeve (and may up-weight the weaker momentum sleeve), so the combined net
Sharpe stays ≤ base+0.15 and/or is not fold-stable.

**One hypothesis only** — the value of a **risk-parity** (vs gross) blend of the
two frozen sleeves. Both signals, the universe, and the cost model are unchanged
from ALX8.

## 2. Universe / data

Identical to ALX8: the frozen 40-major ALX4/ALX5 Binance USDT-M panel
(2019-09-08 … 2026-07-12, cached). No breadth.

## 3. Frozen signals (both reused verbatim — no tuning)

- **Funding sleeve:** `-funding.rolling(7).mean()`, top/bottom 30%, lag 1, hold 1.
- **Momentum sleeve:** `relative_strength(logret, 24)`, top/bottom 30%, lag 1,
  hold 1. Both exactly as in ALX8; lookbacks not touched.

## 4. The ONE thing under test: the blend weighting (locked)

Everything is the ALX5 realistic-maker cost (7 bps + φ = 0.80,
`gross = price + funding carry`). The **only** change from ALX8 is how the two
sleeves are combined:

**Risk-parity blend (LOCKED):**
- For each sleeve, estimate trailing realized vol on its own realistic-maker net
  return over a **60-day** rolling window, **lagged by 1 day** (uses only
  information up to t−1 → no look-ahead). Window = 60 is a standard vol horizon,
  fixed now, not searched.
- Per-day raw weights `a_t ∝ 1/vol_f(t−1)`, `b_t ∝ 1/vol_m(t−1)`; normalize so
  `a_t' + b_t' = 1` (gross-preserving, directly comparable to ALX8's 50/50).
- During the 60-day warm-up (vol undefined), default to **50/50**.
- `combined_applied[t] = a_t'·w_funding[t] + b_t'·w_momentum[t]` (both lag-1);
  price, funding carry, and turnover computed on `combined_applied` (netting),
  then realistic-maker cost + φ applied.

The 60-day window, the 1-day lag, the inverse-vol rule, the gross-preserving
normalization, and the 50/50 warm-up are **locked now**; no adjustment after
seeing results.

## 5. Statistical gates — objective PASS/FAIL committed IN ADVANCE

Staged gauntlet, hard stop between stages.

- **Stage A — anchor / machinery + ingredient re-check (kill-test).** The funding
  sleeve reproduces **ALX5 +1.89 (±0.15)**, rank-IC NW-t(5) ≥ 3. **And** the ALX8
  facts must still hold on this run: momentum-alone net Sharpe **> 0.5** and
  `|corr(net_funding, net_momentum)| < 0.30`. If any fails, **STOP → FAIL** (base
  or ingredient not intact). Call the funding value `base_sharpe`.

- **Stage B — the risk-parity blend lifts the book (decisive).** Both must hold:
  1. risk-parity combined net Sharpe **≥ `base_sharpe` + 0.15**;
  2. combined net Sharpe **≥** the funding book's in **≥ 4 of 6** equal folds.
  Either fails ⇒ **STOP → FAIL** (H0 not rejected).

- **Stage C — real diversification, not leakage / book-averaging (kill-tests).**
  All three must survive:
  1. **leakage tell:** momentum lag-0 gross Sharpe conspicuously **>** lag-1.
  2. **random-sleeve placebo:** replace momentum with a random sleeve (matched
     construction, lag 1), blended by the **same risk-parity rule**, **200 seeds**;
     the real combined net Sharpe must exceed the **97.5th percentile** of the 200
     random-combined Sharpes.
  3. **PIT / no-look-ahead:** funding future-corruption invariance on the combined
     book holds bit-exact before the corruption point.

**Objective verdict rule (LOCKED):**
- **PASS (→ the risk-parity funding+momentum blend graduates as the research
  book):** Stage A all-true **and** both Stage-B conditions **and** all three
  Stage-C kill-tests.
- **FAIL (→ H0 not rejected):** any stage fails. The funding-only ALX5 book stands
  unchanged. Recorded as a full-value result.

**No real money / no production graduation.** A PASS only authorizes the blend as
the research book and a separate forward-paper step before any sizing.

## 6. Single run, no tuning (binding)

Run **once**. The 60-day vol window, 1-day lag, inverse-vol rule, warm-up, the
> 0.5 / < 0.30 / +0.15 thresholds, the 200-seed placebo, and the fold count are
all fixed above. Descriptive outputs (realized average weights, Sharpe-vs-window
sensitivity) are diagnostics, never knobs. Observing any result ends the freedom
to change the spec; a failure *is* the recorded outcome.

## 7. Deliverables

`run.py` (one command, stage-gated, one verdict) + known-answer unit tests under
`tests/test_alx9_riskparity.py` (inverse-vol weight identity, gross-preserving
normalization, warm-up 50/50 fallback, no-look-ahead lag of the vol estimate) +
artifacts under `data/research/alx9/` + `REPORT.md` (honest verdict) + one
squashed commit **after** completion.
