# ALX9 — Risk-parity blend of funding + momentum — REPORT

**VERDICT: FAIL at Stage B (condition B1) → H0 not rejected — but by a whisker.**
The risk-parity blend lifts the book from +1.890 to **+2.016**, a real +0.126
Sharpe gain, but falls **0.024 short** of the locked +2.040 decisive bar. The
funding-only ALX5 book stands unchanged. Single run, no tuning, per
`PREREGISTRATION.md` (LOCKED 2026-07-23). Branch `alpha/alx9-riskparity-blend`.
Reproduce: `py alpha_library/alx9_riskparity_blend/run.py`.

---

## Result

| stage | condition | value | gate | result |
|---|---|---|---|---|
| **A** | funding net Sharpe | +1.890 | ALX5 +1.89 ± 0.15 | OK |
| A | momentum-alone Sharpe | +0.793 | > 0.5 | OK |
| A | corr(funding, momentum) | −0.090 | \|·\| < 0.30 | OK |
| **B1** | risk-parity combined Sharpe | **+2.016** | ≥ base+0.15 = **+2.040** | **FAIL** |
| B2 | folds combined ≥ funding | 4 / 6 | ≥ 4/6 | OK |

Avg realized blend weights: **funding 0.58 / momentum 0.42** (risk parity
correctly tilted toward the lower-vol, stronger funding sleeve). Stage C
(leakage, 200-seed placebo, no-leak) not evaluated per the stop-rule.

## What the numbers say

Risk parity did exactly what it should: it up-weighted the lower-vol funding
sleeve (0.58) over momentum (0.42), lifting the combined Sharpe well above ALX8's
naive equal-gross blend. The lift trajectory across the two locked blend
experiments:

| blend | combined Sharpe | lift over funding (+1.890) |
|---|---|---|
| ALX8 — 50/50 gross | +1.896 | +0.006 |
| **ALX9 — risk parity** | **+2.016** | **+0.126** |
| (in-sample tangency, look-ahead, NOT adoptable) | ~2.12 | +0.23 |

So the orthogonal momentum sleeve **does** add a real, economically meaningful
~7% Sharpe improvement once weighted sensibly — but **not** the +0.15
("meaningful", one anchor-tolerance band) lift the pre-registration demanded. By
the locked objective rule, **H0 is not rejected.**

## Protocol note — the FAIL stands, no retune, and a multiple-testing caution

The 60-day vol window, the inverse-vol rule, and the +0.15 bar were locked before
the run and correctly computed. 2.016 < 2.040 ⇒ FAIL. The temptation to nudge the
bar to +0.12, or to try a 45-day window, or to switch to a Sharpe-tilted weight
until the number crosses 2.040, is precisely the post-hoc tuning the protocol
forbids — **and here it compounds into a meta-risk worth naming explicitly:**

> We have now pre-registered **two** blend weightings (ALX8 50/50, ALX9 risk
> parity) on the **same** funding+momentum combination and the **same** data. Each
> is individually clean, but *continuing to spawn weighting variants until one
> clears 2.040* is a garden-of-forking-paths: with enough locked-but-serial
> attempts, one will cross the bar by chance. **This is where the in-sample search
> must stop.**

## Honest bottom line

- The momentum sleeve is **real and orthogonal** (established in ALX8, reconfirmed
  here), and a sensible risk-parity blend **genuinely lifts** the book by ~+0.13
  Sharpe. This is the most positive finding of the whole lever program — the
  effect is real, just under the pre-set "meaningful" threshold.
- By the locked gate it is a **FAIL**: the validated asset remains the
  **funding-only 40-major ALX5 book (net Sharpe +1.89)**.
- **Recommended next step is NOT a third weighting experiment.** Two honest
  in-sample attempts have bracketed the answer: the blend helps, marginally,
  sub-threshold. The only honest arbiter left is **out-of-sample reality** — a
  *forward paper* run of the risk-parity blend (locked as-is, +2.016 construction)
  alongside the funding-only book, judged on genuinely unseen data. If the forward
  window confirms the blend's edge and lower drawdown, that is real evidence; more
  in-sample re-weighting is not.

## Lever program — final scoreboard

| lever | outcome |
|---|---|
| 1 breadth | ✗ closed (ALX6 abort, ALX7 falsified) |
| 2 vol-targeting | ✗ no help |
| 3 hold | ✗ no help |
| 4 momentum sleeve | ingredient ✓ real+orthogonal; **blend lifts +0.13 but sub-threshold** (ALX8 50/50 ✗, ALX9 risk-parity ✗ by 0.024) |

The funding→price book stands alone as the one validated asset; the momentum
blend is a real-but-marginal enhancement whose fate belongs to a forward test, not
more in-sample tuning.
