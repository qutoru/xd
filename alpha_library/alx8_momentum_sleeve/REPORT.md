# ALX8 — Orthogonal momentum sleeve — REPORT

**VERDICT: FAIL at Stage B (condition B3) → H0 not rejected.** The funding-only
ALX5 book stands unchanged. Single run, no tuning, per `PREREGISTRATION.md`
(LOCKED 2026-07-23). Branch `alpha/alx8-momentum-sleeve`.
Reproduce: `py alpha_library/alx8_momentum_sleeve/run.py`.

**But this is the most encouraging FAIL of the four levers:** the two *hard*
scientific conditions both PASSED — the momentum sleeve is a **real, tradeable
alpha** and it is **genuinely orthogonal** to the funding book. Only the **locked
50/50 gross blend weighting** failed to clear the decisive lift bar. The problem
is now narrowed from "is there a second alpha?" (yes) to "how do we weight it?".

---

## Result

40-major panel, 2500 days. Funding turnover 0.221/day, momentum 0.279/day.

| stage | condition | value | gate | result |
|---|---|---|---|---|
| **A** | funding net Sharpe | **+1.890** | ALX5 +1.89 ± 0.15 | **OK** |
| A | funding rank-IC NW-t | +4.70 | ≥ 3.0 | OK |
| **B1** | momentum-alone net Sharpe | **+0.793** | > 0.5 | **OK** |
| **B2** | corr(funding, momentum) | **−0.090** | \|·\| < 0.30 | **OK** |
| **B3** | combined 50/50 net Sharpe | **+1.896** | ≥ base+0.15 = +2.040 | **FAIL** |
| **B4** | folds combined ≥ funding | 4 / 6 | ≥ 4/6 | OK |

Per the locked stop-rule, Stage C (leakage tell, 200-seed placebo, no-leak) was
not evaluated.

## What the numbers say

1. **Momentum IS a real alpha here.** Net Sharpe **+0.79** under the *same* honest
   realistic-maker cost (7 bps + φ = 0.80) that the funding book survives — despite
   higher turnover (0.28 vs 0.22/day). This is the first of the four levers whose
   ingredient is genuinely alive.
2. **It IS orthogonal.** Correlation **−0.09** — near-zero, slightly negative,
   reproducing ALX4's P3 measurement (−0.091) out of sample. The two sleeves are
   economically distinct (funding-crowding **reversal** vs price **momentum**) and
   statistically nearly independent.
3. **Yet the locked 50/50 gross blend barely moves the book:** combined **+1.896**
   vs funding **+1.890** — a real but tiny lift (and 4/6 folds), far under the
   +2.040 bar. **B3 FAIL → H0 not rejected.**

## Root cause — the FAIL is a WEIGHTING problem, not an ingredient problem

The decisive +0.15 margin was motivated in §0 by the portfolio identity
`Sharpe ≈ √(S₁² + S₂²)` — but **that identity assumes risk-optimal weighting**,
while the pre-registration **locked a 50/50 *gross* split**. Equal gross badly
**under-weights the stronger sleeve**: the funding book (S = 1.89) dominates
momentum (S = 0.79), so a 50/50 gross blend is close to ~half funding + half a
much weaker sleeve, which lands near +1.9, not +2.0+.

**Descriptive diagnostic (NOT a gate; in-sample / look-ahead, do not adopt):** the
full-sample tangency (optimal-weight) combined Sharpe is **~2.12** — above the
+2.04 bar. A no-look-ahead risk-parity blend (rolling inverse-vol) would fall
*between* the 50/50 result (1.90) and this in-sample optimum (2.12), so it **might**
clear the bar — but that is unproven and must not be assumed.

This is a milder cousin of the ALX6 lesson: **when a decisive bar is justified by
optimal-weighting math, the locked blend must use a matching principled weighting,
not equal-gross.**

## Protocol note — the FAIL stands, no retune

The 50/50 split, the +0.15 margin, and all thresholds were locked before the run
and correctly computed. Re-weighting the blend now to clear the bar is exactly the
post-hoc tuning the protocol forbids. **This FAIL is a full-value recorded
result.** The funding-only ALX5 book stands.

## Forward-looking lesson → ALX9 (a NEW experiment, not a retro-edit)

Unlike breadth (a closed dead end), the momentum sleeve is **alive and
orthogonal**, so the honest next step is a *separate*, pre-registered experiment
that fixes only the weighting:

- **ALX9 (proposed):** pre-register a **no-look-ahead risk-parity / vol-scaled
  blend** (each sleeve scaled by its trailing, lagged realized vol; blend the
  vol-normalized sleeves), locked before code, single run. Gate: combined
  no-look-ahead net Sharpe ≥ base+0.15 **and** fold-stable **and** it survives the
  same leakage/placebo kill-tests. This directly tests whether the *proven* second
  alpha lifts the book once weighted sensibly.

## Bottom line

- **Lever #4 is NOT dead** — it is the only lever with a live, orthogonal second
  alpha. What failed is the naive equal-gross blend, a fixable design choice.
- The validated asset remains the **funding-only 40-major ALX5 book (net Sharpe
  +1.89)** until a properly-weighted blend passes its own locked gate.
- Scoreboard: breadth ✗ (closed), vol-targeting ✗, hold ✗, momentum-sleeve ✗ at
  the blend step **but ingredient PASSES** → carried forward to ALX9.
