# E9 — Cross-asset lead-lag — REPORT

**Verdict: FAIL. H1 (tradable cross-asset diffusion) is FALSIFIED.**
Run once, no tuning, exactly per `PREREGISTRATION.md`. Reproduce: `py experiments/e9/run.py`.

Data: 38 follower majors (BTC = leader, excluded), 1h, 26 398 bars (2023–2026).

---

## Stage A — cost-anchored dispersion: PASS

Median cross-sectional dispersion of follower residual returns = **0.4512%** ≥
2× round-trip cost (0.2200%). Effective breadth **N_eff = 27.1** (PC1 = 10.3%),
so the Fundamental-Law threshold is **|IC| ≥ 0.0041** at h = 1. (Signal-agnostic
sanity gate — as expected, there is dispersion to trade.)

## Stage B — rank-IC at h = 1: gates pass on |IC|, but **the SIGN falsifies the hypothesis**

| feature | mean IC | NW t | folds (binom p) | regimes | gap-IC (t) | corr w/ reversal | B1–B4 |
|---|---|---|---|---|---|---|---|
| lead_lag_impulse (LL1) | **−0.0158** | −8.24 | 6/6 neg (0.016) | all neg | −0.0112 (−5.70) | −0.089 | ✅✅✅✅ |
| lead_lag_sustained (LL2) | **−0.0125** | −6.60 | 6/6 neg (0.016) | all neg | −0.0087 (−4.61) | −0.143 | ✅✅✅✅ |

The gates B1–B4 test **|IC|** (sign-agnostic, as pre-registered from the E7
formulation), so they read "PASS". **But the pre-registered sign diagnostic is
decisive: H1 (diffusion) predicts a POSITIVE IC — high-beta followers under-react
to a leader up-move and catch up next bar. The observed IC is NEGATIVE at every
horizon, in every fold, in every volatility regime.** A negative IC means the
opposite of catch-up: followers that just moved *with* BTC subsequently
*reverse*. That is not lead-lag diffusion — it is the cross-sectional reversal
already established in E7, re-expressed through the market factor.

Mechanically this is expected in hindsight: `beta_i · r_BTC(t)` is, by
construction, an estimate of the *systematic* part of follower `i`'s own recent
return. Ranking on it and predicting the residual forward return therefore
recovers the same short-term reversal E7 found — with the reversal sign. The
non-zero (negative) correlation with the E7 `short_term_reversal` feature
(−0.09 / −0.14) confirms the structural overlap.

**IC decay (LL1):** −0.0158 → −0.0175 → −0.0128 → −0.0081 → −0.0061 over
h = 1,2,4,8,24 — the same fast intraday decay as E7 reversal, no evidence of a
distinct multi-bar diffusion channel. The B4 kill-test does *not* kill it (gap-IC
stays significant), exactly because it is the (robust) reversal effect, not
stale-price microstructure — but robust with the wrong sign for H1.

## Stage C — survival under costs: FAIL on every pre-committed criterion

Because |IC| cleared Stage B, the pipeline proceeded to Stage C on LL1 (as
pre-registered). Traded as the hypothesis specified (long high feature):

- **Best GROSS Sharpe over the whole grid = +0.93** (H12/R4/K30%/lag2) — barely
  positive and only at longer holds; 87/270 configs gross > 0.
- **Best NET Sharpe @ Med = −1.71; 0 / 90 configs net > 0** at the realistic cost.
- **Walk-forward (rep config @ Med): −19.3, −16.2, −15.4, −15.7, −10.9, −17.5 → 0 / 6 folds > 0.**
- **Capacity ≈ $6.9M** at 1% participation — below the $10M gate.
- Cost drag +32% to +61% annualised: the same high-turnover wall that killed E7.

**Pre-committed PASS bar = (best net Sharpe > 1.0 @ Med) AND (WF net > 0 in ≥ 4/6)
AND (capacity ≥ $10M). Result: −1.71 / 0-of-6 / $6.9M → FAIL on all three.**

---

## Conclusion (information gain)

**H1 is falsified on two independent grounds:**
1. **Mechanism (Stage B sign):** the only statistically significant
   cross-sectional predictability from the leader's move has the **reversal
   sign, not the diffusion sign**. There is no evidence of tradable lead-lag
   *catch-up*; the beta-scaled leader impulse merely rediscovers E7's
   cross-sectional reversal through the market factor.
2. **Economics (Stage C):** even taken at face value the signal does not survive
   realistic costs — deeply negative net Sharpe, 0/6 folds, sub-gate capacity —
   the identical turnover wall as E7's taker execution.

This is a clean, high-information negative result. Combined with E1–E8 it lets us
state a strong joint claim: **directional cross-sectional predictability in
liquid crypto majors — whether from an asset's own past (E7) or from the
leader's past (E9) — exists only as the intraday short-term-reversal /
liquidity-provision effect, which a taker cannot monetise.** The lead-lag
diffusion channel, if it exists in majors, is not separable from reversal and is
not tradable here.

**Stop rule honoured: E9 FAILS the pre-committed bar → STOP.** No re-parameterisation,
no leader-set search, no added features. Per the backlog, the only economically
distinct continuations remain (a) maker/liquidity-provision execution (needs L2
data — a project, not an experiment) and (b) non-directional volatility/regime as
a Sharpe overlay for any future edge.
