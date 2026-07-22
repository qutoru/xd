# ALX6 — Breadth: wider PIT universe — REPORT

**VERDICT: FAIL at Stage A (anchor / machinery kill-test) → H0 not rejected.**
The 40-name ALX5 book stands unchanged; breadth does **not** graduate. Single
run, no tuning, per `PREREGISTRATION.md` (LOCKED 2026-07-22, before any ALX6
book code). Branch `alpha/alx6-breadth-universe`.
Reproduce: `py alpha_library/alx6_breadth/run.py`.

Per the locked stop-rule ("if Stage A fails, STOP; do not interpret maker numbers
off a broken base"), **Stage B and Stage C were not evaluated.** The expanded /
new-names Sharpe is deliberately not reported — the base it would be measured
against is inconsistent with ALX5, so any such number is uninterpretable.

---

## What happened

The fetched candidate universe (119 with data) was run through the **locked**
PIT-admission + liquidity screen (onboard+30d, ≥252 eligible days, median
eligible-window quote volume ≥ $5M):

| quantity | value |
|---|---|
| admitted universe N | **79** |
| — of which original-40 majors | **28** |
| — new names | 51 |
| cost tiers (7 / 12 / 20 bps) | 49 / 18 / 12 |

**Stage A — orig40 tiered net Sharpe = +1.140** (anchor ALX5 +1.89 ± 0.15 → need
≥ +1.74). **MISMATCH → FAIL.** (Rank-IC Newey-West t = +3.81 ≥ 3 did pass — the
signal is present; the failure is purely the anchor reproduction.)

## Root cause — a design flaw in the locked spec, correctly caught by the kill-test

The Stage-A kill-test exists to verify the pipeline reproduces the frozen base
before any expanded number is trusted. It fired correctly, exposing a flaw in the
pre-registration's own §4 assumption:

> "The 40-name majors sit in the ≥ $50M tier, so the baseline is charged exactly
> as in ALX5 (7 bps) — the anchor must reproduce."

This was **wrong on two counts**, both discovered only by running:

1. The **$5M median-ADV screen excludes 12 of the 40 majors outright** (e.g. the
   smaller/older majors whose median lifetime dollar-volume fell below $5M) — it
   does not merely re-tier them. So the pipeline's "orig40" is a **28-name**
   subset, not 40.
2. Dropping 12 names **collapses the baseline's breadth** (and charges the
   survivors tiered 12/20 bps where they fall below $50M), pushing net Sharpe to
   **+1.14** — far under the +1.74 floor.

Because the base does not reproduce ALX5, the expanded universe cannot be
compared against it on a like-for-like footing, and the experiment correctly
stops. **H0 is not rejected: we have no valid evidence that wider breadth raises
the net Sharpe.**

## Protocol note — the FAIL stands, no retune

Per `CLAUDE.md`: observing the outcome ends the freedom to change the spec. The
screen threshold, the anchor, and the ±0.15 tolerance were locked before the run,
correctly computed, and the kill-test triggered exactly as designed. The
temptation to "just exempt the majors from the screen" or "lower $5M to $1M" is
precisely the post-hoc tuning the protocol forbids. **This FAIL is a full-value
recorded result.**

## Forward-looking lesson (applies to FUTURE pre-registrations only)

*Does not relitigate this locked FAIL.* When a breadth / universe-expansion
experiment reuses a smaller reference universe as its Stage-A anchor, the
**anchor baseline and the new admission screen must be decoupled**, or they can
contradict each other and abort the run before the actual hypothesis is tested.
Concretely, a future design (ALX7) should either:

- **force-include the reference universe** (the original 40) in the baseline
  regardless of the new liquidity screen, so Stage A reproduces by construction
  and the screen governs only the *incremental* names; or
- **anchor Stage A to a screen-consistent baseline** (re-derive the reference
  book on exactly the names that pass the screen, and set the reproduction target
  from that, not from the differently-constructed ALX5 number).

Either decoupling would let the breadth hypothesis actually reach Stage B. That
is a design change for a **new** experiment, never a retro-edit of ALX6.

## Bottom line

- The wider-universe breadth question remains **open** — this run did not answer
  it; it aborted on a self-inflicted anchor contradiction.
- The validated book is **still the 40-name ALX5 realistic-maker book (net Sharpe
  +1.89)**. Nothing here changes it.
- Cost so far: honest. We spent one locked run to learn the screen and the anchor
  were in tension. The re-designed breadth test (ALX7) is the clean way to
  actually measure whether 40 → ~80 helps — on its own branch, its own lock.
