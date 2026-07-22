# ALX7 — Breadth (decoupled anchor) — REPORT

**VERDICT: FAIL at Stage B → H0 not rejected.** Adding PIT-admitted,
liquidity-screened new names to the frozen 40-major book does **not** raise net
Sharpe — it **lowers** it, and the new names carry **no** funding→price signal of
their own. The 40-major ALX5 book stands unchanged; breadth does **not**
graduate. Single run, no tuning, per `PREREGISTRATION.md` (LOCKED 2026-07-23).
Branch `alpha/alx7-breadth-decoupled`.
Reproduce: `py alpha_library/alx7_breadth/run.py`.

**Unlike ALX6, this run was VALID:** the decoupled-anchor design worked — Stage A
reproduced the ALX5 base — so the experiment actually **reached and answered** the
breadth hypothesis instead of aborting on a self-inflicted contradiction.

---

## Result

Universe: 40 majors (reference, 7 bps) **+ 51 admitted new names** (tiered) =
**91** expanded. New-name cost tiers: 24 @7bps, 15 @12bps, 12 @20bps.

| stage | metric | value | gate | result |
|---|---|---|---|---|
| **A** | 40-major net Sharpe | **+1.885** | ALX5 +1.89 ± 0.15 | **OK** |
| A | rank-IC NW-t(5) | +4.70 | ≥ 3.0 | OK |
| **B1** | expanded net Sharpe | **+1.319** | ≥ base+0.15 = +2.035 | **FAIL** |
| **B2** | new-only sub-book Sharpe / IC-t | **−0.072 / −1.37** | net>0 & t≥2 | **FAIL** |
| **B3** | folds expanded ≥ base | **1 / 6** | ≥ 4/6 | **FAIL** |

Per the locked stop-rule, Stage C was not evaluated (Stage B already decisive).

## What the numbers say — a clean, strong falsification

1. **The decoupling fixed ALX6.** Keeping the 40 majors as the untouched
   reference at their ALX5 rate reproduced **+1.885** (anchor OK, IC t = 4.70).
   The machinery is sound; the base is valid; the breadth question was genuinely
   tested.
2. **Adding names HURTS.** Expanded net Sharpe **falls from +1.885 to +1.319** —
   the wider universe is *worse* than the majors alone, the opposite of the
   Fundamental-Law hope. It misses the +2.035 bar by a mile, and beats the base
   in only **1 of 6** folds.
3. **The new names carry no edge.** Their standalone book has Sharpe **−0.072**
   and rank-IC Newey-West t **−1.37** (slightly *negative*). The funding→price
   crowding-reversal signal simply **does not transfer** to the newer / thinner
   names — equal-weighting them into the top/bottom-30% legs dilutes the
   concentrated major edge with noise.

**H0 is not rejected — decisively.** The wider universe does not raise the net
Sharpe; it lowers it. This is a full-value result: it closes the breadth lever.

## Why economically (the mechanism explanation)

The funding→price edge is a **crowding-reversal / funding-carry premium** that
lives where funding is an informative, liquid signal — the **established
majors** with deep perp markets and mature positioning. On the added names
(overwhelmingly 2024–2026 listings, per the manifest) funding is dominated by
**listing dynamics, incentive programs, and thin one-sided flow**, not by the
crowding that mean-reverts. Their funding sign therefore has **no forward
cross-sectional information** (IC t −1.37), so admitting them adds turnover and
noise, not breadth of the *same* signal. Fundamental Law only rewards breadth of
bets that share the IC — these do not.

## Protocol note

The screen, tiers, φ, the +0.15 margin, and the anchor were all locked before the
run and correctly computed. No retune. The temptation to "keep only the few new
names that happened to work" is exactly the post-hoc selection the protocol
forbids. **FAIL stands.**

## Bottom line — breadth is now a CLOSED lever

- **Both breadth attempts are done:** ALX6 (aborted on anchor contradiction) and
  ALX7 (valid run, cleanly falsified). Widening 40 → ~90 Binance USDT-M names
  **does not help** — the edge is intrinsic to the established majors.
- The validated asset remains the **40-major ALX5 realistic-maker book (net
  Sharpe +1.89)**.
- Of the four proposed levers, three are now settled negative on data
  (breadth ✗, vol-targeting ✗, hold ✗). The only remaining honest path to fewer
  red days is an **orthogonal, uncorrelated second alpha** — a genuine discovery
  program, not a knob on this book.
