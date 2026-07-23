# ALX10 — Forward paper protocol: out-of-sample arbiter for the funding book and the momentum blend — PREREGISTRATION

**STATUS: LOCKED 2026-07-23 (owner sign-off).** This registers a *forward,
out-of-sample* validation protocol. Nothing before the lock date is scored; the
whole point is that the arbiter is data generated **after** the lock, which
neither the researcher nor the code has seen. §3–6 are now frozen and the only
permitted action is running the harness's record→evaluate on new bars. **No
in-sample re-fitting, no re-parameterization, no peeking-to-stop.**

Branch: `alpha/alx10-forward-paper` (off the current tip, not `main`: reuses the
frozen infra ALX1 `validate`, ALX3 `replicate`/`data_sources`, ALX4
`characterize` + its forward-ledger pattern, ALX5 `execution`, ALX8
`momentum`, ALX9 `riskparity`, none of which exist on `main`). All new ALX10 code
is isolated under `alpha_library/alx10_forward_paper/`.

---

## 0. Why this protocol, and why now

The in-sample program is complete and honest: the **funding→price book (ALX5,
+1.89)** is the one validated alpha; breadth (ALX6/7), vol-targeting, and hold
FAILed; the **momentum sleeve** is real (+0.79) and orthogonal (−0.09) but the
blend lift is sub-threshold (ALX8 50/50 → 1.896, ALX9 risk-parity → 2.016, bar
2.040). ALX9's REPORT drew the line: **spawning more in-sample weighting variants
would be a garden of forking paths.** ALX5's PASS already authorized exactly this
step — "a forward maker-paper protocol whose positive out-of-sample window is the
sole path toward sizing." So the only remaining honest arbiter is **unseen forward
data**, judged by criteria locked *now*.

## 1. Hypotheses (two frozen books, tracked in parallel)

- **Book A — funding-only (ALX5), verbatim.**
  **H1_A:** on ≥ the committed forward horizon spanning both regimes, Book A is
  net-profitable and signal-positive out-of-sample (forward Sharpe distinguishable
  from 0, IC positive) — an OOS confirmation of ALX5.
  **H0_A:** the in-sample edge does not persist; forward Sharpe is not
  distinguishable from 0 (or drawdown/IC fails). 
- **Book B — risk-parity funding+momentum blend (ALX9), verbatim.**
  **H1_B (graduation):** Book B *beats* Book A out-of-sample by a statistically
  real margin, measured by a **paired** daily-difference test — so the blend
  graduates over funding-only.
  **H0_B:** the blend does not reliably beat funding-only OOS; the +0.13 in-sample
  lift was noise / in-sample luck, and funding-only stands.

Assume both nulls TRUE until the locked criteria say otherwise.

## 2. Universe / data

- The frozen 40-major Binance USDT-M panel, fetched live via the frozen ALX3
  primitives (cache or network). Only **fully-closed** daily bars strictly
  **after `LOCK_DATE`** are scored; a 40-day pre-lock warm-up feeds the 7-day
  funding mean, the 24-day momentum, and the 60-day blend vol, but is **not**
  scored.
- `LOCK_DATE = 2026-07-23` (this pre-registration's lock; set at sign-off).

## 3. Frozen books (verbatim — no re-parameterization)

- **Book A:** `-funding.rolling(7).mean()`, top/bottom 30%, lag 1, hold 1;
  realistic-maker net (7 bps + φ = 0.80). Exactly ALX5.
- **Book B:** the ALX9 risk-parity blend of Book A with the momentum sleeve
  (`relative_strength(logret, 24)`), 60-day lagged inverse-vol gross-preserving
  weights, 50/50 warm-up, realistic-maker net. Exactly ALX9.

No lookback, k%, lag, hold, cost, φ, vol-window, or weight rule may change.

## 4. Execution / cost — two tracks (which criteria use which is locked)

- **Primary track (decisive): frozen realistic-maker net** (φ = 0.80, 7 bps) on
  the post-lock bars. This tests **signal persistence out-of-sample under the
  locked cost model** — data-only, robust, reproducible. All CONFIRM/GRADUATE
  criteria below use this track.
- **Execution-reality track (measurement + one FAIL trip): realized post-only
  fills.** Where the existing Bybit **testnet** execution path (Stage-21/22 infra,
  `TRADE_LIQUID_SYMBOLS` gate) is run alongside, record the **actual fill ratio φ̂**
  and realized adverse-selection per rebalance. This does not set the primary
  verdict, but it is the honest check on the maker assumption and can trip a locked
  FAIL (§5). It is **optional to operate daily** but **mandatory to report** at
  horizon if any testnet fills were collected.

## 5. Locked criteria — objective, committed IN ADVANCE

Append-only, immutable ledger (a realized forward day is never overwritten). The
harness reports **OBSERVING** until the horizon + regime conditions are met,
except that the early-FAIL trips can fire at any time.

**Horizon gate (both required):**
- `MIN_FORWARD_DAYS = 365` scored forward days, **and**
- both a **bull** and a **bear/flat** regime present (30-day mean market return
  > 0 somewhere and ≤ 0 somewhere), as in ALX4.

**Early-FAIL trips (may fire before horizon):**
- Book A forward max drawdown worse than **−30%** (worse than any in-sample year).
- Book A IC flips **negative in a bull** sub-phase.
- Execution-reality: if ≥ 60 testnet rebalances are collected and the realized
  fill ratio **φ̂ < 0.50** (the maker assumption φ = 0.80 was materially
  optimistic), the *maker-execution premise* FAILs — recorded as such.

**Book A — CONFIRM (horizon met):** forward net Sharpe **> 0** with block-bootstrap
95% CI **lower bound > 0**, **and** forward rank-IC Newey-West t **> 0** (positive
in the bull sub-sample too). Else **FAIL_A** (edge not distinguishable OOS).

**Book B — GRADUATE over A (horizon met, only assessed if A CONFIRMs):** the
**paired** daily net difference `d_t = net_B − net_A` has mean **> 0** with
Newey-West t **≥ 2.0**, **and** Book B's forward max drawdown is **not worse** than
Book A's. The paired test is the powered, honest way to ask "does B beat A on the
same days," far more so than comparing two noisy 1-year Sharpes. Else **Book B does
NOT graduate** and funding-only (Book A) stands.

**No peeking-to-stop.** The horizon and regime requirements are fixed; we do not
end early on a favorable interim number. Interim `evaluate` output is OBSERVING
plus current diagnostics only.

## 6. Single procedure, no tuning (binding)

The only permitted operation is `record` (append new post-lock bars) then
`evaluate` (apply §5). Thresholds, horizon, regime rule, cost model, φ, and both
book constructions are fixed above. Nothing is re-fit on forward data. Observing
results never reopens the spec; a FAIL is the recorded outcome. A CONFIRM/GRADUATE
authorizes only a *separate* live-sizing decision, never automatic capital.

## 7. Deliverables

`forward.py` (append-only ledger for both books; `record` + `evaluate`; prints
OBSERVING / CONFIRM / FAIL / GRADUATE) + `run.py` (one command) + known-answer unit
tests under `tests/test_alx10_forward.py` (post-lock-only scoring, ledger
immutability/dedup, paired-difference NW-t correctness, horizon+regime gating) +
append-only ledger under `data/research/alx10/forward/` + `REPORT.md` written **at
horizon** with the real verdict + one squashed commit to LOCK (harness + this
prereg), then periodic ledger-append commits.
