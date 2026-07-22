# ALX7 — Breadth (decoupled anchor): do screened NEW names lift the frozen 40-major book? — PREREGISTRATION

**STATUS: LOCKED 2026-07-23 (owner sign-off).** Committed before any ALX7
experiment/book code exists and before any ALX7 backtest is run. It reuses the
ALX6 data step (fetched universe manifest, no new fetch). From this point the
spec in §3–6 is frozen; observing a result does not reopen it. **Start from the
assumption the breadth gain is FALSE (a survivorship + illiquidity artifact).**

Branch: `alpha/alx7-breadth-decoupled` (off the current tip, not `main`, for the
same reason as ALX5/ALX6: the frozen infra it reuses — ALX1 `validate.backtest`,
ALX3 `data_sources`/`replicate`, ALX4 `characterize`, ALX5 `execution`, and the
ALX6 fetched universe manifest — does not exist on `main`. All *new* ALX7 code is
isolated under `alpha_library/alx7_breadth/`; no frozen alpha/parameters edited.)

---

## 0. Why ALX7 exists — the ALX6 lesson it fixes (not a relitigation)

ALX6 asked the same economic question and **FAILED at Stage A**: its locked $5M
median-ADV screen excluded **12 of the 40 reference majors outright**, so the
pipeline's "orig40 baseline" was a 28-name book (net Sharpe +1.14) that could not
reproduce the ALX5 anchor (+1.89) — the run aborted before the breadth hypothesis
was ever tested. That FAIL **stands** (locked, correctly computed). ALX6's REPORT
recorded the forward-looking fix, which ALX7 implements: **decouple the Stage-A
anchor from the new-name admission screen.**

The decoupling (locked design of ALX7):
- The **reference book is the frozen ALX5 40-major book, charged at its ALX5 rate
  (flat 7 bps maker, φ = 0.80)** — the screen is **never** applied to the 40
  majors. Stage A therefore reproduces ALX5 **by construction**, and remains a
  genuine machinery kill-test (a bug in the new PIT/panel code would still break
  it).
- The PIT + liquidity **screen governs only the INCREMENTAL (new) names.** The
  experiment tests whether *adding* screened new names to the intact 40-major
  book lifts net Sharpe.

## 1. Hypothesis

**H1 (economic claim):** Adding point-in-time-admitted, liquidity-screened **new**
Binance USDT-M names to the frozen 40-major funding→price book **raises** its net
Sharpe — because the identical funding-carry / crowding-reversal mechanism
operates on the additional names (Fundamental-Law breadth of the **same**
signal) — and the uplift survives honest PIT entry, illiquidity-scaled maker cost
on the new names, and survivorship/leakage stresses.

**H0 (null — assume TRUE until rejected):** Adding new names does **not**
meaningfully raise net Sharpe once (a) entry is point-in-time, (b) the new names
are charged their worse-liquidity cost, and (c) the newest-listed / thinnest tail
is stress-removed. Any apparent gain is a **survivorship** (delisted names absent
from the fetch) and **illiquidity-premium** artifact, and/or is confined to a few
recently-listed survivors.

**One hypothesis only** — the effect of adding breadth to the frozen book. The
signal (sign, 7-day lookback, top/bottom 30%, lag 1, hold 1) and the 40-major
reference set are **unchanged**. The only variable is the **set of added names**
(and the locked illiquidity cost they carry).

## 2. Universe / data

- Reuses the ALX6 fetched manifest (`data/research/alx6/universe_manifest.csv`)
  and shared venue cache — **no new fetch.** Per-symbol onboard (listing) dates
  from Binance `exchangeInfo` give the point-in-time entry anchor.
- **Reference universe (fixed):** the 40 ALX4/ALX5 majors, always included, never
  screened.
- **Candidate NEW names:** every fetched non-major symbol.
- **New-name admission (LOCKED, deterministic):** a new name is admitted iff
  1. `t ≥ onboard + 30 calendar days` for its contribution days, **and**
  2. ≥ **252** eligible days of history, **and**
  3. **median eligible-window quote (USDT) volume ≥ $5M.**
  N (number of added names) falls out of this rule, not from choice.
- Panel = dynamic union over 2019-09-08 … present, per-day NaN masking (a name
  contributes only on eligible+valid days), exactly as
  `characterize.build_signal` + `validate.backtest` already handle raggedness.
- **KNOWN residual survivorship limit (documented, gated in §5-C):** the fetch
  sees only currently-listed symbols; delisted coins are absent. Onboard dates
  make *entry* PIT-correct, but the missing delisted tail means any raw breadth
  gain is an **optimistic ceiling**; Stage C stresses this directly.

## 3. Frozen signal (unchanged — reused verbatim)

`signal = -funding.rolling(7).mean()`, dollar-neutral equal-weight top/bottom
**30%**, execution **lag = 1**, **hold = 1**, funding carry to the held leg as in
`validate.backtest`. **None touched.** Rank-IC / folds / helpers are the frozen
ALX4/ALX5 ones.

## 4. The ONE thing under test: the added names (and their honest cost)

The tested object is the **set of added new names**. The cost model is fixed, not
a knob:

- **Reference 40 majors:** flat **7 bps** round-trip maker (their ALX5 tier) +
  φ = 0.80. (This is what makes Stage A reproduce ALX5 exactly.)
- **New names:** locked **illiquidity-tiered** round-trip maker cost by median
  eligible-window quote volume + φ = 0.80:

  | tier | cost |
  |---|---|
  | ADV ≥ $50M | 7 bps |
  | $10M ≤ ADV < $50M | 12 bps |
  | $5M ≤ ADV < $10M | 20 bps |

All tiers, the $5M screen, the 30-day buffer, the 252-day minimum, and φ = 0.80
are **locked now**; no adjustment after seeing results.

## 5. Statistical gates — objective PASS/FAIL committed IN ADVANCE

Staged gauntlet, hard stop between stages.

- **Stage A — anchor / machinery (kill-test).** The 40-major reference book (flat
  7 bps, φ = 0.80, run through the ALX7 PIT pipeline) must reproduce **ALX5's
  +1.89 (±0.15)** and rank-IC Newey-West t(5) **≥ 3.0** positive. If it does not
  reproduce, **STOP → FAIL** (the new pipeline silently corrupted the base). Call
  the reproduced value `base_sharpe`.

- **Stage B — do added names lift the book, really (decisive).** On the expanded
  book (40 majors @7 bps **+** all admitted new names @tiered), **all** must hold:
  1. **margin:** expanded net Sharpe **≥ `base_sharpe` + 0.15** (one anchor-
     tolerance band above the reference — a real, non-noise lift);
  2. **incremental-names attribution:** the book on **only the added new names**
     has net Sharpe **> 0** **and** rank-IC Newey-West t(5) **≥ 2.0** positive —
     the new names genuinely carry the funding→price signal, not noise;
  3. **fold stability:** in **≥ 4 of 6** equal contiguous walk-forward folds, the
     expanded book's net Sharpe **≥** the 40-major reference's over that fold.
  Any one fails ⇒ **STOP → FAIL** (H0 not rejected).

- **Stage C — the lift is not survivorship / illiquidity / leakage (kill-tests).**
  All three must survive:
  1. **survivorship-tail:** drop every **added** name onboarded on/after
     **2023-01-01**; the remaining expanded book (40 majors + pre-2023 new names)
     net Sharpe must stay **≥ `base_sharpe` + 0.15** (the lift is not carried by
     young survivors).
  2. **illiquidity:** drop the entire **$5–10M** new-name tier; the remaining
     expanded book net Sharpe must stay **≥ `base_sharpe` + 0.15**.
  3. **PIT / leakage audit:** funding future-corruption invariance holds
     bit-exact before the corruption point (as ALX3), **and** every added name
     contributes exactly **zero** weight before `onboard + 30d`. Any artifact ⇒
     FAIL.

**Objective verdict rule (LOCKED):**
- **PASS (→ the expanded universe graduates as the research book):** Stage A
  reproduces **and** all three Stage-B conditions hold **and** all three Stage-C
  kill-tests survive.
- **FAIL (→ H0 not rejected):** any stage fails. The 40-major ALX5 book stands
  unchanged. Recorded as a full-value result.

**No real money / no production graduation.** A PASS only authorizes the expanded
universe as the research book and a separate forward-paper step before any sizing.

## 6. Single run, no tuning (binding)

The procedure in §4–5 runs **once** on the frozen signal. The reference set, the
new-name screen, the tiers, φ = 0.80, the +0.15 margin, the 2023 cutoff, and the
fold count are all fixed above. Descriptive outputs (Sharpe-vs-N curve, per-tier
IC, added-name count by year) are diagnostics, never knobs. Observing any result
ends the freedom to change the spec; a failure *is* the recorded outcome.

## 7. Deliverables

`run.py` (one command, stage-gated, one verdict) + known-answer unit tests under
`tests/test_alx7_breadth.py` (reference-book anchor identity vs ALX5, new-name
screen correctness, tiered-vs-flat cost split, incremental partition identity,
PIT zero-weight) + artifacts under `data/research/alx7/` + `REPORT.md` (honest
verdict) + one squashed commit **after** completion.
