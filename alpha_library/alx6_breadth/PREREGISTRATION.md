# ALX6 — Breadth: does a wider PIT universe raise the funding→price net Sharpe? — PREREGISTRATION

**STATUS: LOCKED 2026-07-22 (owner sign-off).** Committed **before** any ALX6
experiment/book code exists and **before** any breadth backtest is run. Only the
data-acquisition step (`fetch_universe.py`, no backtest/Sharpe) preceded this.
Single run, no tuning, per `CLAUDE.md`. From this point the spec in §3–6 is
frozen; observing a result does not reopen it. **Start from the assumption the
breadth gain is FALSE (a survivorship + illiquidity artifact).**

Branch: `alpha/alx6-breadth-universe` (created off the current tip, not `main`,
because the frozen infra this experiment *reuses* — ALX1
`alx_funding_price_validation.validate.backtest`, ALX3
`alx3_external_replication.{data_sources,replicate}`, ALX4
`alx4_regime_analysis.characterize`, ALX5
`alx5_maker_execution.execution` — does not exist on `main`. All *new* ALX6 code
is isolated under `alpha_library/alx6_breadth/`; no frozen alpha/parameters are
edited.)

---

## 0. Motivation & the prior (what we already know)

The frozen ALX book on **40** survivorship-safe majors has realistic-maker net
Sharpe **+1.890** (ALX5, LOCKED). The Fundamental Law of Active Management says
Sharpe ∝ IC · √(breadth). Naively, 40 → ~90 names at *constant* IC would give
1.89 · √(90/40) ≈ **2.8**. We reject that as a foregone conclusion: adding
smaller names almost certainly **dilutes IC** (the funding→reversal signal is
cleaner on liquid majors) and **degrades execution** (thin names have wider
spreads / worse fills than our maker model assumes). So the gate is set to be
**hard**, testing whether *real* breadth survives honest PIT + illiquidity cost,
not to rubber-stamp the √N ceiling.

A **principled** (not rigged) decisive bar: even if breadth genuinely works but
IC drops ~20% at ~2× names, Sharpe ≈ 1.89 · 0.8 · √2 ≈ **2.14**. We therefore
require the expanded net Sharpe to clear **> 2.10** — breadth must deliver a
real, >10% gain, not a marginal one.

## 1. Hypothesis

**H1 (economic claim):** The *same* frozen funding→price signal, applied to a
**point-in-time-admitted, liquidity-screened** universe of N ≈ 80–100 Binance
USDT-M perpetuals, has a higher net Sharpe than the 40-name book — because the
identical funding-carry / crowding-reversal mechanism operates across more
independent names (Fundamental Law breadth of the **same** signal), and the
uplift survives honest PIT entry and illiquidity-scaled maker execution.

**H0 (null — assume TRUE until rejected):** The wider universe does **not**
meaningfully beat the 40-name baseline once (a) entry is point-in-time
(onboard-dated), (b) execution cost is scaled up for the new names' worse
liquidity, and (c) the newest-listed / most-illiquid tail is stress-removed. Any
apparent gain is a **survivorship** (delisted names are absent from the fetch)
and **illiquidity-premium** artifact, not durable breadth.

**One hypothesis only** — the effect of **universe breadth** on the *frozen*
signal. We do **not** re-tune the signal, sign, lookback, k%, lag, or hold. The
**only** variable is the set of names (and the liquidity-scaled cost that honesty
requires when new names are thinner).

## 2. Universe / data

- Source: `alpha_library/alx6_breadth/fetch_universe.py` (already run; data step
  only). Binance USDT-M perpetuals with per-symbol **onboard (listing) dates**
  from `exchangeInfo`, daily klines + funding pulled via the frozen ALX3
  primitives into the shared venue cache.
- **PIT admission rule (LOCKED, deterministic — N falls out of data, not choice):**
  a name is *eligible on day t* iff
  1. `t ≥ onboard_date + 30 calendar days` (warm-up for the 7-day funding mean +
     early-listing instability), **and**
  2. it has ≥ **252** eligible days of history over the full window, **and**
  3. its **median daily quote (USDT) volume over its eligible window ≥ $5M**.
  The set of names passing (1)–(3) is the expanded universe; N is whatever this
  yields. The original 40 majors are a subset of it by construction.
- Panel is the **dynamic union** over 2019-09-08 … present with **per-day NaN
  masking** (a name contributes only on days it is eligible+valid) — exactly the
  ragged-availability handling already in `characterize.build_signal` +
  `validate.backtest`. No faithful-intersection window (that would defeat the
  point of breadth).
- **KNOWN residual survivorship limit (documented, gated in §5-C):** the fetch
  sees only currently-listed symbols, so **delisted** coins are absent. Onboard
  dates make *entry* PIT-correct, but the missing delisted tail means the raw
  expanded Sharpe is an **optimistic ceiling**; Stage C stresses this directly.

## 3. Frozen signal (unchanged — reused verbatim)

`signal = -funding.rolling(7).mean()`, dollar-neutral equal-weight top/bottom
**30%**, execution **lag = 1 day**, **hold = 1 day**, funding carry to the held
leg exactly as in `validate.backtest`. **None of these are touched.** Rank-IC,
folds, and bootstrap use the frozen ALX4/ALX5 helpers.

## 4. The ONE thing under test: the universe (and its honest cost)

The tested object is the **name set**. The cost model is **not** a free knob — it
is fixed to the ALX5 realistic-maker plus a **locked illiquidity scaling** that
honesty demands when admitting thinner names:

**Illiquidity-tiered realistic-maker cost (LOCKED)** — round-trip bps on realized
turnover, **plus** the ALX5 non-fill haircut **φ = 0.80** on gross, per-name by
median eligible-window quote volume:

| liquidity tier | round-trip cost |
|---|---|
| ADV ≥ $50M | 7 bps (= ALX5 realistic maker) |
| $10M ≤ ADV < $50M | 12 bps |
| $5M ≤ ADV < $10M | 20 bps |

These tiers and φ = 0.80 are **locked now**; no adjustment after seeing results.
The 40-name majors sit in the ≥$50M tier, so the baseline is charged exactly as
in ALX5 (7 bps) — the anchor must reproduce.

## 5. Statistical gates — objective PASS/FAIL committed IN ADVANCE

Staged gauntlet, hard stop between stages (research-ladder pattern).

- **Stage A — anchor / machinery (kill-test).** Within the ALX6 pipeline,
  restrict to the original 40 majors and confirm the illiquidity-tiered
  realistic-maker net Sharpe reproduces **ALX5's +1.89 (±0.15)** and rank-IC
  Newey-West t(5) ≥ 3.0 positive. If the base does not reproduce ⇒ **STOP →
  FAIL** (the expanded pipeline silently altered the book; maker numbers off a
  broken base are meaningless).

- **Stage B — does breadth add REAL, non-artifact edge (decisive).** On the full
  PIT-admitted, liquidity-screened universe under the illiquidity-tiered
  realistic-maker cost, **all** must hold:
  1. expanded net Sharpe **> 2.10**;
  2. **incremental-names attribution:** the book built on **only the NEW names**
     (universe minus the original 40) has net Sharpe **> 0** **and** rank-IC
     Newey-West t(5) **≥ 2.0** positive — the new names carry genuine signal, not
     merely diversify the same 40;
  3. **fold stability:** in **≥ 4 of 6** equal contiguous walk-forward folds, the
     expanded book's net Sharpe **≥** the 40-name baseline's over the same fold.
  Any one fails ⇒ **STOP → FAIL** (H0 not rejected).

- **Stage C — the gain is not a survivorship / illiquidity / leakage artifact
  (kill-tests).** All three must survive:
  1. **survivorship-tail stress:** drop every name onboarded **on/after
     2023-01-01** (the young survivors most likely to flatter a
     delisted-blind fetch); the remaining expanded book net Sharpe must stay
     **> baseline 1.89**.
  2. **illiquidity stress:** drop the entire **$5–10M tier** (thinnest names,
     where φ = 0.80 is least trustworthy); the remaining expanded book net Sharpe
     must stay **> baseline 1.89**.
  3. **PIT / leakage audit:** funding future-corruption invariance holds
     bit-exact before the corruption point (as ALX3), **and** each name
     contributes exactly **zero** PnL before `onboard + 30d`. Any artifact ⇒
     FAIL.

**Objective verdict rule (LOCKED):**
- **PASS (→ breadth graduates the wider universe as the new book):** Stage A
  reproduces **and** all three Stage-B conditions hold **and** all three Stage-C
  kill-tests survive.
- **FAIL (→ H0 not rejected):** any stage fails. The 40-name book stands
  unchanged. Recorded as a full-value result.

**No real money and no production graduation from this experiment.** A PASS only
authorizes the wider universe as the research book and a *separate* forward-paper
step before any sizing.

## 6. Single run, no tuning (binding)

The procedure in §4–5 is run **once** on the frozen signal (unchanged). The PIT
rule, the $5M screen, the illiquidity tiers, φ = 0.80, the > 2.10 bar, the 2023
survivorship cutoff, and the fold count are all fixed above. Descriptive outputs
(e.g. Sharpe vs N curve, per-tier IC) are diagnostics, never knobs. Observing any
result ends the freedom to change the spec; a failure *is* the recorded outcome.

## 7. Deliverables

`run.py` (one command, stage-gated, prints the single verdict) + known-answer
unit tests under `tests/test_alx6_breadth.py` (PIT-admission masking correctness,
illiquidity-tier cost monotonicity, incremental-names partition identity) +
artifacts under `data/research/alx6/` + `REPORT.md` (honest verdict) + one
squashed commit **after** completion.
