# ALX8 — Orthogonal momentum sleeve: does a second, uncorrelated alpha lift the book? — PREREGISTRATION

**STATUS: LOCKED 2026-07-23 (owner sign-off).** Committed before any ALX8
experiment/book code exists and before any ALX8 backtest is run. Single run, no
tuning, per `CLAUDE.md`. From this point the spec in §3–6 is frozen; observing a
result does not reopen it. **Start from the assumption the sleeve does NOT help
(momentum is either cost-eaten or not diversifying).**

Branch: `alpha/alx8-momentum-sleeve` (off the current tip, not `main`, for the
same reason as ALX5–7: the frozen infra it reuses — ALX1 `validate.backtest`,
ALX3 `data_sources`/`replicate`, ALX4 `characterize` + the shared factor features
`crypto_signal_bot.research.xsection.features`, ALX5 `execution` — does not exist
on `main`). All *new* ALX8 code is isolated under
`alpha_library/alx8_momentum_sleeve/`; no frozen alpha/parameters edited.

---

## 0. Why this experiment, and why momentum specifically

Levers 1–3 (breadth, vol-targeting, longer hold) are settled **negative** on data
(ALX6/ALX7 FAIL; vol-target and hold descriptive-flat). The only remaining honest
way to raise the book's Sharpe / cut its ~50% red days is an **orthogonal second
alpha**: for two uncorrelated sleeves, portfolio Sharpe ≈ √(S₁² + S₂²).

**Choice of the second signal is data-driven, not fished.** ALX4's P3 already
measured the funding book's correlation with identically-built factor-mimicking
books: **momentum −0.091**, reversal +0.056, beta +0.059, size +0.081,
volatility +0.211. Momentum is (a) the **most orthogonal** (near-zero, slightly
negative), (b) economically the **most distinct** mechanism (trend /
underreaction vs funding-crowding **reversal** — the canonical "carry + momentum"
pair), and (c) **lower-turnover** than short-term reversal, so it has a chance to
survive maker cost. We test momentum and momentum only; reversal / volume-shock
sleeves are separate future experiments if this fails.

**We do not assume momentum is itself a real alpha.** Crypto daily
cross-sectional momentum net of honest maker cost may well be flat or negative
(higher turnover than the funding book). That is exactly what Stage B tests.

## 1. Hypothesis

**H1 (economic claim):** A cross-sectional **momentum** sleeve, blended 50/50
(gross) with the frozen funding→price book, produces a **combined** realistic-
maker net Sharpe **meaningfully above the funding book alone (+1.89)** — because
momentum is itself a net-positive alpha **and** is nearly uncorrelated with the
funding sleeve, so the blend diversifies (√-of-squares benefit).

**H0 (null — assume TRUE until rejected):** The blend does **not** meaningfully
beat +1.89, because either (a) the momentum sleeve is not net-positive under
honest maker cost (turnover eats it), or (b) it does not diversify enough, or (c)
any apparent lift is not attributable to a real momentum signal (a random sleeve
of matched turnover would do the same).

**One hypothesis only** — the value of an **orthogonal momentum sleeve** to the
frozen book. The funding signal and its portfolio construction are unchanged.

## 2. Universe / data

- The frozen 40-major ALX4/ALX5 Binance USDT-M panel (2019-09-08 … 2026-07-12,
  cached), identical to ALX5. No breadth (ALX7 closed that).
- Returns, funding, and the shared factor features from
  `crypto_signal_bot.research.xsection.features`.

## 3. Frozen signals (both reused verbatim — no tuning)

- **Funding sleeve (unchanged):** `signal = -funding.rolling(7).mean()`,
  top/bottom 30%, lag 1, hold 1 — exactly ALX5.
- **Momentum sleeve (frozen factor, NOT tuned here):**
  `signal = relative_strength(logret, 24)` — the shared frozen definition
  (trailing 24-day return, cross-sectionally demeaned), the **same lookback = 24**
  ALX4 used for its momentum factor book. `logret = log1p(ret)`. Same portfolio
  construction as the funding sleeve: dollar-neutral top/bottom **30%**, lag 1,
  hold 1. The lookback is **not** searched; 24 is inherited, locked.

## 4. Execution / cost model (locked = ALX5 realistic maker)

All net Sharpes use the ALX5 **realistic-maker** model: **7 bps** round-trip on
realized turnover **and** φ = 0.80 non-fill haircut on gross, where
`gross = price PnL + funding carry` on the held positions.

**Blend (locked):** combine at the **weight level with netting** — the honest
single-book execution: `combined_applied = 0.5·w_funding + 0.5·w_momentum` (both
lag-1). The combined book's price, funding carry, and **turnover** are computed on
`combined_applied` (overlapping legs net, so turnover is paid once), then the same
realistic-maker cost + φ applied. Each standalone sleeve's net uses the same model
on its own weights. φ = 0.80, 7 bps, and the 50/50 split are **locked now**.

## 5. Statistical gates — objective PASS/FAIL committed IN ADVANCE

Staged gauntlet, hard stop between stages.

- **Stage A — anchor / machinery (kill-test).** The funding sleeve's realistic-
  maker net Sharpe reproduces **ALX5 +1.89 (±0.15)** and rank-IC Newey-West t(5)
  **≥ 3.0** positive. If not, **STOP → FAIL**. Call the value `base_sharpe`.

- **Stage B — momentum is a real, orthogonal, book-lifting alpha (decisive).**
  All four must hold:
  1. **momentum is real:** momentum-sleeve realistic-maker net Sharpe **> 0.5**
     (a genuinely tradeable alpha, not cost-dust);
  2. **orthogonal:** `|corr(net_funding, net_momentum)| < 0.30`;
  3. **the blend lifts the book:** combined net Sharpe **≥ `base_sharpe` + 0.15**
     (one anchor-tolerance band above the funding book);
  4. **fold-stable:** combined net Sharpe **≥** the funding book's in **≥ 4 of 6**
     equal contiguous folds.
  Any one fails ⇒ **STOP → FAIL** (H0 not rejected).

- **Stage C — the lift is real diversification, not leakage or book-averaging
  (kill-tests).** All three must survive:
  1. **leakage tell:** the momentum sleeve's **lag-0** gross Sharpe must be
     conspicuously **>** its traded **lag-1** gross Sharpe (same-bar sanity; the
     traded result must not rely on look-ahead).
  2. **random-sleeve placebo (decisive anti-artifact):** replace the momentum
     signal with a **random** cross-sectional signal (matched top/bottom-30%
     construction, lag 1, same cost), **200 seeds**; blend each with the funding
     book. The **real** combined net Sharpe must exceed the **97.5th percentile**
     of the 200 random-combined Sharpes. If merely averaging the funding book
     with *any* second book of matched turnover reaches base+0.15, the lift is not
     momentum ⇒ FAIL.
  3. **PIT / no-look-ahead:** funding future-corruption invariance on the combined
     book holds bit-exact before the corruption point (as ALX3).

**Objective verdict rule (LOCKED):**
- **PASS (→ the funding+momentum blend graduates as the research book):** Stage A
  reproduces **and** all four Stage-B conditions hold **and** all three Stage-C
  kill-tests survive.
- **FAIL (→ H0 not rejected):** any stage fails. The funding-only ALX5 book
  stands unchanged. Recorded as a full-value result.

**No real money / no production graduation.** A PASS only authorizes the blend as
the research book and a separate forward-paper step before any sizing.

## 6. Single run, no tuning (binding)

Run **once** on the frozen signals. Lookback 24, the 50/50 split, φ = 0.80, 7 bps,
the > 0.5 / < 0.30 / +0.15 thresholds, the 200-seed placebo, and the fold count
are all fixed above. Descriptive outputs (correlation, per-sleeve turnover,
Sharpe-vs-blend-weight curve) are diagnostics, never knobs. Observing any result
ends the freedom to change the spec; a failure *is* the recorded outcome.

## 7. Deliverables

`run.py` (one command, stage-gated, one verdict) + known-answer unit tests under
`tests/test_alx8_momentum.py` (frozen-momentum factor identity, blend-weight
netting/turnover identity, correlation/placebo helper correctness, anchor
reproduction) + artifacts under `data/research/alx8/` + `REPORT.md` (honest
verdict) + one squashed commit **after** completion.
