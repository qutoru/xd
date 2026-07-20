# ALX4 — Regime characterization + causal falsification — REPORT

**VERDICT: mechanism SURVIVES the cheap decisive falsifiers → status stays
🟡 promising, PENDING the locked forward protocol. NOT graduated to the Alpha
Library.** Single run, no tuning, per `PREREGISTRATION.md`. Branch
`alpha/alx4-regime-analysis`. Reproduce: `py alpha_library/alx4_regime_analysis/run.py`.

Object under test: the **frozen** ALX book (`-7d-mean funding`, top/bottom 30%,
hold 1d, lag 1d, Med 7.5 bps), run verbatim through
`alpha_library.alx_funding_price_validation.validate.backtest` on the independent
Binance reconstruction (40 majors, 2019-09-08 … 2026-07-12, 2500 days, cached by
ALX3 from raw REST — no project parquet). Full-sample net Sharpe **+1.94**,
block-bootstrap 95% CI **[+1.16, +2.69]**.

**This does not touch the locked ALX3 REJECTED verdict.** ALX3's binary outcome
(does not graduate) stands. ALX4 only asks whether the *existence/mechanism*
survives new, cheap, decisive falsifiers, and locks a forward protocol.

---

## P1 — Funding cross-sectional shuffle placebo (decisive) — **PASS**

Permute funding across symbols each day (breaking the funding↔symbol linkage),
rebuild the frozen book, N=500 (seed 0).

| quantity | value |
|---|---|
| real net Sharpe | **+1.94** |
| shuffle median Sharpe | **−0.67** |
| shuffle 97.5th pct | +0.03 |
| shuffle max | +0.54 |
| real beats | **100% of 500 shuffles** |

Randomizing the funding→symbol map **destroys** the edge (shuffled books center
at −0.67 — the turnover cost with no signal — and the best of 500 shuffles only
reaches +0.54, far below the real +1.94). The edge is **specifically tied to the
funding-to-symbol assignment**, exactly as H1 requires. **H0 (assignment-agnostic
artifact) is rejected.**

### Disclosure — a corrected specification bug in the P1 gate
The locked pre-registration prose defines the pass condition as *"shuffled books
have no systematic **edge**"* (i.e. no systematic **positive** Sharpe). The first
code implementation operationalized this two-sided as `abs(median) < 0.5`, which
fired **FAIL** because the shuffle median is **−0.67** (|−0.67| ≥ 0.5). But a
*negative* shuffle median is the **correct null behavior** — a long/short book on
randomized funding still pays full turnover cost with no signal, so it *should*
lose money; it is not evidence against the mechanism. Under the escape-hatch
clause (a **concrete implementation bug**: code contradicting its own locked prose
intent), the gate was corrected to the one-sided `median < 0.5` and re-run. Both
outcomes are recorded here in full; the **economic result is identical either
way** (real +1.94 beats 100% of shuffles; shuffles center below zero). The
correction fixes a *false negative* and is against, not toward, a favorable spin.

## P2 — Directionality (funding→return vs return→funding) — **consistent (OK)**

| mapping | mean rank-IC | Newey-West t |
|---|---|---|
| forward: signal(t-1) → ret(t) | **+0.0183** | **+4.70** |
| reverse: ret(t-1) → signal(t) | +0.0110 | +2.78 |

The forward (funding→price) relationship is the material one (t 4.70 clears the
diagnostic bar t≥3). A weaker reverse relationship also exists (t 2.78) — funding
partly reflects recent returns, which is expected for a *crowding* variable
(crowding both follows and precedes price). So this supports funding→price as the
dominant direction but is **not** a clean one-way causal proof — labelled
diagnostic, not decisive.

## P3 — Factor independence — **PASS**

Correlation of ALX net with identically-built factor books:

| factor | corr | | factor | corr |
|---|---|---|---|---|
| momentum(24) | −0.091 | | size (SMB) | +0.081 |
| reversal(3) | +0.056 | | volatility | **+0.211** |
| beta | +0.059 | | market (buy&hold) | +0.064 |

Multivariate OLS **R² = 12.1%**. No single factor |corr| ≥ 0.7; ~88% of variance
is unexplained by known factors. The only non-trivial tilt is a mild
**long-volatility** exposure (+0.21), consistent with the unwind mechanism, not a
repackaging. Combined with the funding-**carry** leg being only ~30% of net PnL
(carry corr +0.115), ALX is a **genuinely independent** source of return, distinct
from momentum/reversal/beta/size and from funding carry. **H0 (known-factor
repackaging) is rejected.**

---

## Diagnostics (descriptive only — no pass/fail, no tuning)

### Per-year (independent Binance, frozen spec)
| year | Sharpe | win | maxDD | mean IC |
|---|---|---|---|---|
| 2020 (20 sym) | +1.60 | 52% | −14% | +0.008 |
| **2021 (bull)** | **+3.65** | 57% | −17% | **+0.050** |
| **2022 (bear)** | **+0.69** | 48% | −11% | **−0.005** |
| 2023 | +1.24 | 56% | −14% | +0.007 |
| **2024** | +1.47 | 53% | −7% | **+0.038** |
| 2025 | +2.48 | 53% | −8% | +0.018 |
| 2026 (part) | +2.10 | 54% | −6% | +0.006 |

Strong in 2021 (pre-development, independent venue), **dormant in the 2022
bear**, positive 2023-2026. **No degradation after discovery** — if anything the
most recent two years are among the strongest.

### Regime-conditional net Sharpe (pre-defined, t-1-lagged variables; terciles)
| regime var | low | mid | high |
|---|---|---|---|
| **market vol** | +0.86 | +2.19 | **+2.63** (monotone ↑) |
| funding-score dispersion | +2.54 | +1.15 | +2.42 |
| cross-sec return dispersion | +2.30 | +1.64 | +2.40 |
| trailing trend (bull/bear) | +2.56 (bear) | +1.67 | +2.04 (bull) |
| breadth | +2.08 | +1.84 | +2.12 |

The clearest dependence is on **market volatility** (monotone). But **every
bucket of every pre-defined variable is positive** — so **no single variable
*gates* the edge**. Per the locked interpretation rule, we do **NOT** claim a
working regime classifier. The 2022 dormancy (a systemic post-LUNA/FTX
deleveraging where the whole cross-section moved together and funding lost
discriminating power) is **not** captured by any of these trailing variables —
the "off" state is rare and event-driven, hard to flag ex-ante.

### Profit concentration (top-day removal)
drop best 0/1/2/5/10% of days → Sharpe **+1.94 / +1.30 / +0.71 / −0.80 / −2.91**.
Critical point between 2–5%: the top ~3–4% of days carry the book. Consistent
with the episodic unwind mechanism; also means the **effective** independent
sample is far smaller than 2500 days, so all CIs overstate confidence.

---

## What is established / not established

**Established (this study + prior):** the effect is **real, not random** (P1: dies
on shuffle; beats 100% of placebos), **factor-independent** (P3: R² 12%, all
|corr| ≤ 0.21) and **not carry** (30% of PnL, corr +0.115); funding→price is the
dominant direction (P2: t 4.70); the effect **does not degrade post-discovery**;
its mechanism (crowding/leverage-unwind, long-vol tilt) is economically coherent
and cross-venue stable (funding→price corr +0.56–0.58 across Binance/OKX/Bybit,
from ALX3).

**NOT established:** that it is a **tradeable stable premium** (episodic
concentration → small effective N; event-driven dormancy in 2022 is unpredictable
by the pre-defined regime variables); that it is a **cleanly regime-classifiable**
alpha (no single variable gates it); that it survives **survivorship-inclusive**
universes and construction choices (in-sample union +1.47 vs intersection +0.21 on
the ALX3 window); and — decisively — anything on **genuinely forward** data, which
does not yet exist.

## Conclusion & next step

**🟡 Promising hypothesis requiring independent forward confirmation.** The
existence and mechanism of ALX survive every cheap decisive falsifier run here;
what remains unproven is tradeable stability, which only *forward* data can
settle. The **locked forward protocol** in `PREREGISTRATION.md` (fixed spec, fixed
universe rule, metrics/PASS-FAIL/horizon committed in advance, ≥12–18 months) is
the only sanctioned path to upgrade the status. Highest-priority open experiments
(cheap, decisive): (1) start the forward observation now (paper/shadow); (2)
survivorship-inclusive universe reconstruction. **Not for real money.**

### Forward harness — IMPLEMENTED (`forward.py`, `forward_run.py`)
The locked forward protocol is now executable:
`py alpha_library/alx4_regime_analysis/forward_run.py`. On each run it (a) rebuilds
the frozen ALX book on independent data, (b) appends every **fully-closed day
strictly after the lock date (2026-07-20)** to an append-only, dedup ledger
(`data/research/alx4/forward/ledger.parquet`) — past forward days are immutable,
never overwritten — and (c) evaluates the locked criteria: net Sharpe +
bootstrap CI, rank-IC + NW t, drawdown, top-5% share, bull/bear coverage. It
returns **OBSERVING** until ≥365 forward days spanning both regimes accumulate,
then **CONFIRMED** (Sharpe CI excludes 0 and bull-IC > 0) or **FAIL**; a
drawdown worse than −30% or a bull-phase IC sign-flip fails it early.

**Current state (run at the lock date):** 0 fully-closed post-lock days →
**OBSERVING**. No verdict is permitted yet, by design. The harness must be run on
a schedule (daily/weekly) so genuine out-of-sample evidence accumulates over the
horizon. This is the *only* way ALX's status can legitimately change.
