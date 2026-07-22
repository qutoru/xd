# ALX5 — Maker-execution / low-turnover viability of the funding→price book — REPORT

**VERDICT: PASS → H0 rejected.** The frozen ALX daily book survives the honest
post-only cost model (4 bps maker + 3 bps adverse-selection on turnover **and**
a φ = 0.80 non-fill haircut on gross), remaining net-profitable and fold-stable.
Single run, no tuning, per `PREREGISTRATION.md` (LOCKED 2026-07-22, before any
ALX5 code). Branch `alpha/alx5-maker-execution`.
Reproduce: `py alpha_library/alx5_maker_execution/run.py`.

**A PASS authorizes only one thing:** drafting a *separate*, forward,
maker-paper pre-registration (the plan's Lever-3 / "ALX6-forward" step). **No
real money, no production graduation, no sizing** flows from this experiment.

Object under test: the **execution/cost model** only. The book itself is frozen
and reused verbatim from ALX4 — `signal = -funding.rolling(7).mean()`,
dollar-neutral top/bottom 30%, hold = 1 day, lag = 1 day — run through
`alpha_library.alx_funding_price_validation.validate.backtest` on the independent
Binance USDT-M reconstruction (40 majors, 2019-09-08 … 2026-07-12, 2500 days,
cached by ALX3 from raw REST; no project parquet). `gross = price PnL + funding
carry` is identical across every cost model; the models differ only in the
per-turnover charge and the fill haircut φ.

---

## Stage A — machinery / anchor sanity (kill-test) — **PASS**

Verifies the reused engine + panel are intact before any maker number is read.

| quantity | value | gate | result |
|---|---|---|---|
| MED (7.5 bps) net Sharpe | **+1.941** | = ALX4 anchor +1.94 ± 0.15 | OK |
| rank-IC mean | +0.0183 | > 0 | OK |
| rank-IC Newey-West t (lag 5) | **+4.70** | ≥ 3.0 | OK |

The base reproduces ALX4 to three decimals — the engine and panel are unaltered.

## Stage B — net viability under honest maker cost (decisive) — **PASS**

All metrics under the **realistic-maker** model (7 bps on turnover, φ = 0.80).

| gate | value | threshold | result |
|---|---|---|---|
| full-window net Sharpe | **+1.890** | > 1.0 | OK |
| walk-forward folds net-positive | **6 / 6** | ≥ 4 / 6 | OK |
| — 6-fold Sharpe | +1.74, +3.30, +1.05, +0.18, +2.24, +2.13 | | |
| untouched pre-2023 fold (2021-07-07…2023-07-07, n=731) | **+1.399** | > 0 | OK |

The φ = 0.80 haircut plus 7 bps does **not** collapse the edge: net Sharpe falls
only from +1.94 (MED) to +1.89, because realized turnover is low (~0.22/day) so
the per-turnover charge is small relative to gross. H0 ("cheap execution, not a
durable premium, was carrying it") is rejected on this window.

## Stage C — the maker story is real, not a leak (kill-tests) — **PASS**

1. **lag-0 vs lag-1 tell** — lag-0 gross Sharpe **+3.10** vs traded lag-1 **+2.25**.
   Lag-0 is conspicuously better (same-bar information exists) and the traded
   lag-1 book does **not** depend on look-ahead. OK.
2. **execution attribution** — the maker-vs-taker net gap equals
   `turnover · (11 − 7) bps` to machine precision, and the realistic net equals
   `0.80 · gross − turnover · 7 bps` bit-for-bit. The maker uplift comes
   **entirely** from the cost/turnover delta — the engine manufactures no edge. OK.
3. **adverse-selection escalation (robustness)** — under a harsher φ = 0.70 and
   +4 bps adverse (8 bps on turnover), net Sharpe stays **+1.780 > 0**. A mild
   worsening of the fill assumption does **not** flip it negative. OK.

## Descriptive — cost × hold ladder (diagnostic only, NOT gated)

Per §4, this **cannot** change the verdict and no "best hold" is selected. Hold >
1 is reported purely to characterize the turnover→net-cost tradeoff; if it looks
live it becomes a *separate* future experiment (ALX6), never folded in here.

| hold | avg turnover | taker (11) | MED (7.5) | ideal maker (4) | realistic maker (7 + φ0.8) |
|---|---|---|---|---|---|
| 1 | 0.221 | +1.80 | +1.94 | +2.08 | +1.89 |
| 2 | 0.178 | +1.79 | +1.92 | +2.05 | +1.87 |
| 3 | 0.153 | +1.79 | +1.89 | +1.99 | +1.86 |
| 5 | 0.128 | +1.90 | +1.99 | +2.07 | +1.95 |

---

## Honest nuance (does NOT change the verdict)

The §0 prior expected the **taker** book to be *marginal-to-negative* — "the
maker route is what makes it real." On this book that expectation did **not**
materialize: taker itself is Sharpe **+1.80**. The reason is turnover: the §0
prior's "−4.1 bps/day after 11 bps taker" assumed a full round-trip **per name**,
whereas the frozen book's signal persistence keeps daily turnover at ~0.22, so 11
bps taker costs only ~2.4 bps/day, not ~4+. So the *decisive* claim H1 required —
survival under **honest maker with the φ haircut** — is confirmed, but the
sharper narrative "taker is dead, maker rescues it" is **not** supported by the
data. The edge is carried by **low turnover**, and maker execution makes it
cheaper still rather than being the sole thing standing between profit and loss.

This nuance is recorded, not acted on. The PASS is defined purely by the locked
Stage A/B/C gates (all satisfied); the taker outcome was descriptive, never a
gate, so it does not reopen the spec.

## Why the alpha exists economically (required before any further work)

The book is **long the most-negative-funding names, short the most-positive**.
Persistently negative funding is the market paying shorts to hold — a crowded-short
/ high-demand-to-borrow state that historically mean-reverts up; persistently
positive funding is crowded-long, paid by longs, and reverts down. The edge is a
**funding-carry + crowding-reversal** premium, not a price-momentum or size
artifact (ALX4 P3: multivariate R² vs momentum/reversal/beta/size/vol books is
low; the funding book is independent). Crucially the premium **persists** — the
7-day funding mean changes slowly — which is exactly why turnover is low and why
maker execution is viable rather than eaten by rebalancing cost.

## Limits / what a PASS does NOT establish

- Not a live-fill model: φ and the adverse-selection bps are **assumptions**, not
  measured post-only fills. The real fill distribution is the open question the
  forward maker-paper protocol must settle.
- No borrow/short-availability, no funding-of-the-hedge, no capacity/impact at
  size beyond ALX3's Stage-7 band. Those are forward-protocol concerns.
- Single window; the pre-2023 fold is the only genuinely out-of-sample check and
  it is positive but lower (+1.40).

**Next step (only sanctioned action):** draft a separate, forward, post-only
maker-paper pre-registration whose positive out-of-sample fill window is the sole
path toward any sizing decision.
