# ALX3 — Independent External Replication — REPORT

**VERDICT: C. REJECTED.** Under the locked criteria the funding→price alpha does
**not** survive independent external replication: on the primary independent venue
(Binance) its net Sharpe is **not distinguishable from zero** at 95%, and on the
untouched pre-development time period the effect **vanishes**. Single run, no
tuning, per `PREREGISTRATION.md`. Branch `alpha/alx3-external-replication`.
Reproduce: `py alpha_library/alx3_external_replication/run.py`.

Object under test: the frozen ALX funding-ranked dollar-neutral daily book
(`−7d-mean funding`, top/bottom 30%, hold 1d, lag 1d, Med 7.5 bps), reused
verbatim through `alpha_library.alx_funding_price_validation.validate`, run
unchanged on data reconstructed from scratch on independent venues.

---

## Stage 1 — Independent data reconstruction — CLEAN

Rebuilt entirely from raw exchange REST APIs (Binance/OKX/Bybit), using none of
the old Bybit parquet or `funding_panel.parquet`.

| audit | result |
|---|---|
| Binance symbols reconstructed | 40, 2019-09-08 … 2026-07-12 |
| off-grid settlement fraction (hours ∉ {0,8,16}) | 0.30% (negligible) |
| funding sign convention | positive = longs pay shorts (same on all venues) |
| timezone / aggregation | UTC epoch-ms; daily = Σ of {0,8,16} settlements |
| funding future-corruption invariance (no look-ahead) | **PASS** |

The reconstruction is point-in-time correct. Note: the frozen ALX
*intersection-universe* rule caps the common window at the earliest-ending
symbol — **EOSUSDT was delisted ~2025-05-21** (on both Binance and Bybit), so the
faithful Stage-2 window is **2023-03-23 … 2025-05-21** (790 days), shorter than
Bybit's original 2023-07…2026-07. This is a faithful consequence of the frozen
construction, not a choice.

## Stage 2 — Different exchange (Binance) — **FAIL**

| | net Sharpe | 95% block-bootstrap CI | DSR (N=100) |
|---|---|---|---|
| Binance faithful window (790d) | **+1.353** | **[−0.01, +2.79]** | 0.29 |

The point estimate is positive and close to ALX's +1.71, but the **95% CI
includes zero** (lower bound −0.01) → by the pre-committed definition the effect
is **statistically indistinguishable from zero** on the primary independent
venue. Deflated Sharpe (0.29) confirms it does not clear multiple-testing. **FAIL.**

## Stage 3 — New time period (pre-2023, untouched) — **FAIL (decisive)**

The most important test: the 2-year Binance window immediately **before** the
development period, never used in any experiment.

| | net Sharpe | 95% CI | DSR |
|---|---|---|---|
| Binance 2021-07-07 … 2023-07-07 (34 names, 730d) | **+0.208** | **[−1.14, +1.64]** | 0.01 |

On genuinely out-of-sample time the effect **essentially disappears** (Sharpe
0.21, CI wide around zero, DSR 0.01). This is the single strongest piece of
evidence in the whole study and it **falsifies the alpha as a stable premium.**

## Stage 4 — Cross-exchange consistency — mixed

| venue | window | net Sharpe | CI-lo | funding→price corr |
|---|---|---|---|---|
| Binance (independent) | 2023-03…2025-05 | +1.353 | −0.01 | **+0.571** |
| OKX (independent, short) | 2026-04…2026-07 (96d) | +1.90 | −1.45 | **+0.581** |
| Bybit (control, fresh re-fetch) | 2023-10…2025-05 | +1.90 | **+0.48** | **+0.562** |

The **funding→price correlation reproduces almost identically on all three
venues (+0.57 / +0.58 / +0.56**, matching ALX's +0.55) — the *economic
relationship* (crowded funding predicts relative underperformance) is real and
venue-independent. But the *tradable Sharpe* is only significant on the **Bybit
control** (same exchange as discovery — a pipeline check, not independence). On
the independent venues it is not distinguishable from zero (Binance CI touches 0;
OKX has only 96 days of funding history → no power).

## Stage 5 — Rolling out-of-sample — time-dependent (red flag)

Per-year Binance Sharpe: **2023: +0.45 → 2024: +1.47 → 2025: +2.83.** 6-fold:
[+2.57, −0.41, −0.60, +2.42, +2.02, +2.71] (4/6 positive). The effect is **absent
early and strengthens toward the present** — consistent with a regime/epoch
effect concentrated in the development window, not a stable, always-on premium.

## Stage 6 — Point-in-time universe (union) — positive but confounded

PIT-universe over full history (2019-09-09 … 2025-05-21, 2082 days): Sharpe **+1.93**, CI
[+1.11, +2.74], DSR 0.99. Strong — **but this window spans the development era**,
and its strength comes from 2023-2026 (per Stage 5). Per the pre-registration's
binding clause, this **does not rescue** the verdict: it is not an independent
new-time test, and the clean pre-2023 test (Stage 3) already failed. Reported for
completeness only.

## Stage 7 — Realistic execution — survives cost

Cost band: optimistic (2 bps) +1.71 | Med (7.5 bps) +1.35 | **pessimistic
(liquidity-aware, ADV impact + spread + short borrow, avg 22.3 bps) +0.38**. The
point estimate stays > 0 under harsh costs — so the Stage-2 failure is a
**statistical-significance / out-of-sample** failure, not a cost failure.

## Stage 8 — Economic mechanism — plausible but insufficient

Daily funding median +0.00027, 17.6% negative observations, funding cash-flow =
37% of net PnL. The proposed mechanism (a positioning / crowding-unwind premium:
fade persistently crowded funding) is economically coherent and is supported by
the stable +0.57 cross-venue correlation. **However**, a genuine risk premium
should persist out-of-sample in time; its disappearance pre-2023 and its
monotone strengthening through 2023-2025 point instead to a **regime-dependent /
development-epoch** phenomenon. The mechanism is *plausible as a correlation* but
does **not** explain a stable tradable alpha, given the replication failure.

---

## Missing data (honesty clause — not approximated)

- **OKX funding history is only ~96 days** (public endpoint depth limit) → OKX
  could not provide a multi-year independent replication, only a short recent
  sign-check. *Needed:* a full OKX funding archive (paid/bulk) to test OKX
  properly over years.
- **Forward (post-development) out-of-sample does not exist** — development runs
  to the present date. Stage 3 used the untouched *pre*-2023 period instead.
  *Needed:* let real time pass and re-test on data generated after this study.
- **Deribit** — unsuitable venue (no broad USDT-perp cross-section); not tested.
- **Delisted-name survivorship** — reconstructable only via onboard dates, not a
  full delisted-inclusive live universe (EOS truncation above illustrates the
  sensitivity). *Needed:* a point-in-time delisted-inclusive constituent history.

Per the pre-registration, none of these gaps were filled with approximations, and
the external-replication failure was **not** compensated with extra internal
stress tests.

---

## Conclusion

**C. REJECTED (at the locked bar).** The funding→price *correlation* is real and
venue-independent (+0.56–0.58 everywhere), but the frozen *tradable daily
strategy* fails independent external replication: not statistically distinguishable
from zero on Binance (CI [−0.01, +2.79], DSR 0.29), and effectively zero on the
untouched pre-2023 period (Sharpe +0.21). Its apparent edge is concentrated in,
and increases through, the 2023-2025 development epoch, indicating regime/epoch
dependence rather than a stable, arbitrage-resistant premium.

The verdict is **not** softened to "PLAUSIBLE": the pre-committed rule
(CI-includes-0 on the independent venue → REJECTED) is triggered, and — decisively
— the most heavily weighted criterion, independent replication on a **new time
period**, failed clearly. Consistent with the discipline applied to ALX2, a locked
criterion binds even when the point estimate looks favorable. The alpha does
**not** graduate to the Alpha Library; it is **rejected on independent data**.

*Re-opening is permitted only under the pre-registered conditions: genuinely new
(forward) data once time has passed, a full independent multi-year OKX/other-venue
funding archive, or a concrete implementation bug — none of which is available now.*
