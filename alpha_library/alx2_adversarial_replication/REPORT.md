# ALX2 — Adversarial Replication — FALSIFICATION REPORT

**VERDICT: FAIL.** First falsifying test: **T2 — temporal concentration.**
Per the locked stop rule the battery halted at T2; T3–T6 were not evaluated.
Single run, no tuning, per `PREREGISTRATION.md`. Branch
`alpha/alx2-adversarial-replication`. Reproduce:
`py alpha_library/alx2_adversarial_replication/run.py`.

Object under test (frozen, unchanged): the ALX funding-ranked dollar-neutral
daily book — long top 30% / short bottom 30% by `−(7-day mean daily funding)`,
hold 1d, lag 1d, Med cost. Frozen net Sharpe **+1.707** (39 symbols × 1100 days,
2023–2026).

---

## T1 — Point-in-time funding reconstruction & alignment audit — **SURVIVED**

The one documented ALX caution was funding-panel provenance (the reused E7 panel
was `merge_asof`/ffill-built; a construction-time leak would evade the
return-corruption test). T1 attacks it by **independently re-fetching** funding
for all 39 symbols and reconstructing the daily realized-funding panel strictly
point-in-time.

| check | result |
|---|---|
| off-grid settlement fraction (hours ∉ {0,8,16}) | 0.81% (negligible) |
| funding-side future-corruption invariance (no forward leak) | **PASS** |
| point-in-time panel vs reused E7 panel correlation | **1.0000** |
| point-in-time net Sharpe @Med | **+1.707** (reused +1.707) |

The freshly re-fetched, strictly point-in-time panel is **identical** to the
reused one and the Sharpe is unchanged. **The funding-provenance caution is
resolved — there is no alignment or look-ahead artifact in the funding input.**
T1 survives cleanly.

## T2 — Temporal concentration (top-day / crisis dependence) — **FALSIFIED**

Pre-registered falsifier: *after dropping the best 5% of PnL days, net Sharpe ≤ 0,
OR excluding the top-volatility decile, net Sharpe ≤ 0.*

| probe | result | falsifier |
|---|---|---|
| drop best 5% (55) PnL days → net Sharpe | **−1.160** | **≤ 0 → FALSIFIES** |
| exclude top-volatility decile → net Sharpe | +1.649 | > 0 (passes) |
| top-1-day PnL share | 7.1% | — |
| top-5%-days PnL share of total | **150.2%** | — |

The strategy is **not** crisis-window dependent (removing the most volatile 10%
of days leaves Sharpe +1.65), and it does **not** hinge on any single day (the
best day is only 7.1%). But it **is** heavily concentrated in a small cluster of
best days: the top 5% of days (≈55 of 1100) contain **150%** of cumulative PnL,
i.e. the remaining 95% of days collectively **lose** money. Removing those best
days flips the book to a negative Sharpe (−1.16). By the locked criterion this
falsifies the alpha at T2, and the stop rule halts the battery.

**This verdict stands and is not overturned.** Two honest points are recorded
around it, neither of which changes the outcome:

1. *Economic reading.* The concentration is consistent with the alpha's own
   proposed mechanism (a positioning / crowding-unwind premium): the book earns
   in episodic bursts when crowded funding-heavy positions unwind and bleeds
   small carry/cost the rest of the time. Episodic-but-real and
   fragile-because-episodic are not mutually exclusive; the test measures the
   latter.
2. *Methodological caveat (recorded, NOT grounds to overturn).* "Drop the best
   5% of days" is an **asymmetric** stress — it removes only right-tail days
   while keeping every loser — so it is severe for any daily strategy whose
   annualized Sharpe is not very high (for a Gaussian Sharpe-1.7 daily book the
   top 5% of days already exceed 100% of total PnL). The bar was, in hindsight,
   stringent. **But it was locked before the run, the computation is correct
   (verified — not an implementation bug), and the protocol forbids
   re-parameterising a criterion after seeing the result.** A poorly-calibrated
   but correct threshold still binds. The escape hatch in §5 (re-run only on new
   data, a concrete implementation bug, or an external contradiction) does **not**
   apply: nothing here is a bug.

## T3–T6 — not evaluated

Per the pre-registered stop rule, the battery stopped at the first falsifying
test. Leave-one-symbol-out (T3), expanded factor attribution (T4),
liquidity-aware costs (T5), and the deflated Sharpe ratio (T6) were not run and
carry no verdict.

## Survivorship bias — limitation only (not scored)

A true point-in-time live universe including delisted assets is unavailable; the
`SURVIVORSHIP_SAFE` list holds survivors only. This is a documented limitation of
the whole ALX/ALX2 line, not a test, and does not affect the verdict.

---

## Conclusion

**FAIL at T2.** The alpha is reproducible and, per T1, free of any
funding-alignment / look-ahead artifact — a genuine result of the audit. It is
nonetheless **rejected by the final adversarial battery** because its return is
concentrated in the best ~5% of days to the degree that removing them turns the
Sharpe negative, breaching the locked T2 concentration bar.

Under the pre-registered stopping rule the alpha does **not** graduate to an
accepted Alpha Library component in this replication. It remains a *candidate*.
No further adversarial test may be invented to rescue it; per §5 a re-run is
permitted only on new data, a concrete implementation bug, or an external
replication that contradicts these results — none of which obtains here.
