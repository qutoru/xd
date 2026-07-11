# E7 — Cross-Sectional Short-Term Reversal: Final Report

**Status: CLOSED (falsified at implementation).** A real signal was found and
then honestly rejected because it does not survive taker execution costs.
Full experiment log and context (E1–E6) live in `RESEARCH.md`.

Reproduce everything from scratch with one command:

```
py reproduce_e7.py
```

(fetch missing 1h/3y universe → funding panel → Stage A/B → Stage C → unit tests)

---

## 1. Hypothesis

After six experiments (E1–E6) confirmed that **directional, single-asset**
prediction of 15m–1h BTC returns from public OHLCV has no exploitable edge, E7
changed the *estimand* to a structurally different one:

> **E[ r_{i,t+h} − r_{market,t+h} | cross-sectional features ]**
> — a symbol's return *relative to the market*, i.e. the residual after removing
> the dominant, unpredictable market factor.

Economic rationale: cross-sectional premia (rotation, liquidity provision,
positioning) are **flow/behavioural**, not information-efficiency, so they
survive arbitrage longer than directional prediction. Grounding: Grinold-Kahn
(Fundamental Law), Jegadeesh-Titman 1993, Asness-Moskowitz-Pedersen 2013,
Liu-Tsyvinski-Wu 2022 (crypto factors), Lehmann 1990 / Lo-MacKinlay 1990
(short-horizon reversal).

## 2. Methodology (falsification-first, pre-registered)

- **Universe:** survivorship-safe — a fixed *ex-ante* list of 40 majors, kept
  only if they cover the full window; 39 symbols × 1h × 3y (2023–2026). Excludes
  newly-listed pumps (conservative).
- **Target:** market-neutral forward return (cross-sectionally demeaned).
- **Predefined features (no additions allowed):** relative strength (CS
  momentum), short-term reversal, funding dispersion.
- **Gates pre-registered with justified thresholds** (theory / literature /
  labelled heuristics — see `RESEARCH.md` E7):
  - **Stage A:** residual dispersion ≥ 2× round-trip cost *(cost-anchored)*.
  - **Stage B:** IC ≥ IR/(TC·√BR) *(Fundamental Law)*; Newey-West t ≥ 3
    *(Harvey-Liu-Zhu 2016)*; fold-sign binomial p < 0.05; **1-bar-gap
    microstructure kill-test**.
  - **Stage C:** full market-neutral long/short grid, net-of-cost Sharpe.
- **Stop rule:** if a stage's gates fail, stop — no feature-adding, no tuning.
- Deterministic, config-driven, unit-tested (`tests/test_xsection.py`, 8 passed).

## 3. Results

**Stage A — PASS.** Median cross-sectional dispersion 0.449% >> 2× round-trip
0.22%. Effective breadth N_eff = 25.7 (PC1 only 11.7% — majors are weakly
one-factor at 1h). Required |IC| ≥ 0.0042.

**Stage B — PASS (signal exists).**

| feature | mean IC | NW t | folds (binom p) | 1-bar-gap IC | verdict |
|---|---|---|---|---|---|
| relative_strength | −0.0266 | −18.8 | 6/6 (0.016) | −0.018 | PASS (inverted → reversion) |
| short_term_reversal | **+0.0349** | **+24.5** | 6/6 (0.016) | +0.017 | **PASS** |
| funding_dispersion | +0.0045 | +4.2 | 4/6 (0.109) | +0.003 | FAIL (B3/B4) |

Both OHLCV features are the same effect from opposite signs: **cross-sectional
short-term reversal** — recent losers outperform recent winners. Statistically
overwhelming (t≈24), stable across all folds and vol regimes. It survives a
1-bar gap (so it is not *only* bid-ask bounce), but loses ~52% of magnitude,
warning that the tradeable part is smaller.

**Stage C — FAIL (does not survive execution).** Full 90-config grid × 3 costs:

- Positive IC **does** translate to positive **gross** Sharpe — but only at the
  shortest horizon: gross Sharpe by holding period H1 **+1.27** → H2 +0.74 →
  H4 −0.17 → worse beyond. The edge is a 1-bar effect that decays in 2–4 bars.
- Capturing it needs **turnover ≈ 7,198×/year** ⇒ cost drag **144%/yr even at an
  optimistic 2 bps**, 540%/yr at realistic 7.5 bps.
- **0 of 90 configs are net-positive at ANY cost level.** Best net Sharpe −4.13.
- Deep dives (rep config, realistic cost): walk-forward all 6 folds −12…−19,
  every year negative, every regime negative. Top-20 most-liquid subset still
  −12.6 (not a small-cap artifact). Capacity ≈ $9.5M.

## 4. Economic cause of failure

The return to short-horizon cross-sectional reversal **is a liquidity-provision
premium**: it is the compensation earned for absorbing order flow and holding
inventory as prices mean-revert over minutes. That premium accrues to whoever
**earns the spread** — a market maker posting passive limit orders — not to
whoever **pays the spread** with taker orders. The signal is real (Stage B), but
we are structurally on the wrong side of the bid-ask spread: the ~1.3 gross
Sharpe is an order of magnitude smaller than the 144%+ annual cost of the
turnover required to harvest a 1-bar effect. This is the textbook Lehmann (1990)
/ Lo-MacKinlay (1990) result — short-horizon reversal ≈ inventory/liquidity
compensation, capturable only via maker execution.

## 5. What we learned

1. **A negative result can be precise.** E7 didn't just say "no" — it located
   the edge, quantified it, and named exactly why it isn't ours to take (wrong
   side of the spread). That is more valuable than a vague failure.
2. **IC ≠ tradeable Sharpe.** A robust, highly-significant IC (t≈24) still died
   after costs. The microstructure kill-test and the gross-Sharpe-vs-holding
   curve were what exposed it; a naive PnL backtest of one config could have
   hidden or faked the story either way.
3. **The estimand matters more than the model.** Switching from directional
   single-asset to cross-sectional residual was the first change in seven
   experiments to produce *any* real signal — validating the "change the class
   of problem, not the parameters" thesis, even though this particular signal is
   unharvestable for us.
4. **Costs, not alpha, are the binding constraint at high frequency.** Any future
   intraday idea must clear a turnover/cost sanity check *first*.
5. **Infrastructure compounded.** The reusable panel/IC/portfolio harness and
   the pre-registered gate discipline are now assets for any next experiment.

**Bottom line:** E7 is a valuable, fully-documented negative result. The
cross-sectional short-term reversal signal is genuine but is a maker-side
liquidity premium that taker execution cannot capture. Branch closed.
