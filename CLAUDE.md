# CLAUDE.md

Guidance for working in this repo. These rules are **binding** — they override
default behavior. This is an alpha-research codebase; the point is to find
signals that are *real*, not signals that *look good*. Self-deception is the
enemy.

## The research protocol (non-negotiable)

Every experiment obeys all of these. No exceptions, no "just this once".

1. **Pre-register before code.** Write `PREREGISTRATION.md` (hypothesis H1, null
   H0, economic mechanism, universe/data, features, target, execution lag, cost
   assumptions, statistical gates, and an objective PASS/FAIL committed *in
   advance*) and lock it **before** writing any experiment code or seeing any
   result. Start from the assumption the alpha is FALSE.
2. **One experiment = one hypothesis.** One pre-registration tests exactly one
   economic claim. Don't bundle.
3. **Every experiment in its own branch.** Branch off `main`. Naming:
   `alpha/<id>-<slug>` for library candidates, experiment id (E1…E9…) for the
   research ladder. One squashed **commit only after the experiment is
   complete**.
4. **Single run. No tuning.** Run the pre-registered procedure once. No grid
   search over the gate, no "one more variant", no re-parameterisation, no
   threshold changes after seeing results. Reported diagnostics are diagnostics,
   not knobs.
5. **Never optimize after seeing results.** Observing the outcome ends your
   freedom to change the spec. If the hypothesis fails, that failure *is* the
   recorded result.
6. **Honest reports.** Write `REPORT.md` after the single run with the real
   verdict (PASS / FAIL → H0 not rejected). Document every falsified hypothesis
   as fully as every survivor. A clean falsification is a full-value outcome.

## E1–E9 are immutable

Closed experiments (E1–E9 and any committed `alpha/*` result) are **frozen**.
Never edit their code, parameters, thresholds, conclusions, or reports. New work
may **reuse their infrastructure only** — the neutral shared machinery under
`crypto_signal_bot/research/` (`xsection.{universe,ic,portfolio,features}`,
`research.metrics`, Newey-West, factor features, `portfolio.backtest_portfolio`)
— **never their alpha, signal, or tuned parameters.** All new experiment code
lives in its own directory (`experiments/<id>/` or
`alpha_library/<slug>/`), self-contained.

## Alpha Library admission

An effect only becomes an Alpha Library candidate **after** it survives a
dedicated falsification experiment (its own branch + pre-registration + single
run). The admission battery, decisive first:

- **Phase 1 — verification (can REJECT):**
  - **A. Reproducibility** — an *independent* from-scratch reimplementation
    (not the discovering code) reproduces the result, triangulated against the
    shared engine and the original number.
  - **B. Leakage / implementation** — future-corruption invariance (no
    look-ahead), sign-reversal flips the Sharpe, random-signal placebo (engine
    must not manufacture Sharpe), timestamp/alignment audit, lag-0-vs-lag-1 tell
    for same-bar leakage. Any artifact ⇒ FAIL.
  - **C. Known-factor attribution** — multivariate OLS vs identically
    constructed factor-mimicking books (reversal, momentum, beta, vol, size,
    …). A statistically and economically meaningful **unexplained** intercept
    must survive. If a known factor explains ~all of it ⇒ FAIL.
  - **If Phase 1 fails, STOP** — do not run Phase 2.
- **Phase 2 — characterization (descriptive only; cannot reject or tune):**
  costs, exec-lag, walk-forward folds, year-by-year, vol regimes, universe
  subsets, bootstrap CI. A lower Sharpe under tougher assumptions is *expected*
  and is not a failure. Only question: does the alpha *disappear*, or *remain
  economically meaningful*?
- The REPORT must explain **why the alpha exists economically** before any
  further work (production graduation, sizing) is proposed.

## Staged gates (research ladder pattern)

Experiments use a Stage A → B → C gauntlet with a hard **stop rule** between
stages: Stage A cost-anchored dispersion sanity → Stage B rank-IC hard gates
(Fundamental-Law breadth, Newey-West t ≥ 3, cross-fold sign stability, and any
decisive kill-test) → Stage C survival under realistic cost (net Sharpe > 1,
≥ 4/6 walk-forward folds positive, capacity). Failing an earlier stage stops the
experiment; do not proceed or add features.

## Designing robustness / adversarial tests (lesson from ALX2)

When pre-registering robustness or concentration tests, **prefer symmetric or
distributional stresses over ones that remove only positive extremes.**

- Removing only the best *k*% of days (as ALX2's T2 did) is an **asymmetric**
  stress: it deletes right-tail winners while keeping every loser. Genuine
  financial strategies naturally concentrate returns in relatively few
  observations, so this stress is over-aggressive — a real daily strategy whose
  annualized Sharpe is not very high can fail it even when it is sound.
- Prefer instead: **symmetric trimming** (both tails), **contribution analysis**
  (report the PnL share of top days as a diagnostic, not a hard gate), or
  **CVaR / tail-based** robustness. Gate on symmetric or economically motivated
  criteria, not on the removal of winners alone.

This lesson is **forward-looking only**. It does **not** change the ALX2 verdict
(FAIL at T2 stands: the criterion was locked before the run, correctly computed,
and the protocol forbids retroactive reinterpretation). Apply it to the design of
*future* pre-registrations, never to relitigate a locked one.

## Environment & conventions

- **Python:** use `py`, **not** `python` (PATH `python` is a broken MS Store
  stub). One-command reproducibility per experiment, e.g.
  `py experiments/e9/run.py`.
- **Shell:** Windows / PowerShell primary; Bash tool available for POSIX.
- **Deliverables per experiment:** isolated code + `run.py` (one command,
  Stage/Phase-gated, prints verdict) + `PREREGISTRATION.md` + `REPORT.md` +
  known-answer unit tests under `tests/` + artifacts under `data/research/<id>/`
  + update the running `RESEARCH.md` log + single commit.
- **Repo map:** `crypto_signal_bot/` — the bot + shared research infra
  (`research/`, `backtest/`, `data/`, `model/`, `features/`, `live/`);
  `experiments/<id>/` — frozen research-ladder experiments;
  `alpha_library/<slug>/` — falsification-validated alpha candidates;
  `RESEARCH.md` / `PROGRESS.md` / `REPORT.md` — logs.
