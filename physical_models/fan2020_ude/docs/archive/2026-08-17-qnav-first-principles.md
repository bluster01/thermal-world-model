# QNAV First-Principles Closure Implementation Plan

**Goal:** Determine whether qnav's apparent gain comes from a defensible residual correction or from duplicated energy injection and access to measured future spray flow.

**Architecture:** Keep the frozen evaporation grey-box, residual MLP, data window and optimizer unchanged. Add one experiment driver that changes only residual injection location and whether residual features contain `W`; evaluate two blocked development folds with seed 0. Linux performs training and returns raw artifacts, while scientific interpretation remains local.

**Tech Stack:** Python, PyTorch, NumPy, pandas, existing `02_train.py`, `09_residual.py`, and `26_fix_evap.py` components.

---

## Frozen scientific questions

1. Does `h_only` retain conditional-rollout accuracy relative to the current `double` injection?
2. Does removing `W` from residual features materially change prediction or counterfactual response?
3. Does an energy-transfer residual (`metal=-z`, `steam=+z`) remain usable?
4. Does the same final model close both wet and dry PI loops when only the measured valve-rate limit is retained?

## Candidate matrix

| ID | Residual injection | Residual reads W | Training |
|---|---|---:|---:|
| `evap_only` | none | no | frozen baseline |
| `double_w` | `(+z,+z)` | yes | yes |
| `h_w` | `(0,+z)` | yes | yes |
| `h_now` | `(0,+z)` | no | yes |
| `conservative_now` | `(-z,+z)` | no | yes |

Folds are blocked and development-only: F0 train `[0,20000)`, validation `[20000,25000)`, evaluation `[25000,30000)`; F1 train `[0,30000)`, validation `[30000,35000)`, evaluation `[35000,40000)`. Each evaluation rollout is 1800 steps. The historical `[40000,50000)` segment is not used by Q32.

## Required artifacts

- one checkpoint and one metrics JSON per learned candidate/fold;
- one summary JSON containing all 10 units;
- conditional-rollout RMSE and per-output RMSE;
- wet/dry valve-only, W-only, coupled, and coupled-with-residual-off step responses;
- wet/dry absolute closed-loop tracking, tail variation, valve range, saturation, and reversal counts;
- manifest with commit, matrix hash, data path/hash, torch version and exact command.

## Implementation tasks

### Task 1: Freeze matrix and pure helpers

Create `configs/qnav_first_principles_matrix.json` and `32_qnav_first_principles.py`. Implement pure helpers for candidate expansion, injection fluxes and residual-feature construction.

### Task 2: Add tests

Create `tests/test_qnav_first_principles.py`. Verify matrix closure, all four injection identities, and that `h_now`/`conservative_now` residual features are invariant to changes in `W`.

### Task 3: Implement execution

Reuse the frozen evaporation checkpoint, train only the residual MLP, run the blocked-fold conditional rollout, and emit the four action-path probes plus absolute PI-loop diagnostics. Do not use the 883 s lag or true future values during constant-condition interventions.

### Task 4: Verify and release

Run compileall, unit tests and `--dry-run`. Commit code/config/plan only and push `adhoc/lumped-enthalpy`. Linux is authorized only for the frozen command printed by `--dry-run`; it must not change candidates, folds, seed, thresholds or conclusions.
