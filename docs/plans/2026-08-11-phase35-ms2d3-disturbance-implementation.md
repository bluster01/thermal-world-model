# Phase 3.5-MS2-D3 Colored-Disturbance Implementation Plan

> Status: IMPLEMENTED / LOCALLY VERIFIED / READY FOR LINUX VALIDATION (2026-08-11)

**Goal:** Implement a fail-closed 21-run D3 validation that tests D2 response recovery under an action-independent stationary AR(1) output disturbance.

**Architecture:** Extend the shared synthetic generator with an opt-in colored-disturbance truth that leaves all previous regimes byte-stable. Add a D3-specific frozen matrix, runner and episode-level validation summary; D3 reuses existing operators but has independent protocol IDs, output paths and test locks.

**Tech Stack:** Python 3.11, PyTorch, JSON, pytest, Git.

---

### Task 1: Add failing generator contract tests

**Files:**
- Modify: `tests/phase35/multistep/test_model_and_synthetic.py`

**Steps:**
1. Test stationary AR(1) determinism, finite shape and truth metadata.
2. Toggle disturbance off with the same seed and assert identical context/action/reference/profile/clean effect.
3. Assert target delta equals the exposed disturbance and existing regimes remain valid.
4. Run the focused tests and confirm RED before implementation.

### Task 2: Implement the opt-in disturbance generator

**Files:**
- Modify: `src/phase35/multistep/synthetic.py`

**Steps:**
1. Add `disturbance_std` and `disturbance_tau_seconds` with zero defaults.
2. Add `disturbed_context_scheduled` validation rules.
3. Generate stationary AR(1) only when enabled; do not consume RNG for legacy regimes.
4. Expose `colored_disturbance` and provenance metadata.
5. Run focused generator and prior D1/D2 tests.

### Task 3: Freeze D3 matrix and runner tests

**Files:**
- Create: `configs/phase3_5/ms2d_disturbance_matrix.json`
- Create: `tests/phase35/multistep/test_ms2d_disturbance_cli.py`

**Steps:**
1. Freeze 7 candidates × 3 seeds, D2 budgets, `sigma_d=0.03`, `tau_d=120 s` and unchanged primary thresholds.
2. Test 21-run dry-run, mutation rejection, test lock and CPU smoke.
3. Test that formal validation writes episode metrics but never test metrics.

### Task 4: Implement D3 validation runner

**Files:**
- Create: `experiments/phase3_5/ms2d_disturbance.py`

**Steps:**
1. Validate the full frozen matrix and exact candidate payloads.
2. Train with validation-effect MAE selection and fail-closed resume rules.
3. Reload the selected checkpoint on validation and write paired episode metrics plus disturbance diagnostics.
4. Preserve manifest/environment/hash/history/test-lock contracts.

### Task 5: Implement episode-level validation summary

**Files:**
- Create: `experiments/phase3_5/summarize_ms2d_disturbance.py`

**Steps:**
1. Verify all 21 manifests/checkpoints/histories/episode files and code equivalence.
2. Reproduce aggregate metrics from episodes.
3. Apply oracle, absolute and paired-bootstrap gates per seed.
4. Report D2 drift, profile/horizon, tau, delay and disturbance diagnostics separately.
5. Exit 2 after preserving a scientifically failed summary.

### Task 6: Update status and Linux handoff

**Files:**
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `TODO.md`
- Modify: `docs/PHASE35_CONTEXT_SNAPSHOT.md`
- Modify: `docs/PHASE35_MAINLINE_CONTEXT.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `results/README.md`
- Modify: `experiments/phase3_5/README.md`

**Steps:**
1. Close D2 with the Supervisor audit and set D3 to local implementation while coding.
2. Run focused tests, CPU smoke, dry-run, compileall and all `tests/phase35`.
3. Only after clean verification set D3 as the sole `ready_for_linux` Gate.
4. Commit and push; Linux may write only `results/phase3_5/ms2d_disturbance/**`.
