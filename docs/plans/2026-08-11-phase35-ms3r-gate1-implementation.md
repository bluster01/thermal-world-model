# Phase 3.5 MS3-R Gate 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the validation-only MS3-R Gate A framework for point contracts, timing/placebo diagnostics, residual excitation, and dual-input rank without authorizing Linux or touching test.

**Architecture:** Reuse the frozen cross-side `Phase35Cache` artifacts and parent MS3 matrix. A pure NumPy analysis core performs past-only rolling cross-fitting and produces deterministic machine artifacts; a thin CLI enforces hashes, clean-worktree execution, and validation-only scope. Local synthetic tests verify signs, placebos, rank diagnostics, split isolation, and CLI output.

**Tech Stack:** Python 3.11+, NumPy, PyTorch only for the existing branch-semantics probe, pytest, JSON/NPZ artifacts.

---

### Task 1: Freeze the Gate A contract

**Files:**
- Create: `configs/phase3_5/ms3r_gate1_point_identifiability.json`
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `TODO.md`

**Steps:**
1. Add a validation-only configuration pinned to the MS3 parent matrix SHA.
2. Freeze point roles, lags, horizons, rolling folds, ridge regularization, rank windows, bootstrap settings, and prohibited claims.
3. Add `ms3_r` to the registry as `implementation`, with no Linux authorization.
4. Update the human queue so MS3-R Gate A is the only active implementation task.
5. Run `python experiments/phase3_5/experiment_status.py --check --json`; expect a valid registry with `active_gate=ms3_r`, `active_status=implementation`, and no Linux authorization.

### Task 2: Implement the analysis core

**Files:**
- Create: `src/phase35/ms3r.py`
- Test: `tests/phase35/test_ms3r.py`

**Steps:**
1. Write failing tests for validation-only config enforcement and aligned cross-side caches.
2. Implement point-quality and branch-semantics contracts.
3. Write failing tests for past-only rolling residualization.
4. Implement standardized ridge cross-fitting for valve innovations and outcome residuals.
5. Write failing tests showing correct-side local response exceeds wrong-side and lead placebos in synthetic data.
6. Implement local-projection diagnostics by side, point, and horizon.
7. Write failing tests for independent versus rank-deficient dual inputs.
8. Implement common/differential energy, covariance condition, and block-Hankel spectra.
9. Run `pytest tests/phase35/test_ms3r.py -q`; expect all tests to pass.

### Task 3: Add the Linux runner and deterministic artifacts

**Files:**
- Create: `experiments/phase3_5/ms3r_gate1_point_identifiability.py`
- Test: `tests/phase35/test_ms3r_cli.py`

**Steps:**
1. Write a CLI smoke test using temporary synthetic A/B caches and a reduced config.
2. Implement `--config`, `--cache-a`, `--cache-b`, `--output-dir`, and `--allow-dirty`.
3. Enforce the parent matrix hash, source SHA, exact timestamp alignment, validation split, and clean committed worktree by default.
4. Write `run_manifest.json`, `point_quality_validation.json`, `path_diagnostics_validation.json`, `rank_diagnostics_validation.json`, `analysis_arrays_validation.npz`, and `summary_validation.json` atomically.
5. Ensure the summary explicitly records `test_accessed=false`, `training_executed=false`, and no automatic scientific PASS.
6. Run `pytest tests/phase35/test_ms3r_cli.py -q`; expect all tests to pass.

### Task 4: Add the local replay boundary

**Files:**
- Create: `experiments/phase3_5/audit_ms3r_gate1_point_identifiability.py`
- Test: `tests/phase35/test_audit_ms3r.py`

**Steps:**
1. Recompute covariance, common/differential energy, path coefficients, and artifact hashes from the saved NPZ and JSON files.
2. Reject test access, configuration drift, missing artifacts, non-finite diagnostics, and inconsistent sample counts.
3. Emit only contract/numeric replay status; do not auto-promote causal or MIMO claims.
4. Run focused replay tests; expect exact deterministic agreement within the frozen numeric tolerance.

### Task 5: Verify and prepare handoff

**Files:**
- Modify: `experiments/phase3_5/README.md`
- Modify: `TODO.md`
- Modify: `configs/phase3_5/experiment_registry.json`

**Steps:**
1. Document one frozen Linux command and the required output bundle.
2. Run all `tests/phase35` tests.
3. Run the repository status checker and inspect `git diff --check`.
4. Keep the registry at `local_verified` until explicit release; the approved release changes only MS3-R Gate A to `ready_for_linux` and leaves Gate B/C frozen.
