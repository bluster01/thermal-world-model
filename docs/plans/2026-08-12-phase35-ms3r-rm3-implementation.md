# MS3-R RM3 Orthogonal Response Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a validation-locked RM3 framework that separates OOF response identification from fair world-model prediction comparison.

**Architecture:** Add a NumPy/Torch orthogonal-response core using expanding OOF nuisance fits, a closed RM3 matrix and dry-run CLI, then add joint-latent physical-interface primitives without authorizing real training. Synthetic tests verify gain recovery, rank refusal, lead/shuffle controls and action-invariant terminal bypass.

**Tech Stack:** Python 3.11, NumPy, PyTorch, pytest, JSON contracts.

---

### Task 1: Freeze RM3 contracts and matrix

**Files:**
- Create: `configs/phase3_5/ms3r_rm3_matrix.json`
- Create: `src/phase35/multistep/rm3_contracts.py`
- Test: `tests/phase35/multistep/test_rm3_contracts.py`

1. Write tests that require train/validation only, H60/H180 orthogonal moments, expanding OOF folds, closed candidates, no test/Linux/MS4 and no automatic scientific pass.
2. Implement immutable candidate/run expansion and fail-closed validation.
3. Run the focused tests and expect PASS.

### Task 2: Implement OOF nuisance residualization

**Files:**
- Create: `src/phase35/multistep/rm3_orthogonal.py`
- Test: `tests/phase35/multistep/test_rm3_orthogonal.py`

1. Test rolling train-before-evaluate enforcement and shape/finite rejection.
2. Reuse the deterministic multi-output ridge contract to produce OOF action/outcome residuals and fold IDs.
3. Verify evaluated rows never use their own/future outcomes in nuisance fitting.

### Task 3: Implement orthogonal moments and R-loss

**Files:**
- Modify: `src/phase35/multistep/rm3_orthogonal.py`
- Test: `tests/phase35/multistep/test_rm3_orthogonal.py`

1. Test known linear-confounding DGP recovery, shuffled-action failure and collinear differential refusal.
2. Implement global/fold/day 2×2 moments, common/differential energy, condition number and Torch R-loss.
3. Persist sufficient row/fold/day arrays for cache-free Supervisor replay.

### Task 4: Add joint-latent physical interfaces

**Files:**
- Create: `src/phase35/multistep/rm3_joint_model.py`
- Test: `tests/phase35/multistep/test_rm3_joint_model.py`

1. Test shared causal latent rollout, stable transition, prefix causality and finite outputs.
2. Add a capacity-controlled terminal residual bypass that reads history context only.
3. Prove the bypass is invariant under future-action perturbations and that every action effect goes through the explicit local response interface.

### Task 5: Add dry-run and local synthetic smoke

**Files:**
- Create: `experiments/phase3_5/ms3r_rm3.py`
- Test: `tests/phase35/multistep/test_rm3_cli.py`

1. Implement `--dry-run` and `--synthetic-smoke`; do not implement/authorize a real matrix execution command yet.
2. Emit the two separate tables: identification candidates and fair prediction candidates.
3. Assert `linux_authorized=false`, `test_authorized=false`, `automatic_scientific_pass=null`.

### Task 6: Verify and update state

**Files:**
- Modify: `TODO.md`
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `experiments/phase3_5/README.md`

1. Run focused RM3 tests, all `tests/phase35`, compileall, JSON validation, status check and `git diff --check`.
2. Mark RM3 as local implementation/local verification only; keep `linux_authorized_gate=null`.
3. Do not push a Linux authorization until the user approves a separately frozen real-data matrix.

### Task 7: Unify H60 prediction adapters and freeze the real envelope

**Files:**
- Create: `src/phase35/multistep/rm3_prediction.py`
- Create: `src/phase35/multistep/rm3_smoke.py`
- Test: `tests/phase35/multistep/test_rm3_prediction.py`
- Test: `tests/phase35/multistep/test_rm3_smoke.py`

1. Rebuild M7-style dense action injection and M9-style prefix-causal action attention on the paired 15-feature H60 contract; do not import old data globals or checkpoints.
2. Put all six candidates behind one fail-closed adapter; logged future valve may enter only P0 oracle.
3. Run finite forward/backward micro-cache smoke for every candidate.
4. Freeze the 48-run envelope but leave real/Linux execution false until a long-training runner and result contract are locally verified.
