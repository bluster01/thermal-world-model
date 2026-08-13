# RM3-AV Independent Audit Validation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build the validation-only RM3-AV framework that turns all independent-audit concerns into a frozen 32-candidate, 64-unit experiment with zero-training replay, fail-closed execution, and auditable diagnostics.

**Architecture:** Keep audited RM3/RM3-A code immutable where practical. Add a compositional RM3-AV layer: strict candidate contracts map one declared intervention axis to a model/training configuration; adapters expose explicit ablation controls and training auxiliaries; one shared diagnostic reporter is mandatory for every candidate. AV0 replays frozen artifacts without changing them, AV1 trains the frozen matrix, and AV2 remains a local Supervisor-only decision layer.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, JSON/NPZ artifact contracts, existing Phase 3.5 cache and RM3 model/training infrastructure.

---

### Task 1: Freeze matrix and contracts

**Files:**
- Create: `configs/phase3_5/ms3r_rm3av_matrix.json`
- Create: `src/phase35/multistep/rm3av_contracts.py`
- Create: `tests/phase35/multistep/test_rm3av_contracts.py`

**Steps:**
1. Write failing tests for protocol version, parent audit pin, 32 unique candidates, exactly one primary intervention per candidate, F0/F1 × seed0 expansion to 64 run IDs, C28–C30 8000-update exceptions, locked test/MS4/Linux flags, and required diagnostic artifacts.
2. Run the focused test and confirm failure because the module/matrix do not exist.
3. Implement immutable dataclasses, validation, and run-spec expansion.
4. Create the full matrix from the frozen RM3-AV design.
5. Run the focused tests and validate JSON parsing and exact unit arithmetic.

### Task 2: Add compositional model interventions

**Files:**
- Create: `src/phase35/multistep/rm3av_model.py`
- Modify: `src/phase35/multistep/rm3_prediction.py`
- Modify: `src/phase35/multistep/rm3_joint_model.py`
- Modify: `src/phase35/multistep/gatec_model.py`
- Create: `tests/phase35/multistep/test_rm3av_model.py`

**Steps:**
1. Write failing shape/contract tests for bypass modes, response-off, free capacity, common/diagonal response, 1/2/3 pole, power, linear, dead-time, signed diagnostic response, PI and PI+GRU valve decoders.
2. Add explicit intervention configuration without changing legacy defaults.
3. Implement a generic RM3-AV adapter that returns the legacy output keys plus component outputs and ablation metadata.
4. Add module-scoped deterministic initialization and prove shared tensor hashes match across C25/C26/C27.
5. Run legacy RM3 tests and the new model tests.

### Task 3: Add training auxiliaries and action shielding

**Files:**
- Create: `src/phase35/multistep/rm3av_training.py`
- Modify: `src/phase35/multistep/rm3_training.py` only through optional hooks if required
- Create: `tests/phase35/multistep/test_rm3av_training.py`

**Steps:**
1. Write failing tests for logged-action auxiliary isolation, OOF R-loss gradient reachability, action-shield projection train/validation separation, delta-valve/roughness loss, and two-window rollout loss.
2. Implement an explicit auxiliary-loss protocol; forecast outputs must never receive logged future valve.
3. Fit action shielding and OOF nuisance only on train-fold anchors; serialize fit hashes and dimensions.
4. Implement per-candidate update caps and unified balanced selector.
5. Verify one-update parameter deltas for the intended branch and zero access for forbidden paths.

### Task 4: Restore mandatory diagnostics and AV0 replay

**Files:**
- Create: `src/phase35/multistep/rm3av_diagnostics.py`
- Create: `src/phase35/multistep/rm3av_replay.py`
- Create: `experiments/phase3_5/audit_ms3r_rm3av0.py`
- Create: `tests/phase35/multistep/test_rm3av_diagnostics.py`
- Create: `tests/phase35/multistep/test_rm3av_replay.py`

**Steps:**
1. Write failing tests for persistence, per-side MAE, valve roughness/span, predicted/logged response, A/B/common/differential effects, finite differences, shuffled/wrong-side/lead placebo, initialization hashes, convergence slopes, and state-closure labels.
2. Implement diagnostics from common episode arrays and explicit model intervention calls.
3. Implement frozen-checkpoint normal/bypass-off/bypass-only/response-off/oracle/placebo replay without rewriting parent artifacts.
4. Regenerate calibration corrections to a new provenance-bearing output path; never overwrite old JSON.
5. Emit AV0 reports with `automatic_scientific_pass=null`.

### Task 5: Implement AV1 runner, reporting, and ledgers

**Files:**
- Create: `src/phase35/multistep/rm3av_execution.py`
- Create: `src/phase35/multistep/rm3av_audit.py`
- Create: `experiments/phase3_5/ms3r_rm3av_train.py`
- Create: `experiments/phase3_5/audit_ms3r_rm3av.py`
- Create: `tests/phase35/multistep/test_rm3av_execution.py`
- Create: `tests/phase35/multistep/test_rm3av_audit.py`

**Steps:**
1. Write failing dry-run tests for 32 candidates/64 units and refusal under the current unauthorized registry.
2. Implement strict parent hash/code hash/cache hash checks and clean-worktree execution boundary.
3. Partition 64 units across declared devices; allow one attempt per unit and no automatic retry or non-empty-output resume.
4. Require manifest/checkpoint/metrics/episodes/diagnostics/ledger for every unit and root manifest/status/summary/ledger.
5. Summarize descriptive paired fold contrasts without a champion or automatic pass; AV2 must reject any Linux-authored Q01-Q33 verdict.

### Task 6: Verification and state handoff

**Files:**
- Modify: `TODO.md`
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `experiments/phase3_5/README.md`
- Modify: `docs/PHASE35_CONTEXT_SNAPSHOT.md`

**Steps:**
1. Run RM3-AV focused tests.
2. Run legacy RM3/RM3-A focused regression tests.
3. Run compileall for all new and modified Python files.
4. Run dry-run and verify 32 candidates, 64 units, test false, automatic pass null, matrix self-authorization false, and execute refusal while Linux authorization is null.
5. Run reduced-cache one-update micro smokes for all C00-C31; do not run long training.
6. Mark implementation `local_verified` only if all gates pass; keep `linux_authorized_gate=null` until a separate authorization commit.
