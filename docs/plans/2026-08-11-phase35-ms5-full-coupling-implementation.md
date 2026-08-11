# Phase 3.5-MS5 Full Coupling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fail-closed 12-run synthetic validation that distinguishes total-prediction success from recovery of the free and action-response components.

**Architecture:** Extend the opt-in synthetic truth with a dynamic action-blind free trajectory and context-correlated action policy while keeping legacy regimes numerically stable. Add a full-model trainer with joint, staged, free-only, and component-oracle modes, then wrap it in a frozen matrix runner and episode-level summary.

**Tech Stack:** Python 3.11, PyTorch, NumPy, JSON, pytest, Git.

**Implementation Status:** COMPLETE / LOCAL VERIFIED；等待 Linux 12-run validation。

---

### Task 1: Freeze generator contract with failing tests

**Files:**
- Modify: `tests/phase35/multistep/test_model_and_synthetic.py`
- Modify: `src/phase35/multistep/synthetic.py`

**Steps:**
1. Add tests for deterministic `clean_free`, `clean_total`, policy-context coupling, exact hold identity, and legacy action/clean-effect stability.
2. Run focused tests and confirm RED.
3. Add opt-in full-coupling truth fields and validation.
4. Run focused tests and confirm GREEN.

### Task 2: Implement full-model metrics and training modes with TDD

**Files:**
- Create: `src/phase35/multistep/full_training.py`
- Create: `tests/phase35/multistep/test_full_training.py`
- Reuse: `src/phase35/multistep/model.py`

**Steps:**
1. Test component metrics, exact free-only zero response, action-blind free interface, finite gradients, and stage trainability.
2. Implement `FullTrainingConfig`, episode metrics, joint/oracle/free-only loops, and A/B/C staged loop.
3. Save per-stage checkpoints, parameter drift, gradient norms, selected checkpoint, history and manifest.
4. Confirm CPU smoke for all four modes.

### Task 3: Freeze the matrix and fail-closed runner

**Files:**
- Create: `configs/phase3_5/ms5_full_coupling_matrix.json`
- Create: `experiments/phase3_5/ms5_full_coupling.py`
- Create: `tests/phase35/multistep/test_ms5_full_coupling_cli.py`

**Steps:**
1. Test exact 4 modes×3 seeds expansion, `test_authorized=false`, protocol drift rejection and source-tree cleanliness.
2. Implement strict matrix validation and 12-run execution.
3. Save validation component episodes and trajectory hashes; reject any test artifact.
4. Run dry-run and one-run CPU smoke.

### Task 4: Implement aggregation and frozen decision tree

**Files:**
- Create: `experiments/phase3_5/summarize_ms5_full_coupling.py`
- Extend: `tests/phase35/multistep/test_ms5_full_coupling_cli.py`

**Steps:**
1. Test artifact replay, same-seed pairing, distinct seeds, oracle failure, joint selection, staged fallback and non-identifiable closure.
2. Implement per-seed absolute gates and the ordered strategy decision.
3. Keep prediction-only, stage drift, profile/horizon and joint-vs-staged comparisons diagnostic unless explicitly frozen.
4. Write deterministic checkpoint archive and exit code 2 on scientific failure.

### Task 5: Close D3 and authorize MS5

**Files:**
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `tests/phase35/test_experiment_status.py`
- Modify: `TODO.md`
- Modify: `README.md`
- Modify: `docs/PHASE35_CONTEXT_SNAPSHOT.md`
- Modify: `docs/PHASE35_MAINLINE_CONTEXT.md`
- Modify: `docs/PHASE35_MS_METHODS_AND_REFERENCES.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `docs/README.md`
- Modify: `docs/WORLD_MODEL_EVIDENCE_LADDER.md`
- Modify: `results/README.md`
- Modify: `experiments/phase3_5/README.md`

**Steps:**
1. Mark D3 `closed` with validation-only/budget boundary and archive advisory.
2. Mark MS5 `ready_for_linux` as the only authorized Gate.
3. Add exact Linux preflight/train/summary commands and result write scope.
4. Update context so a new session cannot re-open D3 test or skip MS5.

### Task 6: Verify and publish

**Steps:**
1. Run focused MS5 tests and compileall.
2. Run `python -m pytest tests/phase35 -q`.
3. Run registry check, MS5 dry-run, Markdown link scan and `git diff --check`.
4. Review staged diff, commit intentionally, fetch, and push `main`.
