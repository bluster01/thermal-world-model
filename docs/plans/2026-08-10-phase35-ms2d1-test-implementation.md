# Phase 3.5-MS2-D1 One-Shot Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a content-addressed, one-shot synthetic test for the 18 frozen MS2-D1 validation checkpoints.

**Architecture:** A versioned authorization pins matrix, validation summary, and checkpoint archive. A fail-closed runner reads checkpoint bytes directly from the tar, evaluates one independent test split without training, and writes root/run ledgers; a separate summarizer verifies all artifacts and applies paired-episode bootstrap gates.

**Tech Stack:** Python 3.11, PyTorch, JSON, tarfile, hashlib, pytest, Git.

---

### Task 1: Freeze authorization and failing protocol tests

**Files:**
- Create: `configs/phase3_5/ms2d_delay_test_authorization.json`
- Create: `tests/phase35/multistep/test_ms2d_delay_test.py`

**Steps:**
1. Pin the exact matrix, validation summary and 18-checkpoint archive hashes.
2. Add tests for a 18-run dry-run, explicit authorization, pin mismatch, completed-ledger refusal and deterministic paired bootstrap.
3. Run the focused test and confirm failure because the runner/summary do not exist.

### Task 2: Implement the one-shot test runner

**Files:**
- Create: `experiments/phase3_5/ms2d_delay_test.py`

**Steps:**
1. Validate authorization scope and repository-relative paths.
2. Preflight all manifests and archive checkpoint payloads before test access.
3. Evaluate all 18 checkpoints on the independent test split without optimizer construction.
4. Write fail-closed root/run ledgers and episode-level metrics.
5. Run dry-run and access-guard tests.

### Task 3: Implement confirmatory aggregation

**Files:**
- Create: `experiments/phase3_5/summarize_ms2d_delay_test.py`
- Modify: `tests/phase35/multistep/test_ms2d_delay_test.py`

**Steps:**
1. Validate manifest/ledger/checkpoint/trajectory pairing for all runs.
2. Apply structural, oracle and paired-stratified response gates.
3. Keep delay-distribution recovery as a separate diagnostic.
4. Exit code 2 on a scientific gate failure while still writing `summary_test.json`.
5. Run fixture tests for PASS and unpaired-episode rejection.

### Task 4: Publish the audited handoff

**Files:**
- Create: `docs/PHASE35_MS2D1_SUPERVISOR_AUDIT_2026-08-10.md`
- Modify: `docs/PHASE35_MS2D1_VALIDATION_REVIEW_2026-08-10.md`
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `TODO.md`
- Modify: `docs/PHASE35_CONTEXT_SNAPSHOT.md`
- Modify: `experiments/phase3_5/README.md`

**Steps:**
1. Record exact local recomputation, provenance correction, validation uncertainty and the 11/11 fallacy scan.
2. Change D1 to `test_authorized`; keep it as the active and only Linux-authorized Gate.
3. Add the exact Linux dry-run, execution and aggregation commands.
4. Run all Phase3.5 tests, compileall, status validation and `git diff --check`.
5. Commit and push; do not start D2.
