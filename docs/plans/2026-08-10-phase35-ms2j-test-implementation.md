# Phase 3.5-MS2-J One-Shot Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fail-closed one-shot synthetic test evaluator and paired-episode summary for the frozen MS2-J validation checkpoints.

**Architecture:** A content-addressed authorization file pins the training matrix, validation summary, and checkpoint tar. A dedicated runner preflights the whole matrix before access, loads weights directly from the tar, writes immutable ledgers and per-episode metrics, and a separate summarizer applies the three frozen bootstrap gates.

**Tech Stack:** Python 3.11, PyTorch, stdlib JSON/tarfile/hashlib/random/statistics, pytest.

---

### Task 1: Freeze authorization and failing tests

**Files:**
- Create: `configs/phase3_5/joint_coupling_test_authorization.json`
- Modify: `tests/phase35/multistep/test_joint_coupling_cli.py`

1. Add tests requiring 27-run preflight, archive-member hashes, explicit authorization, root/run ledgers, repeat refusal, Stage-A evaluation and test-summary gate semantics.
2. Run `python -m pytest tests/phase35/multistep/test_joint_coupling_cli.py -q`; expect failure because the test CLI and summarizer do not exist.

### Task 2: Implement one-shot evaluator

**Files:**
- Create: `experiments/phase3_5/joint_coupling_test.py`

1. Validate the three pinned SHA256 values and frozen-code equivalence before any test access.
2. Preflight all 27 manifests and tar members before writing the root started ledger.
3. Reuse the frozen operator, synthetic generator and metric functions; write per-run aggregate/episode metrics and ledgers atomically.
4. Evaluate Stage A only for the three staged runs in the same test access.
5. Run the targeted CLI tests and require PASS.

### Task 3: Implement paired bootstrap summary

**Files:**
- Create: `experiments/phase3_5/summarize_joint_coupling_test.py`
- Modify: `tests/phase35/multistep/test_joint_coupling_cli.py`

1. Verify root/run ledgers, checkpoint hashes, episode pairing, profile coverage and structural diagnostics.
2. Compute per-seed joint improvements, staged/joint error-ratio CI, and staged/Stage-A improvement using frozen bootstrap settings.
3. Exit code 2 unless artifact/structural gates and all three confirmatory gates pass; preserve the mixed conclusion fields separately.
4. Run targeted tests and require PASS.

### Task 4: Correct claims and publish Linux handoff

**Files:**
- Modify: `docs/PHASE35_MS2J_AUDIT_2026-08-10.md`
- Modify: `docs/PHASE35_MS2J_VALIDATION_REVIEW_2026-08-10.md`
- Modify: `docs/PHASE35_MS_METHODS_AND_REFERENCES.md`
- Modify: `experiments/phase3_5/README.md`
- Modify: `TODO.md`

1. Replace the incorrect `exp(raw_gain)` interpretation with `-softplus(raw_gain)` and report the three oracle base gains.
2. Replace “joint is the correct scheme” and “same solution region” with evidence-limited wording.
3. Record the independent CPU replay tolerance and environment-sensitive trajectory hashes.
4. Add exactly one Linux test command and one summary command; retain code-2-as-scientific-result instructions.

### Task 5: Verify and commit

1. Run targeted tests, `python -m pytest tests/phase35 -q`, and `python -m compileall -q src/phase35 experiments/phase3_5`.
2. Run the MS2-J validation dry-run and the new test CLI without authorization; confirm 27 runs and fail-closed behavior without creating test artifacts.
3. Inspect `git diff --check`, commit the scoped files, push `main`, and report the commit SHA plus Linux commands.
