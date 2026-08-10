# Phase 3.5-MS2-D2 One-Shot Test Implementation Plan

> Execution owner: local Codex for design/code/test/audit; Linux for the frozen one-shot evaluation only.

**Goal:** Freeze and implement a content-addressed one-shot test for all 21 D2 validation-selected checkpoints.

**Architecture:** Reuse the D2 matrix and archived checkpoints. A fail-closed runner generates paired episode metrics and ledgers; a separate summary reconstructs the prespecified oracle, absolute-NMAE and paired-bootstrap gates while keeping tau/delay diagnostics secondary.

**Tech Stack:** Python 3.11, PyTorch, JSON, tar, pytest, Git.

---

### Task 1: Freeze authorization and failing tests

**Files:**
- Create: `configs/phase3_5/ms2d_order_test_authorization.json`
- Create: `tests/phase35/multistep/test_ms2d_order_test.py`

**Steps:**
1. Pin matrix, validation summary and 21-member checkpoint archive by SHA256.
2. Test dry-run, explicit opt-in, content drift, gate drift, pairing and diagnostic separation.
3. Confirm tests fail before the D2 test runner exists.

### Task 2: Implement one-shot runner

**Files:**
- Create: `experiments/phase3_5/ms2d_order_test.py`

**Steps:**
1. Validate authorization, frozen validation status, archive members and all manifests/checkpoints.
2. Refuse any prior/partial test artifact.
3. Evaluate all 21 checkpoints on the independent test split and write root/run ledgers plus episode metrics.
4. Require `--evaluate-test-matrix --allow-synthetic-test`; dry-run must not generate test data.

### Task 3: Implement confirmatory summary

**Files:**
- Create: `experiments/phase3_5/summarize_ms2d_order_test.py`

**Steps:**
1. Verify ledgers, hashes, episode pairing and episode-to-aggregate reproduction.
2. Apply oracle, absolute and CI-lower response gates per seed.
3. Report tau and no-true-delay diagnostics separately.
4. Write `summary_test.json`; exit 2 on scientific failure after preserving the summary.

### Task 4: Freeze status and Linux handoff

**Files:**
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `TODO.md`
- Modify: `docs/PHASE35_CONTEXT_SNAPSHOT.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `README.md`
- Modify: `experiments/phase3_5/README.md`

**Steps:**
1. Set D2 as the sole `test_authorized` Gate; D3/MS5 remain frozen.
2. Add exact preflight, one-shot execution and summary commands.
3. Run focused/full Phase 3.5 tests, compileall, status check, dry-run and diff check.
4. Commit and push; Linux may modify only `results/phase3_5/ms2d_order/**`.
