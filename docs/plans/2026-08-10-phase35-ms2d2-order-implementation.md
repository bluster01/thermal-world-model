# Phase 3.5-MS2-D2 Order Diagnostic Implementation Plan

> Execution owner: local Codex for design/code/test/audit; Linux for the frozen validation run only.

**Goal:** Implement and freeze the 21-run MS2-D2 three-pole structural-pressure validation without propagating the unconfirmed D1 delay result.

**Architecture:** Reuse the existing one-to-three-pole synthetic generator and stable graybox operator. Add a D2-specific frozen matrix, fail-closed runner and summary; keep the primary two-vs-three-pole contrast separate from delay-compensation and flexible-route diagnostics.

**Tech Stack:** Python 3.11, PyTorch, JSON, pytest, Git.

---

### Task 1: Close D1 and harden the handoff boundary

**Files:**
- Create: `docs/PHASE35_MS2D1_TEST_SUPERVISOR_AUDIT_2026-08-10.md`
- Modify: `docs/PHASE35_MS2D1_TEST_REVIEW_2026-08-10.md`
- Modify: `docs/REMOTE_EXPERIMENT_PROTOCOL.md`
- Modify: `configs/phase3_5/experiment_registry.json`

**Steps:**
1. Record the exact artifact/statistical reproduction and 11/11 fallacy scan.
2. Mark the Linux review as an execution report, not an independent audit.
3. Freeze which paths Linux may modify and which only the local Supervisor may update.
4. Close D1 as `TEST_NOT_CONFIRMED_AT_20PCT_MARGIN` and clear its authorization before D2 is ready.

### Task 2: Freeze D2 matrix and write failing tests

**Files:**
- Create: `configs/phase3_5/ms2d_order_matrix.json`
- Create: `tests/phase35/multistep/test_ms2d_order_cli.py`
- Modify: `tests/phase35/multistep/test_model_and_synthetic.py`

**Steps:**
1. Freeze one three-pole context-scheduled R50 truth, seven candidates and three seeds.
2. Test exact matrix expansion, truth order/no-delay, candidate roles and gates.
3. Add a three-pole generator/graybox state-continuation contract test.
4. Add fixture tests for oracle, absolute-error, relative-order and diagnostic separation.
5. Run the focused tests and confirm failure because D2 scripts do not yet exist.

### Task 3: Implement the D2 runner

**Files:**
- Create: `experiments/phase3_5/ms2d_order.py`

**Steps:**
1. Validate every frozen candidate, truth field, seed and threshold.
2. Reuse `SyntheticSpec`, `OperatorConfig` and `train_synthetic_run` without changing D1 code.
3. Add full environment provenance to each formal manifest.
4. Fail closed on partial/incompatible outputs and prohibit synthetic test access.
5. Run a CPU smoke for the primary three-pole candidate.

### Task 4: Implement fail-closed aggregation

**Files:**
- Create: `experiments/phase3_5/summarize_ms2d_order.py`

**Steps:**
1. Validate 21 manifest/checkpoint/history/metrics sets and frozen code equivalence.
2. Apply structural, oracle, absolute-NMAE and relative order gates.
3. Report sorted-τ and no-true-delay compensation diagnostics separately.
4. Refuse any test artifacts and exit code 2 on primary scientific failure after writing the summary.
5. Run fixture PASS/tamper tests.

### Task 5: Freeze the Linux handoff

**Files:**
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `TODO.md`
- Modify: `docs/PHASE35_CONTEXT_SNAPSHOT.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `experiments/phase3_5/README.md`

**Steps:**
1. Set D2 as the sole active/authorized Gate only after tests, smoke and dry-run pass.
2. Add exact preflight, 21-run validation and aggregation commands.
3. Require Linux to modify only result artifacts/manifests and an explicitly named remote execution report.
4. Run all Phase3.5 tests, compileall, status validation and `git diff --check`.
5. Commit and push; do not start D3 or any test.
