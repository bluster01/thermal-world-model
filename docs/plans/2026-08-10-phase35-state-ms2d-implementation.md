# Phase 3.5 State Registry and MS2-D1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fail-closed experiment-state registry and the first MS2-D pure-delay pressure validation framework.

**Architecture:** A machine-readable registry names every active/closed Gate, its scripts, artifacts, owner, and allowed transition; a read-only CLI validates the registry. MS2-D1 extends the existing synthetic/operator contracts with a causal delay buffer and uses a separate frozen runner/matrix so completed MS1/MS2 artifacts remain immutable.

**Tech Stack:** Python 3.11, PyTorch, dataclasses, JSON, pytest, Markdown, Git.

---

### Task 1: Add experiment-state registry and validator

**Files:**
- Create: `configs/phase3_5/experiment_registry.json`
- Create: `experiments/phase3_5/experiment_status.py`
- Create: `tests/phase35/test_experiment_status.py`

**Steps:**
1. Write tests for path existence, one active Linux Gate, deprecated E-track exclusion, and JSON summary output.
2. Run `python -m pytest tests/phase35/test_experiment_status.py -q`; expect missing-script failure.
3. Implement a read-only validator with frozen state enum and fail-closed path checks.
4. Run the test again; expect PASS.

### Task 2: Freeze context and line-ending-safe provenance

**Files:**
- Create: `.gitattributes`
- Create: `docs/PHASE35_CONTEXT_SNAPSHOT.md`
- Modify: `README.md`
- Modify: `TODO.md`

**Steps:**
1. Force LF for tracked JSON/Python/Markdown used by content-addressed protocols.
2. Record the full MS0→MS5 sequence, deprecated E status, MS2-J audited result, provenance advisories, and recovery commands.
3. Remove the unauthorized “MS2-J then paper” stop rule from active entry documents.
4. Run `python experiments/phase3_5/experiment_status.py --check`.

### Task 3: Add delayed synthetic truth

**Files:**
- Modify: `src/phase35/multistep/synthetic.py`
- Modify: `tests/phase35/multistep/test_model_and_synthetic.py`

**Steps:**
1. Add failing tests for deterministic zero-padded 2-step pure delay and unchanged split isolation.
2. Add `input_delay_steps` and `delayed_context_scheduled` validation.
3. Apply delay before the cascade and persist it in `truth` metadata.
4. Run the focused synthetic tests.

### Task 4: Add causal delayed graybox

**Files:**
- Modify: `src/phase35/multistep/contracts.py`
- Modify: `src/phase35/multistep/operators.py`
- Modify: `tests/phase35/multistep/test_operators.py`

**Steps:**
1. Add failing tests for exact identity, future causality, fixed-delay timing, learned simplex constraints, and chunked state continuation.
2. Add `delay_mode`, `fixed_delay_steps`, and `max_delay_steps` to `OperatorConfig`; only graybox accepts delay.
3. Extend graybox state with a causal dose buffer; fixed delay uses one-hot weights, learned delay uses softmax weights.
4. Report weights and expected delay seconds in diagnostics.
5. Run operator tests; expect PASS.

### Task 5: Add frozen MS2-D1 runner and summary

**Files:**
- Create: `configs/phase3_5/ms2d_delay_matrix.json`
- Create: `experiments/phase3_5/ms2d_delay.py`
- Create: `experiments/phase3_5/summarize_ms2d_delay.py`
- Create: `tests/phase35/multistep/test_ms2d_delay_cli.py`
- Modify: `experiments/phase3_5/README.md`

**Steps:**
1. Test exact expansion to 18 validation runs and no test switch.
2. Reuse the frozen MS2-J training contract but write to `results/phase3_5/ms2d_delay/`.
3. Implement artifact/structural gates, oracle gate, learned-vs-no-delay screen, and delay-identification diagnostic.
4. Add CPU smoke for one learned-delay run and a synthetic fixture summary test.
5. Run focused tests and dry-run; expect 18 runs and `test_authorized=false`.

### Task 6: Verify and hand off

**Files:**
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `docs/PHASE35_CONTEXT_SNAPSHOT.md`

**Steps:**
1. Mark MS2-D1 `ready_for_linux` only after tests, compile, smoke, clean-tree and frozen-matrix checks pass.
2. Run `python -m pytest tests/phase35/multistep tests/phase35/test_experiment_status.py -q`.
3. Run `python -m compileall -q src/phase35/multistep experiments/phase3_5`.
4. Run `git diff --check` and status validator.
5. Commit exact files, push `main`, and give Linux only the frozen README command.
