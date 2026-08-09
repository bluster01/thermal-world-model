# Phase 3.5 Multi-step Action Response Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an auditable multi-step A1phys response framework that compares stable gray-box, controlled Koopman, physics-informed ODE, and causal DeepONet operators under one contract.

**Architecture:** Add an isolated `src.phase35.multistep` package so the completed 42-run protocol and its results remain unchanged. Each route implements the same reference-subtracted, causal response interface; a thin A1phys-MS wrapper adds that response to an action-blind free forecast, and a synthetic known-truth benchmark establishes representational and optimization feasibility before any real-data run.

**Tech Stack:** Python 3, PyTorch, dataclasses, pytest, JSON manifests, existing Phase 3.5 utilities.

---

### Task 1: Freeze operator contracts and structural tests

**Files:**
- Create: `src/phase35/multistep/contracts.py`
- Create: `tests/phase35/multistep/test_operators.py`

**Steps:**
1. Write tests that instantiate every route with `[B,C]` context and `[B,H]` action/reference paths.
2. Assert exact reference identity and assert that changing action after index `k` cannot change outputs before `k`.
3. Run `python -m pytest tests/phase35/multistep/test_operators.py -q` and verify missing-module failure.
4. Implement `OperatorConfig`, `ResponseOutput`, input validation, and the common operator base.

### Task 2: Implement stable recursive routes

**Files:**
- Create: `src/phase35/multistep/operators.py`
- Modify: `tests/phase35/multistep/test_operators.py`

**Steps:**
1. Implement one/two-pole gray-box recurrence with `tau>0`, `gain<=0`, and `alpha=exp(-dt/tau)`.
2. Implement diagonal controlled Koopman recurrence with eigenvalues strictly inside `(0,1)`.
3. Implement a two-state nominal ODE with a zero-initialized, action/state-gated neural closure and a reported closure penalty.
4. Implement a causal GRU branch/time-trunk DeepONet and exact `G(action)-G(reference)` subtraction.
5. Add route-specific stability, sign, finite-rollout, and diagnostic tests; run the file until green.

### Task 3: Add A1phys-MS composition and synthetic truth

**Files:**
- Create: `src/phase35/multistep/model.py`
- Create: `src/phase35/multistep/synthetic.py`
- Create: `tests/phase35/multistep/test_model_and_synthetic.py`

**Steps:**
1. Test that the free predictor receives only context and that model output equals `free_mu + effect`.
2. Implement the action-blind baseline contract and `A1PhysMultiStep` wrapper.
3. Test deterministic generation of holds, steps, ramps, pulses, known parameters, and response targets.
4. Implement train/validation/test synthetic splits without reusing noise or action paths.

### Task 4: Add training, evaluation, and CLI

**Files:**
- Create: `src/phase35/multistep/training.py`
- Create: `experiments/phase3_5/multistep_sysid.py`
- Create: `tests/phase35/multistep/test_cli.py`

**Steps:**
1. Test dry-run matrix expansion and a one-epoch CPU smoke run.
2. Implement validation-only checkpoint selection and H1/H6/H18/H60 response metrics.
3. Persist manifest, history, canonical checkpoint, validation metrics, and structural diagnostics.
4. Require an explicit flag for synthetic test evaluation; do not add real-data test access here.

### Task 5: Freeze the experiment matrix and handoff

**Files:**
- Create: `configs/phase3_5/multistep_operator_matrix.json`
- Modify: `experiments/phase3_5/README.md`
- Modify: `TODO.md`
- Create: `src/phase35/multistep/__init__.py`

**Steps:**
1. Enumerate Graybox-1P/2P, Koopman-K2/K4, PI-ODE, and Causal-DeepONet with the same default budget.
2. Document local smoke and Linux dry-run/execute commands and the required returned artifacts.
3. Mark the new track as method-feasibility evidence, not restored E3/E4 causal evidence.
4. Run `python -m pytest tests/phase35/multistep -q`, `python -m compileall -q src/phase35/multistep experiments/phase3_5/multistep_sysid.py`, and a CPU smoke command.
5. Review `git diff --check`, dirty-tree scope, and generated manifest contents before handoff.
