# Phase 3.5 A1phys Core Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an auditable Phase 3.5 framework that runs E1–E5 with absolute valve position as the plant action and produces Linux-trainable, validation-selected, test-isolated results.

**Architecture:** A new `src/phase35` package owns data contracts, the valve-level A1phys model, event construction, metrics, and result schemas. Thin scripts under `experiments/phase3_5` prepare caches, train one run, evaluate a frozen checkpoint, execute the preregistered matrix, and summarize results. Legacy Phase 3 scripts remain read-only historical baselines.

**Tech Stack:** Python 3.10+, NumPy, pandas, PyTorch, pytest, standard-library JSON/subprocess.

---

### Task 1: Freeze data and action contracts

**Files:**
- Create: `src/phase35/__init__.py`
- Create: `src/phase35/schema.py`
- Create: `tests/phase35/test_schema.py`

**Steps:**
1. Write tests asserting plant and supervisory actions cannot be mixed and that required A/B columns resolve by name.
2. Run `python -m pytest tests/phase35/test_schema.py -q`; expect missing-module failure.
3. Implement immutable column/action/split dataclasses and validation errors.
4. Re-run the test; expect PASS.

### Task 2: Implement causal sparse-cache preparation

**Files:**
- Create: `src/phase35/data.py`
- Create: `experiments/phase3_5/prepare_data.py`
- Create: `tests/phase35/test_data.py`

**Steps:**
1. Test asynchronous tag updates, exact-boundary behavior, staleness, and absence of backward fill.
2. Implement causal last-observation reconstruction and NPZ cache metadata.
3. Implement chronological split and valid window sampling without test access.
4. Run data tests and a synthetic cache round trip.

### Task 3: Implement valve-level A1phys

**Files:**
- Create: `src/phase35/model.py`
- Create: `tests/phase35/test_model.py`

**Steps:**
1. Test identity, fixed equal-percentage `R=50`, and learned monotone opening-map endpoints/gradients; keep the fixed curve labeled as a pilot prior, not flow truth.
2. Test constant future valve path gives exactly zero intervention effect.
3. Test opening produces non-positive long-run temperature effect and future actions cannot affect earlier steps.
4. Implement a future-action-isolated free head that still sees pre-treatment valve history, action adapters, monotone effective opening, two-stage inertia, and optional rate residual.
5. Run model tests on CPU.

### Task 4: Implement E3/E5 event construction

**Files:**
- Create: `src/phase35/events.py`
- Create: `tests/phase35/test_events.py`

**Steps:**
1. Test isolated opening/closing detection, minimum separation, stable-load filtering, and split boundaries.
2. Test SP events classify into executed and no-execution groups from valve feedback, not command.
3. Implement quiet-control candidate selection and treatment-prevariable nearest-neighbor matching.
4. Verify matches never use post-event features or cross split boundaries.

### Task 5: Implement preregistered metrics

**Files:**
- Create: `src/phase35/evaluation.py`
- Create: `tests/phase35/test_evaluation.py`

**Steps:**
1. Test horizon MAE, integrated MAE, direction accuracy, onset lag, IRF-WMAE, dose monotonicity, and UTC calendar-day block-bootstrap determinism.
2. Implement matched empirical DiD curves and model actual-vs-constant-valve effects.
3. Keep metrics separate; do not implement CFI or a composite checkpoint selector.

### Task 6: Implement validation-only training

**Files:**
- Create: `src/phase35/training.py`
- Create: `experiments/phase3_5/train.py`
- Create: `tests/phase35/test_training.py`

**Steps:**
1. Test that the trainer receives train/validation datasets only and cannot evaluate test.
2. Test one synthetic optimization step, checkpoint metadata, early stopping, and deterministic validation anchors.
3. Implement Huber/NLL losses, optional free-head freeze, validation-only checkpointing, environment/git manifest, and resume-safe output checks.

### Task 7: Implement explicit frozen evaluation

**Files:**
- Create: `experiments/phase3_5/evaluate.py`
- Test: `tests/phase35/test_evaluation.py`

**Steps:**
1. Load one exact checkpoint/config/cache and require an explicit split.
2. Reject `test` unless `--allow-test-access` is present.
3. Write forecast, valve-event, SP-negative-control metrics and an access ledger.
4. Never update or replace a checkpoint during evaluation.

### Task 8: Implement matrix and conclusion reporting

**Files:**
- Create: `configs/phase3_5/experiment_matrix.json`
- Create: `experiments/phase3_5/run_matrix.py`
- Create: `experiments/phase3_5/summarize.py`
- Create: `tests/phase35/test_reporting.py`

**Steps:**
1. Freeze E1/E2 configurations, A/B sides, seeds, horizons, and output roots in JSON.
2. Expand the 42-run matrix deterministically; print a dry-run by default and execute only with `--execute`.
3. Aggregate seed means/SD separately from calendar-day block confidence intervals.
4. Emit `summary.json` and `summary.md` with PASS/FAIL/INCONCLUSIVE gates and no winner claim when required inputs are absent.

### Task 9: Document Linux handoff and verify

**Files:**
- Create: `experiments/phase3_5/README.md`
- Modify: `experiments/README.md`
- Modify: `README.md`
- Modify: `TODO.md`

**Steps:**
1. Document cache preparation, matrix dry-run, development training, validation summary, explicit test access, and final report commands.
2. Run `python -m pytest tests/phase35 -q`.
3. Run `python experiments/phase3_5/run_matrix.py --matrix configs/phase3_5/experiment_matrix.json` and verify 42 dry-run commands.
4. Run a synthetic CPU smoke train/evaluate/report cycle.
5. Run the existing test suite and document unrelated legacy failures separately.
