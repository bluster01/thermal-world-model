# Q32-R Residual Feedback Probe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the dry-regime sign reversal comes from the learned residual reacting to action-descendant temperature states.

**Architecture:** Reuse the two frozen Q32 `h_now` checkpoints without training. Compare the same valve intervention with the residual live, disabled, replayed from the baseline trajectory, scaled by 0.5, or evaluated after freezing either steam-temperature or metal-temperature features to their baseline values.

**Tech Stack:** Python, PyTorch, NumPy, pandas, existing Q32 integrator and checkpoints.

---

## Alternatives considered

- Repeat a dry pressure gate: rejected because the historical state gate restored some dry behavior but destroyed rollout and timescale.
- Retrain an exogenous-only residual immediately: deferred until the feedback mechanism is demonstrated.
- Baseline replay and feature freezing: selected because it is inference-only and directly separates physical response from residual-mediated response.

## Frozen scope

- Candidate: `h_now` only.
- Folds: Q32 F0 and F1.
- Points: up to eight evenly spaced dry operating points per evaluation fold; no stability cherry-picking.
- Paths: valve-only and valve-plus-training-only-W coupling.
- Modes: `physical`, `live`, `replay`, `half`, `freeze_ts`, `freeze_tm`.
- Historical rows `[40000,50000)` are not loaded.
- No training, model selection, thresholds, PASS/FAIL, document interpretation, or paper claim.

### Task 1: Freeze contract

Create `configs/qnav_residual_feedback_probe.json` with parent artifact hashes, folds, modes, point count and Linux execution boundary.

### Task 2: Implement inference probe

Create `33_qnav_residual_feedback_probe.py`. Add small residual wrappers for recording, replay, scaling and feature freezing; reuse `32_qnav_first_principles.py` for the physical integrator and response metrics.

### Task 3: Test and release

Create `tests/test_qnav_residual_feedback_probe.py`. Test wrapper identities, deterministic point selection and a real-checkpoint smoke path. Run unit tests, compile and dry-run before committing.
