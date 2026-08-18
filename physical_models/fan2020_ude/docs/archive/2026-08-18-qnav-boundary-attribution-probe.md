# Q32-T Boundary Attribution Probe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate object-response, controller-implementation, and initialization-error explanations for the Q32-S wet/dry results without retraining a model.

**Architecture:** Reuse the exact 16 Q32-S points and frozen F0/F1 `h_now` checkpoints. Run three independent inference panels rather than a full factorial matrix: paired open-loop object steps, controller ablations at the existing equilibrium initialization, and absolute initialization-drift diagnostics.

**Tech Stack:** Python, PyTorch, NumPy, pandas, existing Q32/Q32-R/Q32-S integrators and checkpoints.

---

## Frozen scientific boundary

- This is a simulator attribution experiment, not plant causal validation.
- It does not decide whether pure physics is globally valid or invalid.
- Controller reversals are interpreted only after separating deadband, low-pass filtering and anti-windup.
- A first-order low-pass filter is used as the minimal smoothing ablation. A Kalman observer is deferred until a sensor/process-noise model and latent-state correction target are specified.
- No panel selects a winner; all thresholds and conclusions remain supervisor-owned.

### Task 1: Freeze the contract

**Files:**
- Create: `configs/qnav_boundary_attribution_probe.json`
- Create: `docs/plans/2026-08-18-qnav-boundary-attribution-probe.md`

Freeze the Q32-S parent hashes, exact selected points through the parent summary, panel constants and Linux failure boundary.

### Task 2: Implement the three panels

**Files:**
- Create: `35_qnav_boundary_attribution_probe.py`

1. Object panel: coupled valve/W steps of −2% and +2%, with `physical`, `live`, and shared baseline-residual replay. Report sign, gain, time constant and asymmetry.
2. Controller panel: compare anti-windup only, deadband+anti-windup, low-pass+anti-windup, and deadband+low-pass+anti-windup for all three residual modes. Reuse raw PI results from Q32-S as the frozen parent reference.
3. Initialization panel: compare one-step observed initialization, a 180-step logged-history rollout, and constant-input equilibrium offsets at 1/60/180/600 steps for physical and live residuals.

### Task 3: Verify and release

**Files:**
- Create: `tests/test_qnav_boundary_attribution_probe.py`

Test parent/hash closure, controller anti-windup/deadband/filter mechanics, real-checkpoint object and initialization paths, full synthetic execution, no verdict, no training and development-only loading. Run the relevant Q32–Q32-T tests, compile, dry-run, then perform a shortened real-data smoke before commit and push.
