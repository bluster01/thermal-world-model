# FMTS M7 Fusion Implementation Plan

> Execution: use the available executing-plans workflow locally, no delegation.
> The named superpowers aliases are unavailable; use ordinary tests and this
> existing isolated release worktree. User already requested implementation.

**Goal:** Deliver a frozen M7 observer experiment plus the existing GNR1 Linux job.

**Architecture:** Keep parent FMTS v0.2 sources byte-identical. Add an experimental
M7 observer/fusion factory and a separate full-budget trainer/replay auditor.
Reuse the frozen parent data/evaluation and existing historical encoder modules.

**Tech Stack:** Python, PyTorch, NumPy, pytest; Linux formal runs, local synthetic checks.

## Task 1: Contracts and tests

Create `experiments/fmts_m7fusion_20260913/spec.py`, registration
`docs/fmts2026/PREREG_M7_FUSION_20260913.md`, and
`tests/final_wm/test_fmts_m7fusion.py`.
Write tests before model implementation for seed/spec parity, exact anchor,
independent readout, non-observer initialization equality, input isolation,
fixed-patch equivalence and all historical extension gradients.
Run `python -m pytest tests/final_wm/test_fmts_m7fusion.py -q`; expect missing-model
import failure before implementation. Freeze a generated JSON from executable spec.

## Task 2: Model and health diagnostics

Create package `models.py` (M7HistoryEncoder, M7Observer, M7Fusion) and `health.py`.
Test inherited physics/closure parameters unchanged, zero-head identity, bounds,
actual computation paths and finite backward gradients. Health export includes
raw/tanh corrections, tanh derivatives, fixed-last temporal reversal sensitivity,
start/day identities and summaries; no targets enter these calculations.

## Task 3: Runner and read-only return audit

Create `run.py` and `audit.py`. Validate parent sources, hashes, windows and stats;
reject overwrite, train only the one registered new arm, save explicit M7 metadata,
logs and failures. Independently reload/checkpoint-replay all saved prediction,
response and best health arrays; verify old-observer diagnostics on real inputs.
Recompute MAE and paired differences. Validate frozen spec, input, source and
registration hashes. Require all 3 seeds for formal completeness.

Extend tests with a synthetic 4-arm parent (2 updates), 1 M7 arm (2 updates),
old-observer health, reload replay, tamper rejection and overwrite rejection.
Never treat this smoke as evidence that M7 improves real-data predictions.

## Task 4: Linux handoff and verification

Create `RUN_LINUX.md`, `RESULTS.md`, `experiment_state.json`, `VERIFICATION.md`;
update root README/TODO and parent experiment follow-up status. Mark ready, not
running or returned. State exact ordered M7R1/GNR1 commands and return artifacts.
Run targeted suites, then `python -m pytest tests/final_wm -q`. Review changes,
source fingerprint equality and frozen JSON. Leave parent artifacts untouched.
Preserve existing staged paper/GNR1 work and local-only PDF QA images.

## Review checkpoints

1. Encoder/heads tests pass and scientific contract is frozen.
2. Synthetic end-to-end/replay passes for both new arms.
3. Full relevant suite passes; package is ready for Linux. No automatic scientific
   promotion, unscheduled extra experiment, or unattended local training.

## Execution record

Tasks 1–4 completed. Expected missing-module test failed before model creation;
M7 tests then passed (7), combined targeted tests passed (24). Full suite 226
passed / one unchanged historical JEPA queue assertion failed and was reproduced
standalone. See package VERIFICATION.md. Parent source fingerprints unchanged.
Published for Linux under the prior push authorization; no actual execution or
scientific result claimed at preparation time.
