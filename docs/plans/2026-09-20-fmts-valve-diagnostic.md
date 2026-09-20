# FMTS-VD1 Implementation Plan

> Execution: main agent implements locally under the executing-plans workflow; no subagents. The named superpowers wrapper is unavailable; use the installed equivalent. User authorized the diagnostic on 2026-09-20.

**Goal:** Diagnose where fixed FMTS checkpoints use valve history and future valve/spray inputs, without training, changing the paper, or asserting plant-response truth.

**Architecture:** Reuse the frozen BLOCK1 source validation and loaders for the same nine BB/GRU/GNR checkpoints. Add a versioned, validation-only diagnostic with original 64 response origins and 256 prediction origins. Linux performs real inference; local deterministic fixtures verify transformations, timing, accounting and packaging.

**Tech Stack:** Python, PyTorch, NumPy, pytest; private NPZ traces and public aggregate JSON.

---

## Task 1: Freeze and test the perturbations

Create `experiments/fmts_valve_diagnostic_20260920/protocol.json`, `design.py` and `tests/final_wm/test_fmts_valve_diagnostic.py`.
Write failing tests first, then implement: independent history copies; fixed original future inputs during history probes; six 30-second history blocks; future pulse/step starts at 0/60/120 seconds; both valves and signs; actual clipped dose; W-only and explicit artificial joint-input probes.
Run `python -m pytest tests/final_wm/test_fmts_valve_diagnostic.py -q`.

## Task 2: Implement fixed-source inference and saved-array audit

Create `run.py`, `audit.py`, `__init__.py`. Reuse checkpoint/input SHA gates. Replay original prediction and response arrays before interpreting new probes. Preserve raw per-window predictions, changes, support masks, actual doses and private input snapshots. No optimizer or changed model input contract. Verify weights are unchanged and zero-change runs identical. Summarize signed and absolute responses, causal-prefix sensitivity, common elapsed-60-second response, and opening strata separately.
Test anticipation-sensitive and causally ordered fixtures, cancellation, duplicate origins, empty strata, artifact tampering, incomplete cells and output overwrite rejection. Production entry point requires Linux and the exact original real-data fingerprints.

## Task 3: Document execution and state

Create `docs/fmts2026/PREREG_VALVE_DIAGNOSTIC_20260920.md`, package `RUN_LINUX.md`, `VERIFICATION.md`, `experiment_state.json`.
Run the new tests plus existing mainsteam/core/block tests. Do not change existing dirty README/TODO/paper/results. Keep this package self-contained. No publication or Linux execution is claimed until it actually occurs.

## Scope and interpretation

- Primary future tests: original held-boundary baseline, 48 scenarios (2 valves x 2 signs x 2 doses x 3 onsets x 2 shapes).
- History tests: 24 scenarios (2 valves x 2 signs x 6 past blocks), input sensitivity only; no physical counterfactual assertion.
- Spray path: 2 W-only scenarios plus 4 joint valve/W scenarios. W increments are +/-2 t/h, **not** a calibrated valve-to-flow conversion. Do not delete W at inference or call this a retrained baseline.
- Total 78 scenarios per checkpoint, 9 checkpoints; all original windows retained. New low/mid/high strata derive only from the parent training-bank last valve positions, with fixed tertiles; insufficient coverage is reported, not filled by searching.
- Factual prediction diagnostics reuse/replay all 256 original windows, report level and increment errors by measured valve movement, and are not causal event identification.
- No test scoring, training, new model selection, plant intervention, new DID or paper verdict. Scientific null/negative results are retained; weak response or anticipation is an outcome, not an engineering rejection rule.

## Availability preflight

Local formal checkpoints exist, but canonical Side A v2.2 input NPZ is absent from the release worktree. The author-provided BLOCK1 zip exports only main-temperature histories/targets, not actions or the 23-channel model inputs. It cannot serve as a replacement record. Follow the existing local-design/Linux-execution workflow.
