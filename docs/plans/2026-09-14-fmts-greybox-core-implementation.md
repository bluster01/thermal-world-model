# FMTS 同核灰箱 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the user-approved frozen-GRU-core 2×2 inference ablation and target-free temporal-evaluation preflight; no retraining or held-out scoring.

**Architecture:** New independent `experiments/fmts_core_20260914/` package wraps unchanged RichFusion checkpoints. It toggles only past observer correction and neural closure invocation, preserves all physical tensors, and replays the original C11 reference. A separate JSON metadata preflight never opens NPZ/CSV or target arrays.

**Tech Stack:** Python, PyTorch, NumPy, pytest, JSON/Markdown provenance.

---

The installed executing-plans skill supplies the workflow; unavailable superpowers names are compatibility references, not additional installations. User already selected in-session implementation. All scientific source imports/tests bind the existing release worktree, not dirty main sources. New files are staged in the writable main workspace, then copied with explicit filesystem approval to new release paths only.

### Task 1: Freeze the scientific contract

Create `docs/fmts2026/PREREG_CORE_ABLATION_20260914.md` and `experiments/fmts_core_20260914/protocol.json`.
Fix seeds0/1/2, C00/C10/C01/C11, full original prediction/response windows, engineering identity/replay gates, no parameter updates and no automatic winner. Register all null outcomes.

### Task 2: Component wrapper and tests

Create `experiments/fmts_core_20260914/run.py` and `tests/final_wm/test_fmts_core.py`.
Tests first assert C11 equals RichFusion on synthetic histories; C00 never calls observer/closure; physical state_dict is unchanged; all parameters frozen; invalid variants rejected. Implement only initialization/closure dispatch, reuse exact transition and parent metrics.

### Task 3: Bound execution and saved-array audit

Runner checks parent/GNR identities, source hashes and checkpoint hashes before producing new outputs; refuses existing output directories. Copy indices byte-exact, record package/registration/checkpoint hashes and original runtime; save arrays, model internal states and per-run reports. C11 and unchanged GNR must match returned arrays. Separate audit recomputes metrics and optionally replays from actual record; absence of record is never full replay. No optimizer imports or training route.

### Task 4: Target-free evaluation preflight

Create `experiments/fmts_core_20260914/preflight.py` and tests using JSON fixtures.
Only inspect source/schema/time/exposure metadata; reject declaring old test unseen, reject 13-channel packages as direct FMTS inputs, preserve unknown exposure, keep scoring disabled. Do not construct a held-out target loader before the required lineage and final-checkpoint freeze are resolved.

Execution finding (2026-09-14): the source-matched segmented_v2 code and saved outputs establish historical exposure of the candidate. Distinguish confirmed exposure from unknown metadata; even a completed schema/checkpoint/window declaration cannot make it a clean independent test. Save provenance review and one regression guard without changing CORE1 scientific gates.

### Task 5: Verify and hand off

Run `python -B -m pytest tests/final_wm/test_fmts_core.py` then related FMTS tests from release. Save test evidence, RUN_LINUX.md, package state and RESULTS pending. Confirm no frozen source diff and no edits to user-owned work. Commit/push and Linux execution status must remain explicit; do not manufacture a run receipt.

Completion 2026-09-14: tasks 1–5 implemented; final five-file engineering suite
35 passed in 44.00 s. JSON-only exposure preflight saved. The user requested
Linux execution before paper revision; publish this narrow inference package and
its prior-result audit context via the established origin/main handoff. Scientific
results remain pending. See `experiments/fmts_core_20260914/EVIDENCE_PLAN_ZH.md`.
