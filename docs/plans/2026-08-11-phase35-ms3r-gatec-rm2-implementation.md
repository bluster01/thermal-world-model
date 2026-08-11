# Gate C RM2 Linux Parallel Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and authorize one fail-closed 54-run real-data RM2 matrix for Hermes, with full-data training, rolling folds, parallel device workers, replayable checkpoints, and no test access.

**Architecture:** Extend the Gate C model with explicit response-coordinate and downstream-mode ablations, add fold-aware anchor generation, then implement a long-training core and a matrix runner that partitions immutable run specs across CUDA workers. All machine outputs are descriptive; local Supervisor code will make the later decision.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, JSON/NPZ artifacts, Git/versioned experiment registry.

---

### Task 1: Freeze RM2 contracts and 54-run matrix

**Files:**
- Create: `configs/phase3_5/ms3r_gatec_rm2_matrix.json`
- Create: `src/phase35/multistep/gatec_rm2_contracts.py`
- Test: `tests/phase35/multistep/test_gatec_rm2_contracts.py`

1. Write tests requiring exactly groups A/B/C, 9 unique candidates, seeds 0/1/2, folds F0/F1, 54 unique run IDs, test fraction untouched, and Linux authorization fields closed.
2. Run the focused test and confirm failure before implementation.
3. Implement immutable dataclasses, config validation, run expansion and deterministic run IDs.
4. Run focused tests and expect PASS.

### Task 2: Add topology ablations and fold-aware anchors

**Files:**
- Modify: `src/phase35/multistep/gatec_contracts.py`
- Modify: `src/phase35/multistep/gatec_model.py`
- Modify: `src/phase35/multistep/gatec_data.py`
- Test: `tests/phase35/multistep/test_gatec_model.py`
- Test: `tests/phase35/multistep/test_gatec_data.py`

1. Add failing tests for `common_only`, `direct_no_latent`, custom pre-test bounds, and rejection of bounds entering test.
2. Add a response wrapper that removes only differential explicit effect and a direct causal downstream mixer with no recurrent latent state.
3. Add optional bounds to paired anchor generation, preserving existing split behavior by default.
4. Run Gate C model/data tests and expect PASS.

### Task 3: Implement long-training run core

**Files:**
- Create: `src/phase35/multistep/gatec_rm2_training.py`
- Test: `tests/phase35/multistep/test_gatec_rm2_training.py`

1. Write a CPU micro-cache test covering train-only stats, full eligible anchor pool sampling, selector checkpoints, early-stop contract, final metrics and compressed episode arrays.
2. Implement on-demand batch extraction, common score selection, checkpoint serialization, deterministic shuffled-action diagnostics and final NPZ output.
3. Require one attempt, atomic JSON writes, finite gradients, checkpoint/manifest hashes and explicit `test_accessed=false`.
4. Run the focused training test and expect PASS.

### Task 4: Implement parallel matrix runner and fail-closed summary

**Files:**
- Create: `experiments/phase3_5/ms3r_gatec_rm2.py`
- Create: `experiments/phase3_5/summarize_ms3r_gatec_rm2.py`
- Test: `tests/phase35/multistep/test_gatec_rm2_cli.py`

1. Test dry-run count/group/fold/seed closure, registry preflight, per-device static partitioning, incomplete-run rejection, no automatic retry and root ledger generation.
2. Implement `--dry-run`, `--list-runs`, `--run-id`, and `--execute-matrix --devices ...` modes.
3. Make each worker load caches once, bind one device and continue independent runs after writing a failure record.
4. Summarizer verifies all 54 run ledgers, archives checkpoints and emits descriptive completeness only.

### Task 5: Local smoke, regression and Linux authorization

**Files:**
- Modify: `TODO.md`
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `experiments/phase3_5/README.md`
- Modify: `results/README.md`

1. Run a reduced CPU micro-cache matrix without changing the frozen RM2 config.
2. Run focused RM2 tests, all `tests/phase35`, compileall, JSON validation, status checker and `git diff --check`.
3. Document one Hermes command, device-pool control, return artifacts and failure policy.
4. Set `ms3_r.status=ready_for_linux` and `linux_authorized_gate=ms3_r` only after every local check passes.
5. Create one final commit and push once; this is the only webhook-triggering authorization commit.
