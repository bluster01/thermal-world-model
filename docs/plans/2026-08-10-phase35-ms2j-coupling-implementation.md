# Phase 3.5-MS2-J Coupling Implementation Plan

> **For Codex:** implement inline in the shared workspace; subagent delegation is not authorized. Follow the frozen design and stop before synthetic test access.

**Goal:** Build a validation-only 27-run benchmark that tests whether nonlinear opening and context scheduling converge jointly, and whether three-stage training is stable relative to joint-from-scratch.

**Architecture:** Reuse the existing response operators and synthetic generator without modifying the completed MS2-V/C frozen execution files. Add a separate staged trainer, a dedicated MS2-J runner/matrix, and a fail-closed validation summarizer. Joint candidates continue to use the existing trainer; only the staged graybox uses the new trainer.

**Tech Stack:** Python 3.11, PyTorch, JSON manifests, pytest.

---

### Task 1: Freeze the matrix contract

**Files:**
- Create: `configs/phase3_5/joint_coupling_matrix.json`
- Test: `tests/phase35/multistep/test_joint_coupling_cli.py`

1. Add one combined R50 + context-scheduled truth regime, nine candidates and seeds 0/1/2.
2. Encode joint/staged mode outside `OperatorConfig` and freeze Stage A/B/C as 120/90/90 epochs.
3. Write a dry-run test asserting 27 unique runs, exactly one staged candidate and no test switch.
4. Run `python -m pytest tests/phase35/multistep/test_joint_coupling_cli.py -q`; initial runner absence must fail before implementation.

### Task 2: Implement staged training

**Files:**
- Create: `src/phase35/multistep/staging.py`
- Test: `tests/phase35/multistep/test_joint_coupling_cli.py`

1. Add `StagedTrainingConfig` validation and named parameter masks:
   - A: base K/τ + opening trainable, schedules frozen;
   - B: only schedules trainable;
   - C: all trainable at 0.2× learning rate.
2. For each stage, evaluate the boundary state, train with the frozen noisy validation selector, restore the best state and save stage metrics/checkpoint SHA.
3. Write `checkpoint_best_val.pt`, a single history with stage labels, manifest environment fields and validation trajectory digest; keep `test_accessed=false`.
4. Add a CPU smoke assertion that all stages run, trainable parameter names are disjoint as specified, and Stage C checkpoint is loadable.

### Task 3: Implement the validation runner

**Files:**
- Create: `experiments/phase3_5/joint_coupling.py`
- Test: `tests/phase35/multistep/test_joint_coupling_cli.py`

1. Implement strict matrix loading, candidate selection, `--dry-run`, single execution and full-matrix execution.
2. Route joint candidates to `train_synthetic_run` and the staged candidate to the new trainer.
3. Add clean-tree enforcement, compatible-resume checks, checkpoint hashing, environment/provenance augmentation and validation-only refusal semantics.
4. Run joint and staged CPU smoke commands in temporary output roots and assert no test artifacts exist.

### Task 4: Implement fail-closed aggregation

**Files:**
- Create: `experiments/phase3_5/summarize_joint_coupling.py`
- Test: `tests/phase35/multistep/test_joint_coupling_cli.py`

1. Require all 27 manifests, validation metrics, histories and checkpoint hashes; require A/B/C artifacts for staged runs.
2. Reuse structural gates and reject any test artifact.
3. Compute the pre-registered joint-vs-single-module improvements and staged non-inferiority/final-vs-Stage-A improvements per seed.
4. Exit non-zero unless all structural, 20% module and 10% staged stability gates pass.

### Task 5: Documentation and release verification

**Files:**
- Modify: `TODO.md`
- Modify: `docs/README.md`
- Modify: `experiments/phase3_5/README.md`
- Modify: `docs/PROJECT_STATUS.md`

1. Add the design/review links and Linux validation commands; state explicitly that test is unauthorized.
2. Run `python -m pytest tests/phase35/multistep -q` and `python -m pytest tests/phase35 -q`.
3. Run compile checks, `git diff --check`, Markdown link checks and one formal dry-run.
4. Commit and push only after the worktree is clean apart from the intentional MS2-J files.

