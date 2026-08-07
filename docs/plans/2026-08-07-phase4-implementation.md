# Phase 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an auditable Phase 4 code path for a Fan20-centered main-steam-temperature world model, with separate plant/supervisory estimands, validation-only model selection, three fair dynamics routes, and reproducible Linux handoff.

**Architecture:** Preserve all legacy experiments as historical evidence. Add an isolated `src/phase4/` package whose dependencies flow from immutable, task-qualified manifests into datasets, models, evaluation, and a single runner. The runner never receives a final split during training; final/internal-final evaluation is a separate command that consumes a frozen run manifest and blindly builds final events. Fan20 is the common plant contract candidate, Fan17 explicit-metal/Fan21 closure are optional components, and structured ODE/fixed-operator controlled-Koopman/time-varying gray-box implement one shared route interface.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, SciPy, PyYAML, IAPWS, pyarrow, pytest, ruff, JSON Schema, Git/SHA256 provenance.

---

## Preconditions

- Do not edit or delete historical result files.
- Do not use the last 15% as a fresh lockbox; obtain a new future period/other unit or label the study internal validation.
- Before Task 2, obtain engineering confirmation for action tag meanings and units.
- Before remote Batch R1, freeze `delta_pred`, `delta_event`, `delta_gain_pred/event`, dose/response deadbands, physical ranges, one primary metric/family, MDE/power result and failure thresholds in `configs/phase4/decision_thresholds.yaml`.
- Complete each task with tests and a focused commit. Do not bundle remote results into implementation commits.

### Task 1: Restore a deterministic local test and dependency baseline

**Files:**
- Create: `pyproject.toml`
- Create: `requirements/base.txt`
- Create: `requirements/dev.txt`
- Modify: `tests/test_eval_protocol.py`
- Modify: `experiments/phase2_mpc/eval_protocol.py`

**Step 1: Capture the current failure**

Run:

```bash
python -m pytest tests/test_eval_protocol.py -q
```

Expected: collection fails because the test stub does not export `TimeXerWM`.

**Step 2: Fix only the collection contract**

Export a harmless `TimeXerWM` alias/class from the stub, or remove the unused import in `eval_protocol.py` if repository-wide search proves it unused. Do not load real data during import.

**Step 3: Add failing PID behavior tests**

In `tests/test_eval_protocol.py`, assert:

- hot PV causes more cooling action than cold PV;
- zero error preserves the declared working point;
- a changing error produces a non-zero derivative contribution when `kd != 0`.

Run the three tests and confirm they fail before changing PID code.

**Step 4: Repair PID sign, bias and derivative order**

Use a cooling-positive convention documented in the class. Compute derivative before updating `e_prev`, and express output as `u_bias + correction`.

**Step 5: Lock the environment**

Declare runtime and test dependencies, set pytest test paths, and make imports side-effect free for test collection.

**Step 6: Verify**

Run:

```bash
python -m pytest tests/test_eval_protocol.py -q
python -m pytest -q
```

Expected: collection succeeds; any remaining legacy failures are recorded explicitly rather than hidden.

**Step 7: Commit**

```bash
git add pyproject.toml requirements tests/test_eval_protocol.py experiments/phase2_mpc/eval_protocol.py
git commit -m "fix: restore protocol test baseline"
```

### Task 2: Implement immutable data and action contracts

**Files:**
- Create: `src/phase4/__init__.py`
- Create: `src/phase4/contracts.py`
- Create: `src/phase4/hashing.py`
- Create: `schemas/phase4/data_manifest.schema.json`
- Create: `schemas/phase4/action_manifest.schema.json`
- Create: `configs/phase4/column_mapping.yaml`
- Create: `tests/phase4/test_contracts.py`

**Step 1: Write failing schema tests**

Test that a manifest is rejected when it lacks task ID, units, time range, sample interval, file SHA256, action layer, or verified tag semantics. Test that `spray_flow` cannot use valve-position units, and that Task S separates full SP input from `delta_sp_exposure` event metadata.

**Step 2: Implement typed contracts**

Define `PlantAction`, `SupervisoryAction`, `ColumnSpec`, `DataManifest` and validation errors. Keep plant and supervisory actions as different types, not a string flag passed through model code.

**Step 3: Implement stable hashing**

Canonicalize JSON/YAML before SHA256. Include raw-file hashes and mapping hash in `data_id`.

**Step 4: Populate the mapping as unverified**

Transfer current 40-column hypotheses into YAML with `status: unverified` unless supported by DCS documentation. Do not infer units.

**Step 5: Verify**

```bash
python -m pytest tests/phase4/test_contracts.py -q
```

**Step 6: Commit**

```bash
git add src/phase4 schemas/phase4 configs/phase4 tests/phase4/test_contracts.py
git commit -m "feat: add phase4 data and action contracts"
```

### Task 3: Build episode-aware preprocessing and purged splits

**Files:**
- Create: `src/phase4/data/episodes.py`
- Create: `src/phase4/data/preprocess.py`
- Create: `src/phase4/data/splits.py`
- Create: `src/phase4/data/windows.py`
- Create: `configs/phase4/splits.yaml`
- Create: `tests/phase4/test_episodes.py`
- Create: `tests/phase4/test_splits.py`
- Create: `tests/phase4/test_windows.py`

**Step 1: Write failing synthetic tests**

Cover duplicate timestamps, gaps, startup boundaries, long NaN runs, frozen sensors, `W + H` purge, and a window attempting to cross an episode/split.

**Step 2: Implement preprocessing without silent imputation**

Return values plus missingness masks. Make per-column interpolation limits explicit. Reject unresolved NaN in required physical inputs.

**Step 3: Implement rolling split manifests**

Create three development folds and an optional opaque external-lockbox reference. Persist development episode IDs and exact row/time boundaries; hash the result. Do not enumerate final rows/events into a training-visible object.

**Step 4: Implement manifest-driven windows**

Datasets accept a split manifest object, never percentages. Training constructors expose only train/validation handles; test access lives in a different module.

**Step 5: Verify**

```bash
python -m pytest tests/phase4/test_episodes.py tests/phase4/test_splits.py tests/phase4/test_windows.py -q
```

**Step 6: Commit**

```bash
git add src/phase4/data configs/phase4/splits.yaml tests/phase4
git commit -m "feat: add episode-aware purged data splits"
```

### Task 4: Replace CFE fallback with a fail-closed observational event protocol

**Files:**
- Create: `src/phase4/events.py`
- Create: `src/phase4/evaluation/event_metrics.py`
- Create: `schemas/phase4/event_manifest.schema.json`
- Create: `configs/phase4/events.yaml`
- Create: `tests/phase4/test_events.py`
- Create: `tests/phase4/test_event_metrics.py`

**Step 1: Write failure-mode tests**

Assert hard failure on missing reference file, split mismatch, event-ID mismatch, length mismatch, future-variable matching, control reuse beyond policy, and metric-version mismatch.

**Step 2: Implement onset and matching**

Select onset from task-specific action exposure only. Match on declared pre-event covariates; compute overlap, standardized differences, pre-trend and placebo diagnostics. Build development event pools only. Freeze a pure final-event builder/config hash; `final_evaluate` or an independent steward invokes it after model freeze.

**Step 3: Implement a vector metric panel**

Return raw event-response-curve WMAE, gain bias, direction consistency, TTP and shape separately. Enforce a dose floor; score direction only where the reference CI excludes zero or exceeds the frozen response deadband. Do not expose a checkpoint-selection `CFI` scalar.

**Step 4: Verify**

```bash
python -m pytest tests/phase4/test_events.py tests/phase4/test_event_metrics.py -q
```

**Step 5: Commit**

```bash
git add src/phase4/events.py src/phase4/evaluation schemas/phase4/event_manifest.schema.json configs/phase4/events.yaml tests/phase4
git commit -m "feat: add fail-closed observational event evaluation"
```

### Task 5: Implement and verify the steam-property layer

**Files:**
- Create: `src/phase4/physics/units.py`
- Create: `src/phase4/physics/water.py`
- Create: `tests/phase4/test_units.py`
- Create: `tests/phase4/test_water.py`
- Create: `tests/fixtures/iapws_reference_points.yaml`

**Step 1: Add table-point tests**

Use independent IAPWS reference points for `h(p,T)` and `T(p,h)`, including relevant superheated-steam ranges and unit conversions. Set tolerances before implementation.

**Step 2: Implement a single property API**

All model code calls SI-typed wrappers. Reject ambiguous pressure/temperature units and out-of-region states. Provide a differentiable approximation only after comparing it against the reference library.

**Step 3: Add gradient and round-trip tests**

Verify finite gradients and `T(p, h(p,T))` recovery across the declared operating envelope.

**Step 4: Verify**

```bash
python -m pytest tests/phase4/test_units.py tests/phase4/test_water.py -q
```

**Step 5: Commit**

```bash
git add src/phase4/physics tests/phase4/test_units.py tests/phase4/test_water.py tests/fixtures
git commit -m "feat: add verified steam property layer"
```

### Task 6: Implement the Fan20-SST common physical contract

**Files:**
- Create: `src/phase4/physics/fan20.py`
- Create: `src/phase4/models/base.py`
- Create: `configs/phase4/fan20.yaml`
- Create: `tests/phase4/test_fan20_equations.py`
- Create: `tests/phase4/test_fan20_synthetic.py`
- Create: `scripts/phase4/generate_synthetic_fan20.py`

**Step 1: Transcribe equations with source anchors**

For every equation, record the Fan20 equation number/page and project variable mapping in docstrings/config. Keep measured inputs distinct from fitted states. Add a source-lock table checked independently against the PDF/full text; do not transcribe from project notes alone.

**Step 2: Write conservation and invariant tests**

Test positive flows/enthalpies, two-stage mixing limits, zero-spray limits, monotonic cooling direction under fixed conditions, finite gradients and state bounds. Split `spray_flow` and `valve_proxy` specs: only the former may expose spray mass/energy residuals.

**Step 3: Implement constrained parameters and RHS/output functions**

Expose a route-neutral `PhysicalSpec` with state names, input names, `rhs`, `observe`, `residuals` and bounds. No neural residual yet.

**Step 4: Add synthetic recovery**

Generate known trajectories; compare Euler/RK4/reference integration at 10 s; recover identifiable parameter groups under noise. At least one hand-calculated conservation fixture or second independent RHS must produce the reference; a generator importing `fan20.rhs` is only a self-consistency smoke.

**Step 5: Verify**

```bash
python -m pytest tests/phase4/test_fan20_equations.py tests/phase4/test_fan20_synthetic.py -q
python scripts/phase4/generate_synthetic_fan20.py --smoke
```

**Step 6: Commit**

```bash
git add src/phase4/physics/fan20.py src/phase4/models/base.py configs/phase4/fan20.yaml tests/phase4 scripts/phase4
git commit -m "feat: implement Fan20 SST physical core"
```

### Task 7: Add Fan17 and Fan21 as optional, testable components

**Files:**
- Create: `src/phase4/physics/fan17.py`
- Create: `src/phase4/physics/fan21.py`
- Create: `src/phase4/physics/components.py`
- Create: `configs/phase4/components.yaml`
- Create: `tests/phase4/test_fan17_component.py`
- Create: `tests/phase4/test_fan21_component.py`

**Step 1: Add equation regression tests**

Include the correct Fan21 mismatch term and the corrected Fan17 turbine-flow function with its `pst^-0.743` factor:

```text
Q1 = k1*rB + m/(Ne + g) * (Dfw - a*rB)
```

Test that the implementation cannot regress to the old erroneous additive formula or double-count total heat across Fan20 segments.

**Step 2: Implement composable components**

Fan20 already contains pulverizer delay/inertia. Fan17 may contribute only an explicit metal-storage `Tj` component after re-deriving segmented energy balances. Fan21 contributes bounded load scheduling or one whole-boiler mismatch closure with a frozen replacement/allocation rule across `k11/k12/k13`. Do not implement throttle loss in SST Task P; reserve it for a later full CCS/turbine model.

**Step 3: Test nesting and identifiability flags**

Turning a component off must reproduce Fan20-core numerically within a frozen tolerance. Assert total heat is counted once. Unidentifiable parameter groups must be surfaced in diagnostics, not silently optimized.

**Step 4: Verify**

```bash
python -m pytest tests/phase4/test_fan17_component.py tests/phase4/test_fan21_component.py -q
```

**Step 5: Commit**

```bash
git add src/phase4/physics configs/phase4/components.yaml tests/phase4
git commit -m "feat: add nested Fan17 and Fan21 components"
```

### Task 8: Implement the three fair representation/closure routes

**Files:**
- Create: `src/phase4/models/structured_ode.py`
- Create: `src/phase4/models/controlled_koopman.py`
- Create: `src/phase4/models/time_varying_graybox.py`
- Create: `src/phase4/models/observer.py`
- Create: `src/phase4/models/factory.py`
- Create: `configs/phase4/routes/ode.yaml`
- Create: `configs/phase4/routes/koopman.yaml`
- Create: `configs/phase4/routes/time_varying.yaml`
- Create: `tests/phase4/test_route_contract.py`
- Create: `tests/phase4/test_action_causality.py`

**Step 1: Define a shared route interface**

Every route consumes the same `PhysicalSpec`, history observer output, current action and exogenous inputs; returns state trajectory, observations and diagnostics.

**Step 2: Write route-neutral tests**

Check shapes, finite rollout, common state/output names, parameter counts, zero-step identity, and identical physical residual API.

**Step 3: Write the future-action leakage test**

For all routes and direct baselines, compute `d y_hat[k] / d u[j]`; require zero within tolerance for every `j > k`.

**Step 4: Implement Route A**

Restrict neural corrections to configured closure terms/coefficients and cap capacity. Record the bypass-capacity audit.

**Step 5: Implement Route B**

Use fixed-operator `z[k+1] = A z[k] + B u[k]`, preserve stable real/complex modes, and decode to the shared physical state. Treat load as a declared state/input; if `A/B` are externally scheduled, rename the route LPV Koopman. Do not reuse `KoopmanFreeHead`.

**Step 6: Implement Route C**

Schedule only predeclared low-dimensional Fan parameters by load; constrain range and smoothness.

**Step 7: Verify**

```bash
python -m pytest tests/phase4/test_route_contract.py tests/phase4/test_action_causality.py -q
```

**Step 8: Commit**

```bash
git add src/phase4/models configs/phase4/routes tests/phase4
git commit -m "feat: implement phase4 dynamics routes"
```

### Task 9: Add baselines without mixing estimands

**Files:**
- Create: `src/phase4/baselines/statistical.py`
- Create: `src/phase4/baselines/legacy_adapters.py`
- Create: `src/phase4/tasks.py`
- Create: `configs/phase4/baselines.yaml`
- Create: `tests/phase4/test_task_boundaries.py`
- Create: `tests/phase4/test_baselines.py`

**Step 1: Test task separation**

Assert that A1phys cannot register in a plant-action leaderboard and that valve/spray models cannot silently register as supervisory models. Task S models consume a full SP-level trajectory; `ΔSP` is event exposure metadata, not a substitute model input.

**Step 2: Implement statistical baselines**

Add persistence, ARX and N4SID/state-space under the same manifests and windows.

**Step 3: Wrap legacy M7 and A1phys**

Adapters must retrain on Phase 4 splits, declare action type, and expose the same prediction output. Do not import scripts with top-level training side effects.

**Step 4: Verify**

```bash
python -m pytest tests/phase4/test_task_boundaries.py tests/phase4/test_baselines.py -q
```

**Step 5: Commit**

```bash
git add src/phase4/baselines src/phase4/tasks.py configs/phase4/baselines.yaml tests/phase4
git commit -m "feat: add estimand-safe phase4 baselines"
```

### Task 10: Implement validation-only training and canonical checkpoint selection

**Files:**
- Create: `src/phase4/training.py`
- Create: `src/phase4/selection.py`
- Create: `src/phase4/evaluation/forecast_metrics.py`
- Create: `src/phase4/evaluation/physics_metrics.py`
- Create: `src/phase4/evaluation/statistics.py`
- Create: `configs/phase4/decision_thresholds.yaml`
- Create: `tests/phase4/test_no_test_access.py`
- Create: `tests/phase4/test_selection.py`
- Create: `tests/phase4/test_statistics.py`

**Step 1: Write a no-test-access test**

Pass a sentinel test object that raises on access. Complete a smoke training/selection run and assert the sentinel is untouched.

**Step 2: Write canonical-selection cases**

Test hard physics rejection, event catastrophic rejection, prediction non-inferiority, deterministic tie-breaks and exactly one selected checkpoint. Test that a CFI scalar cannot be a selector.

**Step 3: Implement metric panels**

Compute horizon-wise/integrated MAE, event metrics, physical residuals, invalid states, solver failures and subgroup tables from immutable prediction files.

**Step 4: Implement clustered uncertainty**

Use episode/time-block paired bootstrap for forecasts and event/episode clustering for responses. Keep seed variability separate; implement Holm correction for the three route comparisons.

**Step 5: Verify**

```bash
python -m pytest tests/phase4/test_no_test_access.py tests/phase4/test_selection.py tests/phase4/test_statistics.py -q
```

**Step 6: Commit**

```bash
git add src/phase4/training.py src/phase4/selection.py src/phase4/evaluation configs/phase4/decision_thresholds.yaml tests/phase4
git commit -m "feat: enforce validation-only canonical selection"
```

### Task 11: Build the reproducible runner and artifact validator

**Files:**
- Create: `src/phase4/runner.py`
- Create: `src/phase4/artifacts.py`
- Create: `src/phase4/final_evaluate.py`
- Create: `schemas/phase4/run_manifest.schema.json`
- Create: `scripts/phase4/run.py`
- Create: `scripts/phase4/aggregate.py`
- Create: `scripts/phase4/final_evaluate.py`
- Create: `scripts/phase4/validate_artifacts.py`
- Create: `tests/phase4/test_artifacts.py`
- Create: `tests/phase4/test_final_lock.py`

**Step 1: Write overwrite/provenance tests**

Reject an existing output directory, dirty/unrecorded code unless explicitly allowed for smoke, missing hashes, incomplete outputs and a checkpoint whose SHA256 differs.

**Step 2: Implement content-addressed runs**

Build paths from experiment/config hash/fold/seed. Save resolved config, git/dirty hash, environment, command, data/split/event hashes, logs, predictions, diagnostics and status.

**Step 3: Separate final evaluation**

`src/phase4/final_evaluate.py` and its CLI accept only a frozen final/internal-final manifest and canonical checkpoints. They never train or tune. The CLI writes an access ledger before opening data, blindly builds task-qualified final events, and evaluates the entire frozen model/seed batch before exposing summaries.

**Step 4: Implement aggregation from raw predictions**

Recompute metrics rather than trusting training summaries. Fail the entire aggregate if a required run or hash is invalid.

**Step 5: Verify**

```bash
python -m pytest tests/phase4/test_artifacts.py tests/phase4/test_final_lock.py -q
python scripts/phase4/run.py --config configs/phase4/routes/ode.yaml --smoke
python scripts/phase4/validate_artifacts.py results/phase4
```

**Step 6: Commit**

```bash
git add src/phase4/runner.py src/phase4/artifacts.py src/phase4/final_evaluate.py schemas/phase4 scripts/phase4 tests/phase4
git commit -m "feat: add reproducible phase4 runner and final lock"
```

### Task 12: Perform local Gate 0 smoke and prepare staged Linux R0/R1 handoff

**Files:**
- Create: `configs/phase4/matrix_l0_smoke.yaml`
- Create: `configs/phase4/matrix_r0.yaml`
- Create: `configs/phase4/matrix_r1.yaml`
- Create: `docs/PHASE4_RUNBOOK.md`
- Create: `results/phase4/README.md`
- Modify: `TODO.md`

**Step 1: Run the complete local suite**

```bash
python -m pytest -q
python -m ruff check src/phase4 tests/phase4 scripts/phase4
```

Expected: all Phase 4 tests pass; no import starts training or loads private data.

**Step 2: Run CPU numerical smoke**

```bash
python scripts/phase4/run.py --matrix configs/phase4/matrix_l0_smoke.yaml --smoke
python scripts/phase4/validate_artifacts.py results/phase4
```

Expected: Fan20, Fan17-metal and Fan21-mismatch synthetic runs finish with finite trajectories and valid local-smoke manifests. This is `smoke_passed`, not `ready_for_remote`.

**Step 3: Audit Gate 0/1 evidence**

Check off a TODO item only with a linked test/manifest/report. Record unresolved engineering semantics as blockers; do not mark `ready_for_remote` while they remain.

**Step 4: Freeze and push the run commit**

```bash
git status --short
git add configs/phase4 docs/PHASE4_RUNBOOK.md results/phase4/README.md TODO.md
git commit -m "docs: prepare phase4 Linux experiment handoff"
git tag -a phase4-r0-v1 -m "Phase 4 R0 handoff"
git push origin HEAD phase4-r0-v1
```

`matrix_r0.yaml` stores the predeclared tag name `phase4-r0-v1`, not its self-referential commit SHA. Remote verifies that `HEAD` equals the tag target; the runtime manifest records the actual 40-character SHA. R1 receives a new tag only after R0 and Gate 0 are audited.

**Step 5: Linux preflight**

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/phase4 -q
python scripts/phase4/run.py --matrix configs/phase4/matrix_r0.yaml --preflight
```

Expected: data/split/action/event hashes match the signed handoff; GPU/environment details are recorded; no formal training begins during preflight.

**Step 6: Release R0, then R1 only after separate Supervisor sign-offs**

Run R0 numerical smoke first, preserve stdout/stderr, and return the content-addressed directories. Only after R0/Gate 0 audit, freeze/tag `matrix_r1.yaml` for Gate 1 Fan20 validation. Generate R2–R6 matrices only after the prior Gate is audited. Update TODO states through `remote_running → results_returned → audited`; never jump from “run finished” to “concluded”.

## Final verification checklist

- [ ] `python -m pytest -q` passes.
- [ ] All manifests validate and hashes recompute.
- [ ] No training code imports or reads test data.
- [ ] Plant and supervisory leaderboards reject the wrong action type.
- [ ] Fan20 standard points, synthetic recovery and solver convergence pass.
- [ ] Future-action Jacobian is zero for every route.
- [ ] Each seed has exactly one validation-selected canonical checkpoint.
- [ ] Metric aggregation reproduces from raw predictions.
- [ ] Remote run paths cannot overwrite.
- [ ] Final command is separate, frozen and access-ledgered; without new data it labels output `internal_final`, never `lockbox`.
- [ ] README/PROJECT_STATUS/TODO state the model remains unfinalized until Gate 5.
