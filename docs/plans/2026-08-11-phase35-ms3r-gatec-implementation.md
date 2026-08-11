# MS3-R Gate C Measured-Boundary MIMO Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and locally verify a dual-interface measured-boundary latent MIMO framework with pluggable response operators, frozen ablations, and no Linux authorization.

**Architecture:** A paired A/B history encoder feeds a causal SP→valve policy decoder, a past-only Tin boundary forecaster, an explicit stable local MIMO response operator, and a stable downstream latent mixer. Oracle Tin is an audit ceiling; forecast/scenario Tin are deployable interfaces. Four operator routes share the same outer model and selector.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, existing `src/phase35` cache/contracts, pytest, JSON/NPZ machine artifacts.

---

### Task 1: Freeze Gate C contracts and matrix schema

**Files:**
- Create: `src/phase35/multistep/gatec_contracts.py`
- Create: `configs/phase3_5/ms3r_gatec_model_matrix.json`
- Test: `tests/phase35/multistep/test_gatec_contracts.py`

**Step 1: Write failing contract tests**

Test that only `validation` is accepted; `test_allowed=false`; boundary modes are exactly `oracle_boundary/forecast_boundary/scenario_boundary`; oracle cannot be primary selector; response routes are the four frozen IDs; future logged Tin/valve access is rejected in forecast mode; RM1 contains exactly six attribution candidates and four operator candidates.

**Step 2: Run the tests and confirm failure**

```bash
python -m pytest tests/phase35/multistep/test_gatec_contracts.py -q
```

Expected: import/file-not-found failure.

**Step 3: Implement immutable dataclasses and validators**

Provide `GateCModelConfig`, `GateCTrainingConfig`, `GateCRunSpec`, `validate_gatec_matrix`, and `gatec_run_specs`. Unknown fields, mixed splits, oracle selection, duplicate IDs, non-closed budgets, and invalid response-route semantics must raise `Phase35ProtocolError`.

**Step 4: Run tests and commit the task**

Expected: all contract tests PASS.

### Task 2: Build paired A/B causal windows

**Files:**
- Create: `src/phase35/multistep/gatec_data.py`
- Test: `tests/phase35/multistep/test_gatec_data.py`

**Step 1: Write failing tests**

Construct aligned synthetic `Phase35Cache` pairs. Verify shared features occur once, side features preserve cross mapping, windows never cross gaps/splits, forecast-boundary inputs contain no future Tin, scenario inputs are explicit, and future SP prefix changes cannot alter earlier inputs.

**Step 2: Implement `PairedGateCBatch` and extraction**

The batch must expose history, future SP, logged future valve, logged future Tin, local drop, Tout and terminal targets separately. Boundary/action oracle fields must remain named and cannot be silently substituted.

**Step 3: Run tests and commit the task**

### Task 3: Implement shared model modules

**Files:**
- Create: `src/phase35/multistep/gatec_model.py`
- Test: `tests/phase35/multistep/test_gatec_model.py`

**Step 1: Write failing structural tests**

Test tensor shapes for A/B outputs; SP prefix causality; residual output invariance to future valve permutation; constant-action identity; zero-effect behavior for `paired_free`; finite 600 s rollout; stable latent poles; full terminal mixing; oracle/forecast boundary separation; gradients reaching valve, Tin, local response and downstream modules.

**Step 2: Implement minimal modules**

Implement:

```python
class PairedHistoryEncoder(nn.Module): ...
class CausalValvePolicyDecoder(nn.Module): ...
class TinBoundaryForecaster(nn.Module): ...
class StableLocalMIMOResponse(nn.Module): ...
class StableDownstreamLatentMixer(nn.Module): ...
class MeasuredBoundaryMIMOWorldModel(nn.Module): ...
```

Use stable time constants/poles rather than an unconstrained recurrent transition. Return a dictionary with `valve_prediction`, `tin_prediction`, `local_drop_prediction`, `tout_prediction`, `terminal_prediction`, `local_effect`, `terminal_effect`, `latent_state`, and diagnostics.

**Step 3: Add pluggable operator adapters**

Add builders for `a1phys_three_pole`, `stable_koopman_lpv`, `pi_neural_ode`, and `deeponet_response`. Before scientific training, every adapter must pass the same identity, prefix, finite and shape contract.

**Step 4: Run tests and commit the task**

### Task 4: Freeze multitask losses and checkpoint selector

**Files:**
- Create: `src/phase35/multistep/gatec_training.py`
- Test: `tests/phase35/multistep/test_gatec_training.py`

**Step 1: Write failing tests**

Verify train-only robust scales; near-zero floors; weights sum to one; oracle metrics cannot enter selector; a checkpoint with better terminal MAE but failed identity/collapse is ineligible; future-action leakage fails closed; warm-up updates are at most 10%; all modules unfreeze for joint training.

**Step 2: Implement losses and lexicographic eligibility**

Implement `GateCLossBreakdown`, `GateCStructuralMetrics`, `GateCSelectorRecord`, `compute_gatec_loss`, and `select_gatec_checkpoint`. Eligibility gates run before the dimensionless composite score.

**Step 3: Run tests and commit the task**

### Task 5: Add known-truth component recovery

**Files:**
- Create: `src/phase35/multistep/gatec_synthetic.py`
- Test: `tests/phase35/multistep/test_gatec_synthetic.py`

**Step 1: Create a paired known-truth generator**

Generate common/differential valve excitation, nonlinear opening, observed Tin boundary, latent cross-side terminal mixing and colored unmeasured disturbance. Preserve exact component truth.

**Step 2: Test identifiability and failure controls**

The main model must recover local response direction/amplitude under supported excitation. A collinear-input generator must refuse independent channel claims. A future-Tin leakage mutant and a response-collapse mutant must fail structural gates.

**Step 3: Run tests and commit the task**

### Task 6: Implement deterministic local runner and artifacts

**Files:**
- Create: `experiments/phase3_5/ms3r_gatec_model_screen.py`
- Create: `experiments/phase3_5/summarize_ms3r_gatec_model_screen.py`
- Test: `tests/phase35/test_ms3r_gatec_cli.py`

**Step 1: Write failing CLI tests**

Dry-run must print the exact RM0/RM1 matrices, run count, budget, validation-only boundary and `linux_authorized=false`. Machine artifacts must pin config/source/cache/Git hashes, boundary mode, operator route, selector eligibility and resource contract.

**Step 2: Implement dry-run and one-run execution**

No `--skip-existing` scientific retry behavior is allowed until a remote matrix is authorized. Partial runs remain visible. Summary code may produce diagnostics but never a Supervisor decision.

**Step 3: Run tests and commit the task**

### Task 7: Close local verification without Linux release

**Files:**
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify: `TODO.md`
- Modify: `experiments/phase3_5/README.md`
- Modify: `tests/phase35/test_experiment_status.py`

**Step 1: Run focused and full verification**

```bash
python -m pytest tests/phase35/multistep/test_gatec_contracts.py \
  tests/phase35/multistep/test_gatec_data.py \
  tests/phase35/multistep/test_gatec_model.py \
  tests/phase35/multistep/test_gatec_training.py \
  tests/phase35/multistep/test_gatec_synthetic.py \
  tests/phase35/test_ms3r_gatec_cli.py -q
python -m pytest tests/phase35 -q
python -m compileall -q src/phase35 experiments/phase3_5
```

Expected: all PASS; no test access or real-data training.

**Step 2: Update state**

Set `ms3_r.status=local_verified`, keep `linux_authorized_gate=null`, and record exact scripts/config/tests. Do not create a remote command or `ready_for_linux` state yet.

**Step 3: Commit and push the local framework**

Commit message: `feat(phase35): add MS3-R Gate C model framework`.

Remote execution requires a later, explicit authorization commit after local Supervisor inspection of the frozen matrix.
