"""Training loop for discrimination-matrix arms.

One `TrainSpec` = one frozen (unit, arm, seed) run.  The loop is deliberately
plain: Adam + grad clipping + early stopping on validation NLL, with the
best checkpoint written to the artifacts directory and a JSONL ledger per
epoch.  No LR scheduling, no sweeps -- the matrix forbids retry culture.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    BOUNDARY_ELEMENTS,
    OBSERVATION_ELEMENTS,
    PHYSICAL_STATE_ELEMENTS,
    BoundaryModelConfig,
    ClosureConfig,
    FinalWMProtocolError,
    ObserverConfig,
    TransitionConfig,
    WorldModelConfig,
)
from src.final_wm.data import SPLIT_TRAIN, CanonicalRecord, sample_windows
from src.final_wm.evaluation import evaluate_windows
from src.final_wm.model import FinalWorldModel
from src.final_wm.properties import AnalyticThermoProperties, ThermoProperties


@dataclass(frozen=True)
class TrainSpec:
    unit: str                       # dsyn | o1 | b1 | t1 | r1 | j1
    arm: str
    seed: int
    epochs: int = 30
    batch_size: int = 32
    batches_per_epoch: int = 200
    lr: float = 1e-3
    horizon: int = 18
    history_steps: int = 96
    patience: int = 5
    boundary_mode: str = "oracle"   # training-time boundary information mode
    train_boundary: bool = False    # add boundary-forecast NLL to the loss
    boundary_loss_only: bool = False  # train the boundary model alone (B1, J1 stage B)
    init_checkpoint: str | None = None  # warm start from a previous arm's checkpoint
    # Amendment v0.6-B (2026-08-26, anchor evidence pack): warm-start ONLY the
    # transition physics constants (transition.raw) from a reference checkpoint;
    # all networks init fresh.  Basin-rescue protocol for multi-seed runs.
    anchor_constants_checkpoint: str | None = None
    initial_state_mode: str = "steady"
    closure_mode: str = "none"
    latent_dim: int = 0
    eval_windows: int = 128
    eval_batch: int = 32

    def validate(self) -> None:
        if self.boundary_mode not in ("oracle", "forecast"):
            raise FinalWMProtocolError(f"unknown boundary_mode: {self.boundary_mode}")
        if self.boundary_mode == "forecast" and not (self.train_boundary or self.boundary_loss_only):
            raise FinalWMProtocolError("forecast-mode training must include the boundary loss")
        if self.boundary_loss_only and self.boundary_mode != "forecast":
            raise FinalWMProtocolError("boundary_loss_only requires forecast boundary mode")
        if self.epochs < 1 or self.batch_size < 1 or self.batches_per_epoch < 1:
            raise FinalWMProtocolError("training sizes must be positive")
        if self.init_checkpoint is not None and self.anchor_constants_checkpoint is not None:
            raise FinalWMProtocolError(
                "init_checkpoint already carries the physics constants; "
                "anchor_constants_checkpoint is for fresh-network constant anchoring only")
        if self.anchor_constants_checkpoint is not None and self.closure_mode.endswith("_norew"):
            raise FinalWMProtocolError(
                "anchor_constants_checkpoint would clobber the pinned rewetting-ablation "
                "constants (aW raw = -30); anchoring is not composable with _norew arms")


def build_world_model(spec: TrainSpec, properties: ThermoProperties | None = None) -> FinalWorldModel:
    spec.validate()
    # Amendment v0.4: the `_norew` suffix selects the first-class rewetting
    # ablation arm (aW frozen ~0, audit F3).  The closure injection mode
    # itself is unchanged; the suffix only toggles the transition config.
    closure_mode = spec.closure_mode.removesuffix("_norew")
    norew = closure_mode != spec.closure_mode
    config = WorldModelConfig(
        transition=TransitionConfig(latent_dim=spec.latent_dim, rewet_ablate=bool(norew)),
        closure=ClosureConfig(injection_mode=closure_mode),
        observer=ObserverConfig(history_steps=spec.history_steps, latent_dim=spec.latent_dim),
        boundary=BoundaryModelConfig(history_steps=spec.history_steps),
        boundary_mode=spec.boundary_mode,
        initial_state_mode=spec.initial_state_mode,
    )
    return FinalWorldModel(config, properties or AnalyticThermoProperties())


def apply_anchor_constants(model: FinalWorldModel, checkpoint_path: str | Path) -> None:
    """Warm-start ONLY the transition physics constants from a reference
    checkpoint (amendment v0.6-B).  Fail-closed: every raw parameter must be
    present in the reference state dict with a matching shape; nothing else
    (observer/closure/boundary/observation networks, latent_rho_raw) is
    touched, so multi-seed runs share the identified-constant basin while
    network noise stays seed-fresh."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    missing: list[str] = []
    for name, param in model.transition.raw.items():
        key = f"transition.raw.{name}"
        if key not in state or state[key].shape != param.shape:
            missing.append(key)
    if missing:
        raise FinalWMProtocolError(f"anchor checkpoint lacks transition constants: {missing}")
    with torch.no_grad():
        for name, param in model.transition.raw.items():
            param.copy_(state[f"transition.raw.{name}"])


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def boundary_nll(model: FinalWorldModel, history_boundary: torch.Tensor, history_actions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    seq = model.boundary_model.forecast(
        history_boundary, history_actions, horizon=target.shape[1]
    )
    var = torch.exp(seq.logvar)
    return (0.5 * (target - seq.mu) ** 2 / var + 0.5 * seq.logvar).mean()


def train_arm(
    spec: TrainSpec,
    record: CanonicalRecord,
    out_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    properties: ThermoProperties | None = None,
    compile_substep: bool = False,
) -> dict:
    """Train one arm; returns the final ledger entry (also appended to JSONL)."""
    spec.validate()
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "ledger.jsonl"

    torch.manual_seed(spec.seed)
    model = build_world_model(spec, properties).to(device)
    if compile_substep:
        # Speed lever for launch-bound rollouts (2026-08-21, user-reported
        # 15 h T1+R1 estimate): fuse the physics substep.  Numerics unchanged
        # (fp32, same graph); budgets/protocol untouched.
        model.transition._substep = torch.compile(model.transition._substep, dynamic=False)
    if spec.init_checkpoint is not None:
        payload = torch.load(spec.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
    if spec.anchor_constants_checkpoint is not None:
        apply_anchor_constants(model, spec.anchor_constants_checkpoint)
    if spec.boundary_loss_only:
        params = list(model.boundary_model.parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=spec.lr)
    gen = torch.Generator().manual_seed(spec.seed)

    run_id = f"{spec.unit}_{spec.arm}_seed{spec.seed}"
    base_entry = {
        "run_id": run_id,
        "spec": asdict(spec),
        "commit": _git_commit(),
        "properties": type(model.transition.properties).__name__,
        "device": str(device),
    }
    validation_anchor_seed = 10_000 + spec.seed
    base_entry["validation_anchor_seed"] = validation_anchor_seed

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    val_history: list[float] = []
    stop_reason = "cap"
    t0 = time.time()
    # Per-phase wall-clock accounting (2026-08-21: the 19.4ks seed0 arms made
    # clear we must stop guessing where training time goes).
    t_data = t_step = t_eval = 0.0
    with ledger_path.open("a", encoding="utf-8") as ledger:
        for epoch in range(spec.epochs):
            model.train()
            train_loss = 0.0
            for _ in range(spec.batches_per_epoch):
                _t = time.time()
                batch = sample_windows(
                    record, SPLIT_TRAIN, spec.batch_size, spec.history_steps, spec.horizon, gen
                )
                t_data += time.time() - _t
                history = batch.history.__class__(
                    obs=batch.history.obs.to(device),
                    actions=batch.history.actions.to(device),
                    boundary=batch.history.boundary.to(device),
                )
                future_actions = batch.future_actions.to(device)
                future_obs = batch.future_obs.to(device)
                future_boundary = batch.future_boundary.to(device)
                if spec.boundary_loss_only:
                    loss = boundary_nll(model, history.boundary, history.actions, future_boundary)
                else:
                    result = model.forecast(
                        history,
                        future_actions,
                        boundary_mode=spec.boundary_mode,
                        true_future_boundary=future_boundary if spec.boundary_mode == "oracle" else None,
                    )
                    loss = model.observation_nll(result.temps_mu, result.temps_sigma, future_obs)
                    if spec.train_boundary:
                        loss = loss + boundary_nll(model, history.boundary, history.actions, future_boundary)
                _t = time.time()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                train_loss += float(loss)
                if torch.cuda.is_available() and "cuda" in str(device):
                    torch.cuda.synchronize()
                t_step += time.time() - _t
            train_loss /= spec.batches_per_epoch

            _t = time.time()
            val = evaluate_windows(
                model, record, 1,
                n_windows=spec.eval_windows, batch_size=spec.eval_batch,
                history_steps=spec.history_steps, horizon=spec.horizon,
                boundary_mode=spec.boundary_mode, seed=validation_anchor_seed, device=device,
            )
            val_nll = float(val.nll.mean())
            val_history.append(val_nll)
            if torch.cuda.is_available() and "cuda" in str(device):
                torch.cuda.synchronize()
            t_eval += time.time() - _t
            entry = dict(base_entry, epoch=epoch, train_loss=train_loss, val_nll=val_nll,
                         wall_seconds=time.time() - t0)
            ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")
            ledger.flush()

            if val_nll < best_val - 1e-4:
                best_val = val_nll
                best_epoch = epoch
                epochs_since_best = 0
                torch.save(
                    {"state_dict": model.state_dict(), "spec": asdict(spec), "val_nll": best_val},
                    ckpt_dir / f"{run_id}.pt",
                )
            else:
                epochs_since_best += 1
                if epochs_since_best >= spec.patience:
                    stop_reason = "patience"
                    break

    final = dict(base_entry, final=True, best_val_nll=best_val, best_epoch=best_epoch,
                 epochs_run=epoch + 1, wall_seconds=time.time() - t0,
                 # Convergence diagnostics (matrix v0.2): stop_reason=cap with a still-
                 # descending val_tail flags undertrained arms; converged = early stop.
                 stop_reason=stop_reason, converged=stop_reason == "patience",
                 val_tail=val_history[-5:],
                 # Runtime speed flags for audit uniformity: all arms feeding one
                 # verdict must share the same flag state (2026-08-21).
                 flags={"compile_substep": compile_substep,
                        "matmul_precision": torch.get_float32_matmul_precision()},
                 timing={"data_s": round(t_data, 1), "step_s": round(t_step, 1),
                         "eval_s": round(t_eval, 1)})
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(final, ensure_ascii=False) + "\n")
    return final


def model_structure_fingerprint() -> str:
    """Structural fingerprint of the world-model code, independent of TrainSpec.

    Root-cause fix for the 2026-08-20 Hermes rerun failure
    (results/final_wm/rerun_failure_report_20260820.md §3): the resume
    fingerprint covered only the TrainSpec fields, so a repair batch that
    changes the model structure (state layout 9->11, new parameters, prior
    re-anchoring) silently "RESUMED" onto incompatible v0.2 artifacts and
    crashed on state-dict load.  Any change to the state/boundary/action/
    observation registries or to the transition prior table now busts the
    resume cache automatically.
    """
    from src.final_wm.transition import TRANSITION_PARAM_PRIORS

    payload = json.dumps(
        {
            "states": PHYSICAL_STATE_ELEMENTS,
            "boundary": BOUNDARY_ELEMENTS,
            "actions": ACTION_ELEMENTS,
            "observations": OBSERVATION_ELEMENTS,
            "priors": TRANSITION_PARAM_PRIORS,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _git_tree_hash(path: str) -> str:
    """Tree hash of a tracked directory at HEAD (changes iff its content changes)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", f"HEAD:{path}"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _content_identity(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return f"MISSING:{candidate}"
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(
    spec: TrainSpec,
    *,
    record_path: str | Path | None = None,
    properties_path: str | Path | None = None,
) -> str:
    # The code tree hashes close the bisection blind spot: a repair that only
    # changes dynamics (no registry/prior change) still busts the resume cache.
    payload = json.dumps(
        {
            "spec": asdict(spec),
            "structure": model_structure_fingerprint(),
            "code": [_git_tree_hash("src/final_wm"), _git_tree_hash("experiments/final_wm")],
            "inputs": {
                "record_sha256": _content_identity(record_path),
                "properties_sha256": _content_identity(properties_path),
                "init_checkpoint_sha256": _content_identity(spec.init_checkpoint),
                "anchor_checkpoint_sha256": _content_identity(spec.anchor_constants_checkpoint),
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
