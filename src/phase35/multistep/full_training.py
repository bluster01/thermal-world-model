"""Known-truth training for the complete action-blind-free + response model."""

from __future__ import annotations

import copy
import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .contracts import (
    ActionResponseOperator,
    OperatorCapabilities,
    OperatorConfig,
    ResponseOutput,
)
from .model import A1PhysMultiStep, ContextFreePredictor
from .operators import build_response_operator
from .staging import environment_payload
from .synthetic import SyntheticBatch, SyntheticSpec, generate_synthetic_split
from .training import _json_dump, _sha256, set_seed, structural_diagnostics


TRAINING_MODES = {
    "free_only",
    "joint_total",
    "staged_total",
    "component_oracle",
}


@dataclass(frozen=True)
class FullCouplingTrainingConfig:
    batch_size: int = 64
    epochs: int = 300
    patience: int = 30
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    free_hidden_dim: int = 32
    component_weight: float = 1.0
    stage_a_epochs: int = 80
    stage_b_epochs: int = 140
    stage_c_epochs: int = 80
    stage_patience: int = 20
    joint_learning_rate_scale: float = 0.2

    def validate(self) -> None:
        counts = (
            self.batch_size,
            self.epochs,
            self.patience,
            self.free_hidden_dim,
            self.stage_a_epochs,
            self.stage_b_epochs,
            self.stage_c_epochs,
            self.stage_patience,
        )
        if min(counts) < 1:
            raise ValueError("full-coupling training counts must be positive")
        if self.stage_a_epochs + self.stage_b_epochs + self.stage_c_epochs != self.epochs:
            raise ValueError("MS5 stage epoch caps must sum to the total epoch cap")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.gradient_clip <= 0
            or self.component_weight <= 0
        ):
            raise ValueError("MS5 optimizer and component weights are invalid")
        if not 0 < self.joint_learning_rate_scale <= 1:
            raise ValueError("MS5 joint learning-rate scale must be in (0,1]")


@dataclass
class FullTrainResult:
    output_dir: Path
    checkpoint: Path
    best_epoch: int
    validation_metrics: dict[str, Any]


class ZeroResponseOperator(ActionResponseOperator):
    """Exact no-response negative control for the free-only mode."""

    capabilities = OperatorCapabilities(
        stateful_rollout=True,
        fixed_horizon=False,
        direction_constrained=True,
        continuous_time=False,
    )

    def forward(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> ResponseOutput:
        self._validate_inputs(context, action, reference)
        effect = torch.zeros_like(action)
        state = effect.unsqueeze(-1)
        return ResponseOutput(effect, state, {"zero_response_control": torch.ones((), device=effect.device)})


def build_full_model(
    operator_config: OperatorConfig,
    *,
    free_hidden_dim: int,
    mode: str,
) -> A1PhysMultiStep:
    if mode not in TRAINING_MODES:
        raise ValueError(f"unknown MS5 training mode={mode!r}")
    operator_config.validate()
    free = ContextFreePredictor(
        operator_config.context_dim,
        operator_config.horizon,
        hidden_dim=free_hidden_dim,
    )
    response: ActionResponseOperator
    if mode == "free_only":
        response = ZeroResponseOperator(operator_config)
    else:
        response = build_response_operator(operator_config)
    return A1PhysMultiStep(free, response)


def trajectory_design_sha256(batch: SyntheticBatch) -> str:
    digest = hashlib.sha256()
    for tensor in (
        batch.context,
        batch.action,
        batch.reference,
        batch.clean_free,
        batch.clean_effect,
        batch.profile_ids,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(numerator / denominator.clamp_min(1e-8))


def component_metrics(
    batch: SyntheticBatch,
    prediction: torch.Tensor,
    free_prediction: torch.Tensor,
    response_prediction: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction = prediction.detach().cpu()
    free_prediction = free_prediction.detach().cpu()
    response_prediction = response_prediction.detach().cpu()
    noisy_target = batch.target_temperature.detach().cpu()
    clean_total = batch.clean_total.detach().cpu()
    clean_free = batch.clean_free.detach().cpu()
    clean_response = batch.clean_effect.detach().cpu()
    total_error = (prediction - clean_total).abs()
    free_error = (free_prediction - clean_free).abs()
    response_error = (response_prediction - clean_response).abs()
    total_scale = clean_total.abs().mean()
    free_scale = clean_free.abs().mean()
    response_scale = clean_response.abs().mean()
    predicted_response_abs = response_prediction.abs().mean()
    response_amplitude_ratio = _safe_ratio(predicted_response_abs, response_scale)
    metrics = {
        "sample_count": int(len(batch.context)),
        "horizon": int(clean_total.shape[1]),
        "total_noisy_mae": float((prediction - noisy_target).abs().mean()),
        "total_clean_mae": float(total_error.mean()),
        "total_clean_scale": float(total_scale),
        "total_clean_nmae": _safe_ratio(total_error.mean(), total_scale),
        "free_clean_mae": float(free_error.mean()),
        "free_clean_scale": float(free_scale),
        "free_clean_nmae": _safe_ratio(free_error.mean(), free_scale),
        "response_clean_mae": float(response_error.mean()),
        "response_clean_scale": float(response_scale),
        "response_clean_nmae": _safe_ratio(response_error.mean(), response_scale),
        "response_amplitude_ratio": response_amplitude_ratio,
        "absorption_flag": bool(
            _safe_ratio(response_error.mean(), response_scale) > 0.15
            or response_amplitude_ratio < 0.80
        ),
        "over_attribution_flag": bool(response_amplitude_ratio > 1.20),
    }
    episodes = {
        "episode_ids": list(range(len(batch.context))),
        "profile_ids": batch.profile_ids.detach().cpu().tolist(),
        "profile_names": list(batch.profile_names),
        "trajectory_design_sha256": trajectory_design_sha256(batch),
        "total_clean_mae": total_error.mean(dim=1).tolist(),
        "total_clean_scale": clean_total.abs().mean(dim=1).tolist(),
        "free_clean_mae": free_error.mean(dim=1).tolist(),
        "free_clean_scale": clean_free.abs().mean(dim=1).tolist(),
        "response_clean_mae": response_error.mean(dim=1).tolist(),
        "response_clean_scale": clean_response.abs().mean(dim=1).tolist(),
        "predicted_response_abs": response_prediction.abs().mean(dim=1).tolist(),
        "true_response_abs": clean_response.abs().mean(dim=1).tolist(),
    }
    return metrics, episodes


def _subset(batch: SyntheticBatch, mask: torch.Tensor) -> SyntheticBatch:
    return SyntheticBatch(
        context=batch.context[mask],
        action=batch.action[mask],
        reference=batch.reference[mask],
        clean_effect=batch.clean_effect[mask],
        target_effect=batch.target_effect[mask],
        target_temperature=batch.target_temperature[mask],
        clean_free=batch.clean_free[mask],
        clean_total=batch.clean_total[mask],
        colored_disturbance=batch.colored_disturbance[mask],
        profile_ids=batch.profile_ids[mask],
        profile_names=batch.profile_names,
        truth=batch.truth,
    )


@torch.no_grad()
def evaluate_full_model(
    model: A1PhysMultiStep,
    batch: SyntheticBatch,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    predictions = []
    free_predictions = []
    response_predictions = []
    for start in range(0, len(batch.context), batch_size):
        stop = start + batch_size
        output = model(
            batch.context[start:stop].to(device),
            batch.action[start:stop].to(device),
            batch.reference[start:stop].to(device),
        )
        predictions.append(output["prediction"].cpu())
        free_predictions.append(output["free_prediction"].cpu())
        response_predictions.append(output["effect"].cpu())
    prediction = torch.cat(predictions)
    free = torch.cat(free_predictions)
    response = torch.cat(response_predictions)
    metrics, episodes = component_metrics(batch, prediction, free, response)
    response_diagnostics = structural_diagnostics(
        model.response_operator, batch, device
    )
    count = min(8, len(batch.context))
    context = batch.context[:count].to(device)
    action = batch.action[:count].to(device)
    reference = batch.reference[:count].to(device)
    normal = model(context, action, reference)
    changed = action.clone()
    boundary = max(1, action.shape[1] // 2)
    changed[:, boundary:] = (changed[:, boundary:] + 7.0).clamp(0.0, 100.0)
    changed_output = model(context, changed, reference)
    response_diagnostics.update(
        {
            "free_future_action_leakage_max_error": float(
                (normal["free_prediction"] - changed_output["free_prediction"])
                .abs()
                .max()
            ),
            "finite_prediction": bool(torch.isfinite(normal["prediction"]).all()),
            "finite_free": bool(torch.isfinite(normal["free_prediction"]).all()),
        }
    )
    metrics["structural_diagnostics"] = response_diagnostics
    metrics["truth"] = batch.truth
    return metrics, episodes


def _set_trainability(
    model: A1PhysMultiStep, *, train_free: bool, train_response: bool
) -> list[torch.nn.Parameter]:
    for parameter in model.free_predictor.parameters():
        parameter.requires_grad_(train_free)
        parameter.grad = None
    for parameter in model.response_operator.parameters():
        parameter.requires_grad_(train_response)
        parameter.grad = None
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("MS5 phase has no trainable parameters")
    return parameters


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _loader(batch: SyntheticBatch, batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(
        batch.context,
        batch.action,
        batch.reference,
        batch.target_temperature,
        batch.clean_free,
        batch.clean_effect,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _run_phase(
    *,
    model: A1PhysMultiStep,
    phase: str,
    train_batch: SyntheticBatch,
    selector_batch: SyntheticBatch,
    device: torch.device,
    config: FullCouplingTrainingConfig,
    epochs: int,
    patience: int,
    learning_rate: float,
    train_free: bool,
    train_response: bool,
    component_supervision: bool,
    history: list[dict[str, Any]],
    global_epoch: int,
    history_path: Path,
    loader_seed: int,
) -> tuple[dict[str, Any], int, int]:
    parameters = _set_trainability(
        model, train_free=train_free, train_response=train_response
    )
    free_parameters = list(model.free_predictor.parameters())
    response_parameters = list(model.response_operator.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_metrics, _ = evaluate_full_model(model, selector_batch, device)
    best_score = float(initial_metrics["total_noisy_mae"])
    best_state = copy.deepcopy(model.state_dict())
    best_phase_epoch = 0
    best_global_epoch = global_epoch
    wait = 0
    updates = 0
    epochs_ran = 0
    for phase_epoch in range(1, epochs + 1):
        model.train()
        losses = []
        free_gradients = []
        response_gradients = []
        for context, action, reference, target, clean_free, clean_effect in _loader(
            train_batch, config.batch_size, loader_seed + phase_epoch
        ):
            context = context.to(device)
            action = action.to(device)
            reference = reference.to(device)
            target = target.to(device)
            clean_free = clean_free.to(device)
            clean_effect = clean_effect.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(context, action, reference)
            loss = F.smooth_l1_loss(output["prediction"], target, beta=0.2)
            if component_supervision:
                loss = loss + config.component_weight * (
                    F.smooth_l1_loss(
                        output["free_prediction"], clean_free, beta=0.2
                    )
                    + F.smooth_l1_loss(output["effect"], clean_effect, beta=0.2)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MS5 loss at {phase}/{phase_epoch}")
            loss.backward()
            free_gradients.append(_gradient_norm(free_parameters))
            response_gradients.append(_gradient_norm(response_parameters))
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach().cpu()))
        epochs_ran = phase_epoch
        global_epoch += 1
        validation_metrics, _ = evaluate_full_model(model, selector_batch, device)
        score = float(validation_metrics["total_noisy_mae"])
        history.append(
            {
                "epoch": global_epoch,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "train_loss": float(np.mean(losses)),
                "validation_total_noisy_mae": score,
                "validation_response_clean_nmae": validation_metrics[
                    "response_clean_nmae"
                ],
                "free_gradient_norm": float(np.mean(free_gradients)),
                "response_gradient_norm": float(np.mean(response_gradients)),
                "learning_rate": learning_rate,
            }
        )
        _json_dump(history_path, history)
        if score < best_score - 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_phase_epoch = phase_epoch
            best_global_epoch = global_epoch
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return (
        {
            "stage": phase,
            "epoch_cap": epochs,
            "epochs_ran": epochs_ran,
            "best_stage_epoch": best_phase_epoch,
            "best_validation_total_noisy_mae": best_score,
            "optimizer_updates": updates,
            "learning_rate": learning_rate,
            "train_free": train_free,
            "train_response": train_response,
            "component_supervision": component_supervision,
        },
        global_epoch,
        best_global_epoch,
    )


def _git_sha(repo_root: str | Path | None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _branch_drift(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], prefix: str
) -> float:
    values = [
        (after[key].detach().float() - value.detach().float()).square().sum()
        for key, value in before.items()
        if key.startswith(prefix)
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def train_full_synthetic_run(
    *,
    operator_config: OperatorConfig,
    full_config: FullCouplingTrainingConfig,
    synthetic_spec: SyntheticSpec,
    validation_samples: int,
    seed: int,
    mode: str,
    route_id: str,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    repo_root: str | Path | None = None,
    overwrite: bool = False,
    protocol_version: str = "phase3.5-ms5-v1",
) -> FullTrainResult:
    operator_config.validate()
    full_config.validate()
    synthetic_spec.validate()
    if mode not in TRAINING_MODES:
        raise ValueError(f"unknown MS5 mode={mode!r}")
    if synthetic_spec.truth_regime != "full_coupled_context_scheduled":
        raise ValueError("MS5 requires the frozen full-coupling synthetic truth")
    out = Path(output_dir)
    checkpoint = out / "checkpoint_best_val.pt"
    if checkpoint.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite completed MS5 run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    dev = torch.device(device)
    model = build_full_model(
        operator_config,
        free_hidden_dim=full_config.free_hidden_dim,
        mode=mode,
    ).to(dev)
    initial_state = copy.deepcopy(model.state_dict())
    seeded_spec = replace(synthetic_spec, seed=synthetic_spec.seed + seed * 1_000_003)
    train_batch = generate_synthetic_split(seeded_spec, "train")
    validation_batch = generate_synthetic_split(
        replace(seeded_spec, samples=validation_samples), "validation"
    )
    history: list[dict[str, Any]] = []
    stages = []
    stage_checkpoints = []
    global_epoch = 0
    selected_epoch = 0
    started = time.time()
    if mode == "staged_total":
        hold_train = _subset(train_batch, train_batch.profile_ids == 0)
        hold_validation = _subset(
            validation_batch, validation_batch.profile_ids == 0
        )
        specs = (
            (
                "stage_a_free_hold",
                hold_train,
                hold_validation,
                full_config.stage_a_epochs,
                full_config.learning_rate,
                True,
                False,
            ),
            (
                "stage_b_response_frozen_free",
                train_batch,
                validation_batch,
                full_config.stage_b_epochs,
                full_config.learning_rate,
                False,
                True,
            ),
            (
                "stage_c_low_lr_joint",
                train_batch,
                validation_batch,
                full_config.stage_c_epochs,
                full_config.learning_rate * full_config.joint_learning_rate_scale,
                True,
                True,
            ),
        )
        for index, (
            stage,
            stage_train,
            stage_validation,
            epochs,
            learning_rate,
            train_free,
            train_response,
        ) in enumerate(specs):
            summary, global_epoch, selected_epoch = _run_phase(
                model=model,
                phase=stage,
                train_batch=stage_train,
                selector_batch=stage_validation,
                device=dev,
                config=full_config,
                epochs=epochs,
                patience=full_config.stage_patience,
                learning_rate=learning_rate,
                train_free=train_free,
                train_response=train_response,
                component_supervision=False,
                history=history,
                global_epoch=global_epoch,
                history_path=out / "history.json",
                loader_seed=seed * 100 + index * 10_000,
            )
            stages.append(summary)
            stage_path = out / f"checkpoint_{stage}.pt"
            torch.save(
                {
                    "protocol_version": protocol_version,
                    "route_id": route_id,
                    "seed": seed,
                    "mode": mode,
                    "stage": stage,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                },
                stage_path,
            )
            stage_checkpoints.append(
                {
                    "stage": stage,
                    "path": stage_path.name,
                    "sha256": _sha256(stage_path),
                }
            )
    else:
        summary, global_epoch, selected_epoch = _run_phase(
            model=model,
            phase=mode,
            train_batch=train_batch,
            selector_batch=validation_batch,
            device=dev,
            config=full_config,
            epochs=full_config.epochs,
            patience=full_config.patience,
            learning_rate=full_config.learning_rate,
            train_free=True,
            train_response=mode != "free_only",
            component_supervision=mode == "component_oracle",
            history=history,
            global_epoch=global_epoch,
            history_path=out / "history.json",
            loader_seed=seed * 100,
        )
        stages.append(summary)
    validation_metrics, episodes = evaluate_full_model(
        model, validation_batch, dev
    )
    validation_metrics["parameter_drift"] = {
        "free_l2": _branch_drift(
            initial_state, model.state_dict(), "free_predictor."
        ),
        "response_l2": _branch_drift(
            initial_state, model.state_dict(), "response_operator."
        ),
    }
    _json_dump(out / "metrics_validation.json", validation_metrics)
    _json_dump(out / "episode_metrics_validation.json", episodes)
    git_sha = _git_sha(repo_root)
    payload = {
        "protocol_version": protocol_version,
        "route_id": route_id,
        "seed": seed,
        "mode": mode,
        "operator_config": operator_config.to_dict(),
        "full_training_config": asdict(full_config),
        "synthetic_spec": asdict(seeded_spec),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "validation_metrics": validation_metrics,
        "git_sha": git_sha,
    }
    torch.save(payload, checkpoint)
    manifest = {
        "protocol_version": protocol_version,
        "evidence_scope": "synthetic_full_free_response_coupling_validation_not_field_causality",
        "run_id": f"synthetic_{route_id}_s{seed}",
        "route_id": route_id,
        "seed": seed,
        "training_mode": mode,
        "operator_config": operator_config.to_dict(),
        "full_training_config": asdict(full_config),
        "synthetic_spec": asdict(seeded_spec),
        "train_samples": len(train_batch.context),
        "validation_samples": len(validation_batch.context),
        "checkpoint_selector": "validation_total_noisy_mae",
        "best_epoch": selected_epoch,
        "epochs_ran": global_epoch,
        "git_sha": git_sha,
        "device": str(dev),
        "environment": environment_payload(dev),
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": _sha256(checkpoint),
        "validation_trajectory_design_sha256": episodes[
            "trajectory_design_sha256"
        ],
        "stage_summaries": stages,
        "stage_checkpoints": stage_checkpoints,
        "test_accessed": False,
        "test_authorized": False,
    }
    _json_dump(out / "manifest.json", manifest)
    return FullTrainResult(out, checkpoint, selected_epoch, validation_metrics)
