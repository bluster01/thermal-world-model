"""Three-stage training for the Phase 3.5-MS2-J scheduled graybox."""

from __future__ import annotations

import copy
import hashlib
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .contracts import OperatorConfig
from .operators import StableGrayboxOperator, build_response_operator
from .synthetic import SyntheticBatch, SyntheticSpec, generate_synthetic_split
from .training import (
    MultiStepTrainResult,
    TrainingConfig,
    _git_sha,
    _json_dump,
    _sha256,
    evaluate_operator,
    set_seed,
    structural_diagnostics,
)


@dataclass(frozen=True)
class StagedTrainingConfig:
    stage_a_epochs: int = 120
    stage_b_epochs: int = 90
    stage_c_epochs: int = 90
    stage_patience: int = 20
    joint_learning_rate_scale: float = 0.2

    def validate(self, total_epoch_cap: int) -> None:
        counts = (
            self.stage_a_epochs,
            self.stage_b_epochs,
            self.stage_c_epochs,
            self.stage_patience,
        )
        if min(counts) < 1:
            raise ValueError("staged training counts must be positive")
        if sum(counts[:3]) != total_epoch_cap:
            raise ValueError("staged epoch caps must sum to the joint epoch cap")
        if not 0 < self.joint_learning_rate_scale <= 1:
            raise ValueError("joint_learning_rate_scale must be in (0, 1]")


def environment_payload(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "platform": platform.platform(),
    }


def trajectory_design_sha256(batch: SyntheticBatch) -> str:
    digest = hashlib.sha256()
    for tensor in (
        batch.context,
        batch.action,
        batch.reference,
        batch.clean_effect,
        batch.profile_ids,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _dataset(batch: SyntheticBatch) -> TensorDataset:
    return TensorDataset(batch.context, batch.action, batch.reference, batch.target_effect)


def _set_trainable(operator: StableGrayboxOperator, stage: str) -> list[str]:
    if stage not in {"stage_a", "stage_b", "stage_c"}:
        raise ValueError(f"unknown training stage={stage}")
    trainable = []
    for name, parameter in operator.named_parameters():
        is_schedule = name.startswith("gain_schedule.") or name.startswith(
            "tau_schedule."
        )
        enabled = (
            (stage == "stage_a" and not is_schedule)
            or (stage == "stage_b" and is_schedule)
            or stage == "stage_c"
        )
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise RuntimeError(f"stage={stage} has no trainable parameters")
    return trainable


def _run_stage(
    operator: StableGrayboxOperator,
    stage: str,
    epochs: int,
    patience: int,
    learning_rate: float,
    training_config: TrainingConfig,
    loader: DataLoader,
    validation_batch: SyntheticBatch,
    device: torch.device,
    history: list[dict[str, Any]],
    global_epoch: int,
    history_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    trainable_names = _set_trainable(operator, stage)
    parameters = [parameter for parameter in operator.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=training_config.weight_decay,
    )
    boundary_metrics, _ = evaluate_operator(operator, validation_batch, device)
    best_score = float(boundary_metrics["effect_mae"])
    best_state = copy.deepcopy(operator.state_dict())
    best_stage_epoch = 0
    wait = 0
    optimizer_updates = 0
    epochs_ran = 0
    for stage_epoch in range(1, epochs + 1):
        operator.train()
        losses = []
        for context, action, reference, target in loader:
            context = context.to(device)
            action = action.to(device)
            reference = reference.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = operator(context, action, reference)
            data_loss = F.smooth_l1_loss(output.effect, target, beta=0.2)
            physics_residual = output.diagnostics.get("physics_residual_mse")
            if physics_residual is None:
                physics_residual = torch.zeros(
                    (), dtype=data_loss.dtype, device=device
                )
            loss = data_loss + training_config.physics_weight * physics_residual
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite staged loss at {stage}/epoch={stage_epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, training_config.gradient_clip
            )
            optimizer.step()
            optimizer_updates += 1
            losses.append(float(loss.detach().cpu()))
        epochs_ran = stage_epoch
        global_epoch += 1
        validation_metrics, _ = evaluate_operator(operator, validation_batch, device)
        score = float(validation_metrics["effect_mae"])
        history.append(
            {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch,
                "train_loss": sum(losses) / len(losses),
                "validation_effect_mae": score,
                "learning_rate": learning_rate,
                "trainable_parameters": trainable_names,
            }
        )
        _json_dump(history_path, history)
        if score < best_score - 1e-8:
            best_score = score
            best_stage_epoch = stage_epoch
            best_state = copy.deepcopy(operator.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    operator.load_state_dict(best_state)
    best_metrics, _ = evaluate_operator(operator, validation_batch, device)
    best_metrics["structural_diagnostics"] = structural_diagnostics(
        operator, validation_batch, device
    )
    best_metrics["truth"] = validation_batch.truth
    summary = {
        "stage": stage,
        "epoch_cap": epochs,
        "epochs_ran": epochs_ran,
        "best_stage_epoch": best_stage_epoch,
        "boundary_validation_effect_mae": boundary_metrics["effect_mae"],
        "best_validation_effect_mae": best_metrics["effect_mae"],
        "optimizer_updates": optimizer_updates,
        "learning_rate": learning_rate,
        "trainable_parameters": trainable_names,
    }
    return summary, best_metrics, global_epoch


def train_staged_synthetic_run(
    operator_config: OperatorConfig,
    training_config: TrainingConfig,
    staged_config: StagedTrainingConfig,
    synthetic_spec: SyntheticSpec,
    validation_samples: int,
    seed: int,
    route_id: str,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    repo_root: str | Path | None = None,
    overwrite: bool = False,
    protocol_version: str = "phase3.5-ms2j-v1",
) -> MultiStepTrainResult:
    operator_config.validate()
    training_config.validate()
    staged_config.validate(training_config.epochs)
    synthetic_spec.validate()
    if not (
        operator_config.route == "graybox"
        and operator_config.context_scheduled
        and operator_config.opening_map == "monotone"
    ):
        raise ValueError(
            "staged training requires a scheduled graybox with learned monotone opening"
        )
    out = Path(output_dir)
    checkpoint_path = out / "checkpoint_best_val.pt"
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite completed staged run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    dev = torch.device(device)
    operator = build_response_operator(operator_config).to(dev)
    if not isinstance(operator, StableGrayboxOperator):
        raise TypeError("staged operator did not build a StableGrayboxOperator")
    seeded_spec = replace(synthetic_spec, seed=synthetic_spec.seed + seed * 1_000_003)
    train_batch = generate_synthetic_split(seeded_spec, "train")
    validation_batch = generate_synthetic_split(
        replace(seeded_spec, samples=validation_samples), "validation"
    )
    loader = DataLoader(
        _dataset(train_batch),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    stage_specs = (
        ("stage_a", staged_config.stage_a_epochs, training_config.learning_rate),
        ("stage_b", staged_config.stage_b_epochs, training_config.learning_rate),
        (
            "stage_c",
            staged_config.stage_c_epochs,
            training_config.learning_rate * staged_config.joint_learning_rate_scale,
        ),
    )
    history: list[dict[str, Any]] = []
    stage_summaries = []
    stage_checkpoints = {}
    global_epoch = 0
    started = time.time()
    git_sha = _git_sha(repo_root)
    for stage, epochs, learning_rate in stage_specs:
        summary, metrics, global_epoch = _run_stage(
            operator=operator,
            stage=stage,
            epochs=epochs,
            patience=staged_config.stage_patience,
            learning_rate=learning_rate,
            training_config=training_config,
            loader=loader,
            validation_batch=validation_batch,
            device=dev,
            history=history,
            global_epoch=global_epoch,
            history_path=out / "history.json",
        )
        stage_summaries.append(summary)
        metrics_path = out / f"metrics_{stage}_validation.json"
        _json_dump(metrics_path, metrics)
        stage_checkpoint = out / f"checkpoint_{stage}.pt"
        payload = {
            "protocol_version": protocol_version,
            "route_id": route_id,
            "seed": seed,
            "stage": stage,
            "operator_config": operator_config.to_dict(),
            "training_config": asdict(training_config),
            "staged_training_config": asdict(staged_config),
            "synthetic_spec": asdict(seeded_spec),
            "model_state_dict": copy.deepcopy(operator.state_dict()),
            "validation_metrics": metrics,
            "stage_summary": summary,
            "git_sha": git_sha,
        }
        torch.save(payload, stage_checkpoint)
        stage_checkpoints[stage] = {
            "file": stage_checkpoint.name,
            "sha256": _sha256(stage_checkpoint),
        }
        if stage == "stage_c":
            torch.save(payload, checkpoint_path)

    final_metrics, _ = evaluate_operator(operator, validation_batch, dev)
    final_metrics["structural_diagnostics"] = structural_diagnostics(
        operator, validation_batch, dev
    )
    final_metrics["truth"] = validation_batch.truth
    _json_dump(out / "metrics_validation.json", final_metrics)
    best_epoch = sum(
        summary["epochs_ran"] for summary in stage_summaries[:-1]
    ) + stage_summaries[-1]["best_stage_epoch"]
    manifest = {
        "protocol_version": protocol_version,
        "evidence_scope": "synthetic_joint_coupling_validation_not_field_causality",
        "run_id": f"synthetic_{route_id}_s{seed}",
        "route_id": route_id,
        "seed": seed,
        "training_mode": "staged",
        "operator_config": operator_config.to_dict(),
        "training_config": asdict(training_config),
        "staged_training_config": asdict(staged_config),
        "synthetic_spec": asdict(seeded_spec),
        "train_samples": len(train_batch.context),
        "validation_samples": len(validation_batch.context),
        "checkpoint_selector": "stage_c_validation_effect_mae_with_boundary_epoch0",
        "best_epoch": best_epoch,
        "git_sha": git_sha,
        "environment": environment_payload(dev),
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "stage_summaries": stage_summaries,
        "stage_checkpoints": stage_checkpoints,
        "validation_trajectory_design_sha256": trajectory_design_sha256(
            validation_batch
        ),
        "test_accessed": False,
    }
    _json_dump(out / "manifest.json", manifest)
    return MultiStepTrainResult(
        out, checkpoint_path, best_epoch, final_metrics, None
    )
