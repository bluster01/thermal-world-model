"""Validation-selected training for the known-truth multi-step benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .contracts import ActionResponseOperator, OperatorConfig
from .operators import build_response_operator
from .synthetic import SyntheticBatch, SyntheticSpec, generate_synthetic_split


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    patience: int = 15
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    physics_weight: float = 1e-2
    gradient_clip: float = 1.0

    def validate(self) -> None:
        if min(self.batch_size, self.epochs, self.patience) < 1:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.physics_weight < 0:
            raise ValueError("optimizer and physics weights are outside supported ranges")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


@dataclass
class MultiStepTrainResult:
    output_dir: Path
    checkpoint: Path
    best_epoch: int
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha(repo_root: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(float(value.detach().cpu()))
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def response_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    clean_target: torch.Tensor | None = None,
) -> dict[str, Any]:
    target = target.detach().cpu()
    prediction = prediction.detach().cpu()
    error = prediction - target
    horizon = target.shape[1]
    horizon_points = sorted(set(min(point, horizon) for point in (1, 6, 18, 60)))
    active = target.abs() > 0.01
    if active.any():
        direction_accuracy = float((torch.sign(target[active]) == torch.sign(prediction[active])).float().mean())
    else:
        direction_accuracy = None
    metrics = {
        "effect_mae": float(error.abs().mean()),
        "effect_rmse": float(error.square().mean().sqrt()),
        "integrated_absolute_error": float(error.abs().sum(dim=1).mean()),
        "direction_accuracy_nonzero": direction_accuracy,
        "horizon_mae": {
            f"H{point}": float(error[:, point - 1].abs().mean()) for point in horizon_points
        },
        "sample_count": int(target.shape[0]),
        "horizon": int(horizon),
    }
    if clean_target is not None:
        clean_target = clean_target.detach().cpu()
        clean_error = prediction - clean_target
        clean_active = clean_target.abs() > 0.01
        clean_scale = clean_target.abs().mean()
        metrics.update({
            "clean_effect_mae": float(clean_error.abs().mean()),
            "clean_effect_rmse": float(clean_error.square().mean().sqrt()),
            "clean_effect_scale": float(clean_scale),
            "clean_effect_nmae": float(clean_error.abs().mean() / clean_scale.clamp_min(1e-8)),
            "clean_effect_mae_active": float(clean_error[clean_active].abs().mean())
            if clean_active.any() else None,
            "noise_mae": float((target - clean_target).abs().mean()),
            "direction_accuracy_clean_nonzero": float(
                (torch.sign(clean_target[clean_active]) == torch.sign(prediction[clean_active]))
                .float()
                .mean()
            ) if clean_active.any() else None,
            "clean_horizon_mae": {
                f"H{point}": float(clean_error[:, point - 1].abs().mean())
                for point in horizon_points
            },
        })
    return metrics


@torch.no_grad()
def evaluate_operator(
    operator: ActionResponseOperator,
    batch: SyntheticBatch,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, Any], torch.Tensor]:
    operator.eval()
    predictions = []
    for start in range(0, len(batch.context), batch_size):
        stop = start + batch_size
        output = operator(
            batch.context[start:stop].to(device),
            batch.action[start:stop].to(device),
            batch.reference[start:stop].to(device),
        )
        predictions.append(output.effect.cpu())
    prediction = torch.cat(predictions, dim=0)
    return response_metrics(batch.target_effect, prediction, batch.clean_effect), prediction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def structural_diagnostics(
    operator: ActionResponseOperator, batch: SyntheticBatch, device: torch.device
) -> dict[str, Any]:
    operator.eval()
    count = min(8, len(batch.context))
    context = batch.context[:count].to(device)
    action = batch.action[:count].to(device)
    reference = batch.reference[:count].to(device)
    identity = operator(context, reference, reference)
    normal = operator(context, action, reference)
    boundary = max(1, operator.config.horizon // 2)
    changed = action.clone()
    changed[:, boundary:] = (changed[:, boundary:] + 7.0).clamp(0.0, 100.0)
    changed_output = operator(context, changed, reference)
    changed_difference = (normal.effect - changed_output.effect).abs()
    leakage = changed_difference[:, :boundary].max()
    post_change_sensitivity = changed_difference[:, boundary:].max()
    positive_step = (reference + 5.0).clamp(0.0, 100.0)
    positive_step_output = operator(context, positive_step, reference)
    diagnostics = {
        "reference_identity_max_error": identity.effect.abs().max(),
        "future_action_leakage_max_error": leakage,
        "post_change_sensitivity_max_c": post_change_sensitivity,
        "positive_step_terminal_effect_max_c": positive_step_output.effect[:, -1].max(),
        "finite_effect": bool(torch.isfinite(normal.effect).all()),
        "finite_state": bool(torch.isfinite(normal.state_trajectory).all()),
        "capabilities": asdict(operator.capabilities),
        "operator": normal.diagnostics,
    }
    return _json_safe(diagnostics)


def _dataset(batch: SyntheticBatch) -> TensorDataset:
    return TensorDataset(batch.context, batch.action, batch.reference, batch.target_effect)


def train_synthetic_run(
    operator_config: OperatorConfig,
    training_config: TrainingConfig,
    synthetic_spec: SyntheticSpec,
    validation_samples: int,
    seed: int,
    route_id: str,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    repo_root: str | Path | None = None,
    overwrite: bool = False,
    protocol_version: str = "phase3.5-ms-v1",
) -> MultiStepTrainResult:
    operator_config.validate()
    training_config.validate()
    synthetic_spec.validate()
    out = Path(output_dir)
    checkpoint_path = out / "checkpoint_best_val.pt"
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite completed run: {checkpoint_path}")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    dev = torch.device(device)
    operator = build_response_operator(operator_config).to(dev)
    seeded_spec = replace(synthetic_spec, seed=synthetic_spec.seed + seed * 1_000_003)
    train_batch = generate_synthetic_split(seeded_spec, "train")
    validation_batch = generate_synthetic_split(replace(seeded_spec, samples=validation_samples), "validation")
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _dataset(train_batch),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(
        operator.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    history: list[dict[str, Any]] = []
    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    wait = 0
    started = time.time()
    for epoch in range(1, training_config.epochs + 1):
        operator.train()
        losses = []
        for context, action, reference, target in loader:
            context = context.to(dev)
            action = action.to(dev)
            reference = reference.to(dev)
            target = target.to(dev)
            optimizer.zero_grad(set_to_none=True)
            output = operator(context, action, reference)
            data_loss = F.smooth_l1_loss(output.effect, target, beta=0.2)
            physics_residual = output.diagnostics.get("physics_residual_mse")
            if physics_residual is None:
                physics_residual = torch.zeros((), dtype=data_loss.dtype, device=dev)
            loss = data_loss + training_config.physics_weight * physics_residual
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss for route={route_id} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(operator.parameters(), training_config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_metrics, _ = evaluate_operator(operator, validation_batch, dev)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_effect_mae": validation_metrics["effect_mae"],
        }
        history.append(record)
        score = validation_metrics["effect_mae"]
        if score < best_mae - 1e-8:
            best_mae = score
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(operator.state_dict())
            torch.save(
                {
                    "protocol_version": protocol_version,
                    "route_id": route_id,
                    "seed": seed,
                    "epoch": epoch,
                    "operator_config": operator_config.to_dict(),
                    "training_config": asdict(training_config),
                    "synthetic_spec": asdict(seeded_spec),
                    "model_state_dict": best_state,
                    "validation_metrics": validation_metrics,
                    "git_sha": _git_sha(repo_root),
                },
                checkpoint_path,
            )
        else:
            wait += 1
        _json_dump(out / "history.json", history)
        if wait >= training_config.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    operator.load_state_dict(best_state)
    validation_metrics, _ = evaluate_operator(operator, validation_batch, dev)
    validation_metrics["structural_diagnostics"] = structural_diagnostics(operator, validation_batch, dev)
    validation_metrics["truth"] = validation_batch.truth
    _json_dump(out / "metrics_validation.json", validation_metrics)
    manifest = {
        "protocol_version": protocol_version,
        "evidence_scope": "synthetic_method_feasibility_not_field_causality",
        "run_id": f"synthetic_{route_id}_s{seed}",
        "route_id": route_id,
        "seed": seed,
        "operator_config": operator_config.to_dict(),
        "training_config": asdict(training_config),
        "synthetic_spec": asdict(seeded_spec),
        "train_samples": len(train_batch.context),
        "validation_samples": len(validation_batch.context),
        "checkpoint_selector": "validation_effect_mae",
        "best_epoch": best_epoch,
        "git_sha": _git_sha(repo_root),
        "device": str(dev),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "test_accessed": False,
    }
    _json_dump(out / "manifest.json", manifest)
    return MultiStepTrainResult(out, checkpoint_path, best_epoch, validation_metrics, None)


def evaluate_synthetic_test_checkpoint(
    output_dir: str | Path,
    test_samples: int,
    device: str | torch.device = "cpu",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Open synthetic test once for an already frozen validation checkpoint."""

    out = Path(output_dir)
    checkpoint_path = out / "checkpoint_best_val.pt"
    manifest_path = out / "manifest.json"
    ledger_path = out / "synthetic_test_access_ledger.json"
    metrics_path = out / "metrics_test.json"
    if ledger_path.exists() or metrics_path.exists():
        raise RuntimeError(f"refusing repeat synthetic test access for {out}")
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"frozen checkpoint/manifest missing in {out}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("test_accessed") is not False:
        raise RuntimeError(f"manifest does not authorize first synthetic test access for {out}")
    current_sha = _git_sha(repo_root)
    if manifest.get("git_sha") != current_sha:
        raise RuntimeError(
            f"checkpoint git_sha={manifest.get('git_sha')} differs from current git_sha={current_sha}"
        )
    dev = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    expected_checkpoint_sha = manifest.get("checkpoint_sha256")
    if expected_checkpoint_sha is not None and _sha256(checkpoint_path) != expected_checkpoint_sha:
        raise RuntimeError(f"checkpoint sha256 mismatch for {out}")
    for key in ("protocol_version", "route_id", "seed", "git_sha"):
        if checkpoint.get(key) != manifest.get(key):
            raise RuntimeError(f"checkpoint/manifest mismatch for {key}")
    operator_config = OperatorConfig.from_mapping(checkpoint["operator_config"])
    operator = build_response_operator(operator_config).to(dev)
    operator.load_state_dict(checkpoint["model_state_dict"])
    synthetic_spec = SyntheticSpec(**checkpoint["synthetic_spec"])
    if test_samples < 1:
        raise ValueError("test_samples must be positive")
    ledger = {
        "protocol_version": manifest["protocol_version"],
        "status": "started",
        "evidence_scope": "synthetic_method_feasibility_not_field_causality",
        "route_id": manifest["route_id"],
        "seed": manifest["seed"],
        "git_sha": current_sha,
        "checkpoint": checkpoint_path.name,
        "checkpoint_selector": manifest["checkpoint_selector"],
        "test_samples": test_samples,
    }
    _json_dump(ledger_path, ledger)
    test_batch = generate_synthetic_split(replace(synthetic_spec, samples=test_samples), "test")
    metrics, _ = evaluate_operator(operator, test_batch, dev)
    metrics["structural_diagnostics"] = structural_diagnostics(operator, test_batch, dev)
    metrics["truth"] = test_batch.truth
    _json_dump(metrics_path, metrics)
    ledger["status"] = "completed"
    _json_dump(ledger_path, ledger)
    manifest["test_accessed"] = True
    manifest["test_access_note"] = "synthetic_known_truth_only"
    manifest["test_access_ledger"] = ledger_path.name
    _json_dump(manifest_path, manifest)
    return metrics
