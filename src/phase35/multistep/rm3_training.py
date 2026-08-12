"""Validation-only RM3 prediction training with scope-qualified selection."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ..data import Phase35Cache, deterministic_anchor_subset
from ..schema import Phase35ProtocolError
from .gatec_data import extract_gatec_batch, paired_valid_anchors
from .rm3_contracts import RM3PredictionRunSpec
from .rm3_prediction import RM3FairPredictionAdapter, RM3PredictionConfig


TARGET_KEYS_BY_SCOPE = {
    "terminal_only": ("terminal",),
    "valve_and_terminal": ("valve", "terminal"),
    "full_multitask": ("valve", "tin", "local", "terminal"),
}
PREDICTION_KEYS = {
    "valve": "valve_prediction",
    "tin": "tin_prediction",
    "local": "local_drop_prediction",
    "terminal": "terminal_prediction",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _bounds(n_rows: int, fraction: tuple[float, float]) -> tuple[int, int]:
    return int(n_rows * fraction[0]), int(n_rows * fraction[1])


def rm3_run_bounds(
    n_rows: int, spec: RM3PredictionRunSpec, *, actual_test_start: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    train = _bounds(n_rows, spec.train_fraction)
    raw_validation = _bounds(n_rows, spec.validation_fraction)
    validation = (raw_validation[0], min(raw_validation[1], int(actual_test_start)))
    if not 0 <= train[0] < train[1] <= validation[0]:
        raise Phase35ProtocolError("RM3 train/validation ordering is invalid")
    if not validation[0] < validation[1] <= actual_test_start:
        raise Phase35ProtocolError("RM3 validation touches the test lockbox")
    return train, validation


def _targets(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "valve": torch.as_tensor(batch.logged_future_valve, dtype=torch.float32, device=device),
        "tin": torch.as_tensor(batch.logged_future_tin, dtype=torch.float32, device=device),
        "local": torch.as_tensor(batch.local_drop_target, dtype=torch.float32, device=device),
        "terminal": torch.as_tensor(batch.terminal_target, dtype=torch.float32, device=device),
    }


def _inputs(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(batch.history, dtype=torch.float32, device=device),
        torch.as_tensor(batch.future_sp, dtype=torch.float32, device=device),
    )


def _forward(
    model: RM3FairPredictionAdapter,
    batch: Any,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    history, future_sp = _inputs(batch, device)
    targets = _targets(batch, device)
    kwargs = (
        {"logged_future_valve": targets["valve"]}
        if model.config.candidate_id == "P0_m7_oracle_valve"
        else {}
    )
    return model(history, future_sp, **kwargs), targets


def fit_rm3_train_statistics(batch: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    flat = batch.history.reshape(-1, batch.history.shape[-1]).astype(np.float64)
    center = np.median(flat, axis=0).astype(np.float32)
    scale = np.maximum(np.median(np.abs(flat - center), axis=0), 1e-3).astype(np.float32)
    raw = {
        "valve": batch.logged_future_valve,
        "tin": batch.logged_future_tin,
        "local": batch.local_drop_target,
        "terminal": batch.terminal_target,
    }
    target_scales = {}
    for key, value in raw.items():
        flattened = np.asarray(value, dtype=np.float64).reshape(-1)
        median = np.median(flattened)
        target_scales[key] = max(float(np.median(np.abs(flattened - median))), 1e-3)
    return center, scale, target_scales


def rm3_scope_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, torch.Tensor],
    *,
    output_scope: str,
    target_scales: Mapping[str, float],
    component_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    if output_scope not in TARGET_KEYS_BY_SCOPE:
        raise Phase35ProtocolError("RM3 output scope is invalid")
    keys = TARGET_KEYS_BY_SCOPE[output_scope]
    if component_weights is None:
        weights = {key: 1.0 / len(keys) for key in keys}
    else:
        if set(component_weights) != set(keys):
            raise Phase35ProtocolError("RM3 component loss weights do not match output scope")
        weights = {key: float(component_weights[key]) for key in keys}
        if any(value < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-12:
            raise Phase35ProtocolError("RM3 component loss weights must be nonnegative and sum to one")
    components = []
    for key in keys:
        prediction_key = PREDICTION_KEYS[key]
        if prediction_key not in output:
            raise Phase35ProtocolError(f"RM3 {output_scope} output lacks {prediction_key}")
        components.append(
            weights[key] * F.smooth_l1_loss(
                output[prediction_key] / float(target_scales[key]),
                targets[key] / float(target_scales[key]),
            )
        )
    return torch.stack(components).sum()


def _split_validation_anchors(
    anchors: np.ndarray, *, selector_count: int, reporting_count: int, fold_id: str
) -> tuple[np.ndarray, np.ndarray]:
    if len(anchors) < selector_count + reporting_count:
        raise Phase35ProtocolError("RM3 validation has insufficient disjoint anchors")
    rng = np.random.default_rng(35300 + int(fold_id[1:]))
    selected = rng.choice(anchors, size=selector_count + reporting_count, replace=False)
    return np.sort(selected[:selector_count]), np.sort(selected[selector_count:])


def _evaluate_selector(
    model: RM3FairPredictionAdapter,
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    *,
    window: int,
    horizon: int,
    batch_size: int,
    target_scales: Mapping[str, float],
    output_scope: str,
    device: torch.device,
    component_loss_weights: Mapping[str, float] | None = None,
) -> float:
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            batch = extract_gatec_batch(
                caches, anchors[start : start + batch_size], window=window, horizon=horizon,
                validate_pair=False,
            )
            output, targets = _forward(model, batch, device)
            total += float(
                rm3_scope_loss(
                    output, targets, output_scope=output_scope, target_scales=target_scales,
                    component_weights=component_loss_weights,
                ).cpu()
            ) * len(batch.anchors)
    model.train()
    return total / len(anchors)


def _report(
    model: RM3FairPredictionAdapter,
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    *,
    window: int,
    horizon: int,
    batch_size: int,
    target_scales: Mapping[str, float],
    output_scope: str,
    device: torch.device,
    component_loss_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sums = {key: 0.0 for key in TARGET_KEYS_BY_SCOPE[output_scope]}
    terminal_sum = 0.0
    arrays: dict[str, list[np.ndarray]] = {
        "terminal_target": [], "terminal_prediction": [], "future_sp": [],
        "logged_valve": [],
    }
    optional = {"valve": [], "tin": [], "local": []}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            batch = extract_gatec_batch(
                caches, anchors[start : start + batch_size], window=window, horizon=horizon,
                validate_pair=False,
            )
            output, targets = _forward(model, batch, device)
            terminal_sum += float((output["terminal_prediction"] - targets["terminal"]).abs().sum().cpu())
            for key in TARGET_KEYS_BY_SCOPE[output_scope]:
                sums[key] += float((output[PREDICTION_KEYS[key]] - targets[key]).abs().sum().cpu())
            arrays["terminal_target"].append(targets["terminal"].cpu().numpy().astype(np.float32))
            arrays["terminal_prediction"].append(output["terminal_prediction"].cpu().numpy().astype(np.float32))
            arrays["future_sp"].append(np.asarray(batch.future_sp, dtype=np.float32))
            arrays["logged_valve"].append(np.asarray(batch.logged_future_valve, dtype=np.float32))
            for key in optional:
                prediction_key = PREDICTION_KEYS[key]
                if prediction_key in output:
                    optional[key].append(output[prediction_key].cpu().numpy().astype(np.float32))
    elements = len(anchors) * horizon * 2
    metrics: dict[str, Any] = {
        "terminal_mae_c": terminal_sum / elements,
        "scope_selector_score": _evaluate_selector(
            model, caches, anchors, window=window, horizon=horizon, batch_size=batch_size,
            target_scales=target_scales, output_scope=output_scope, device=device,
            component_loss_weights=component_loss_weights,
        ),
        "output_scope": output_scope,
        "finite": True,
    }
    for key, value in sums.items():
        metrics[f"{key}_mae" + ("_c" if key != "valve" else "")] = value / elements
    episode_arrays = {
        "anchors": np.asarray(anchors, dtype=np.int64),
        "timestamps_ns": caches["A"].timestamps_ns[anchors].astype(np.int64),
        **{key: np.concatenate(parts) for key, parts in arrays.items()},
    }
    for key, parts in optional.items():
        if parts:
            episode_arrays[f"{key}_prediction"] = np.concatenate(parts)
    return metrics, episode_arrays


def run_rm3_prediction_training(
    caches: Mapping[str, Phase35Cache],
    matrix: Mapping[str, Any],
    spec: RM3PredictionRunSpec,
    *,
    device: str,
    output_dir: Path,
    provenance: Mapping[str, Any],
    model_candidate_id: str | None = None,
    model_d_model: int | None = None,
    model_latent_dim: int | None = None,
    component_loss_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run one immutable train/validation prediction job; never touches test."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"RM3 refuses existing non-empty run directory: {output_dir}")
    _set_seed(spec.seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise Phase35ProtocolError("RM3 requested unavailable CUDA")
    data, training = matrix["data_contract"], matrix["training"]
    window, horizon = int(data["window_steps"]), int(data["horizon_steps"])
    n_rows = len(caches["A"].timestamps_ns)
    train_bounds, validation_bounds = rm3_run_bounds(
        n_rows, spec, actual_test_start=caches["A"].split_bounds()["test"][0]
    )
    train_pool = paired_valid_anchors(
        caches, "train", window=window, horizon=horizon, max_age_s=float(data["max_age_s"]),
        bounds_override=train_bounds,
    )
    validation_pool = paired_valid_anchors(
        caches, "validation", window=window, horizon=horizon,
        max_age_s=float(data["max_age_s"]), bounds_override=validation_bounds,
    )
    fold_index = int(spec.fold_id[1:])
    stats_anchors = deterministic_anchor_subset(
        train_pool, int(training["stats_anchor_count"]), 35400 + fold_index
    )
    selector_anchors, reporting_anchors = _split_validation_anchors(
        validation_pool, selector_count=int(training["selector_anchor_count"]),
        reporting_count=int(training["reporting_anchor_count"]), fold_id=spec.fold_id,
    )
    stats_batch = extract_gatec_batch(
        caches, stats_anchors, window=window, horizon=horizon, validate_pair=False
    )
    center, scale, target_scales = fit_rm3_train_statistics(stats_batch)
    architecture_id = model_candidate_id or spec.candidate_id
    config = RM3PredictionConfig(
        candidate_id=architecture_id, window=window, horizon=horizon,
        n_features=stats_batch.history.shape[-1],
        d_model=int(model_d_model or matrix["model"]["d_model"]),
        latent_dim=int(model_latent_dim or matrix["model"]["latent_dim"]),
        dropout=float(matrix["model"]["dropout"]),
    )
    model = RM3FairPredictionAdapter(config, stats_batch.history_feature_names).to(torch_device)
    model.set_history_normalization(torch.from_numpy(center).to(torch_device), torch.from_numpy(scale).to(torch_device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(35500 + fold_index * 100 + spec.seed)
    best_score, best_update, best_state = math.inf, 0, None
    stale = 0
    selector_history: list[dict[str, float | int]] = []
    loss_curve: list[float] = []
    started = time.perf_counter()
    model.train()
    for update in range(1, int(training["optimizer_updates_cap"]) + 1):
        chosen = train_pool[rng.integers(0, len(train_pool), size=int(training["batch_size"]))]
        batch = extract_gatec_batch(caches, chosen, window=window, horizon=horizon, validate_pair=False)
        optimizer.zero_grad(set_to_none=True)
        output, targets = _forward(model, batch, torch_device)
        loss = rm3_scope_loss(
            output, targets, output_scope=spec.output_scope, target_scales=target_scales,
            component_weights=component_loss_weights,
        )
        if not torch.isfinite(loss):
            raise Phase35ProtocolError("RM3 training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        loss_curve.append(float(loss.detach().cpu()))
        if update % int(training["evaluation_interval_updates"]):
            continue
        score = _evaluate_selector(
            model, caches, selector_anchors, window=window, horizon=horizon,
            batch_size=int(training["evaluation_batch_size"]), target_scales=target_scales,
            output_scope=spec.output_scope, device=torch_device,
            component_loss_weights=component_loss_weights,
        )
        selector_history.append({"update": update, "scope_selector_score": score})
        if score < best_score - float(training["minimum_score_improvement"]):
            best_score, best_update = score, update
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if update >= int(training["minimum_updates"]) and stale >= int(training["early_stopping_patience_evaluations"]):
            break
    if best_state is None:
        raise Phase35ProtocolError("RM3 produced no validation checkpoint")
    model.load_state_dict(best_state)
    metrics, episodes = _report(
        model, caches, reporting_anchors, window=window, horizon=horizon,
        batch_size=int(training["evaluation_batch_size"]), target_scales=target_scales,
        output_scope=spec.output_scope, device=torch_device,
        component_loss_weights=component_loss_weights,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint_best_validation.pt"
    _atomic_checkpoint(checkpoint, {
        "protocol_version": matrix["protocol_version"], "run_spec": asdict(spec),
        "model_config": asdict(config), "feature_names": list(stats_batch.history_feature_names),
        "model_state_dict": best_state, "history_center": center, "history_scale": scale,
        "target_scales": target_scales, "best_update": best_update,
        "best_selector_score": best_score,
    })
    _atomic_npz(output_dir / "episodes_validation.npz", episodes)
    metrics_payload = {
        "run_id": spec.run_id, "candidate_id": spec.candidate_id,
        "output_scope": spec.output_scope, "metrics": metrics,
        "selector_history": selector_history, "loss_curve": loss_curve,
        "best_update": best_update, "best_selector_score": best_score,
        "optimizer_updates_completed": len(loss_curve), "test_accessed": False,
        "automatic_scientific_pass": None,
    }
    _atomic_json(output_dir / "metrics_validation.json", metrics_payload)
    manifest = {
        "protocol_version": matrix["protocol_version"], "run_id": spec.run_id,
        "run_spec": asdict(spec), "provenance": dict(provenance),
        "architecture_candidate_id": architecture_id,
        "component_loss_weights": (
            None if component_loss_weights is None else dict(component_loss_weights)
        ),
        "train_anchor_pool_count": len(train_pool), "stats_anchor_count": len(stats_anchors),
        "selector_anchor_count": len(selector_anchors), "reporting_anchor_count": len(reporting_anchors),
        "selector_anchor_sha256": hashlib.sha256(np.asarray(selector_anchors, dtype="<i8").tobytes()).hexdigest(),
        "reporting_anchor_sha256": hashlib.sha256(np.asarray(reporting_anchors, dtype="<i8").tobytes()).hexdigest(),
        "selector_reporting_disjoint": bool(not np.intersect1d(selector_anchors, reporting_anchors).size),
        "checkpoint_selector": (
            f"validation_{spec.output_scope}_normalized_loss"
            if component_loss_weights is None
            else f"validation_{spec.output_scope}_declared_component_weighted_loss"
        ),
        "checkpoint_sha256": _sha256(checkpoint), "test_accessed": False,
        "automatic_scientific_pass": None, "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    artifact_names = ("manifest.json", "checkpoint_best_validation.pt", "metrics_validation.json", "episodes_validation.npz")
    _atomic_json(output_dir / "artifact_ledger.json", {name: _sha256(output_dir / name) for name in artifact_names})
    return {"run_id": spec.run_id, "status": "complete", "metrics": metrics_payload}
