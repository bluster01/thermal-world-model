"""Validation-only training loop and forecast evaluation."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    Phase35Cache,
    deterministic_anchor_subset,
    extract_windows,
    valid_window_anchors,
)
from .evaluation import forecast_metrics
from .model import A1PhysValveWM, assert_constant_valve_identity
from .schema import ExperimentConfig, TARGET_COLUMN, VALVE_COLUMN


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_sha(repo_root: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_model(config: ExperimentConfig, cache: Phase35Cache, device: torch.device) -> tuple[A1PhysValveWM, list[str]]:
    features = list(config.history_features)
    target_index = features.index(TARGET_COLUMN)
    model = A1PhysValveWM(config, n_features=len(features), target_index=target_index).to(device)
    return model, features


def loss_value(output: dict[str, torch.Tensor], target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mae":
        return torch.mean(torch.abs(output["mu"] - target))
    if loss_name == "huber":
        return F.smooth_l1_loss(output["mu"], target, beta=1.0)
    variance = output["sigma"].square().clamp_min(1e-4)
    return torch.mean(0.5 * (torch.log(variance) + (target - output["mu"]).square() / variance))


@torch.no_grad()
def evaluate_forecast(
    model: A1PhysValveWM,
    cache: Phase35Cache,
    anchors: np.ndarray,
    feature_columns: list[str],
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    model.eval()
    predictions, targets = [], []
    for start in range(0, len(anchors), batch_size):
        batch = extract_windows(
            cache,
            anchors[start:start + batch_size],
            feature_columns,
            TARGET_COLUMN,
            VALVE_COLUMN,
            model.config.window,
            model.config.horizon,
        )
        history = torch.from_numpy(batch["history"]).to(device)
        future_valve = torch.from_numpy(batch["future_valve"]).to(device)
        baseline = torch.from_numpy(batch["baseline_valve"]).to(device)
        output = model(history, future_valve, baseline)
        predictions.append(output["mu"].cpu().numpy())
        targets.append(batch["target"])
    if not predictions:
        raise RuntimeError("forecast evaluation received no valid anchors")
    return forecast_metrics(np.concatenate(targets), np.concatenate(predictions))


@dataclass
class TrainResult:
    output_dir: Path
    checkpoint: Path
    best_epoch: int
    validation_metrics: dict


def train_one(
    cache: Phase35Cache,
    config: ExperimentConfig,
    side: str,
    seed: int,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
    repo_root: str | Path | None = None,
) -> TrainResult:
    config.validate()
    if side not in {"A", "B"}:
        raise ValueError("side must be A or B")
    cached_side = str(cache.metadata.get("side", side))
    if cached_side != side:
        raise ValueError(f"cache side={cached_side!r} does not match requested side={side!r}")
    out = Path(output_dir)
    checkpoint_path = out / "checkpoint_best_val.pt"
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite completed run: {checkpoint_path}")
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    dev = torch.device(device)
    model, features = build_model(config, cache, dev)
    train_anchors = valid_window_anchors(
        cache, "train", features, TARGET_COLUMN, VALVE_COLUMN, config.window, config.horizon
    )
    val_anchors = valid_window_anchors(
        cache, "validation", features, TARGET_COLUMN, VALVE_COLUMN, config.window, config.horizon
    )
    train_anchors = deterministic_anchor_subset(train_anchors, config.max_train_anchors, seed)
    val_anchors = deterministic_anchor_subset(val_anchors, config.max_eval_anchors, 10_000 + seed)
    if len(train_anchors) < config.batch_size or len(val_anchors) == 0:
        raise RuntimeError(
            f"insufficient anchors: train={len(train_anchors)} validation={len(val_anchors)} "
            f"batch_size={config.batch_size}"
        )
    probe = extract_windows(
        cache, val_anchors[: min(4, len(val_anchors))], features, TARGET_COLUMN, VALVE_COLUMN,
        config.window, config.horizon,
    )
    assert_constant_valve_identity(
        model,
        torch.from_numpy(probe["history"]).to(dev),
        torch.from_numpy(probe["baseline_valve"]).to(dev),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed)
    history_log: list[dict] = []
    best_mae, best_epoch, wait = float("inf"), 0, 0
    started = time.time()
    free_parameters = list(model.free_head.parameters()) if model.free_head is not None else []
    if config.freeze_free_epochs:
        for parameter in free_parameters:
            parameter.requires_grad = False
    for epoch in range(1, config.epochs + 1):
        if config.freeze_free_epochs and epoch == config.freeze_free_epochs + 1:
            for parameter in free_parameters:
                parameter.requires_grad = True
        model.train()
        losses = []
        for _ in range(config.steps_per_epoch):
            chosen = rng.choice(train_anchors, size=config.batch_size, replace=False)
            batch = extract_windows(
                cache, chosen, features, TARGET_COLUMN, VALVE_COLUMN, config.window, config.horizon
            )
            history = torch.from_numpy(batch["history"]).to(dev)
            future_valve = torch.from_numpy(batch["future_valve"]).to(dev)
            baseline = torch.from_numpy(batch["baseline_valve"]).to(dev)
            target = torch.from_numpy(batch["target"]).to(dev)
            optimizer.zero_grad(set_to_none=True)
            output = model(history, future_valve, baseline)
            loss = loss_value(output, target, config.loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_forecast(model, cache, val_anchors, features, dev)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_integrated_mae": val_metrics["integrated_mae"],
        }
        history_log.append(record)
        score = val_metrics["integrated_mae"]
        if score < best_mae - config.min_delta:
            best_mae, best_epoch, wait = score, epoch, 0
            torch.save({
                "protocol_version": "phase3.5-v1",
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "feature_columns": features,
                "target_column": TARGET_COLUMN,
                "valve_column": VALVE_COLUMN,
                "side": side,
                "seed": seed,
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "git_sha": git_sha(repo_root),
            }, checkpoint_path)
        else:
            wait += 1
        _json_dump(out / "history.json", history_log)
        if wait >= config.patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val = evaluate_forecast(model, cache, val_anchors, features, dev)
    manifest = {
        "protocol_version": "phase3.5-v1",
        "run_id": f"{side}_{config.config_id}_s{seed}",
        "side": side,
        "seed": seed,
        "config": config.to_dict(),
        "cache_metadata": cache.metadata,
        "git_sha": git_sha(repo_root),
        "device": str(dev),
        "torch_version": torch.__version__,
        "train_anchor_count": int(len(train_anchors)),
        "validation_anchor_count": int(len(val_anchors)),
        "best_epoch": best_epoch,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "test_accessed": False,
        "checkpoint_selector": "validation_integrated_mae",
    }
    _json_dump(out / "manifest.json", _finite_json(manifest))
    _json_dump(out / "metrics_validation.json", _finite_json(final_val))
    return TrainResult(out, checkpoint_path, best_epoch, final_val)
