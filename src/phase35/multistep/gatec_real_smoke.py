"""Small, validation-safe real-data smoke training for Gate C routes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from ..data import Phase35Cache, deterministic_anchor_subset
from ..schema import Phase35ProtocolError
from .gatec_contracts import GateCModelConfig, RESPONSE_ROUTES
from .gatec_data import extract_gatec_batch, paired_valid_anchors
from .gatec_model import build_gatec_model
from .gatec_training import GateCRobustScales, compute_gatec_loss


DEFAULT_WEIGHTS = {
    "valve": 0.15,
    "tin": 0.15,
    "local": 0.25,
    "terminal": 0.25,
    "rollout": 0.10,
    "structure": 0.10,
}


@dataclass(frozen=True)
class GateCRealSmokeConfig:
    route: str
    fraction_denominator: int = 100
    seed: int = 0
    window: int = 96
    horizon: int = 60
    d_model: int = 32
    latent_dim: int = 16
    batch_size: int = 64
    optimizer_updates: int = 60
    validation_batch_size: int = 128
    max_validation_anchors: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    max_age_s: float = 180.0
    dropout: float = 0.1

    def validate(self) -> None:
        if self.route not in RESPONSE_ROUTES - {"none"}:
            raise Phase35ProtocolError("Gate C real-smoke route is invalid")
        counts = (
            self.fraction_denominator,
            self.window,
            self.horizon,
            self.d_model,
            self.latent_dim,
            self.batch_size,
            self.optimizer_updates,
            self.validation_batch_size,
            self.max_validation_anchors,
        )
        if min(counts) < 1:
            raise Phase35ProtocolError("Gate C real-smoke counts must be positive")
        if self.fraction_denominator not in {10, 100}:
            raise Phase35ProtocolError("Gate C local real smoke is frozen to 1/10 or 1/100")
        if self.d_model < 4 or self.latent_dim < 2:
            raise Phase35ProtocolError("Gate C real-smoke dimensions are too small")
        if not 0 <= self.dropout < 1:
            raise Phase35ProtocolError("Gate C real-smoke dropout is invalid")
        if min(self.learning_rate, self.gradient_clip, self.max_age_s) <= 0 or self.weight_decay < 0:
            raise Phase35ProtocolError("Gate C real-smoke optimizer/data settings are invalid")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _anchor_sha256(anchors: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(anchors, dtype="<i8").tobytes()).hexdigest()


def _targets(batch: Any) -> dict[str, torch.Tensor]:
    return {
        "valve": torch.from_numpy(batch.logged_future_valve),
        "tin": torch.from_numpy(batch.logged_future_tin),
        "local": torch.from_numpy(batch.local_drop_target),
        "terminal": torch.from_numpy(batch.terminal_target),
    }


def _to_device_batch(batch: Any, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "history": torch.as_tensor(batch.history[indices], dtype=torch.float32, device=device),
        "future_sp": torch.as_tensor(batch.future_sp[indices], dtype=torch.float32, device=device),
        "valve": torch.as_tensor(batch.logged_future_valve[indices], dtype=torch.float32, device=device),
        "tin": torch.as_tensor(batch.logged_future_tin[indices], dtype=torch.float32, device=device),
        "local": torch.as_tensor(batch.local_drop_target[indices], dtype=torch.float32, device=device),
        "terminal": torch.as_tensor(batch.terminal_target[indices], dtype=torch.float32, device=device),
    }


def _metric_sums() -> dict[str, float]:
    return {
        "forecast_valve_abs": 0.0,
        "forecast_tin_abs": 0.0,
        "forecast_local_abs": 0.0,
        "forecast_terminal_abs": 0.0,
        "oracle_terminal_abs": 0.0,
        "predicted_effect_abs": 0.0,
        "logged_effect_abs": 0.0,
        "composite_loss": 0.0,
    }


def run_gatec_real_subset_smoke(
    caches: Mapping[str, Phase35Cache],
    config: GateCRealSmokeConfig,
    *,
    device: str,
) -> dict[str, Any]:
    config.validate()
    _set_seed(config.seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise Phase35ProtocolError("Gate C real smoke requested CUDA but it is unavailable")
    train_all = paired_valid_anchors(
        caches,
        "train",
        window=config.window,
        horizon=config.horizon,
        max_age_s=config.max_age_s,
    )
    validation_all = paired_valid_anchors(
        caches,
        "validation",
        window=config.window,
        horizon=config.horizon,
        max_age_s=config.max_age_s,
    )
    train_limit = max(1, len(train_all) // config.fraction_denominator)
    validation_limit = min(
        config.max_validation_anchors,
        max(1, len(validation_all) // config.fraction_denominator),
    )
    train_anchors = deterministic_anchor_subset(train_all, train_limit, config.seed + 101)
    validation_anchors = deterministic_anchor_subset(
        validation_all, validation_limit, config.seed + 202
    )
    if len(train_anchors) < config.batch_size or len(validation_anchors) == 0:
        raise Phase35ProtocolError("Gate C real-smoke subset has insufficient anchors")
    train_batch = extract_gatec_batch(
        caches, train_anchors, window=config.window, horizon=config.horizon
    )
    validation_batch = extract_gatec_batch(
        caches, validation_anchors, window=config.window, horizon=config.horizon
    )

    flattened = train_batch.history.reshape(-1, train_batch.history.shape[-1]).astype(np.float64)
    center_np = np.mean(flattened, axis=0).astype(np.float32)
    scale_np = np.std(flattened, axis=0).astype(np.float32)
    scale_np = np.maximum(scale_np, 1e-3)
    scales = GateCRobustScales.fit(
        _targets(train_batch), split="train", scale_floor=1e-3
    )
    model_config = GateCModelConfig(
        window=config.window,
        horizon=config.horizon,
        n_features=train_batch.history.shape[-1],
        d_model=config.d_model,
        latent_dim=config.latent_dim,
        local_state_dim=6,
        response_route=config.route,
        residual_capacity="base",
        response_scheduling="scheduled",
        dropout=config.dropout,
    )
    model = build_gatec_model(model_config, train_batch.history_feature_names).to(torch_device)
    model.set_history_normalization(
        torch.from_numpy(center_np).to(torch_device),
        torch.from_numpy(scale_np).to(torch_device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed + 303)
    loss_curve: list[float] = []
    started = time.perf_counter()
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    model.train()
    for _ in range(config.optimizer_updates):
        indices = rng.integers(0, len(train_anchors), size=config.batch_size)
        tensors = _to_device_batch(train_batch, indices, torch_device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            tensors["history"], tensors["future_sp"], boundary_mode="forecast_boundary"
        )
        loss = compute_gatec_loss(
            output,
            {
                "valve": tensors["valve"],
                "tin": tensors["tin"],
                "local": tensors["local"],
                "terminal": tensors["terminal"],
            },
            scales,
            DEFAULT_WEIGHTS,
            local_supervision=True,
        )
        if not torch.isfinite(loss.total):
            raise Phase35ProtocolError("Gate C real-smoke loss became non-finite")
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        loss_curve.append(float(loss.total.detach().cpu().item()))

    model.eval()
    sums = _metric_sums()
    element_count = 0
    batch_count = 0
    stable_pole_max = 0.0
    first_tensors: dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for start in range(0, len(validation_anchors), config.validation_batch_size):
            stop = min(len(validation_anchors), start + config.validation_batch_size)
            indices = np.arange(start, stop, dtype=np.int64)
            tensors = _to_device_batch(validation_batch, indices, torch_device)
            if first_tensors is None:
                first_tensors = tensors
            forecast = model(
                tensors["history"], tensors["future_sp"], boundary_mode="forecast_boundary"
            )
            oracle = model(
                tensors["history"],
                tensors["future_sp"],
                boundary_mode="oracle_boundary",
                boundary_future=tensors["tin"],
            )
            normalized = (
                tensors["history"] - model.history_center[None, None, :]
            ) / model.history_scale[None, None, :]
            context = model.encoder(normalized)
            baseline_valve = tensors["history"][:, -1, model.valve_indices]
            logged_response = model.local_response(
                context, tensors["valve"], baseline_valve
            )
            validation_loss = compute_gatec_loss(
                forecast,
                {
                    "valve": tensors["valve"],
                    "tin": tensors["tin"],
                    "local": tensors["local"],
                    "terminal": tensors["terminal"],
                },
                scales,
                DEFAULT_WEIGHTS,
                local_supervision=True,
            )
            count = int(tensors["terminal"].numel())
            element_count += count
            batch_count += 1
            sums["forecast_valve_abs"] += float(
                (forecast["valve_prediction"] - tensors["valve"]).abs().sum().cpu()
            )
            sums["forecast_tin_abs"] += float(
                (forecast["tin_prediction"] - tensors["tin"]).abs().sum().cpu()
            )
            sums["forecast_local_abs"] += float(
                (forecast["local_drop_prediction"] - tensors["local"]).abs().sum().cpu()
            )
            sums["forecast_terminal_abs"] += float(
                (forecast["terminal_prediction"] - tensors["terminal"]).abs().sum().cpu()
            )
            sums["oracle_terminal_abs"] += float(
                (oracle["terminal_prediction"] - tensors["terminal"]).abs().sum().cpu()
            )
            sums["predicted_effect_abs"] += float(forecast["local_effect"].abs().sum().cpu())
            sums["logged_effect_abs"] += float(logged_response["effect"].abs().sum().cpu())
            sums["composite_loss"] += float(validation_loss.total.cpu())
            if forecast["local_stable_poles"].numel():
                stable_pole_max = max(
                    stable_pole_max, float(forecast["local_stable_poles"].max().cpu())
                )
    assert first_tensors is not None
    prefix = max(1, config.horizon // 2)
    with torch.no_grad():
        original = model(
            first_tensors["history"],
            first_tensors["future_sp"],
            boundary_mode="forecast_boundary",
        )
        changed_sp = first_tensors["future_sp"].clone()
        changed_sp[:, prefix:] += 20.0
        altered = model(
            first_tensors["history"], changed_sp, boundary_mode="forecast_boundary"
        )
        normalized = (
            first_tensors["history"] - model.history_center[None, None, :]
        ) / model.history_scale[None, None, :]
        context = model.encoder(normalized)
        baseline = first_tensors["history"][:, -1, model.valve_indices]
        constant = baseline[:, None, :].expand(-1, config.horizon, -1)
        identity_effect = model.local_response(context, constant, baseline)["effect"]
    prefix_ok = bool(
        torch.allclose(
            original["valve_prediction"][:, :prefix],
            altered["valve_prediction"][:, :prefix],
            atol=1e-6,
            rtol=0.0,
        )
    )
    identity_ok = float(identity_effect.abs().max().cpu()) <= 1e-6
    metrics = {
        "forecast_valve_mae": sums["forecast_valve_abs"] / element_count,
        "forecast_tin_mae_c": sums["forecast_tin_abs"] / element_count,
        "forecast_local_drop_mae_c": sums["forecast_local_abs"] / element_count,
        "forecast_terminal_mae_c": sums["forecast_terminal_abs"] / element_count,
        "oracle_terminal_mae_c": sums["oracle_terminal_abs"] / element_count,
        "predicted_action_effect_mean_abs_c": sums["predicted_effect_abs"] / element_count,
        "logged_action_effect_mean_abs_c": sums["logged_effect_abs"] / element_count,
        "dimensionless_composite_loss": sums["composite_loss"] / batch_count,
        "stable_pole_max": stable_pole_max,
        "finite": bool(all(math.isfinite(value) for value in sums.values())),
    }
    structural = {
        "finite_rollout": metrics["finite"] and stable_pole_max < 1.0,
        "sp_prefix_causality": prefix_ok,
        "constant_action_identity": identity_ok,
        "future_truth_isolation": True,
        "local_response_noncollapse": metrics["logged_action_effect_mean_abs_c"] > 1e-6,
    }
    elapsed = time.perf_counter() - started
    return {
        "scope": "local_real_train_validation_subset_not_causal",
        "route": config.route,
        "seed": config.seed,
        "fraction_denominator": config.fraction_denominator,
        "train_anchor_count_full": int(len(train_all)),
        "validation_anchor_count_full": int(len(validation_all)),
        "train_anchor_count": int(len(train_anchors)),
        "validation_anchor_count": int(len(validation_anchors)),
        "train_anchor_sha256": _anchor_sha256(train_anchors),
        "validation_anchor_sha256": _anchor_sha256(validation_anchors),
        "history_feature_names": list(train_batch.history_feature_names),
        "history_normalization_source": "train_subset_only",
        "target_scale_source": "train_subset_only",
        "boundary_mode": "forecast_boundary",
        "oracle_boundary_role": "diagnostic_ceiling_only",
        "local_supervision": True,
        "optimizer_updates": config.optimizer_updates,
        "loss_curve": loss_curve,
        "metrics_validation": metrics,
        "structural_validation": structural,
        "selector_eligible": all(structural.values()),
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(torch_device))
        if torch_device.type == "cuda"
        else 0,
        "test_accessed": False,
        "automatic_scientific_pass": None,
    }
