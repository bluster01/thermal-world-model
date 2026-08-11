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
import torch.nn.functional as F

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

COMMON_PREDICTION_WEIGHTS = {
    key: value / (1.0 - DEFAULT_WEIGHTS["structure"])
    for key, value in DEFAULT_WEIGHTS.items()
    if key != "structure"
} | {"structure": 0.0}


@dataclass(frozen=True)
class GateCRealSmokeConfig:
    route: str
    candidate_id: str | None = None
    residual_capacity: str = "base"
    response_scheduling: str = "scheduled"
    local_supervision: bool = True
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
        if self.route not in RESPONSE_ROUTES:
            raise Phase35ProtocolError("Gate C real-smoke route is invalid")
        if self.residual_capacity not in {"small", "base", "large"}:
            raise Phase35ProtocolError("Gate C real-smoke residual capacity is invalid")
        if self.response_scheduling not in {"none", "additive", "scheduled"}:
            raise Phase35ProtocolError("Gate C real-smoke response scheduling is invalid")
        if self.route == "none" and self.response_scheduling != "none":
            raise Phase35ProtocolError("Gate C paired-free candidate cannot schedule a response")
        if self.route != "none" and self.response_scheduling == "none":
            raise Phase35ProtocolError("Gate C response candidate requires additive/scheduled mode")
        if not isinstance(self.local_supervision, bool):
            raise Phase35ProtocolError("Gate C real-smoke local supervision must be boolean")
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
        "logged_aux_local_abs": 0.0,
        "shuffled_aux_local_abs": 0.0,
        "composite_loss": 0.0,
        "candidate_objective_loss": 0.0,
        "persistence_valve_abs": 0.0,
        "persistence_tin_abs": 0.0,
        "persistence_local_abs": 0.0,
        "persistence_terminal_abs": 0.0,
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
        residual_capacity=config.residual_capacity,
        response_scheduling=config.response_scheduling,
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
    use_logged_action_aux = config.local_supervision and config.route != "none"
    training_weights = dict(DEFAULT_WEIGHTS)
    if not config.local_supervision:
        training_weights["local"] = 0.0
    if not use_logged_action_aux:
        training_weights["structure"] = 0.0
    active_weight = sum(training_weights.values())
    training_weights = {
        key: value / active_weight for key, value in training_weights.items()
    }
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
            tensors["history"],
            tensors["future_sp"],
            boundary_mode="forecast_boundary",
            logged_future_valve_for_aux=tensors["valve"] if use_logged_action_aux else None,
        )
        if use_logged_action_aux:
            output["structure_penalty"] = F.smooth_l1_loss(
                output["logged_local_drop_prediction"] / scales.values["local"],
                tensors["local"] / scales.values["local"],
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
            training_weights,
            local_supervision=config.local_supervision,
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
    shuffle_rng = np.random.default_rng(config.seed + 404)
    shuffled_rows = shuffle_rng.permutation(len(validation_anchors))
    with torch.no_grad():
        for start in range(0, len(validation_anchors), config.validation_batch_size):
            stop = min(len(validation_anchors), start + config.validation_batch_size)
            indices = np.arange(start, stop, dtype=np.int64)
            tensors = _to_device_batch(validation_batch, indices, torch_device)
            shuffled_valve = torch.as_tensor(
                validation_batch.logged_future_valve[shuffled_rows[indices]],
                dtype=torch.float32,
                device=torch_device,
            )
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
            baseline_tin = tensors["history"][:, -1, model.tin_indices]
            baseline_tout = tensors["history"][:, -1, model.tout_indices]
            baseline_local = baseline_tin - baseline_tout
            baseline_terminal = tensors["history"][:, -1, model.terminal_indices]
            logged_response = model.local_response(
                context, tensors["valve"], baseline_valve
            )
            shuffled_response = model.local_response(context, shuffled_valve, baseline_valve)
            logged_aux_local = forecast["residual_local_prediction"] + logged_response["effect"]
            shuffled_aux_local = (
                forecast["residual_local_prediction"] + shuffled_response["effect"]
            )
            common_prediction_loss = compute_gatec_loss(
                forecast,
                {
                    "valve": tensors["valve"],
                    "tin": tensors["tin"],
                    "local": tensors["local"],
                    "terminal": tensors["terminal"],
                },
                scales,
                COMMON_PREDICTION_WEIGHTS,
                local_supervision=True,
            )
            candidate_objective_loss = compute_gatec_loss(
                forecast,
                {
                    "valve": tensors["valve"],
                    "tin": tensors["tin"],
                    "local": tensors["local"],
                    "terminal": tensors["terminal"],
                },
                scales,
                training_weights,
                local_supervision=config.local_supervision,
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
            sums["logged_aux_local_abs"] += float(
                (logged_aux_local - tensors["local"]).abs().sum().cpu()
            )
            sums["shuffled_aux_local_abs"] += float(
                (shuffled_aux_local - tensors["local"]).abs().sum().cpu()
            )
            sums["composite_loss"] += float(common_prediction_loss.total.cpu())
            sums["candidate_objective_loss"] += float(
                candidate_objective_loss.total.cpu()
            )
            sums["persistence_valve_abs"] += float(
                (tensors["valve"] - baseline_valve[:, None]).abs().sum().cpu()
            )
            sums["persistence_tin_abs"] += float(
                (tensors["tin"] - baseline_tin[:, None]).abs().sum().cpu()
            )
            sums["persistence_local_abs"] += float(
                (tensors["local"] - baseline_local[:, None]).abs().sum().cpu()
            )
            sums["persistence_terminal_abs"] += float(
                (tensors["terminal"] - baseline_terminal[:, None]).abs().sum().cpu()
            )
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
        "logged_aux_local_mae_c": sums["logged_aux_local_abs"] / element_count,
        "shuffled_aux_local_mae_c": sums["shuffled_aux_local_abs"] / element_count,
        "dimensionless_composite_loss": sums["composite_loss"] / batch_count,
        "candidate_objective_validation_loss": (
            sums["candidate_objective_loss"] / batch_count
        ),
        "persistence_valve_mae": sums["persistence_valve_abs"] / element_count,
        "persistence_tin_mae_c": sums["persistence_tin_abs"] / element_count,
        "persistence_local_drop_mae_c": sums["persistence_local_abs"] / element_count,
        "persistence_terminal_mae_c": sums["persistence_terminal_abs"] / element_count,
        "stable_pole_max": stable_pole_max,
        "finite": bool(all(math.isfinite(value) for value in sums.values())),
    }
    metrics["valve_to_persistence_ratio"] = (
        metrics["forecast_valve_mae"] / max(metrics["persistence_valve_mae"], 1e-9)
    )
    metrics["tin_to_persistence_ratio"] = (
        metrics["forecast_tin_mae_c"] / max(metrics["persistence_tin_mae_c"], 1e-9)
    )
    metrics["local_to_persistence_ratio"] = (
        metrics["forecast_local_drop_mae_c"]
        / max(metrics["persistence_local_drop_mae_c"], 1e-9)
    )
    metrics["terminal_to_persistence_ratio"] = (
        metrics["forecast_terminal_mae_c"]
        / max(metrics["persistence_terminal_mae_c"], 1e-9)
    )
    metrics["logged_effect_to_local_delta_ratio"] = (
        metrics["logged_action_effect_mean_abs_c"]
        / max(metrics["persistence_local_drop_mae_c"], 1e-9)
    )
    metrics["logged_vs_shuffled_local_advantage_c"] = (
        metrics["shuffled_aux_local_mae_c"] - metrics["logged_aux_local_mae_c"]
    )
    response_expected = config.route != "none"
    structural = {
        "finite_rollout": metrics["finite"] and stable_pole_max < 1.0,
        "sp_prefix_causality": prefix_ok,
        "constant_action_identity": identity_ok,
        "future_truth_isolation": True,
        "local_response_noncollapse": (
            metrics["logged_action_effect_mean_abs_c"] > 1e-6
            if response_expected
            else True
        ),
    }
    baseline_diagnostics = {
        "valve_not_worse_than_1p05_persistence": metrics["valve_to_persistence_ratio"] <= 1.05,
        "tin_not_worse_than_1p05_persistence": metrics["tin_to_persistence_ratio"] <= 1.05,
        "local_not_worse_than_1p05_persistence": metrics["local_to_persistence_ratio"] <= 1.05,
        "terminal_not_worse_than_1p05_persistence": metrics["terminal_to_persistence_ratio"] <= 1.05,
    }
    elapsed = time.perf_counter() - started
    return {
        "scope": "local_real_train_validation_subset_not_causal",
        "candidate_id": config.candidate_id or config.route,
        "route": config.route,
        "residual_capacity": config.residual_capacity,
        "response_scheduling": config.response_scheduling,
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
        "local_supervision": config.local_supervision,
        "logged_action_auxiliary_used_for_training": (
            config.local_supervision and response_expected
        ),
        "training_weights": training_weights,
        "common_prediction_weights": COMMON_PREDICTION_WEIGHTS,
        "score_comparability": {
            "dimensionless_composite_loss": "shared_across_all_candidates",
            "candidate_objective_validation_loss": "not_comparable_when_supervision_differs",
        },
        "structural_applicability": {
            "local_response_noncollapse": "required" if response_expected else "not_applicable_paired_free"
        },
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "optimizer_updates": config.optimizer_updates,
        "loss_curve": loss_curve,
        "metrics_validation": metrics,
        "structural_validation": structural,
        "baseline_diagnostics": baseline_diagnostics,
        "selector_eligible": all(structural.values()),
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(torch_device))
        if torch_device.type == "cuda"
        else 0,
        "test_accessed": False,
        "automatic_scientific_pass": None,
    }
