"""Validation-only A/B adaptation of the MS5-selected additive model."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import (
    Phase35Cache,
    deterministic_anchor_subset,
    extract_windows,
    valid_window_anchors,
)
from ..model import HistoryEncoder
from ..schema import (
    LOAD_COLUMN,
    PRESSURE_COLUMN,
    SP_COLUMN,
    TARGET_COLUMN,
    VALVE_COLUMN,
)
from .contracts import OperatorConfig
from .full_training import ZeroResponseOperator
from .operators import build_response_operator
from .training import _json_dump, _sha256


REAL_MODES = {"joint_total", "free_only"}


@dataclass(frozen=True)
class RealModelConfig:
    window: int = 96
    horizon: int = 60
    d_model: int = 32
    n_heads: int = 4
    dropout: float = 0.10
    opening_map: str = "monotone"
    poles: int = 3
    tau_min_seconds: float = 20.0
    tau_max_seconds: float = 900.0
    context_scheduled: bool = True
    schedule_log_scale: float = 0.5

    def validate(self) -> None:
        if self.window < 2 or self.horizon < 2 or self.d_model < 4:
            raise ValueError("MS3 model dimensions are outside supported ranges")
        if self.d_model % self.n_heads:
            raise ValueError("MS3 d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("MS3 dropout must be in [0,1)")
        self.operator_config(self.d_model * 2).validate()

    def operator_config(self, context_dim: int) -> OperatorConfig:
        return OperatorConfig(
            route="graybox",
            horizon=self.horizon,
            context_dim=context_dim,
            dt_seconds=10.0,
            opening_map=self.opening_map,
            poles=self.poles,
            latent_dim=4,
            hidden_dim=32,
            tau_min_seconds=self.tau_min_seconds,
            tau_max_seconds=self.tau_max_seconds,
            ode_substeps=2,
            closure_scale=0.02,
            context_scheduled=self.context_scheduled,
            schedule_log_scale=self.schedule_log_scale,
            delay_mode="none",
            fixed_delay_steps=0,
            max_delay_steps=0,
        )


@dataclass(frozen=True)
class RealTrainingConfig:
    batch_size: int = 128
    epochs: int = 40
    patience: int = 8
    steps_per_epoch: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    min_delta: float = 1e-4
    max_train_anchors: int = 100_000
    max_selector_anchors: int = 2_048
    max_validation_anchors: int = 8_192
    max_age_seconds: float = 180.0
    dynamic_dose_threshold_pct: float = 1.0
    min_operating_load_mw: float = 250.0
    min_operating_pressure_mpa: float = 10.0
    target_temperature_min_c: float = 500.0
    target_temperature_max_c: float = 600.0
    sp_temperature_min_c: float = 500.0
    sp_temperature_max_c: float = 600.0
    valve_min_pct: float = -2.0
    valve_max_pct: float = 102.0

    def validate(self) -> None:
        counts = (
            self.batch_size,
            self.epochs,
            self.patience,
            self.steps_per_epoch,
            self.max_train_anchors,
            self.max_selector_anchors,
            self.max_validation_anchors,
        )
        if min(counts) < 1:
            raise ValueError("MS3 training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("MS3 optimizer settings are invalid")
        if self.gradient_clip <= 0 or self.min_delta < 0:
            raise ValueError("MS3 clipping/min_delta settings are invalid")
        if self.max_age_seconds <= 0 or self.dynamic_dose_threshold_pct <= 0:
            raise ValueError("MS3 data thresholds must be positive")
        if self.min_operating_load_mw <= 0 or self.min_operating_pressure_mpa <= 0:
            raise ValueError("MS3 operating thresholds must be positive")
        if not self.target_temperature_min_c < self.target_temperature_max_c:
            raise ValueError("MS3 target-temperature range is invalid")
        if not self.sp_temperature_min_c < self.sp_temperature_max_c:
            raise ValueError("MS3 SP-temperature range is invalid")
        if not self.valve_min_pct < self.valve_max_pct:
            raise ValueError("MS3 valve range is invalid")


class RealA1PhysMultiStep(nn.Module):
    """Past-only free forecast plus the frozen three-pole response operator."""

    def __init__(
        self,
        config: RealModelConfig,
        n_features: int,
        target_index: int,
        mode: str,
    ):
        super().__init__()
        config.validate()
        if mode not in REAL_MODES:
            raise ValueError(f"unknown MS3 mode={mode!r}")
        self.config = config
        self.mode = mode
        self.encoder = HistoryEncoder(
            n_features=n_features,
            target_index=target_index,
            window=config.window,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        context_dim = self.encoder.context_dim
        self.free_head = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(context_dim * 2, config.horizon),
        )
        operator_config = config.operator_config(context_dim)
        self.response_operator = (
            ZeroResponseOperator(operator_config)
            if mode == "free_only"
            else build_response_operator(operator_config)
        )

    def forward(
        self,
        history: torch.Tensor,
        future_valve: torch.Tensor,
        baseline_valve: torch.Tensor,
    ) -> dict[str, Any]:
        context, target_mean, target_std, _ = self.encoder(history)
        normalized_free = self.free_head(context)
        free = target_mean[:, None] + target_std[:, None] * normalized_free
        reference = baseline_valve[:, None].expand_as(future_valve)
        response = self.response_operator(context, future_valve, reference)
        return {
            "prediction": free + response.effect,
            "free_prediction": free,
            "effect": response.effect,
            "response_state": response.state_trajectory,
            "response_diagnostics": response.diagnostics,
        }


def build_real_model(
    model_config: RealModelConfig,
    feature_columns: Sequence[str],
    mode: str,
) -> RealA1PhysMultiStep:
    features = list(feature_columns)
    if TARGET_COLUMN not in features:
        raise ValueError("MS3 history features must contain the target")
    return RealA1PhysMultiStep(
        model_config,
        n_features=len(features),
        target_index=features.index(TARGET_COLUMN),
        mode=mode,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _trajectory_sha256(
    anchors: np.ndarray,
    target: np.ndarray,
    future_valve: np.ndarray,
    baseline_valve: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for value in (anchors, target, future_valve, baseline_valve):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def operating_anchor_subset(
    cache: Phase35Cache,
    anchors: np.ndarray,
    *,
    window: int,
    horizon: int,
    config: RealTrainingConfig,
) -> np.ndarray:
    """Keep anchors whose treatment-preceding history is in the running regime.

    Load, pressure, target, and SP filters use history only.  The future check is
    limited to actuator plausibility and does not condition on future temperature.
    """

    anchors = np.asarray(anchors, dtype=np.int64)
    if len(anchors) == 0:
        return anchors
    load = cache.values[:, cache.index(LOAD_COLUMN)]
    pressure = cache.values[:, cache.index(PRESSURE_COLUMN)]
    target = cache.values[:, cache.index(TARGET_COLUMN)]
    sp = cache.values[:, cache.index(SP_COLUMN)]
    valve = cache.values[:, cache.index(VALVE_COLUMN)]
    history_bad = (
        (load < config.min_operating_load_mw)
        | (pressure < config.min_operating_pressure_mpa)
        | (target < config.target_temperature_min_c)
        | (target > config.target_temperature_max_c)
        | (sp < config.sp_temperature_min_c)
        | (sp > config.sp_temperature_max_c)
    )
    prefix = np.concatenate(([0], np.cumsum(history_bad, dtype=np.int64)))
    start = anchors - window + 1
    history_ok = (prefix[anchors + 1] - prefix[start]) == 0
    action_ok = np.ones(len(anchors), dtype=bool)
    for step in range(1, horizon + 1):
        future = valve[anchors + step]
        action_ok &= (future >= config.valve_min_pct) & (
            future <= config.valve_max_pct
        )
    return anchors[history_ok & action_ok]


def shuffled_delta_paths(
    future_valve: np.ndarray,
    baseline_valve: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Permute delta paths within baseline bins; report unavoidable singleton fixed points."""

    future = np.asarray(future_valve, dtype=np.float32)
    baseline = np.asarray(baseline_valve, dtype=np.float32)
    deltas = future - baseline[:, None]
    bins = np.floor(np.clip(baseline, 0.0, 99.999) / 10.0).astype(np.int64)
    rng = np.random.default_rng(seed)
    source = np.arange(len(baseline), dtype=np.int64)
    permuted_groups = 0
    singleton_count = 0
    for bin_id in np.unique(bins):
        indices = np.flatnonzero(bins == bin_id)
        if len(indices) < 2:
            singleton_count += len(indices)
            continue
        order = rng.permutation(indices)
        source[order] = np.roll(order, 1)
        permuted_groups += 1
    shuffled = np.clip(baseline[:, None] + deltas[source], 0.0, 100.0).astype(np.float32)
    return shuffled, {
        "method": "cyclic_delta_path_permutation_within_baseline_10pct_bins",
        "seed": int(seed),
        "permuted_group_count": int(permuted_groups),
        "singleton_count": int(singleton_count),
        "fixed_point_count": int(np.count_nonzero(source == np.arange(len(source)))),
    }


@torch.no_grad()
def evaluate_logged_mae(
    model: RealA1PhysMultiStep,
    cache: Phase35Cache,
    anchors: np.ndarray,
    feature_columns: Sequence[str],
    device: torch.device,
    batch_size: int = 256,
) -> float:
    model.eval()
    absolute_error = 0.0
    count = 0
    for start in range(0, len(anchors), batch_size):
        window = extract_windows(
            cache,
            anchors[start : start + batch_size],
            feature_columns,
            TARGET_COLUMN,
            VALVE_COLUMN,
            model.config.window,
            model.config.horizon,
        )
        output = model(
            torch.from_numpy(window["history"]).to(device),
            torch.from_numpy(window["future_valve"]).to(device),
            torch.from_numpy(window["baseline_valve"]).to(device),
        )
        target = torch.from_numpy(window["target"]).to(device)
        absolute_error += float((output["prediction"] - target).abs().sum().cpu())
        count += int(target.numel())
    if count == 0:
        raise RuntimeError("MS3 validation received no anchors")
    return absolute_error / count


@torch.no_grad()
def evaluate_real_model(
    model: RealA1PhysMultiStep,
    cache: Phase35Cache,
    anchors: np.ndarray,
    feature_columns: Sequence[str],
    device: torch.device,
    *,
    dynamic_dose_threshold_pct: float,
    shuffle_seed: int,
    batch_size: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    window = extract_windows(
        cache,
        anchors,
        feature_columns,
        TARGET_COLUMN,
        VALVE_COLUMN,
        model.config.window,
        model.config.horizon,
    )
    shuffled, shuffle_design = shuffled_delta_paths(
        window["future_valve"], window["baseline_valve"], shuffle_seed
    )
    logged_predictions: list[torch.Tensor] = []
    baseline_predictions: list[torch.Tensor] = []
    shuffled_predictions: list[torch.Tensor] = []
    free_predictions: list[torch.Tensor] = []
    effects: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for start in range(0, len(anchors), batch_size):
        stop = start + batch_size
        history = torch.from_numpy(window["history"][start:stop]).to(device)
        future = torch.from_numpy(window["future_valve"][start:stop]).to(device)
        baseline = torch.from_numpy(window["baseline_valve"][start:stop]).to(device)
        constant = baseline[:, None].expand_as(future)
        shuffled_action = torch.from_numpy(shuffled[start:stop]).to(device)
        logged = model(history, future, baseline)
        reference = model(history, constant, baseline)
        placebo = model(history, shuffled_action, baseline)
        logged_predictions.append(logged["prediction"].cpu())
        baseline_predictions.append(reference["prediction"].cpu())
        shuffled_predictions.append(placebo["prediction"].cpu())
        free_predictions.append(logged["free_prediction"].cpu())
        effects.append(logged["effect"].cpu())
        states.append(logged["response_state"].cpu())
    target = torch.from_numpy(window["target"])
    prediction = torch.cat(logged_predictions)
    baseline_prediction = torch.cat(baseline_predictions)
    shuffled_prediction = torch.cat(shuffled_predictions)
    free = torch.cat(free_predictions)
    effect = torch.cat(effects)
    state = torch.cat(states)
    error_logged = (prediction - target).abs()
    error_baseline = (baseline_prediction - target).abs()
    error_shuffled = (shuffled_prediction - target).abs()
    dose = np.max(
        np.abs(window["future_valve"] - window["baseline_valve"][:, None]), axis=1
    )
    dynamic = dose >= dynamic_dose_threshold_pct
    horizon_points = sorted(set(min(point, model.config.horizon) for point in (1, 6, 18, 60)))
    days = cache.timestamps_ns[anchors].astype("datetime64[ns]").astype("datetime64[D]")
    day_labels = np.datetime_as_string(days, unit="D").tolist()

    probe_count = min(8, len(anchors))
    probe_history = torch.from_numpy(window["history"][:probe_count]).to(device)
    probe_future = torch.from_numpy(window["future_valve"][:probe_count]).to(device)
    probe_baseline = torch.from_numpy(window["baseline_valve"][:probe_count]).to(device)
    probe_reference = probe_baseline[:, None].expand_as(probe_future)
    identity = model(probe_history, probe_reference, probe_baseline)
    normal = model(probe_history, probe_future, probe_baseline)
    changed = probe_future.clone()
    boundary = max(1, changed.shape[1] // 2)
    changed[:, boundary:] = (changed[:, boundary:] + 7.0).clamp(0.0, 100.0)
    changed_output = model(probe_history, changed, probe_baseline)
    positive = (probe_reference + 5.0).clamp(0.0, 100.0)
    positive_output = model(probe_history, positive, probe_baseline)
    structural = {
        "reference_identity_max_error": float(identity["effect"].abs().max().cpu()),
        "free_future_action_leakage_max_error": float(
            (normal["free_prediction"] - changed_output["free_prediction"]).abs().max().cpu()
        ),
        "future_action_prefix_leakage_max_error": float(
            (normal["prediction"][:, :boundary] - changed_output["prediction"][:, :boundary])
            .abs()
            .max()
            .cpu()
        ),
        "positive_step_terminal_effect_max_c": float(
            positive_output["effect"][:, -1].max().cpu()
        ),
        "finite_prediction": bool(torch.isfinite(prediction).all()),
        "finite_free": bool(torch.isfinite(free).all()),
        "finite_effect": bool(torch.isfinite(effect).all()),
        "finite_state": bool(torch.isfinite(state).all()),
    }
    dynamic_tensor = torch.from_numpy(dynamic)
    dynamic_count = int(dynamic.sum())
    metrics = {
        "sample_count": int(len(anchors)),
        "horizon": int(model.config.horizon),
        "logged_mae_c": float(error_logged.mean()),
        "baseline_action_mae_c": float(error_baseline.mean()),
        "shuffled_action_mae_c": float(error_shuffled.mean()),
        "logged_rmse_c": float((prediction - target).square().mean().sqrt()),
        "horizon_mae_c": {
            f"H{point}": float(error_logged[:, point - 1].mean())
            for point in horizon_points
        },
        "dynamic_support": {
            "threshold_pct": float(dynamic_dose_threshold_pct),
            "window_count": dynamic_count,
            "day_count": int(len(set(np.asarray(day_labels)[dynamic].tolist()))),
        },
        "dynamic_logged_mae_c": float(error_logged[dynamic_tensor].mean())
        if dynamic_count
        else None,
        "dynamic_baseline_action_mae_c": float(error_baseline[dynamic_tensor].mean())
        if dynamic_count
        else None,
        "dynamic_shuffled_action_mae_c": float(error_shuffled[dynamic_tensor].mean())
        if dynamic_count
        else None,
        "dynamic_mean_abs_effect_c": float(effect[dynamic_tensor].abs().mean())
        if dynamic_count
        else None,
        "max_abs_effect_c": float(effect.abs().max()),
        "structural_diagnostics": structural,
        "shuffle_design": shuffle_design,
    }
    episodes = {
        "anchors": anchors.astype(np.int64).tolist(),
        "utc_days": day_labels,
        "dynamic_mask": dynamic.tolist(),
        "action_dose_pct": dose.astype(float).tolist(),
        "logged_mae_c": error_logged.mean(dim=1).tolist(),
        "baseline_action_mae_c": error_baseline.mean(dim=1).tolist(),
        "shuffled_action_mae_c": error_shuffled.mean(dim=1).tolist(),
        "mean_abs_effect_c": effect.abs().mean(dim=1).tolist(),
        "terminal_effect_c": effect[:, -1].tolist(),
        "validation_trajectory_sha256": _trajectory_sha256(
            anchors,
            window["target"],
            window["future_valve"],
            window["baseline_valve"],
        ),
    }
    return metrics, episodes


@dataclass
class RealTrainResult:
    output_dir: Path
    checkpoint: Path
    best_epoch: int
    validation_metrics: dict[str, Any]


def train_real_run(
    *,
    cache: Phase35Cache,
    cache_path: str | Path,
    feature_columns: Sequence[str],
    model_config: RealModelConfig,
    training_config: RealTrainingConfig,
    side: str,
    seed: int,
    mode: str,
    run_id: str,
    output_dir: str | Path,
    device: str | torch.device,
    protocol_version: str,
    matrix_sha256: str,
    repo_git_sha: str,
    overwrite: bool = False,
) -> RealTrainResult:
    model_config.validate()
    training_config.validate()
    if side not in {"A", "B"} or mode not in REAL_MODES:
        raise ValueError("MS3 side/mode is invalid")
    if str(cache.metadata.get("side")) != side:
        raise ValueError("MS3 cache side does not match run side")
    if cache.metadata.get("cross_pairing_frozen") is not True:
        raise ValueError("MS3 requires a cache with frozen cross pairing")
    out = Path(output_dir)
    checkpoint_path = out / "checkpoint_best_val.pt"
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite completed MS3 run: {checkpoint_path}")
    out.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    dev = torch.device(device)
    model = build_real_model(model_config, feature_columns, mode).to(dev)
    train_anchors = valid_window_anchors(
        cache,
        "train",
        feature_columns,
        TARGET_COLUMN,
        VALVE_COLUMN,
        model_config.window,
        model_config.horizon,
        training_config.max_age_seconds,
    )
    validation_anchors = valid_window_anchors(
        cache,
        "validation",
        feature_columns,
        TARGET_COLUMN,
        VALVE_COLUMN,
        model_config.window,
        model_config.horizon,
        training_config.max_age_seconds,
    )
    train_anchors = operating_anchor_subset(
        cache,
        train_anchors,
        window=model_config.window,
        horizon=model_config.horizon,
        config=training_config,
    )
    validation_anchors = operating_anchor_subset(
        cache,
        validation_anchors,
        window=model_config.window,
        horizon=model_config.horizon,
        config=training_config,
    )
    train_anchors = deterministic_anchor_subset(
        train_anchors, training_config.max_train_anchors, 1000 + seed
    )
    validation_anchors = deterministic_anchor_subset(
        validation_anchors, training_config.max_validation_anchors, 10_000 + seed
    )
    selector_anchors = deterministic_anchor_subset(
        validation_anchors, training_config.max_selector_anchors, 20_000 + seed
    )
    if len(train_anchors) < training_config.batch_size or len(selector_anchors) < 1:
        raise RuntimeError(
            f"insufficient MS3 anchors train={len(train_anchors)} validation={len(validation_anchors)}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    response_parameters = list(model.response_operator.parameters())
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    wait = 0
    started = time.time()
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        losses: list[float] = []
        response_gradients: list[float] = []
        for _ in range(training_config.steps_per_epoch):
            chosen = rng.choice(
                train_anchors, size=training_config.batch_size, replace=False
            )
            window = extract_windows(
                cache,
                chosen,
                feature_columns,
                TARGET_COLUMN,
                VALVE_COLUMN,
                model_config.window,
                model_config.horizon,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(
                torch.from_numpy(window["history"]).to(dev),
                torch.from_numpy(window["future_valve"]).to(dev),
                torch.from_numpy(window["baseline_valve"]).to(dev),
            )
            target = torch.from_numpy(window["target"]).to(dev)
            loss = F.smooth_l1_loss(output["prediction"], target, beta=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MS3 loss at epoch={epoch}")
            loss.backward()
            response_gradients.append(_gradient_norm(response_parameters))
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = evaluate_logged_mae(
            model,
            cache,
            selector_anchors,
            feature_columns,
            dev,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_logged_mae_c": score,
                "response_gradient_norm": float(np.mean(response_gradients))
                if response_gradients
                else 0.0,
            }
        )
        _json_dump(out / "history.json", history)
        if score < best_score - training_config.min_delta:
            best_score = score
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "protocol_version": protocol_version,
                    "run_id": run_id,
                    "side": side,
                    "seed": seed,
                    "mode": mode,
                    "model_config": asdict(model_config),
                    "operator_config": model_config.operator_config(
                        model.encoder.context_dim
                    ).to_dict(),
                    "training_config": asdict(training_config),
                    "feature_columns": list(feature_columns),
                    "model_state_dict": best_state,
                    "validation_logged_mae_c": score,
                    "git_sha": repo_git_sha,
                    "matrix_sha256": matrix_sha256,
                },
                checkpoint_path,
            )
        else:
            wait += 1
        if wait >= training_config.patience:
            break
    if best_state is None:
        raise RuntimeError("MS3 training produced no finite checkpoint")
    model.load_state_dict(best_state)
    metrics, episodes = evaluate_real_model(
        model,
        cache,
        validation_anchors,
        feature_columns,
        dev,
        dynamic_dose_threshold_pct=training_config.dynamic_dose_threshold_pct,
        shuffle_seed=91_000 + seed,
    )
    cache_file = Path(cache_path)
    manifest = {
        "protocol_version": protocol_version,
        "evidence_scope": "real_ab_observational_validation_not_causal",
        "run_id": run_id,
        "side": side,
        "seed": seed,
        "mode": mode,
        "model_config": asdict(model_config),
        "operator_config": model_config.operator_config(
            model.encoder.context_dim
        ).to_dict(),
        "training_config": asdict(training_config),
        "feature_columns": list(feature_columns),
        "cache_path": str(cache_file.resolve()),
        "cache_sha256": _sha256(cache_file),
        "cache_metadata": cache.metadata,
        "matrix_sha256": matrix_sha256,
        "git_sha": repo_git_sha,
        "device": str(dev),
        "torch_version": torch.__version__,
        "train_anchor_count": int(len(train_anchors)),
        "validation_anchor_count": int(len(validation_anchors)),
        "selector_anchor_count": int(len(selector_anchors)),
        "validation_trajectory_sha256": episodes["validation_trajectory_sha256"],
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "checkpoint_selector": "validation_logged_mae_c",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "test_accessed": False,
        "test_authorized": False,
    }
    _json_dump(out / "manifest.json", manifest)
    _json_dump(out / "metrics_validation.json", metrics)
    _json_dump(out / "episode_metrics_validation.json", episodes)
    return RealTrainResult(out, checkpoint_path, best_epoch, metrics)
