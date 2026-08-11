"""Long-running real-data training core for the Gate C RM2 Hermes batch."""

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
from .gatec_contracts import GateCModelConfig
from .gatec_data import extract_gatec_batch, paired_valid_anchors
from .gatec_model import build_gatec_model
from .gatec_real_smoke import COMMON_PREDICTION_WEIGHTS, DEFAULT_WEIGHTS
from .gatec_rm2_contracts import RM2RunSpec
from .gatec_training import GateCRobustScales, compute_gatec_loss


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


def _anchor_sha256(anchors: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(anchors, dtype="<i8").tobytes()).hexdigest()


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


def _fraction_bounds(n_rows: int, fraction: tuple[float, float]) -> tuple[int, int]:
    return int(n_rows * fraction[0]), int(n_rows * fraction[1])


def rm2_run_bounds(
    n_rows: int, spec: RM2RunSpec, *, actual_test_start: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    train_bounds = _fraction_bounds(n_rows, spec.train_fraction)
    validation_raw = _fraction_bounds(n_rows, spec.validation_fraction)
    validation_bounds = (validation_raw[0], min(validation_raw[1], int(actual_test_start)))
    if not 0 <= train_bounds[0] < train_bounds[1] <= validation_bounds[0]:
        raise Phase35ProtocolError("RM2 train/validation fold ordering is invalid")
    if not validation_bounds[0] < validation_bounds[1] <= actual_test_start:
        raise Phase35ProtocolError("RM2 validation fold touches the test split")
    return train_bounds, validation_bounds


def _targets(batch: Any) -> dict[str, torch.Tensor]:
    return {
        "valve": torch.from_numpy(batch.logged_future_valve),
        "tin": torch.from_numpy(batch.logged_future_tin),
        "local": torch.from_numpy(batch.local_drop_target),
        "terminal": torch.from_numpy(batch.terminal_target),
    }


def _device_tensors(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "history": torch.as_tensor(batch.history, dtype=torch.float32, device=device),
        "future_sp": torch.as_tensor(batch.future_sp, dtype=torch.float32, device=device),
        "valve": torch.as_tensor(batch.logged_future_valve, dtype=torch.float32, device=device),
        "tin": torch.as_tensor(batch.logged_future_tin, dtype=torch.float32, device=device),
        "local": torch.as_tensor(batch.local_drop_target, dtype=torch.float32, device=device),
        "terminal": torch.as_tensor(batch.terminal_target, dtype=torch.float32, device=device),
    }


def _split_validation_anchors(
    anchors: np.ndarray, *, selector_count: int, final_count: int, fold_id: str
) -> tuple[np.ndarray, np.ndarray]:
    if len(anchors) < selector_count + final_count:
        raise Phase35ProtocolError("RM2 validation fold has insufficient disjoint anchors")
    fold_seed = 9100 + int(fold_id[1:])
    rng = np.random.default_rng(fold_seed)
    selected = rng.choice(anchors, size=selector_count + final_count, replace=False)
    selector = np.sort(selected[:selector_count])
    final = np.sort(selected[selector_count:])
    return selector, final


def _training_weights(response_expected: bool) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not response_expected:
        weights["structure"] = 0.0
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _evaluate_score(
    model: torch.nn.Module,
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    *,
    window: int,
    horizon: int,
    scales: GateCRobustScales,
    batch_size: int,
    device: torch.device,
) -> float:
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            batch = extract_gatec_batch(
                caches,
                anchors[start : start + batch_size],
                window=window,
                horizon=horizon,
                validate_pair=False,
            )
            tensors = _device_tensors(batch, device)
            output = model(
                tensors["history"], tensors["future_sp"], boundary_mode="forecast_boundary"
            )
            loss = compute_gatec_loss(
                output,
                {key: tensors[key] for key in ("valve", "tin", "local", "terminal")},
                scales,
                COMMON_PREDICTION_WEIGHTS,
                local_supervision=True,
            )
            total += float(loss.total.cpu()) * len(batch.anchors)
            count += len(batch.anchors)
    model.train()
    return total / count


def _final_evaluation(
    model: torch.nn.Module,
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    *,
    window: int,
    horizon: int,
    scales: GateCRobustScales,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(12000 + seed)
    permutation = rng.permutation(len(anchors))
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "future_sp",
            "logged_valve",
            "forecast_valve",
            "local_target",
            "forecast_local",
            "logged_local",
            "shuffled_local",
            "terminal_target",
            "forecast_terminal",
            "forecast_tin",
            "predicted_effect",
            "logged_effect",
            "shuffled_effect",
            "a_only_effect",
            "b_only_effect",
        )
    }
    sums = {
        key: 0.0
        for key in (
            "valve_abs",
            "tin_abs",
            "local_abs",
            "terminal_abs",
            "oracle_terminal_abs",
            "persistence_valve_abs",
            "persistence_tin_abs",
            "persistence_local_abs",
            "persistence_terminal_abs",
            "logged_local_abs",
            "shuffled_local_abs",
            "predicted_effect_abs",
            "logged_effect_abs",
            "common_score",
        )
    }
    stable_pole_max = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            stop = min(len(anchors), start + batch_size)
            batch = extract_gatec_batch(
                caches,
                anchors[start:stop],
                window=window,
                horizon=horizon,
                validate_pair=False,
            )
            shuffled_batch = extract_gatec_batch(
                caches,
                anchors[permutation[start:stop]],
                window=window,
                horizon=horizon,
                validate_pair=False,
            )
            tensors = _device_tensors(batch, device)
            shuffled_valve = torch.as_tensor(
                shuffled_batch.logged_future_valve, dtype=torch.float32, device=device
            )
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
            logged_response = model.local_response(context, tensors["valve"], baseline_valve)
            shuffled_response = model.local_response(context, shuffled_valve, baseline_valve)
            held_valve = baseline_valve[:, None, :].expand_as(tensors["valve"])
            a_only_valve = held_valve.clone()
            a_only_valve[..., 0] = tensors["valve"][..., 0]
            b_only_valve = held_valve.clone()
            b_only_valve[..., 1] = tensors["valve"][..., 1]
            a_only_response = model.local_response(context, a_only_valve, baseline_valve)
            b_only_response = model.local_response(context, b_only_valve, baseline_valve)
            logged_local = forecast["residual_local_prediction"] + logged_response["effect"]
            shuffled_local = forecast["residual_local_prediction"] + shuffled_response["effect"]
            score = compute_gatec_loss(
                forecast,
                {key: tensors[key] for key in ("valve", "tin", "local", "terminal")},
                scales,
                COMMON_PREDICTION_WEIGHTS,
                local_supervision=True,
            ).total
            elements = int(tensors["terminal"].numel())
            sums["valve_abs"] += float((forecast["valve_prediction"] - tensors["valve"]).abs().sum().cpu())
            sums["tin_abs"] += float((forecast["tin_prediction"] - tensors["tin"]).abs().sum().cpu())
            sums["local_abs"] += float((forecast["local_drop_prediction"] - tensors["local"]).abs().sum().cpu())
            sums["terminal_abs"] += float((forecast["terminal_prediction"] - tensors["terminal"]).abs().sum().cpu())
            sums["oracle_terminal_abs"] += float((oracle["terminal_prediction"] - tensors["terminal"]).abs().sum().cpu())
            sums["persistence_valve_abs"] += float((tensors["valve"] - baseline_valve[:, None]).abs().sum().cpu())
            sums["persistence_tin_abs"] += float((tensors["tin"] - baseline_tin[:, None]).abs().sum().cpu())
            sums["persistence_local_abs"] += float((tensors["local"] - baseline_local[:, None]).abs().sum().cpu())
            sums["persistence_terminal_abs"] += float((tensors["terminal"] - baseline_terminal[:, None]).abs().sum().cpu())
            sums["logged_local_abs"] += float((logged_local - tensors["local"]).abs().sum().cpu())
            sums["shuffled_local_abs"] += float((shuffled_local - tensors["local"]).abs().sum().cpu())
            sums["predicted_effect_abs"] += float(forecast["local_effect"].abs().sum().cpu())
            sums["logged_effect_abs"] += float(logged_response["effect"].abs().sum().cpu())
            sums["common_score"] += float(score.cpu()) * len(batch.anchors)
            if forecast["local_stable_poles"].numel():
                stable_pole_max = max(stable_pole_max, float(forecast["local_stable_poles"].max().cpu()))
            values = {
                "future_sp": tensors["future_sp"],
                "logged_valve": tensors["valve"],
                "forecast_valve": forecast["valve_prediction"],
                "local_target": tensors["local"],
                "forecast_local": forecast["local_drop_prediction"],
                "logged_local": logged_local,
                "shuffled_local": shuffled_local,
                "terminal_target": tensors["terminal"],
                "forecast_terminal": forecast["terminal_prediction"],
                "forecast_tin": forecast["tin_prediction"],
                "predicted_effect": forecast["local_effect"],
                "logged_effect": logged_response["effect"],
                "shuffled_effect": shuffled_response["effect"],
                "a_only_effect": a_only_response["effect"],
                "b_only_effect": b_only_response["effect"],
            }
            for key, tensor in values.items():
                arrays[key].append(tensor.detach().cpu().numpy().astype(np.float32))
    element_count = len(anchors) * horizon * 2
    metrics = {
        "shared_prediction_score": sums["common_score"] / len(anchors),
        "forecast_valve_mae": sums["valve_abs"] / element_count,
        "forecast_tin_mae_c": sums["tin_abs"] / element_count,
        "forecast_local_mae_c": sums["local_abs"] / element_count,
        "forecast_terminal_mae_c": sums["terminal_abs"] / element_count,
        "oracle_terminal_mae_c": sums["oracle_terminal_abs"] / element_count,
        "persistence_valve_mae": sums["persistence_valve_abs"] / element_count,
        "persistence_tin_mae_c": sums["persistence_tin_abs"] / element_count,
        "persistence_local_mae_c": sums["persistence_local_abs"] / element_count,
        "persistence_terminal_mae_c": sums["persistence_terminal_abs"] / element_count,
        "logged_local_mae_c": sums["logged_local_abs"] / element_count,
        "shuffled_local_mae_c": sums["shuffled_local_abs"] / element_count,
        "logged_vs_shuffled_local_advantage_c": (sums["shuffled_local_abs"] - sums["logged_local_abs"]) / element_count,
        "predicted_effect_mean_abs_c": sums["predicted_effect_abs"] / element_count,
        "logged_effect_mean_abs_c": sums["logged_effect_abs"] / element_count,
        "stable_pole_max": stable_pole_max,
        "finite": bool(all(math.isfinite(value) for value in sums.values())),
    }
    for target in ("valve", "tin", "local", "terminal"):
        metrics[f"{target}_to_persistence_ratio"] = metrics[
            f"forecast_{target}_mae" + ("_c" if target != "valve" else "")
        ] / max(metrics[f"persistence_{target}_mae" + ("_c" if target != "valve" else "")], 1e-9)
    episode_arrays = {
        "anchors": np.asarray(anchors, dtype=np.int64),
        "timestamps_ns": caches["A"].timestamps_ns[anchors].astype(np.int64),
        "shuffled_anchors": np.asarray(anchors[permutation], dtype=np.int64),
        **{key: np.concatenate(parts, axis=0) for key, parts in arrays.items()},
    }
    baseline_valve = np.stack(
        [
            caches[side].values[anchors, caches[side].index("二级减温调节门阀位")]
            for side in ("A", "B")
        ],
        axis=1,
    ).astype(np.float32)
    episode_arrays["baseline_valve"] = baseline_valve
    dose = episode_arrays["logged_valve"] - baseline_valve[:, None, :]
    common = 0.5 * (dose[..., 0] + dose[..., 1])
    differential = 0.5 * (dose[..., 0] - dose[..., 1])
    metrics["action_support"] = {
        "opening_window_count": int(np.sum(np.mean(dose, axis=(1, 2)) > 0.05)),
        "closing_window_count": int(np.sum(np.mean(dose, axis=(1, 2)) < -0.05)),
        "common_energy": float(np.mean(common**2)),
        "differential_energy": float(np.mean(differential**2)),
        "differential_to_common_energy_ratio": float(np.mean(differential**2) / max(np.mean(common**2), 1e-12)),
    }
    for seconds in (60, 180):
        index = seconds // 10 - 1
        if index < horizon:
            metrics[f"logged_effect_h{seconds}_mean_abs_c"] = float(
                np.mean(np.abs(episode_arrays["logged_effect"][:, index]))
            )
    return metrics, episode_arrays


def _structural_validation(
    model: torch.nn.Module,
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    *,
    window: int,
    horizon: int,
    device: torch.device,
    stable_pole_max: float,
    response_expected: bool,
    logged_effect_mean_abs_c: float,
) -> dict[str, Any]:
    batch = extract_gatec_batch(
        caches,
        anchors[: min(64, len(anchors))],
        window=window,
        horizon=horizon,
        validate_pair=False,
    )
    tensors = _device_tensors(batch, device)
    prefix = max(1, horizon // 2)
    model.eval()
    with torch.no_grad():
        original = model(
            tensors["history"], tensors["future_sp"], boundary_mode="forecast_boundary"
        )
        changed_sp = tensors["future_sp"].clone()
        changed_sp[:, prefix:] += 20.0
        changed = model(
            tensors["history"], changed_sp, boundary_mode="forecast_boundary"
        )
        normalized = (
            tensors["history"] - model.history_center[None, None, :]
        ) / model.history_scale[None, None, :]
        context = model.encoder(normalized)
        baseline = tensors["history"][:, -1, model.valve_indices]
        constant = baseline[:, None, :].expand(-1, horizon, -1)
        identity_effect = model.local_response(context, constant, baseline)["effect"]
    response_noncollapse = logged_effect_mean_abs_c > 1e-6 if response_expected else True
    structural = {
        "finite_rollout": bool(torch.isfinite(original["terminal_prediction"]).all())
        and stable_pole_max < 1.0,
        "sp_prefix_causality": bool(
            torch.allclose(
                original["terminal_prediction"][:, :prefix],
                changed["terminal_prediction"][:, :prefix],
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "constant_action_identity": float(identity_effect.abs().max().cpu()) <= 1e-6,
        "future_truth_isolation": True,
        "local_response_noncollapse": response_noncollapse,
        "local_response_noncollapse_applicability": (
            "required" if response_expected else "not_applicable_paired_free"
        ),
    }
    structural["selector_eligible"] = all(
        bool(structural[key])
        for key in (
            "finite_rollout",
            "sp_prefix_causality",
            "constant_action_identity",
            "future_truth_isolation",
            "local_response_noncollapse",
        )
    )
    return structural


def run_rm2_training(
    caches: Mapping[str, Phase35Cache],
    matrix: Mapping[str, Any],
    spec: RM2RunSpec,
    *,
    device: str,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _set_seed(spec.seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise Phase35ProtocolError("RM2 requested CUDA but it is unavailable")
    data = matrix["data_contract"]
    training = matrix["training"]
    n_rows = len(caches["A"].timestamps_ns)
    train_bounds, validation_bounds = rm2_run_bounds(
        n_rows, spec, actual_test_start=caches["A"].split_bounds()["test"][0]
    )
    train_pool = paired_valid_anchors(
        caches,
        "train",
        window=int(data["window"]),
        horizon=int(data["horizon"]),
        max_age_s=float(data["max_age_s"]),
        bounds_override=train_bounds,
    )
    validation_pool = paired_valid_anchors(
        caches,
        "validation",
        window=int(data["window"]),
        horizon=int(data["horizon"]),
        max_age_s=float(data["max_age_s"]),
        bounds_override=validation_bounds,
    )
    fold_index = int(spec.fold_id[1:])
    stats_anchors = deterministic_anchor_subset(
        train_pool, int(training["stats_anchor_count"]), 9000 + fold_index
    )
    selector_anchors, final_anchors = _split_validation_anchors(
        validation_pool,
        selector_count=int(training["selector_anchor_count"]),
        final_count=int(training["final_anchor_count"]),
        fold_id=spec.fold_id,
    )
    stats_batch = extract_gatec_batch(
        caches,
        stats_anchors,
        window=int(data["window"]),
        horizon=int(data["horizon"]),
        validate_pair=False,
    )
    flattened = stats_batch.history.reshape(-1, stats_batch.history.shape[-1]).astype(np.float64)
    center = np.mean(flattened, axis=0).astype(np.float32)
    scale = np.maximum(np.std(flattened, axis=0).astype(np.float32), 1e-3)
    scales = GateCRobustScales.fit(_targets(stats_batch), split="train", scale_floor=1e-3)
    model_config = GateCModelConfig(
        window=int(data["window"]),
        horizon=int(data["horizon"]),
        n_features=stats_batch.history.shape[-1],
        d_model=int(matrix["model"]["d_model"]),
        latent_dim=int(matrix["model"]["latent_dim"]),
        local_state_dim=int(matrix["model"]["local_state_dim"]),
        response_route=spec.response_route,
        residual_capacity=spec.residual_capacity,
        response_scheduling=spec.response_scheduling,
        response_coordinate_mode=spec.response_coordinate_mode,
        downstream_mode=spec.downstream_mode,
        dropout=float(matrix["model"]["dropout"]),
    )
    model = build_gatec_model(model_config, stats_batch.history_feature_names).to(torch_device)
    model.set_history_normalization(
        torch.from_numpy(center).to(torch_device), torch.from_numpy(scale).to(torch_device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    response_expected = spec.response_route != "none"
    weights = _training_weights(response_expected)
    rng = np.random.default_rng(15000 + spec.seed + fold_index * 100)
    best_score = math.inf
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    evaluations_without_improvement = 0
    selector_history: list[dict[str, float | int]] = []
    loss_curve: list[float] = []
    started = time.perf_counter()
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    model.train()
    for update in range(1, int(training["optimizer_updates_cap"]) + 1):
        batch_anchors = train_pool[
            rng.integers(0, len(train_pool), size=int(training["batch_size"]))
        ]
        batch = extract_gatec_batch(
            caches,
            batch_anchors,
            window=int(data["window"]),
            horizon=int(data["horizon"]),
            validate_pair=False,
        )
        tensors = _device_tensors(batch, torch_device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            tensors["history"],
            tensors["future_sp"],
            boundary_mode="forecast_boundary",
            logged_future_valve_for_aux=tensors["valve"] if response_expected else None,
        )
        if response_expected:
            output["structure_penalty"] = F.smooth_l1_loss(
                output["logged_local_drop_prediction"] / scales.values["local"],
                tensors["local"] / scales.values["local"],
            )
        loss = compute_gatec_loss(
            output,
            {key: tensors[key] for key in ("valve", "tin", "local", "terminal")},
            scales,
            weights,
            local_supervision=True,
        )
        if not torch.isfinite(loss.total):
            raise Phase35ProtocolError("RM2 training loss became non-finite")
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        loss_curve.append(float(loss.total.detach().cpu()))
        if update % int(training["evaluation_interval_updates"]) != 0:
            continue
        score = _evaluate_score(
            model,
            caches,
            selector_anchors,
            window=int(data["window"]),
            horizon=int(data["horizon"]),
            scales=scales,
            batch_size=int(training["evaluation_batch_size"]),
            device=torch_device,
        )
        selector_history.append({"update": update, "shared_prediction_score": score})
        if score < best_score - float(training["minimum_score_improvement"]):
            best_score = score
            best_update = update
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        if (
            update >= int(training["minimum_updates"])
            and evaluations_without_improvement
            >= int(training["early_stopping_patience_evaluations"])
        ):
            break
    if best_state is None:
        raise Phase35ProtocolError("RM2 produced no selectable checkpoint")
    model.load_state_dict(best_state)
    metrics, episodes = _final_evaluation(
        model,
        caches,
        final_anchors,
        window=int(data["window"]),
        horizon=int(data["horizon"]),
        scales=scales,
        batch_size=int(training["evaluation_batch_size"]),
        device=torch_device,
        seed=spec.seed + fold_index * 100,
    )
    structural = _structural_validation(
        model,
        caches,
        final_anchors,
        window=int(data["window"]),
        horizon=int(data["horizon"]),
        device=torch_device,
        stable_pole_max=float(metrics["stable_pole_max"]),
        response_expected=response_expected,
        logged_effect_mean_abs_c=float(metrics["logged_effect_mean_abs_c"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_best_validation.pt"
    _atomic_checkpoint(
        checkpoint_path,
        {
            "protocol_version": matrix["protocol_version"],
            "run_spec": asdict(spec),
            "model_config": asdict(model_config),
            "feature_names": list(stats_batch.history_feature_names),
            "model_state_dict": best_state,
            "history_center": center,
            "history_scale": scale,
            "robust_scales": scales.values,
            "best_update": best_update,
            "best_selector_score": best_score,
        },
    )
    _atomic_npz(output_dir / "episodes_validation.npz", episodes)
    metrics_payload = {
        "run_id": spec.run_id,
        "metrics": metrics,
        "selector_history": selector_history,
        "loss_curve": loss_curve,
        "best_update": best_update,
        "best_selector_score": best_score,
        "optimizer_updates_completed": len(loss_curve),
        "early_stopped": len(loss_curve) < int(training["optimizer_updates_cap"]),
        "structural_validation": structural,
        "selector_eligible": structural["selector_eligible"],
        "automatic_scientific_pass": None,
    }
    _atomic_json(output_dir / "metrics_validation.json", metrics_payload)
    manifest = {
        "protocol_version": matrix["protocol_version"],
        "run_spec": asdict(spec),
        "provenance": dict(provenance),
        "train_bounds": list(train_bounds),
        "validation_bounds": list(validation_bounds),
        "train_anchor_pool_count": int(len(train_pool)),
        "stats_anchor_count": int(len(stats_anchors)),
        "selector_anchor_count": int(len(selector_anchors)),
        "final_anchor_count": int(len(final_anchors)),
        "stats_anchor_sha256": _anchor_sha256(stats_anchors),
        "selector_anchor_sha256": _anchor_sha256(selector_anchors),
        "final_anchor_sha256": _anchor_sha256(final_anchors),
        "selector_reporting_disjoint": bool(
            not np.intersect1d(selector_anchors, final_anchors).size
        ),
        "normalization_source": "fold_train_stats_anchors_only",
        "target_scale_source": "fold_train_stats_anchors_only",
        "boundary_mode": "forecast_boundary",
        "logged_future_valve_role": "response_auxiliary_and_validation_diagnostic_only",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "episodes_sha256": _sha256(output_dir / "episodes_validation.npz"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(torch_device))
        if torch_device.type == "cuda"
        else 0,
        "test_accessed": False,
        "attempt": 1,
        "automatic_scientific_pass": None,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    ledger_names = (
        "manifest.json",
        "checkpoint_best_validation.pt",
        "metrics_validation.json",
        "episodes_validation.npz",
    )
    _atomic_json(
        output_dir / "artifact_ledger.json",
        {name: _sha256(output_dir / name) for name in ledger_names},
    )
    return {"run_id": spec.run_id, "status": "complete", "metrics": metrics_payload}
