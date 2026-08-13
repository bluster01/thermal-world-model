"""Validation-only training executor for one frozen RM3-AV unit."""

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

from ..data import Phase35Cache, deterministic_anchor_subset
from ..schema import Phase35ProtocolError
from .gatec_data import extract_gatec_batch, paired_valid_anchors
from .rm3_training import fit_rm3_train_statistics
from .rm3av_contracts import RM3AVRunSpec
from .rm3av_diagnostics import (
    build_assumption_ledger,
    build_manual_verdict_template,
    build_prediction_diagnostics,
    build_state_closure_audit,
    convergence_diagnostics,
    daily_gain_context_diagnostics,
    dependence_diagnostics,
    mechanism_residual_dependence,
    response_trajectory_diagnostics,
    stratified_error_diagnostics,
    valve_trajectory_diagnostics,
    valve_innovation_rank,
    valve_policy_probe_diagnostics,
)
from .rm3av_model import (
    RM3AVModel,
    RM3AVModelConfig,
    build_rm3av_model,
    module_state_hashes,
)
from .rm3av_training import (
    OOFActionProjection,
    OOFRModelSet,
    fit_oof_action_projection,
    fit_oof_action_outcome_audit,
    fit_oof_r_model,
    rm3av_multitask_loss,
)


TASKS = ("valve", "tin", "local", "terminal")
PREDICTION_KEYS = {
    "valve": "valve_prediction",
    "tin": "tin_prediction",
    "local": "local_drop_prediction",
    "terminal": "terminal_prediction",
}
DAY_NS = 86_400_000_000_000
FINITE_DIFFERENCE_VALVE_POINTS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
FINITE_DIFFERENCE_HORIZONS = (6, 18, 60)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
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


def _groups(caches: Mapping[str, Phase35Cache], anchors: np.ndarray) -> np.ndarray:
    return np.floor_divide(caches["A"].timestamps_ns[anchors], DAY_NS).astype(np.int64)


def _split_validation(
    anchors: np.ndarray,
    timestamps_ns: np.ndarray,
    selector_count: int,
    reporting_count: int,
    fold_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(anchors) < selector_count + reporting_count:
        raise Phase35ProtocolError("RM3-AV validation has insufficient disjoint anchors")
    days = np.floor_divide(timestamps_ns[anchors], DAY_NS)
    unique_days = np.unique(days)
    if len(unique_days) < 2:
        raise Phase35ProtocolError(
            "RM3-AV validation requires at least two UTC days for selector/reporting isolation"
        )
    rng = np.random.default_rng(36300 + int(fold_id[1:]))
    valid_splits = []
    for split_index in range(1, len(unique_days)):
        selector_pool = anchors[np.isin(days, unique_days[:split_index])]
        reporting_pool = anchors[np.isin(days, unique_days[split_index:])]
        if len(selector_pool) >= selector_count and len(reporting_pool) >= reporting_count:
            valid_splits.append((split_index, selector_pool, reporting_pool))
    if not valid_splits:
        raise Phase35ProtocolError(
            "RM3-AV UTC-day selector/reporting split lacks declared anchor counts"
        )
    split_index, selector_pool, reporting_pool = min(
        valid_splits,
        key=lambda item: (
            abs(item[0] / len(unique_days) - selector_count / (selector_count + reporting_count)),
            item[0],
        ),
    )
    selector = np.sort(rng.choice(selector_pool, selector_count, replace=False))
    reporting = np.sort(rng.choice(reporting_pool, reporting_count, replace=False))
    return selector, reporting, unique_days[:split_index], unique_days[split_index:]


def _tensor_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    history = torch.as_tensor(batch.history, dtype=torch.float32, device=device)
    future_sp = torch.as_tensor(batch.future_sp, dtype=torch.float32, device=device)
    targets = {
        "valve": torch.as_tensor(batch.logged_future_valve, dtype=torch.float32, device=device),
        "tin": torch.as_tensor(batch.logged_future_tin, dtype=torch.float32, device=device),
        "local": torch.as_tensor(batch.local_drop_target, dtype=torch.float32, device=device),
        "terminal": torch.as_tensor(batch.terminal_target, dtype=torch.float32, device=device),
    }
    return history, future_sp, targets


def _reconstructed_second_history(
    model: RM3AVModel,
    first_history: torch.Tensor,
    first_sp: torch.Tensor,
    first_output: Mapping[str, Any],
) -> torch.Tensor:
    horizon = first_sp.shape[1]
    generated = first_history[:, -1:, :].expand(-1, horizon, -1).clone()
    generated[:, :, model.valve_indices] = first_output["valve_prediction"]
    generated[:, :, model.tin_indices] = first_output["tin_prediction"]
    generated[:, :, model.tout_indices] = (
        first_output["tin_prediction"] - first_output["local_drop_prediction"]
    )
    generated[:, :, model.terminal_indices] = first_output["terminal_prediction"]
    sp_indices = [
        model.feature_names.index(f"{side}::二级减温调节阀设定") for side in ("A", "B")
    ]
    generated[:, :, sp_indices] = first_sp
    if first_history.shape[1] >= horizon:
        return torch.cat((first_history[:, horizon:], generated), dim=1)
    return generated[:, -first_history.shape[1] :]


def _forward_loss(
    model: RM3AVModel,
    batch: Any,
    device: torch.device,
    *,
    candidate_id: str,
    target_scales: Mapping[str, float],
    r_model: OOFRModelSet | None,
    groups: np.ndarray,
    rollout_weight: float,
    second_batch: Any | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    history, future_sp, targets = _tensor_batch(batch, device)
    kwargs: dict[str, torch.Tensor] = {}
    action_residual_tensor = outcome_residual_tensor = None
    if candidate_id in {"C10", "C13"}:
        kwargs["logged_future_valve"] = targets["valve"]
    elif candidate_id in {"C11", "C12"}:
        if r_model is None:
            raise Phase35ProtocolError("RM3-AV R-loss model was not train-fitted")
        action_residual, outcome_residual = r_model.residualize(
            batch.history,
            batch.logged_future_valve
            - batch.history[:, -1, model.valve_indices][:, None],
            batch.local_drop_target
            - (
                batch.history[:, -1, model.tin_indices]
                - batch.history[:, -1, model.tout_indices]
            )[:, None],
            groups,
            future_sp=batch.future_sp,
        )
        action_residual_tensor = torch.as_tensor(action_residual, device=device)
        outcome_residual_tensor = torch.as_tensor(outcome_residual, device=device)
        kwargs = {
            "logged_future_valve": targets["valve"],
            "oof_action_residual": action_residual_tensor,
        }
    output = model(history, future_sp, **kwargs)
    losses = rm3av_multitask_loss(
        output,
        targets,
        candidate_id=candidate_id,
        target_scales=target_scales,
        action_residual=action_residual_tensor,
        outcome_residual=outcome_residual_tensor,
    )
    if candidate_id == "C31":
        if second_batch is None:
            raise Phase35ProtocolError("C31 needs a consecutive second window")
        _, second_sp, second_targets = _tensor_batch(second_batch, device)
        second_history = _reconstructed_second_history(model, history, future_sp, output)
        rollout = model.forward_two_window(history, future_sp, second_history, second_sp)
        second_losses = rm3av_multitask_loss(
            rollout["second"],
            second_targets,
            candidate_id="C31",
            target_scales=target_scales,
        )
        losses["two_window_rollout"] = second_losses["total"]
        losses["total"] = losses["total"] + rollout_weight * second_losses["total"]
        output = {**output, "two_window_second": rollout["second"]}
    return output, targets, losses


def _candidate_pool(
    caches: Mapping[str, Phase35Cache],
    split: str,
    *,
    bounds: tuple[int, int],
    window: int,
    horizon: int,
    max_age_s: float,
    candidate_id: str,
) -> np.ndarray:
    # Every candidate uses the same H120-eligible anchor universe so that C31's
    # true two-window rollout remains fold-paired with its H60 controls.
    # Non-C31 candidates still train and score only the declared H60 outputs.
    del candidate_id
    required_horizon = horizon * 2
    return paired_valid_anchors(
        caches,
        split,
        window=window,
        horizon=required_horizon,
        max_age_s=max_age_s,
        bounds_override=bounds,
    )


def _evaluate(
    model: RM3AVModel,
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    *,
    window: int,
    horizon: int,
    batch_size: int,
    candidate_id: str,
    target_scales: Mapping[str, float],
    r_model: OOFRModelSet | None,
    rollout_weight: float,
    device: torch.device,
) -> float:
    del r_model, rollout_weight
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            selected = anchors[start : start + batch_size]
            batch = extract_gatec_batch(caches, selected, window=window, horizon=horizon, validate_pair=False)
            history, future_sp, targets = _tensor_batch(batch, device)
            output = model(history, future_sp)
            common_loss = sum(
                torch.nn.functional.smooth_l1_loss(
                    output[PREDICTION_KEYS[key]] / float(target_scales[key]),
                    targets[key] / float(target_scales[key]),
                )
                for key in TASKS
            ) / len(TASKS)
            total += float(common_loss.cpu()) * len(selected)
    model.train()
    return total / len(anchors)


def _report(
    model: RM3AVModel,
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    *,
    diagnostic_count: int,
    window: int,
    horizon: int,
    batch_size: int,
    candidate_id: str,
    target_scales: Mapping[str, float],
    r_model: OOFRModelSet | None,
    rollout_weight: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    predictions = {key: [] for key in TASKS}
    targets_all = {key: [] for key in TASKS}
    baselines = {key: [] for key in TASKS}
    rollout_predictions = {key: [] for key in TASKS}
    rollout_teacher_forced = {key: [] for key in TASKS}
    rollout_targets = {key: [] for key in TASKS}
    rollout_baselines = {key: [] for key in TASKS}
    timestamps: list[np.ndarray] = []
    anchors_all: list[np.ndarray] = []
    contexts_all: list[np.ndarray] = []
    mode_sums = {mode: {"terminal": 0.0, "local": 0.0, "n": 0} for mode in (
        "normal", "bypass_off", "bypass_only", "response_off", "predicted_valve",
        "logged_valve", "logged_valve_oracle_tin", "oracle_local", "shuffled",
        "wrong_side", "lead",
    )}
    explicit_effects: list[np.ndarray] = []
    identity_effects: list[np.ndarray] = []
    diagnostic_local_predictions: list[np.ndarray] = []
    diagnostic_local_baselines: list[np.ndarray] = []
    stable_poles: np.ndarray | None = None
    finite_difference_sums = {
        coordinate: {
            str(point): {
                str(step): {
                    "local_signed_side_sum": np.zeros(2, dtype=np.float64),
                    "terminal_signed_side_sum": np.zeros(2, dtype=np.float64),
                    "count": 0,
                }
                for step in FINITE_DIFFERENCE_HORIZONS
            }
            for point in FINITE_DIFFERENCE_VALVE_POINTS
        }
        for coordinate in ("A_only", "B_only", "common", "differential")
    }
    branch_invariance = {
        "free_residual_logged_vs_shuffled_max_abs": 0.0,
        "terminal_bypass_logged_vs_shuffled_max_abs": 0.0,
        "free_residual_comparison_count": 0,
        "terminal_bypass_comparison_count": 0,
    }
    alignment_sums = {
        str(seconds): {"local": 0.0, "terminal": 0.0, "n": 0}
        for seconds in (-30, -20, -10, 0, 10, 20, 30)
    }
    boundary_placebo_sums = {
        mode: {"local": 0.0, "terminal": 0.0, "n": 0}
        for mode in ("logged_tin_correct_side", "logged_tin_wrong_side", "logged_tin_lead")
    }
    diagnostic_anchors = set(
        deterministic_anchor_subset(
            anchors,
            min(diagnostic_count, len(anchors)),
            36700,
        ).tolist()
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            selected = anchors[start : start + batch_size]
            batch = extract_gatec_batch(caches, selected, window=window, horizon=horizon, validate_pair=False)
            second = (
                extract_gatec_batch(caches, selected + horizon, window=window, horizon=horizon, validate_pair=False)
                if candidate_id == "C31" else None
            )
            output, targets, _ = _forward_loss(
                model, batch, device, candidate_id=candidate_id,
                target_scales=target_scales, r_model=r_model,
                groups=_groups(caches, selected), rollout_weight=rollout_weight,
                second_batch=second,
            )
            if candidate_id == "C31":
                assert second is not None
                second_history, second_sp, second_targets = _tensor_batch(second, device)
                recursive = output["two_window_second"]
                teacher_forced = model(second_history, second_sp)
                for key in TASKS:
                    rollout_predictions[key].append(
                        recursive[PREDICTION_KEYS[key]].cpu().numpy().astype(np.float32)
                    )
                    rollout_teacher_forced[key].append(
                        teacher_forced[PREDICTION_KEYS[key]].cpu().numpy().astype(np.float32)
                    )
                    rollout_targets[key].append(
                        second_targets[key].cpu().numpy().astype(np.float32)
                    )
                rollout_baselines["valve"].append(
                    second_history[:, -1, model.valve_indices].cpu().numpy()
                )
                rollout_baselines["tin"].append(
                    second_history[:, -1, model.tin_indices].cpu().numpy()
                )
                rollout_baselines["local"].append(
                    (
                        second_history[:, -1, model.tin_indices]
                        - second_history[:, -1, model.tout_indices]
                    ).cpu().numpy()
                )
                rollout_baselines["terminal"].append(
                    second_history[:, -1, model.terminal_indices].cpu().numpy()
                )
            for key in TASKS:
                predictions[key].append(output[PREDICTION_KEYS[key]].cpu().numpy().astype(np.float32))
                targets_all[key].append(targets[key].cpu().numpy().astype(np.float32))
            history = torch.as_tensor(batch.history, dtype=torch.float32, device=device)
            baselines["valve"].append(history[:, -1, model.valve_indices].cpu().numpy())
            baselines["tin"].append(history[:, -1, model.tin_indices].cpu().numpy())
            baselines["local"].append(
                (history[:, -1, model.tin_indices] - history[:, -1, model.tout_indices]).cpu().numpy()
            )
            baselines["terminal"].append(history[:, -1, model.terminal_indices].cpu().numpy())
            timestamps.append(caches["A"].timestamps_ns[selected].astype(np.int64))
            anchors_all.append(selected.astype(np.int64))
            contexts_all.append(batch.history[:, -1].astype(np.float32))

            mask = np.asarray([anchor in diagnostic_anchors for anchor in selected])
            if mask.any():
                indices = np.flatnonzero(mask)
                modes = model.diagnostic_forward(
                    history[indices],
                    torch.as_tensor(batch.future_sp[indices], device=device),
                    logged_future_valve=torch.as_tensor(batch.logged_future_valve[indices], device=device),
                    logged_future_tin=torch.as_tensor(batch.logged_future_tin[indices], device=device),
                    local_target=torch.as_tensor(batch.local_drop_target[indices], device=device),
                )
                aligned_modes = model.action_alignment_sensitivity(
                    history[indices],
                    torch.as_tensor(batch.future_sp[indices], device=device),
                    logged_future_valve=torch.as_tensor(batch.logged_future_valve[indices], device=device),
                    logged_future_tin=torch.as_tensor(batch.logged_future_tin[indices], device=device),
                    local_target=torch.as_tensor(batch.local_drop_target[indices], device=device),
                )
                logged_tin_tensor = torch.as_tensor(batch.logged_future_tin[indices], device=device)
                logged_valve_tensor = torch.as_tensor(batch.logged_future_valve[indices], device=device)
                diagnostic_sp = torch.as_tensor(batch.future_sp[indices], device=device)
                boundary_modes = {
                    "logged_tin_correct_side": model._diagnostic_action_forward(
                        history[indices], diagnostic_sp, logged_valve_tensor,
                        boundary_tin=logged_tin_tensor,
                    ),
                    "logged_tin_wrong_side": model._diagnostic_action_forward(
                        history[indices], diagnostic_sp, logged_valve_tensor,
                        boundary_tin=logged_tin_tensor.flip(-1),
                    ),
                    "logged_tin_lead": model._diagnostic_action_forward(
                        history[indices], diagnostic_sp, logged_valve_tensor,
                        boundary_tin=torch.cat(
                            (logged_tin_tensor[:, 1:], logged_tin_tensor[:, -1:]), dim=1
                        ),
                    ),
                }
                if (
                    "residual_local_prediction" in modes["logged_valve"]
                    and "residual_local_prediction" in modes["shuffled"]
                ):
                    branch_invariance["free_residual_logged_vs_shuffled_max_abs"] = max(
                        branch_invariance["free_residual_logged_vs_shuffled_max_abs"],
                        float((
                            modes["logged_valve"]["residual_local_prediction"]
                            - modes["shuffled"]["residual_local_prediction"]
                        ).abs().max().cpu()),
                    )
                    branch_invariance["free_residual_comparison_count"] += len(indices)
                if (
                    "terminal_bypass" in modes["logged_valve"]
                    and "terminal_bypass" in modes["shuffled"]
                ):
                    branch_invariance["terminal_bypass_logged_vs_shuffled_max_abs"] = max(
                        branch_invariance["terminal_bypass_logged_vs_shuffled_max_abs"],
                        float((
                            modes["logged_valve"]["terminal_bypass"]
                            - modes["shuffled"]["terminal_bypass"]
                        ).abs().max().cpu()),
                    )
                    branch_invariance["terminal_bypass_comparison_count"] += len(indices)
                explicit_effects.append(
                    modes["normal"]["explicit_local_effect"].cpu().numpy().astype(np.float32)
                )
                baseline_valve = history[indices][:, -1][:, model.valve_indices]
                baseline_local = (
                    history[indices][:, -1][:, model.tin_indices]
                    - history[indices][:, -1][:, model.tout_indices]
                )
                diagnostic_local_predictions.append(
                    modes["normal"]["local_drop_prediction"].cpu().numpy().astype(np.float32)
                )
                diagnostic_local_baselines.append(baseline_local.cpu().numpy().astype(np.float32))
                constant_valve = baseline_valve[:, None].expand(-1, horizon, -1)
                response_context = (
                    model._p5_context(history[indices])
                    if model.base_candidate_id == "P5_hybrid_joint_latent"
                    else model.base.model.encoder(
                        (history[indices] - model.base.model.history_center)
                        / model.base.model.history_scale
                    )
                )
                constant_response = model.explicit_response(
                    response_context, constant_valve, baseline_valve
                )
                identity_effects.append(
                    constant_response["effect"].cpu().numpy().astype(np.float32)
                )
                constant_terminal = model._diagnostic_action_forward(
                    history[indices],
                    torch.as_tensor(batch.future_sp[indices], device=device),
                    constant_valve,
                )["terminal_prediction"]
                for coordinate, direction in (
                    ("A_only", (1.0, 0.0)),
                    ("B_only", (0.0, 1.0)),
                    ("common", (1.0, 1.0)),
                    ("differential", (1.0, -1.0)),
                ):
                    direction_tensor = torch.as_tensor(
                        direction, dtype=constant_valve.dtype, device=device
                    ).view(1, 1, 2)
                    for point in FINITE_DIFFERENCE_VALVE_POINTS:
                        intervened_valve = constant_valve + float(point) * direction_tensor
                        response = model.explicit_response(
                            response_context, intervened_valve, baseline_valve
                        )["effect"]
                        intervened_terminal = model._diagnostic_action_forward(
                            history[indices],
                            torch.as_tensor(batch.future_sp[indices], device=device),
                            intervened_valve,
                        )["terminal_prediction"]
                        terminal_delta = intervened_terminal - constant_terminal
                        for step in FINITE_DIFFERENCE_HORIZONS:
                            endpoint = min(step, horizon) - 1
                            row = finite_difference_sums[coordinate][str(point)][str(step)]
                            row["local_signed_side_sum"] += (
                                response[:, endpoint].sum(dim=0).cpu().numpy()
                            )
                            row["terminal_signed_side_sum"] += (
                                terminal_delta[:, endpoint].sum(dim=0).cpu().numpy()
                            )
                            row["count"] += len(indices)
                poles_tensor = constant_response["stable_poles"]
                stable_poles = poles_tensor.detach().cpu().numpy()
                terminal_target = torch.as_tensor(batch.terminal_target[indices], device=device)
                local_target = torch.as_tensor(batch.local_drop_target[indices], device=device)
                for mode, value in modes.items():
                    mode_sums[mode]["terminal"] += float(
                        (value["terminal_prediction"] - terminal_target).abs().sum().cpu()
                    )
                    mode_sums[mode]["local"] += float(
                        (value["local_drop_prediction"] - local_target).abs().sum().cpu()
                    )
                    mode_sums[mode]["n"] += int(terminal_target.numel())
                for seconds, value in aligned_modes.items():
                    alignment_sums[seconds]["terminal"] += float(
                        (value["terminal_prediction"] - terminal_target).abs().sum().cpu()
                    )
                    alignment_sums[seconds]["local"] += float(
                        (value["local_drop_prediction"] - local_target).abs().sum().cpu()
                    )
                    alignment_sums[seconds]["n"] += int(terminal_target.numel())
                for mode, value in boundary_modes.items():
                    boundary_placebo_sums[mode]["terminal"] += float(
                        (value["terminal_prediction"] - terminal_target).abs().sum().cpu()
                    )
                    boundary_placebo_sums[mode]["local"] += float(
                        (value["local_drop_prediction"] - local_target).abs().sum().cpu()
                    )
                    boundary_placebo_sums[mode]["n"] += int(terminal_target.numel())
    arrays = {
        f"{key}_prediction": np.concatenate(predictions[key]) for key in TASKS
    }
    arrays.update({f"{key}_target": np.concatenate(targets_all[key]) for key in TASKS})
    arrays.update({f"{key}_baseline": np.concatenate(baselines[key]).astype(np.float32) for key in TASKS})
    arrays["anchors"] = np.concatenate(anchors_all)
    arrays["timestamps_ns"] = np.concatenate(timestamps)
    arrays["history_last"] = np.concatenate(contexts_all)
    if candidate_id == "C31":
        for key in TASKS:
            arrays[f"rollout_second_{key}_prediction"] = np.concatenate(
                rollout_predictions[key]
            )
            arrays[f"teacher_forced_second_{key}_prediction"] = np.concatenate(
                rollout_teacher_forced[key]
            )
            arrays[f"rollout_second_{key}_target"] = np.concatenate(rollout_targets[key])
            arrays[f"rollout_second_{key}_baseline"] = np.concatenate(
                rollout_baselines[key]
            ).astype(np.float32)
    metrics = {
        f"{key}_mae" + ("_c" if key != "valve" else ""): float(
            np.mean(np.abs(arrays[f"{key}_prediction"] - arrays[f"{key}_target"]))
        )
        for key in TASKS
    }
    mode_records = {
        mode: {
            "status": "computed",
            "terminal_mae_c": values["terminal"] / values["n"],
            "local_mae_c": values["local"] / values["n"],
            "element_count": values["n"],
        }
        for mode, values in mode_sums.items()
    }
    diagnostics = build_prediction_diagnostics(
        {key: arrays[f"{key}_prediction"] for key in TASKS},
        {key: arrays[f"{key}_target"] for key in TASKS},
        {key: arrays[f"{key}_baseline"] for key in TASKS},
        mode_records=mode_records,
    )
    logged = arrays["valve_target"]
    valve_baseline = arrays["valve_baseline"]
    diagnostics.update({
        "candidate_id": candidate_id,
        "raw_valve_change_rank": valve_innovation_rank(
            (logged - valve_baseline[:, None]).reshape(-1, 2)
        ),
        "raw_valve_change_dependence": dependence_diagnostics(
            (logged - valve_baseline[:, None]).reshape(-1, 2)
        ),
        "mechanism_prediction_residual_dependence": mechanism_residual_dependence({
            key: arrays[f"{key}_target"] - arrays[f"{key}_prediction"]
            for key in TASKS
        }),
        "state_closure": build_state_closure_audit(
            generated={"valve", "tin", "tout", "terminal"},
            declared_external={"sp", "load", "pressure", "feedwater", "coal", "steam"},
            required={"valve", "tin", "tout", "terminal", "sp", "load", "pressure", "feedwater", "coal", "steam"},
        ),
        "assumption_ledger": build_assumption_ledger(),
        "manual_audit_verdicts": build_manual_verdict_template(),
        "claim_boundary": {
            "model_champion": False,
            "causal_identification": False,
            "arbitrary_do_valve": False,
            "state_closed_simulator": False,
        },
        "test_accessed": False,
        "automatic_scientific_pass": None,
    })
    diagnostics["valve_trajectory"] = valve_trajectory_diagnostics(
        arrays["valve_prediction"], arrays["valve_target"], arrays["valve_baseline"]
    )
    if explicit_effects:
        response_values = np.concatenate(explicit_effects)
        local_values = np.concatenate(diagnostic_local_predictions)
        local_baseline_values = np.concatenate(diagnostic_local_baselines)[:, None]
        diagnostics["response_trajectory"] = response_trajectory_diagnostics(
            response_values,
            constant_action_effect=np.concatenate(identity_effects),
            stable_poles=np.asarray([] if stable_poles is None else stable_poles),
        )
        total_local_magnitude = float(np.mean(np.abs(local_values - local_baseline_values)))
        explicit_magnitude = float(np.mean(np.abs(response_values)))
        diagnostics["response_trajectory"]["explicit_to_total_local_change_ratio"] = (
            explicit_magnitude / total_local_magnitude
            if total_local_magnitude > 0.0 else None
        )
        diagnostics["response_trajectory"]["ratio_is_descriptive_not_identified"] = True
    else:
        diagnostics["response_trajectory"] = {
            "status": "not_applicable",
            "reason": "no diagnostic anchors",
        }
    diagnostics["finite_difference_response"] = {
        "coordinate_definition": {
            "A_only": "delta_v_A!=0, delta_v_B=0",
            "B_only": "delta_v_A=0, delta_v_B!=0",
            "common": "delta_v_A=delta_v_B",
            "differential": "delta_v_A=-delta_v_B",
        },
        "perturbation_points_percent_valve": list(FINITE_DIFFERENCE_VALVE_POINTS),
        "endpoint_horizons_steps": list(FINITE_DIFFERENCE_HORIZONS),
        "responses": {
            coordinate: {
                point: {
                    step: {
                        "explicit_local_signed_by_side": (
                            row["local_signed_side_sum"] / row["count"]
                        ).tolist(),
                        "terminal_signed_delta_by_side": (
                            row["terminal_signed_side_sum"] / row["count"]
                        ).tolist(),
                        "anchor_count": row["count"],
                    }
                    for step, row in point_rows.items()
                }
                for point, point_rows in coordinate_rows.items()
            }
            for coordinate, coordinate_rows in finite_difference_sums.items()
        },
        "test_accessed": False,
        "causal_identification_claim": False,
    }
    diagnostics["branch_action_invariance"] = {
        **branch_invariance,
        "free_residual_action_invariant": bool(
            branch_invariance["free_residual_comparison_count"] == 0
            or branch_invariance["free_residual_logged_vs_shuffled_max_abs"] <= 1e-7
        ),
        "terminal_bypass_action_invariant": bool(
            branch_invariance["terminal_bypass_comparison_count"] == 0
            or branch_invariance["terminal_bypass_logged_vs_shuffled_max_abs"] <= 1e-7
        ),
        "zero_comparison_means_not_applicable": True,
        "semantic_physics_claim": False,
    }
    if candidate_id == "C31":
        rollout_target_map = {
            key: arrays[f"rollout_second_{key}_target"] for key in TASKS
        }
        rollout_baseline_map = {
            key: arrays[f"rollout_second_{key}_baseline"] for key in TASKS
        }
        diagnostics["two_window_rollout"] = {
            "recursive_second_window": build_prediction_diagnostics(
                {
                    key: arrays[f"rollout_second_{key}_prediction"]
                    for key in TASKS
                },
                rollout_target_map,
                rollout_baseline_map,
            ),
            "teacher_forced_second_window": build_prediction_diagnostics(
                {
                    key: arrays[f"teacher_forced_second_{key}_prediction"]
                    for key in TASKS
                },
                rollout_target_map,
                rollout_baseline_map,
            ),
            "recursive_horizon_seconds": 1200,
            "reported_endpoints_seconds": [600, 1200],
            "unavailable_endpoints_seconds": [1800, 3600],
            "unavailable_reason": (
                "the frozen AV1 candidate carries state for two H60 windows only; "
                "30/60-minute simulation remains NOT_TESTABLE"
            ),
            "uses_true_future_temperature_in_recursive_second_history": False,
            "uses_declared_held_context_in_recursive_second_history": True,
            "state_closed_simulator_claim": False,
        }
    else:
        diagnostics["two_window_rollout"] = {
            "status": "not_applicable",
            "reason": "candidate does not train the declared two-window intervention",
            "state_closed_simulator_claim": False,
        }
    timestamps_ns = arrays["timestamps_ns"]
    dates = np.floor_divide(timestamps_ns, DAY_NS)
    terminal_target = arrays["terminal_target"]
    terminal_prediction = arrays["terminal_prediction"]
    load_index = model.feature_names.index("机组负荷")
    reporting_history = []
    for start in range(0, len(anchors), batch_size):
        selected = anchors[start : start + batch_size]
        reporting_history.append(
            extract_gatec_batch(
                caches, selected, window=window, horizon=horizon, validate_pair=False
            ).history[:, -1, load_index]
        )
    load = np.concatenate(reporting_history)
    load_threshold = float(np.median(load))
    valve_activity = np.mean(
        np.abs(arrays["valve_target"] - arrays["valve_baseline"][:, None]), axis=(1, 2)
    )
    activity_threshold = float(np.median(valve_activity))
    diagnostics["terminal_strata"] = stratified_error_diagnostics(
        terminal_prediction,
        terminal_target,
        {
            "utc_date": dates,
            "load_bin": (load > load_threshold).astype(int),
            "valve_activity_bin": (valve_activity > activity_threshold).astype(int),
        },
    )
    diagnostics["action_alignment_sensitivity_seconds"] = {
        seconds: {
            "local_mae_c": values["local"] / values["n"],
            "terminal_mae_c": values["terminal"] / values["n"],
            "element_count": values["n"],
        }
        for seconds, values in alignment_sums.items()
    }
    diagnostics["boundary_tin_placebo"] = {
        **{
            mode: {
                "local_mae_c": values["local"] / values["n"],
                "terminal_mae_c": values["terminal"] / values["n"],
                "element_count": values["n"],
            }
            for mode, values in boundary_placebo_sums.items()
        },
        "causal_boundary_claim": False,
    }
    sp_indices = (
        model.feature_names.index("A::二级减温调节阀设定"),
        model.feature_names.index("B::二级减温调节阀设定"),
    )
    temperature_indices = tuple(model.terminal_indices)
    reporting_batches = [
        extract_gatec_batch(
            caches,
            anchors[start : start + batch_size],
            window=window,
            horizon=horizon,
            validate_pair=False,
        )
        for start in range(0, len(anchors), batch_size)
    ]
    probe_groups = np.floor_divide(timestamps_ns, DAY_NS)
    reporting_history_full = np.concatenate([batch.history for batch in reporting_batches])
    reporting_sp_full = np.concatenate([batch.future_sp for batch in reporting_batches])
    reporting_action_change = (
        np.concatenate([batch.logged_future_valve for batch in reporting_batches])
        - arrays["valve_baseline"][:, None]
    )
    reporting_local_change = (
        np.concatenate([batch.local_drop_target for batch in reporting_batches])
        - arrays["local_baseline"][:, None]
    )
    if len(np.unique(probe_groups)) < 2:
        diagnostics["valve_policy_probes"] = {
            "status": "not_applicable",
            "reason": "reporting anchors contain fewer than two independent UTC days",
            "causal_direction_claim": False,
        }
        diagnostics["residualized_valve_innovation"] = {
            "status": "not_applicable",
            "reason": "reporting anchors contain fewer than two independent UTC days",
            "causal_identification_claim": False,
        }
    else:
        diagnostics["valve_policy_probes"] = valve_policy_probe_diagnostics(
            reporting_history_full,
            reporting_sp_full,
            np.concatenate([batch.logged_future_valve for batch in reporting_batches]),
            probe_groups,
            sp_indices=sp_indices,
            temperature_indices=temperature_indices,
            ridge=1e-3,
        )
        innovation_audit = fit_oof_action_outcome_audit(
            reporting_history_full,
            reporting_sp_full,
            reporting_action_change,
            reporting_local_change,
            probe_groups,
            ridge=1e-3,
        )
        diagnostics["residualized_valve_innovation"] = {
            "status": "computed",
            "conditioning_information": (
                "per_timestep_history_last_and_mean_plus_future_sp_level_delta_integral"
            ),
            "group_unit": "UTC_day",
            "fold_records": list(innovation_audit.fold_records),
            "action_prediction_r2_by_side": {
                "A": innovation_audit.action_r2_by_side[0],
                "B": innovation_audit.action_r2_by_side[1],
            },
            "local_prediction_r2_by_side": {
                "A": innovation_audit.outcome_r2_by_side[0],
                "B": innovation_audit.outcome_r2_by_side[1],
            },
            "rank": valve_innovation_rank(
                innovation_audit.action_innovation.reshape(-1, 2)
            ),
            "dependence": dependence_diagnostics(
                innovation_audit.action_innovation.reshape(-1, 2)
            ),
            "action_local_innovation_cross_covariance": np.cov(
                np.concatenate(
                    (
                        innovation_audit.action_innovation.reshape(-1, 2),
                        innovation_audit.outcome_innovation.reshape(-1, 2),
                    ),
                    axis=1,
                ),
                rowvar=False,
            )[:2, 2:].tolist(),
            "causal_identification_claim": False,
        }
    unique_days = np.unique(probe_groups)
    if len(unique_days) < 2:
        diagnostics["daily_gain_context_activity"] = {
            "status": "not_applicable",
            "reason": "fewer than two UTC days",
            "causal_gain_explanation": False,
        }
    else:
        daily_gain = []
        daily_context = []
        daily_activity = []
        daily_groups = []
        shared_names = ("机组负荷", "主蒸汽压力", "未校正总煤量")
        shared_indices = [model.feature_names.index(name) for name in shared_names]
        for day in unique_days:
            mask = probe_groups == day
            action_delta = innovation_audit.action_innovation[mask].reshape(-1, 2)
            local_delta = innovation_audit.outcome_innovation[mask].reshape(-1, 2)
            gain_side = []
            for side in range(2):
                denominator = float(np.dot(action_delta[:, side], action_delta[:, side])) + 1e-6
                gain_side.append(float(np.dot(action_delta[:, side], local_delta[:, side]) / denominator))
            daily_gain.append(gain_side)
            daily_context.append(np.mean(arrays["history_last"][mask][:, shared_indices], axis=0))
            daily_activity.append(np.mean(np.abs(action_delta), axis=0))
            daily_groups.append(day)
        diagnostics["daily_gain_context_activity"] = daily_gain_context_diagnostics(
            gain=np.asarray(daily_gain),
            context=np.asarray(daily_context),
            activity=np.asarray(daily_activity),
            groups=np.asarray(daily_groups),
            context_names=shared_names,
            ridge=1e-3,
        )
    return metrics, arrays, diagnostics


def run_rm3av_training(
    caches: Mapping[str, Phase35Cache],
    matrix: Mapping[str, Any],
    spec: RM3AVRunSpec,
    *,
    device: str,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"RM3-AV refuses existing non-empty run directory: {output_dir}")
    _set_seed(spec.seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise Phase35ProtocolError("RM3-AV requested unavailable CUDA")
    data, training = matrix["data_contract"], matrix["training"]
    window, horizon = int(data["window_steps"]), int(data["horizon_steps"])
    n_rows = len(caches["A"].timestamps_ns)
    actual_test_start = caches["A"].split_bounds()["test"][0]
    train_bounds = _bounds(n_rows, spec.train_fraction)
    validation_raw = _bounds(n_rows, spec.validation_fraction)
    validation_bounds = (validation_raw[0], min(validation_raw[1], actual_test_start))
    if not 0 <= train_bounds[0] < train_bounds[1] <= validation_bounds[0] < validation_bounds[1] <= actual_test_start:
        raise Phase35ProtocolError("RM3-AV fold touches test or breaks chronology")
    train_pool = _candidate_pool(
        caches, "train", bounds=train_bounds, window=window, horizon=horizon,
        max_age_s=float(data["max_age_s"]), candidate_id=spec.candidate_id,
    )
    validation_pool = _candidate_pool(
        caches, "validation", bounds=validation_bounds, window=window, horizon=horizon,
        max_age_s=float(data["max_age_s"]), candidate_id=spec.candidate_id,
    )
    stats_anchors = deterministic_anchor_subset(
        train_pool, int(training["stats_anchor_count"]), 36400 + int(spec.fold_id[1:])
    )
    selector_anchors, reporting_anchors, selector_days, reporting_days = _split_validation(
        validation_pool,
        caches["A"].timestamps_ns,
        int(training["selector_anchor_count"]),
        int(training["reporting_anchor_count"]),
        spec.fold_id,
    )
    stats_batch = extract_gatec_batch(caches, stats_anchors, window=window, horizon=horizon, validate_pair=False)
    center, scale, target_scales = fit_rm3_train_statistics(stats_batch)
    config = RM3AVModelConfig(
        candidate_id=spec.candidate_id,
        window=window,
        horizon=horizon,
        n_features=stats_batch.history.shape[-1],
        d_model=int(matrix["model"]["d_model"]),
        latent_dim=int(matrix["model"]["latent_dim"]),
        dropout=float(matrix["model"]["dropout"]),
    )
    model = build_rm3av_model(config, stats_batch.history_feature_names).to(torch_device)
    initialization_hashes = module_state_hashes(model)
    model.set_history_normalization(
        torch.from_numpy(center).to(torch_device), torch.from_numpy(scale).to(torch_device)
    )
    nuisance_metadata: dict[str, Any] = {"fitted_on": None}
    r_model: OOFRModelSet | None = None
    train_groups = _groups(caches, stats_anchors)
    if spec.candidate_id == "C09":
        projection: OOFActionProjection = fit_oof_action_projection(
            stats_batch.history,
            stats_batch.logged_future_valve - stats_batch.history[:, -1, model.valve_indices][:, None],
            train_groups,
            ridge=1e-3,
        )
        model.set_action_shield(torch.from_numpy(projection.projector).to(torch_device))
        nuisance_metadata = {
            "fitted_on": "train_stats_anchors_only",
            "kind": "OOF_action_projection",
            "group_unit": "UTC_day",
            "fold_count": len(projection.fold_records),
        }
    if spec.candidate_id in {"C11", "C12"}:
        r_model, r_residuals = fit_oof_r_model(
            stats_batch.history,
            stats_batch.logged_future_valve - stats_batch.history[:, -1, model.valve_indices][:, None],
            stats_batch.local_drop_target
            - (
                stats_batch.history[:, -1, model.tin_indices]
                - stats_batch.history[:, -1, model.tout_indices]
            )[:, None],
            train_groups,
            ridge=1e-3,
            future_sp=stats_batch.future_sp,
        )
        nuisance_metadata = {
            "fitted_on": "train_stats_anchors_only",
            "kind": "OOF_R_loss",
            "group_unit": "UTC_day",
            "fold_count": len(r_residuals.fold_records),
            "conditioning_includes_future_sp": r_model.uses_future_sp,
            "action_definition": "logged_future_valve_minus_history_last_valve",
            "outcome_definition": "future_local_drop_minus_history_last_local_drop",
        }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    update_cap = int(spec.optimizer_updates_cap)
    if update_cap == 4000:
        update_cap = int(training.get("default_optimizer_updates_cap", update_cap))
    rng = np.random.default_rng(36500 + int(spec.fold_id[1:]) * 100 + spec.seed)
    best_score, best_update, best_state = math.inf, 0, None
    selector_history: list[dict[str, Any]] = []
    loss_curve: list[float] = []
    loss_component_curve: list[dict[str, float]] = []
    gradient_audit: dict[str, Any] | None = None
    started = time.perf_counter()
    model.train()
    for update in range(1, update_cap + 1):
        chosen = train_pool[rng.integers(0, len(train_pool), size=int(training["batch_size"]))]
        batch = extract_gatec_batch(caches, chosen, window=window, horizon=horizon, validate_pair=False)
        second = (
            extract_gatec_batch(caches, chosen + horizon, window=window, horizon=horizon, validate_pair=False)
            if spec.candidate_id == "C31" else None
        )
        optimizer.zero_grad(set_to_none=True)
        _, _, losses = _forward_loss(
            model, batch, torch_device, candidate_id=spec.candidate_id,
            target_scales=target_scales, r_model=r_model, groups=_groups(caches, chosen),
            rollout_weight=float(training["rollout_weight"]), second_batch=second,
        )
        loss = losses["total"]
        if not torch.isfinite(loss):
            raise Phase35ProtocolError("RM3-AV training loss became non-finite")
        loss.backward()
        if gradient_audit is None:
            response_before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if "local_response" in name
            }
            gradient_audit = {
                "response_trainable_parameter_count": sum(
                    parameter.numel()
                    for name, parameter in model.named_parameters()
                    if "local_response" in name and parameter.requires_grad
                ),
                "response_nonzero_gradient_tensor_count": sum(
                    1
                    for name, parameter in model.named_parameters()
                    if "local_response" in name
                    and parameter.grad is not None
                    and bool(torch.count_nonzero(parameter.grad).item())
                ),
                "logged_future_valve_role": (
                    "training_auxiliary_only"
                    if spec.candidate_id in {"C10", "C11", "C12", "C13"}
                    else "not_provided"
                ),
                "forecast_path_reads_logged_future_valve": False,
                "parameter_before": response_before,
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        if update == 1 and gradient_audit is not None:
            before = gradient_audit.pop("parameter_before")
            gradient_audit["response_parameter_delta_l1_after_one_update"] = float(
                sum(
                    (parameter.detach().cpu() - before[name]).abs().sum()
                    for name, parameter in model.named_parameters()
                    if name in before
                )
            )
            gradient_audit["response_training_path_reachable"] = bool(
                gradient_audit["response_nonzero_gradient_tensor_count"] > 0
                and gradient_audit["response_parameter_delta_l1_after_one_update"] > 0.0
            )
        loss_curve.append(float(loss.detach().cpu()))
        loss_component_curve.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        if update % int(training["evaluation_interval_updates"]):
            continue
        score = _evaluate(
            model, caches, selector_anchors, window=window, horizon=horizon,
            batch_size=int(training["evaluation_batch_size"]), candidate_id=spec.candidate_id,
            target_scales=target_scales, r_model=r_model,
            rollout_weight=float(training["rollout_weight"]), device=torch_device,
        )
        selector_history.append({"update": update, "validation_full_multitask_score": score})
        if score < best_score - float(training["minimum_score_improvement"]):
            best_score, best_update = score, update
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise Phase35ProtocolError("RM3-AV produced no validation checkpoint")
    model.load_state_dict(best_state)
    metrics, episodes, diagnostics = _report(
        model, caches, reporting_anchors,
        diagnostic_count=int(training["diagnostic_anchor_count"]),
        window=window, horizon=horizon, batch_size=int(training["evaluation_batch_size"]),
        candidate_id=spec.candidate_id, target_scales=target_scales,
        r_model=r_model, rollout_weight=float(training["rollout_weight"]), device=torch_device,
    )
    diagnostics["convergence"] = convergence_diagnostics(
        loss_curve, best_update=best_update, update_cap=update_cap
    )
    diagnostics["training_graph"] = gradient_audit or {
        "response_training_path_reachable": False,
        "reason": "no optimizer update completed",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_best_validation.pt"
    _atomic_checkpoint(checkpoint_path, {
        "protocol_version": matrix["protocol_version"],
        "run_spec": asdict(spec),
        "model_config": asdict(config),
        "feature_names": list(stats_batch.history_feature_names),
        "model_state_dict": best_state,
        "history_center": center,
        "history_scale": scale,
        "target_scales": target_scales,
        "best_update": best_update,
        "best_selector_score": best_score,
        "nuisance_metadata": nuisance_metadata,
        "oof_r_model": r_model,
        "initialization_hashes": initialization_hashes,
    })
    _atomic_npz(output_dir / "episodes_validation.npz", episodes)
    _atomic_json(output_dir / "diagnostics_validation.json", diagnostics)
    metrics_payload = {
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
        "metrics": metrics,
        "selector_history": selector_history,
        "loss_curve": loss_curve,
        "loss_component_curve": loss_component_curve,
        "best_update": best_update,
        "best_selector_score": best_score,
        "optimizer_updates_completed": len(loss_curve),
        "test_accessed": False,
        "automatic_scientific_pass": None,
    }
    _atomic_json(output_dir / "metrics_validation.json", metrics_payload)
    manifest = {
        "protocol_version": matrix["protocol_version"],
        "run_id": spec.run_id,
        "run_spec": asdict(spec),
        "provenance": dict(provenance),
        "nuisance_metadata": nuisance_metadata,
        "initialization_hashes": initialization_hashes,
        "train_anchor_pool_count": len(train_pool),
        "stats_anchor_count": len(stats_anchors),
        "selector_anchor_count": len(selector_anchors),
        "reporting_anchor_count": len(reporting_anchors),
        "diagnostic_anchor_count": min(int(training["diagnostic_anchor_count"]), len(reporting_anchors)),
        "selector_reporting_disjoint": bool(not np.intersect1d(selector_anchors, reporting_anchors).size),
        "selector_reporting_utc_day_disjoint": bool(
            not set(selector_days.tolist()) & set(reporting_days.tolist())
        ),
        "selector_utc_days": selector_days.tolist(),
        "reporting_utc_days": reporting_days.tolist(),
        "selector_anchor_sha256": hashlib.sha256(np.asarray(selector_anchors, dtype="<i8").tobytes()).hexdigest(),
        "reporting_anchor_sha256": hashlib.sha256(np.asarray(reporting_anchors, dtype="<i8").tobytes()).hexdigest(),
        "checkpoint_selector": "validation_full_multitask_common_four_task_loss",
        "early_stopping_enabled": False,
        "checkpoint_sha256": _sha(checkpoint_path),
        "maximum_attempts_per_run": 1,
        "test_accessed": False,
        "automatic_scientific_pass": None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    artifact_names = {
        "manifest.json",
        "checkpoint_best_validation.pt",
        "metrics_validation.json",
        "episodes_validation.npz",
        "diagnostics_validation.json",
    }
    _atomic_json(
        output_dir / "artifact_ledger.json",
        {name: _sha(output_dir / name) for name in sorted(artifact_names)},
    )
    return {"run_id": spec.run_id, "status": "complete", "metrics": metrics_payload}
