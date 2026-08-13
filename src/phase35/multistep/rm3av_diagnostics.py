"""Common validation diagnostics for RM3-AV replay and trained candidates."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..schema import Phase35ProtocolError


REQUIRED_DIAGNOSTIC_MODES = (
    "normal",
    "bypass_off",
    "bypass_only",
    "response_off",
    "predicted_valve",
    "logged_valve",
    "logged_valve_oracle_tin",
    "oracle_local",
    "shuffled",
    "wrong_side",
    "lead",
)

ASSUMPTION_LEDGER_METHODS = (
    "CD-NOD",
    "Nonstationary linear SEM",
    "TDRL",
    "CtrlNS",
    "IDOL",
    "CaRiNG",
    "LEAP",
)


def _finite_three_dimensional(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 2 or not np.isfinite(array).all():
        raise Phase35ProtocolError(f"RM3-AV {label} must be finite [episode,horizon,side]")
    return array


def _validate_modes(mode_records: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    if mode_records is None:
        return {
            mode: {"status": "not_applicable", "reason": "not supplied to this diagnostic call"}
            for mode in REQUIRED_DIAGNOSTIC_MODES
        }
    if set(mode_records) != set(REQUIRED_DIAGNOSTIC_MODES):
        raise Phase35ProtocolError("RM3-AV diagnostic mode fields are incomplete")
    result = {}
    for mode, record in mode_records.items():
        status = record.get("status")
        if status not in {"computed", "not_applicable"}:
            raise Phase35ProtocolError(f"RM3-AV mode {mode} status is invalid")
        if status == "not_applicable" and not record.get("reason"):
            raise Phase35ProtocolError(f"RM3-AV mode {mode} needs a not-applicable reason")
        result[mode] = dict(record)
    return result


def build_prediction_diagnostics(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    baselines: Mapping[str, np.ndarray],
    *,
    mode_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    required = {"valve", "tin", "local", "terminal"}
    if set(predictions) != required or set(targets) != required or set(baselines) != required:
        raise Phase35ProtocolError("RM3-AV prediction diagnostic task fields changed")
    result: dict[str, Any] = {}
    for key in sorted(required):
        prediction = _finite_three_dimensional(predictions[key], f"{key} prediction")
        target = _finite_three_dimensional(targets[key], f"{key} target")
        baseline = np.asarray(baselines[key], dtype=np.float64)
        if prediction.shape != target.shape or baseline.shape != (len(target), 2):
            raise Phase35ProtocolError(f"RM3-AV {key} diagnostic shape mismatch")
        persistent = np.broadcast_to(baseline[:, None], target.shape)
        mae_side = np.mean(np.abs(prediction - target), axis=(0, 1))
        persistence_side = np.mean(np.abs(persistent - target), axis=(0, 1))
        pooled = float(np.mean(np.abs(prediction - target)))
        persistence_pooled = float(np.mean(np.abs(persistent - target)))
        result[key] = {
            "mae_pooled": pooled,
            "mae_by_side": {"A": float(mae_side[0]), "B": float(mae_side[1])},
            "persistence_mae_pooled": persistence_pooled,
            "persistence_mae_by_side": {
                "A": float(persistence_side[0]),
                "B": float(persistence_side[1]),
            },
            "skill_vs_persistence_pooled": (
                float(1.0 - pooled / persistence_pooled) if persistence_pooled > 0 else None
            ),
            "horizon_curve_steps": {
                str(horizon): {
                    "mae_pooled": float(
                        np.mean(
                            np.abs(
                                prediction[:, : min(horizon, prediction.shape[1])]
                                - target[:, : min(horizon, target.shape[1])]
                            )
                        )
                    ),
                    "persistence_mae_pooled": float(
                        np.mean(
                            np.abs(
                                persistent[:, : min(horizon, persistent.shape[1])]
                                - target[:, : min(horizon, target.shape[1])]
                            )
                        )
                    ),
                }
                for horizon in (6, 18, 60)
            },
        }
    result["mode_records"] = _validate_modes(mode_records)
    result["test_accessed"] = False
    result["automatic_scientific_pass"] = None
    return result


def valve_trajectory_diagnostics(
    prediction: np.ndarray, target: np.ndarray, baseline: np.ndarray
) -> dict[str, Any]:
    prediction = _finite_three_dimensional(prediction, "valve trajectory prediction")
    target = _finite_three_dimensional(target, "valve trajectory target")
    baseline = np.asarray(baseline, dtype=np.float64)
    if prediction.shape != target.shape or baseline.shape != (len(target), 2):
        raise Phase35ProtocolError("RM3-AV valve trajectory shapes changed")

    def delta(value: np.ndarray) -> float:
        return float(np.mean(np.abs(np.diff(value, axis=1))))

    def span(value: np.ndarray) -> float:
        return float(np.mean(np.max(value, axis=1) - np.min(value, axis=1)))

    def roughness(value: np.ndarray) -> float:
        values = []
        for stride in (1, 3, 6):
            if value.shape[1] <= 2 * stride:
                continue
            second = value[:, 2 * stride :] - 2 * value[:, stride:-stride] + value[:, : -2 * stride]
            values.append(np.mean(np.abs(second)))
        return float(np.mean(values))

    persistent = np.broadcast_to(baseline[:, None], target.shape)
    prediction_mae = float(np.mean(np.abs(prediction - target)))
    persistence_mae = float(np.mean(np.abs(persistent - target)))
    prediction_centered = prediction - prediction.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=1, keepdims=True)
    prediction_power = np.abs(np.fft.rfft(prediction_centered, axis=1)) ** 2
    target_power = np.abs(np.fft.rfft(target_centered, axis=1)) ** 2
    high_start = max(1, prediction_power.shape[1] // 3)
    return {
        "prediction_mean_abs_delta": delta(prediction),
        "target_mean_abs_delta": delta(target),
        "prediction_mean_range": span(prediction),
        "target_mean_range": span(target),
        "prediction_static_fraction": float(np.mean(np.abs(np.diff(prediction, axis=1)) < 1e-3)),
        "target_static_fraction": float(np.mean(np.abs(np.diff(target, axis=1)) < 1e-3)),
        "prediction_multiscale_roughness": roughness(prediction),
        "target_multiscale_roughness": roughness(target),
        "prediction_high_frequency_power": float(np.mean(prediction_power[:, high_start:])),
        "target_high_frequency_power": float(np.mean(target_power[:, high_start:])),
        "persistence_mae": persistence_mae,
        "persistence_skill": (
            float(1.0 - prediction_mae / persistence_mae) if persistence_mae > 0 else None
        ),
        "prediction_lower_saturation_fraction": float(np.mean(prediction <= 1.0)),
        "prediction_upper_saturation_fraction": float(np.mean(prediction >= 99.0)),
        "target_lower_saturation_fraction": float(np.mean(target <= 1.0)),
        "target_upper_saturation_fraction": float(np.mean(target >= 99.0)),
        "prediction_small_step_fraction_lt_0_05_point": float(
            np.mean(np.abs(np.diff(prediction, axis=1)) < 0.05)
        ),
        "target_small_step_fraction_lt_0_05_point": float(
            np.mean(np.abs(np.diff(target, axis=1)) < 0.05)
        ),
        "deadband_is_descriptive_not_controller_identification": True,
    }


def response_trajectory_diagnostics(
    effect: np.ndarray,
    *,
    constant_action_effect: np.ndarray,
    stable_poles: np.ndarray,
) -> dict[str, Any]:
    effect = _finite_three_dimensional(effect, "response effect")
    identity = _finite_three_dimensional(constant_action_effect, "constant-action response")
    if effect.shape != identity.shape:
        raise Phase35ProtocolError("RM3-AV response identity shape mismatch")
    poles = np.asarray(stable_poles, dtype=np.float64)
    if not np.isfinite(poles).all():
        raise Phase35ProtocolError("RM3-AV response poles must be finite")
    horizons = {
        str(horizon): float(np.mean(np.abs(effect[:, : min(horizon, effect.shape[1])])))
        for horizon in (6, 18, 60)
    }
    common = 0.5 * (effect[..., 0] + effect[..., 1])
    differential = 0.5 * (effect[..., 0] - effect[..., 1])
    timing_grid = {}
    for step in (0, 1, 3, 6, 12, 18, 30, 60):
        endpoint = (
            np.zeros((len(effect), 2), dtype=np.float64)
            if step == 0 else effect[:, min(step, effect.shape[1]) - 1]
        )
        timing_grid[str(step)] = {
            "seconds": int(step * 10),
            "signed_effect_by_side": np.mean(endpoint, axis=0).tolist(),
            "absolute_effect_by_side": np.mean(np.abs(endpoint), axis=0).tolist(),
            "signed_common": float(np.mean(0.5 * (endpoint[:, 0] + endpoint[:, 1]))),
            "signed_differential": float(np.mean(0.5 * (endpoint[:, 0] - endpoint[:, 1]))),
        }
    valid_poles = poles[(poles > 0.0) & (poles < 1.0)]
    equivalent_tau = (-10.0 / np.log(valid_poles)).tolist()
    return {
        "mean_absolute_effect_by_horizon_steps": horizons,
        "integrated_absolute_effect_by_horizon_steps": {
            str(horizon): float(np.mean(np.sum(np.abs(effect[:, : min(horizon, effect.shape[1])]), axis=1)))
            for horizon in (6, 18, 60)
        },
        "common_mode_energy": float(np.mean(common**2)),
        "differential_mode_energy": float(np.mean(differential**2)),
        "signed_absolute_timing_grid": timing_grid,
        "constant_action_identity_max_abs": float(np.max(np.abs(identity))),
        "stable_poles": poles.tolist(),
        "operator_has_poles": bool(poles.size),
        "all_poles_stable": (
            bool(np.all((poles >= 0.0) & (poles < 1.0))) if poles.size else None
        ),
        "equivalent_time_constants_seconds": equivalent_tau,
        "time_constant_boundary_diagnostic": {
            "at_or_below_20_seconds": int(np.sum(np.asarray(equivalent_tau) <= 20.5)),
            "at_or_above_1200_seconds": int(np.sum(np.asarray(equivalent_tau) >= 1188.0)),
            "horizon_seconds": int(effect.shape[1] * 10),
            "tau_exceeds_horizon_count": int(
                np.sum(np.asarray(equivalent_tau) > effect.shape[1] * 10)
            ),
            "architecture_order_claim": False,
        },
        "finite": True,
    }


def mechanism_residual_dependence(residuals: Mapping[str, np.ndarray]) -> dict[str, Any]:
    expected = {"valve", "tin", "local", "terminal"}
    if set(residuals) != expected:
        raise Phase35ProtocolError("RM3-AV residual dependence task fields changed")
    features = []
    names = []
    episode_count = None
    for task in ("valve", "tin", "local", "terminal"):
        value = _finite_three_dimensional(residuals[task], f"{task} residual")
        if episode_count is None:
            episode_count = len(value)
        elif len(value) != episode_count:
            raise Phase35ProtocolError("RM3-AV residual dependence episode count changed")
        features.append(value.mean(axis=1))
        names.extend((f"{task}_A", f"{task}_B"))
    matrix = np.concatenate(features, axis=1)
    covariance = np.cov(matrix, rowvar=False)
    correlation = np.corrcoef(matrix, rowvar=False)
    return {
        "feature_names": names,
        "episode_pooled_covariance": covariance.tolist(),
        "episode_pooled_correlation": correlation.tolist(),
        "episode_count": int(episode_count or 0),
        "independent_mechanism_noise_claim": False,
        "note": "prediction-residual dependence diagnostic; not recovered structural noise",
    }


def stratified_error_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    strata: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    prediction = _finite_three_dimensional(prediction, "stratified prediction")
    target = _finite_three_dimensional(target, "stratified target")
    if prediction.shape != target.shape:
        raise Phase35ProtocolError("RM3-AV stratified prediction shape mismatch")
    result: dict[str, Any] = {}
    for name, labels_raw in strata.items():
        labels = np.asarray(labels_raw)
        if labels.ndim != 1 or len(labels) != len(target):
            raise Phase35ProtocolError("RM3-AV stratum labels shape mismatch")
        rows = {}
        for value in np.unique(labels):
            mask = labels == value
            label = str(value.item() if hasattr(value, "item") else value)
            side_mae = np.mean(np.abs(prediction[mask] - target[mask]), axis=(0, 1))
            rows[label] = {
                "episode_count": int(mask.sum()),
                "mae_pooled": float(np.mean(np.abs(prediction[mask] - target[mask]))),
                "mae_by_side": {"A": float(side_mae[0]), "B": float(side_mae[1])},
            }
        result[name] = rows
    result["automatic_invariance_pass"] = None
    return result


def convergence_diagnostics(
    loss_curve: list[float], *, best_update: int, update_cap: int
) -> dict[str, Any]:
    values = np.asarray(loss_curve, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise Phase35ProtocolError("RM3-AV convergence curve is invalid")
    tail = values[-min(500, len(values)) :]
    slope = float(np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0]) if len(tail) > 1 else 0.0
    return {
        "optimizer_updates_completed": len(values),
        "update_cap": int(update_cap),
        "best_update": int(best_update),
        "best_update_distance_to_cap": int(update_cap - best_update),
        "last_500_update_slope": slope,
        "last_loss": float(values[-1]),
        "best_observed_training_loss": float(values.min()),
        "last_minus_best_training_loss": float(values[-1] - values.min()),
        "converged": None,
    }


def valve_policy_probe_diagnostics(
    history: np.ndarray,
    future_sp: np.ndarray,
    target_valve: np.ndarray,
    groups: np.ndarray,
    *,
    sp_indices: tuple[int, int],
    temperature_indices: tuple[int, int],
    ridge: float,
) -> dict[str, Any]:
    history = np.asarray(history, dtype=np.float64)
    future_sp = _finite_three_dimensional(future_sp, "valve-probe future SP")
    target = _finite_three_dimensional(target_valve, "valve-probe target")
    groups = np.asarray(groups)
    if (
        history.ndim != 3
        or len(history) != len(target)
        or future_sp.shape != target.shape
        or groups.shape != (len(target),)
        or len(np.unique(groups)) < 2
        or ridge <= 0
    ):
        raise Phase35ProtocolError("RM3-AV valve policy probe contract changed")
    current_sp = history[:, -1, sp_indices]
    current_temperature = history[:, -1, temperature_indices]
    error = current_sp - current_temperature
    historical_error = history[:, :, sp_indices] - history[:, :, temperature_indices]
    integral_error = historical_error.sum(axis=1)
    future_sp_flat = future_sp.reshape(len(target), -1)
    history_summary = np.concatenate(
        (
            history[:, -1],
            history.mean(axis=1),
            history.std(axis=1),
            history[:, -1] - history[:, 0],
        ),
        axis=1,
    )
    features = {
        "sp_only": future_sp_flat,
        "sp_plus_current_temperature": np.concatenate(
            (future_sp_flat, current_temperature), axis=1
        ),
        "pi_features": np.concatenate(
            (future_sp_flat, current_temperature, error, integral_error), axis=1
        ),
        "full_history": np.concatenate((future_sp_flat, history_summary), axis=1),
    }
    target_flat = target.reshape(len(target), -1)
    unique_groups = np.unique(groups)
    fold_count = min(5, len(unique_groups))
    group_folds = np.array_split(unique_groups, fold_count)
    probe_rows = {}
    maximum_overlap = 0
    for name, x in features.items():
        prediction = np.empty_like(target_flat)
        for held_groups in group_folds:
            held = np.isin(groups, held_groups)
            train = ~held
            maximum_overlap = max(
                maximum_overlap,
                len(set(groups[train].tolist()) & set(groups[held].tolist())),
            )
            train_x = x[train]
            train_y = target_flat[train]
            center_x = train_x.mean(axis=0, keepdims=True)
            center_y = train_y.mean(axis=0, keepdims=True)
            centered_x = train_x - center_x
            coefficients = np.linalg.solve(
                centered_x.T @ centered_x + ridge * np.eye(centered_x.shape[1]),
                centered_x.T @ (train_y - center_y),
            )
            prediction[held] = (x[held] - center_x) @ coefficients + center_y
        error_value = prediction - target_flat
        residual = target_flat - target_flat.mean(axis=0, keepdims=True)
        probe_rows[name] = {
            "mae": float(np.mean(np.abs(error_value))),
            "mse": float(np.mean(error_value**2)),
            "incremental_r2": float(
                1.0 - np.sum(error_value**2) / np.sum(residual**2)
            ) if np.sum(residual**2) > 0 else None,
        }
    return {
        "probes": probe_rows,
        "oof_group_unit": "caller_declared_independent_group",
        "group_overlap_count": maximum_overlap,
        "causal_direction_claim": False,
        "interpretation": "predictive action-proxy audit only",
    }


def dependence_diagnostics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3 or not np.isfinite(values).all():
        raise Phase35ProtocolError("RM3-AV dependence values must be finite [sample,2]")
    covariance = np.cov(values, rowvar=False)
    correlation = np.corrcoef(values, rowvar=False)
    return {
        "cross_covariance": float(covariance[0, 1]),
        "pearson_correlation": float(correlation[0, 1]),
        "sample_count": len(values),
        "independence_claim": False,
        "note": "second-order observational diagnostic; not a mechanism-noise independence test",
    }


def daily_gain_context_diagnostics(
    *,
    gain: np.ndarray,
    context: np.ndarray,
    activity: np.ndarray,
    groups: np.ndarray,
    context_names: tuple[str, ...],
    ridge: float,
) -> dict[str, Any]:
    gain = np.asarray(gain, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    activity = np.asarray(activity, dtype=np.float64)
    groups = np.asarray(groups)
    if (
        gain.ndim != 2
        or gain.shape[1] != 2
        or context.shape != (len(gain), len(context_names))
        or activity.shape != gain.shape
        or groups.shape != (len(gain),)
        or len(np.unique(groups)) < 2
        or ridge <= 0
        or not all(np.isfinite(value).all() for value in (gain, context, activity))
    ):
        raise Phase35ProtocolError("RM3-AV daily gain/context contract changed")
    x = np.concatenate((context, activity), axis=1)
    feature_names = (*context_names, "activity_A", "activity_B")
    unique = np.unique(groups)
    folds = np.array_split(unique, min(5, len(unique)))
    prediction = np.empty_like(gain)
    maximum_overlap = 0
    for held_groups in folds:
        held = np.isin(groups, held_groups)
        train = ~held
        maximum_overlap = max(
            maximum_overlap,
            len(set(groups[train].tolist()) & set(groups[held].tolist())),
        )
        center_x = x[train].mean(axis=0, keepdims=True)
        center_y = gain[train].mean(axis=0, keepdims=True)
        centered = x[train] - center_x
        coefficients = np.linalg.solve(
            centered.T @ centered + ridge * np.eye(centered.shape[1]),
            centered.T @ (gain[train] - center_y),
        )
        prediction[held] = (x[held] - center_x) @ coefficients + center_y
    centered_all = x - x.mean(axis=0, keepdims=True)
    gain_centered = gain - gain.mean(axis=0, keepdims=True)
    coefficients_all = np.linalg.solve(
        centered_all.T @ centered_all + ridge * np.eye(centered_all.shape[1]),
        centered_all.T @ gain_centered,
    )
    return {
        "feature_names": list(feature_names),
        "coefficients_by_side": {
            "A": coefficients_all[:, 0].tolist(),
            "B": coefficients_all[:, 1].tolist(),
        },
        "blocked_oof_mae_by_side": {
            "A": float(np.mean(np.abs(prediction[:, 0] - gain[:, 0]))),
            "B": float(np.mean(np.abs(prediction[:, 1] - gain[:, 1]))),
        },
        "group_overlap_count": maximum_overlap,
        "causal_gain_explanation": False,
        "note": "descriptive context/activity regression only",
    }


def valve_innovation_rank(innovations: np.ndarray) -> dict[str, Any]:
    values = np.asarray(innovations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3 or not np.isfinite(values).all():
        raise Phase35ProtocolError("RM3-AV valve innovations must be finite [sample,2]")
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    singular = np.linalg.svd(covariance, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    threshold = singular[0] * 1e-3 if singular[0] > 0 else 0.0
    effective_rank = int(np.sum(singular > threshold))
    common = 0.5 * (centered[:, 0] + centered[:, 1])
    differential = 0.5 * (centered[:, 0] - centered[:, 1])
    common_energy = float(np.mean(common**2))
    differential_energy = float(np.mean(differential**2))
    total = common_energy + differential_energy
    differential_fraction = differential_energy / total if total > 0 else 0.0
    return {
        "covariance": covariance.tolist(),
        "singular_values": singular.tolist(),
        "condition_number": condition,
        "effective_rank": effective_rank,
        "common_energy": common_energy,
        "differential_energy": differential_energy,
        "differential_energy_fraction": differential_fraction,
        "independent_channels_supported": bool(
            effective_rank == 2 and differential_fraction >= 0.01 and condition is not None and condition <= 1000.0
        ),
        "claim_boundary": "rank diagnostic only; does not identify do(valve)",
    }


def build_state_closure_audit(
    *, generated: set[str], declared_external: set[str], required: set[str]
) -> dict[str, Any]:
    if generated & declared_external:
        raise Phase35ProtocolError("state-closure variables cannot be both generated and external")
    missing = sorted(required - generated - declared_external)
    if missing:
        status = "STATE_CLOSURE_BLOCKED"
    elif declared_external:
        status = "DECLARED_CONTEXT_ROLLOUT_ONLY"
    else:
        status = "STATE_CLOSED_BY_DECLARED_OUTPUTS"
    return {
        "status": status,
        "required": sorted(required),
        "generated": sorted(generated),
        "declared_external": sorted(declared_external),
        "missing": missing,
        "state_closed_simulator": bool(status == "STATE_CLOSED_BY_DECLARED_OUTPUTS"),
        "uses_true_future_context": False,
    }


def build_assumption_ledger() -> dict[str, dict[str, Any]]:
    assumptions = {
        "CD-NOD": "domain/time index is observed and separated from endogenous measured context",
        "Nonstationary linear SEM": "fully observed linear instantaneous acyclic SEM with theorem-level coefficient/noise assumptions",
        "TDRL": "finite discrete first-order Markov domain state with known domain variable and full-rank variation",
        "CtrlNS": "mechanism-separable changing support with sufficient variation and bounded model complexity",
        "IDOL": "sparse latent process and justified instantaneous/lagged parent sets",
        "CaRiNG": "non-invertible observation assumptions and independent latent dynamics/noise",
        "LEAP": "independent mechanism noise and identifiable temporal latent process assumptions",
    }
    return {
        method: {
            "assumption": assumptions[method],
            "status": "not_testable" if method in {"TDRL", "CtrlNS", "IDOL", "CaRiNG", "LEAP"} else "unmet",
            "identification_claim_allowed": False,
            "engineering_candidate_only": True,
        }
        for method in ASSUMPTION_LEDGER_METHODS
    }


def build_manual_verdict_template() -> dict[str, None]:
    return {f"Q{index:02d}": None for index in range(1, 34)}
