#!/usr/bin/env python3
"""Independently replay and diagnose Phase 3.5-MS3 validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from experiments.phase3_5.ms3_real_adaptation import (  # noqa: E402
    DEFAULT_MATRIX,
    _validate_cache,
    expand_runs,
    load_matrix,
)
from src.phase35.data import (  # noqa: E402
    Phase35Cache,
    deterministic_anchor_subset,
    extract_windows,
    load_cache,
    valid_window_anchors,
)
from src.phase35.multistep.real_training import (  # noqa: E402
    RealModelConfig,
    RealTrainingConfig,
    build_real_model,
    evaluate_real_model,
    operating_anchor_subset,
)
from src.phase35.multistep.training import _json_dump, _sha256  # noqa: E402
from src.phase35.schema import TARGET_COLUMN, VALVE_COLUMN  # noqa: E402


DEFAULT_RESULTS = ROOT / "results/phase3_5/ms3_real_adaptation"
DEFAULT_OUTPUT_NAME = "supervisor_replay_validation.json"
NUMERIC_EPISODE_KEYS = (
    "action_dose_pct",
    "logged_mae_c",
    "baseline_action_mae_c",
    "shuffled_action_mae_c",
    "mean_abs_effect_c",
    "terminal_effect_c",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {key: None for key in ("min", "q05", "q25", "median", "q75", "q95", "max")}
    return {
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def circular_day_block_bootstrap(
    day_values: np.ndarray,
    *,
    block_length: int,
    samples: int,
    seed: int,
) -> list[float]:
    """Return a circular consecutive-day block-bootstrap 95% interval."""
    values = np.asarray(day_values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("day bootstrap needs at least two one-dimensional day values")
    if block_length < 1 or block_length > len(values) or samples < 1:
        raise ValueError("invalid day-block bootstrap settings")
    rng = np.random.default_rng(seed)
    block_count = math.ceil(len(values) / block_length)
    starts = rng.integers(0, len(values), size=(samples, block_count))
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets) % len(values)
    indices = indices.reshape(samples, -1)[:, : len(values)]
    boot = values[indices].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def _day_improvements(
    episodes: dict[str, Any], comparator_key: str
) -> tuple[np.ndarray, np.ndarray, int]:
    days = np.asarray(episodes["utc_days"])
    dynamic = np.asarray(episodes["dynamic_mask"], dtype=bool)
    logged = np.asarray(episodes["logged_mae_c"], dtype=float)
    comparator = np.asarray(episodes[comparator_key], dtype=float)
    if not (days.shape == dynamic.shape == logged.shape == comparator.shape):
        raise ValueError("MS3 episode arrays do not share one observation axis")
    selected_days = np.unique(days[dynamic])
    day_values = np.asarray(
        [
            np.mean(comparator[(days == day) & dynamic] - logged[(days == day) & dynamic])
            for day in selected_days
        ],
        dtype=float,
    )
    return selected_days, day_values, int(dynamic.sum())


def day_diagnostics(
    episodes: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, (label, comparator) in enumerate(
        (
            ("logged_vs_baseline", "baseline_action_mae_c"),
            ("logged_vs_shuffled", "shuffled_action_mae_c"),
        )
    ):
        days, values, window_count = _day_improvements(episodes, comparator)
        leave_one_out = np.asarray(
            [np.delete(values, index).mean() for index in range(len(values))],
            dtype=float,
        )
        block_intervals = {
            str(block): circular_day_block_bootstrap(
                values,
                block_length=block,
                samples=samples,
                seed=seed + offset * 100 + block,
            )
            for block in (1, 2, 3, 5)
            if block <= len(values)
        }
        result[label] = {
            "day_count": int(len(days)),
            "window_count": window_count,
            "day_start": str(days[0]),
            "day_end": str(days[-1]),
            "mean_improvement_c": float(values.mean()),
            "median_improvement_c": float(np.median(values)),
            "positive_day_count": int((values > 0).sum()),
            "negative_day_count": int((values < 0).sum()),
            "zero_day_count": int((values == 0).sum()),
            "day_improvement_quantiles_c": _quantiles(values),
            "leave_one_day_out_mean_range_c": [
                float(leave_one_out.min()),
                float(leave_one_out.max()),
            ],
            "circular_block_bootstrap_ci95_c": block_intervals,
            "bootstrap_samples": int(samples),
            "diagnostic_only": True,
        }
    return result


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_numeric_leaves(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            leaves[prefix] = numeric
    return leaves


def _max_shared_numeric_error(stored: dict[str, Any], replayed: dict[str, Any]) -> float:
    left = _numeric_leaves(stored)
    right = _numeric_leaves(replayed)
    shared = set(left) & set(right)
    if not shared:
        return math.inf
    return max(abs(left[key] - right[key]) for key in shared)


def _validation_anchors(
    cache: Phase35Cache,
    feature_columns: list[str],
    model: RealModelConfig,
    training: RealTrainingConfig,
    seed: int,
) -> np.ndarray:
    anchors = valid_window_anchors(
        cache,
        "validation",
        feature_columns,
        TARGET_COLUMN,
        VALVE_COLUMN,
        model.window,
        model.horizon,
        training.max_age_seconds,
    )
    anchors = operating_anchor_subset(
        cache,
        anchors,
        window=model.window,
        horizon=model.horizon,
        config=training,
    )
    return deterministic_anchor_subset(
        anchors, training.max_validation_anchors, 10_000 + seed
    )


@torch.no_grad()
def standardized_step_diagnostics(
    model,
    cache: Phase35Cache,
    anchors: np.ndarray,
    feature_columns: list[str],
    *,
    batch_size: int = 512,
) -> dict[str, Any]:
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
    baseline = window["baseline_valve"]
    eligible = (baseline >= 5.0) & (baseline <= 95.0)
    history = window["history"][eligible]
    baseline = baseline[eligible]
    plus_effects: list[np.ndarray] = []
    minus_effects: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    taus: list[np.ndarray] = []
    plus_doses: list[np.ndarray] = []
    minus_doses: list[np.ndarray] = []
    for start in range(0, len(baseline), batch_size):
        stop = start + batch_size
        history_tensor = torch.from_numpy(history[start:stop])
        baseline_tensor = torch.from_numpy(baseline[start:stop])
        context, _, _, _ = model.encoder(history_tensor)
        reference = baseline_tensor[:, None].expand(
            len(baseline_tensor), model.config.horizon
        )
        plus = reference + 5.0
        minus = reference - 5.0
        plus_response = model.response_operator(context, plus, reference)
        minus_response = model.response_operator(context, minus, reference)
        plus_effects.append(plus_response.effect.cpu().numpy())
        minus_effects.append(minus_response.effect.cpu().numpy())
        gain, tau = model.response_operator.physical_parameters(context)
        gains.append(gain.detach().cpu().numpy())
        taus.append(tau.detach().cpu().numpy())
        opening = model.response_operator.opening_map
        plus_doses.append((opening(plus[:, 0]) - opening(baseline_tensor)).cpu().numpy())
        minus_doses.append((opening(minus[:, 0]) - opening(baseline_tensor)).cpu().numpy())
    plus_effect = np.concatenate(plus_effects)
    minus_effect = np.concatenate(minus_effects)
    gain_values = np.concatenate(gains)
    tau_values = np.concatenate(taus)
    horizons = {"H6": 5, "H18": 17, "H60": 59}
    opening = model.response_operator.opening_map
    return {
        "eligible_anchor_count": int(len(baseline)),
        "baseline_valve_pct": _quantiles(baseline),
        "effective_opening_delta_plus5_pct": _quantiles(np.concatenate(plus_doses)),
        "effective_opening_delta_minus5_pct": _quantiles(np.concatenate(minus_doses)),
        "plus5_effect_c": {
            label: _quantiles(plus_effect[:, index])
            for label, index in horizons.items()
        },
        "minus5_effect_c": {
            label: _quantiles(minus_effect[:, index])
            for label, index in horizons.items()
        },
        "scheduled_gain_c_per_effective_pct": _quantiles(gain_values),
        "tau_seconds": {
            f"pole_{index + 1}": _quantiles(tau_values[:, index])
            for index in range(tau_values.shape[1])
        },
        "opening_map_knots_pct": opening.knots.detach().cpu().tolist(),
        "opening_map_values_pct": opening.knot_values().detach().cpu().tolist(),
        "interpretation": "checkpoint_diagnostic_not_field_intervention_evidence",
    }


def _episode_summary(episodes: dict[str, Any]) -> dict[str, Any]:
    dynamic = np.asarray(episodes["dynamic_mask"], dtype=bool)
    return {
        "sample_count": int(len(dynamic)),
        "dynamic_count": int(dynamic.sum()),
        "utc_day_count": int(len(set(np.asarray(episodes["utc_days"])[dynamic]))),
        "action_dose_pct": _quantiles(np.asarray(episodes["action_dose_pct"])[dynamic]),
        "mean_abs_effect_c": _quantiles(
            np.asarray(episodes["mean_abs_effect_c"])[dynamic]
        ),
        "terminal_effect_c": _quantiles(
            np.asarray(episodes["terminal_effect_c"])[dynamic]
        ),
    }


def _pairwise_anchor_overlap(runs: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for left in sorted(runs):
        for right in sorted(runs):
            if right <= left:
                continue
            left_set = set(runs[left]["episodes"]["anchors"])
            right_set = set(runs[right]["episodes"]["anchors"])
            overlap = len(left_set & right_set)
            union = len(left_set | right_set)
            output.append(
                {
                    "seed_pair": [left, right],
                    "overlap_count": overlap,
                    "left_fraction": overlap / len(left_set),
                    "jaccard": overlap / union,
                }
            )
    return output


def run_audit(
    *,
    matrix_path: Path,
    results_root: Path,
    cache_paths: dict[str, Path],
    bootstrap_samples: int,
) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    matrix_sha = _sha256(matrix_path)
    summary = _read_json(results_root / "summary_validation.json")
    caches = {side: load_cache(path) for side, path in cache_paths.items()}
    for side in ("A", "B"):
        _validate_cache(caches[side], side, matrix, matrix_sha)

    archive_path = results_root / "checkpoints_validation.tar"
    expected_members = {
        f"{run['side']}_{run['candidate_id']}_s{run['seed']}/checkpoint_best_val.pt"
        for run in expand_runs(matrix)
    }
    reported_members = {
        item["path"]: item["sha256"]
        for item in summary["checkpoint_archive"]["members"]
    }
    archive_sha = _sha256(archive_path)
    run_outputs: list[dict[str, Any]] = []
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    with tarfile.open(archive_path, "r") as archive:
        actual_members = {member.name for member in archive.getmembers() if member.isfile()}
        if actual_members != expected_members:
            raise RuntimeError("MS3 checkpoint archive member set changed")
        for run_index, run in enumerate(expand_runs(matrix)):
            side = run["side"]
            seed = int(run["seed"])
            run_id = f"{side}_{run['candidate_id']}_s{seed}"
            run_dir = results_root / run_id
            manifest = _read_json(run_dir / "manifest.json")
            history = _read_json(run_dir / "history.json")
            stored_metrics = _read_json(run_dir / "metrics_validation.json")
            stored_episodes = _read_json(run_dir / "episode_metrics_validation.json")
            member_name = f"{run_id}/checkpoint_best_val.pt"
            extracted = archive.extractfile(member_name)
            if extracted is None:
                raise RuntimeError(f"MS3 archive member cannot be read: {member_name}")
            checkpoint_bytes = extracted.read()
            checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
            checkpoint = torch.load(
                io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
            )
            if not (
                checkpoint_sha == reported_members[member_name]
                == manifest["checkpoint_sha256"]
            ):
                raise RuntimeError(f"MS3 checkpoint hash mismatch: {run_id}")
            for key, expected in (
                ("run_id", run_id),
                ("side", side),
                ("seed", seed),
                ("mode", run["mode"]),
                ("matrix_sha256", matrix_sha),
                ("git_sha", summary["execution_git_sha"]),
            ):
                if checkpoint.get(key) != expected:
                    raise RuntimeError(f"MS3 checkpoint payload mismatch: {run_id}:{key}")

            model_config = RealModelConfig(**checkpoint["model_config"])
            training_config = RealTrainingConfig(**checkpoint["training_config"])
            feature_columns = list(checkpoint["feature_columns"])
            model = build_real_model(model_config, feature_columns, run["mode"])
            model.load_state_dict(checkpoint["model_state_dict"])
            anchors = _validation_anchors(
                caches[side], feature_columns, model_config, training_config, seed
            )
            stored_anchor_array = np.asarray(stored_episodes["anchors"], dtype=np.int64)
            anchors_exact = bool(np.array_equal(anchors, stored_anchor_array))
            replayed_metrics, replayed_episodes = evaluate_real_model(
                model,
                caches[side],
                anchors,
                feature_columns,
                torch.device("cpu"),
                dynamic_dose_threshold_pct=training_config.dynamic_dose_threshold_pct,
                shuffle_seed=91_000 + seed,
                batch_size=512,
            )
            episode_errors = {
                key: float(
                    np.max(
                        np.abs(
                            np.asarray(stored_episodes[key], dtype=float)
                            - np.asarray(replayed_episodes[key], dtype=float)
                        )
                    )
                )
                for key in NUMERIC_EPISODE_KEYS
            }
            exact_episode_contract = bool(
                anchors_exact
                and stored_episodes["utc_days"] == replayed_episodes["utc_days"]
                and stored_episodes["dynamic_mask"] == replayed_episodes["dynamic_mask"]
                and stored_episodes["validation_trajectory_sha256"]
                == replayed_episodes["validation_trajectory_sha256"]
            )
            primary_seed = (
                int(matrix["gates"]["day_block_bootstrap_seed"])
                + (0 if side == "A" else 100)
                + seed * 10
            )
            day_report = day_diagnostics(
                stored_episodes,
                samples=bootstrap_samples,
                seed=primary_seed + 10_000,
            )
            replayed_day_report = day_diagnostics(
                replayed_episodes,
                samples=bootstrap_samples,
                seed=primary_seed + 10_000,
            )
            primary_ci_replay: dict[str, Any] = {}
            for offset, label in enumerate(
                ("logged_vs_baseline", "logged_vs_shuffled"), start=1
            ):
                comparator = (
                    "baseline_action_mae_c"
                    if label == "logged_vs_baseline"
                    else "shuffled_action_mae_c"
                )
                _, day_values, _ = _day_improvements(stored_episodes, comparator)
                day_report[label]["frozen_primary_bootstrap_ci95_c"] = (
                    circular_day_block_bootstrap(
                        day_values,
                        block_length=1,
                        samples=int(matrix["gates"]["day_block_bootstrap_samples"]),
                        seed=primary_seed + offset,
                    )
                )
                replayed_comparator = (
                    "baseline_action_mae_c"
                    if label == "logged_vs_baseline"
                    else "shuffled_action_mae_c"
                )
                _, replayed_day_values, _ = _day_improvements(
                    replayed_episodes, replayed_comparator
                )
                replayed_ci = circular_day_block_bootstrap(
                    replayed_day_values,
                    block_length=1,
                    samples=int(matrix["gates"]["day_block_bootstrap_samples"]),
                    seed=primary_seed + offset,
                )
                reported_ci = day_report[label]["frozen_primary_bootstrap_ci95_c"]
                primary_ci_replay[label] = {
                    "reported_ci95_c": reported_ci,
                    "checkpoint_replayed_ci95_c": replayed_ci,
                    "lower_bound_shift_c": replayed_ci[0] - reported_ci[0],
                    "reported_positive": reported_ci[0] > 0,
                    "checkpoint_replayed_positive": replayed_ci[0] > 0,
                }
            output = {
                "run_id": run_id,
                "side": side,
                "seed": seed,
                "candidate_id": run["candidate_id"],
                "checkpoint_sha256": checkpoint_sha,
                "anchors_exact": anchors_exact,
                "trajectory_contract_exact": exact_episode_contract,
                "metric_replay_max_abs_error": _max_shared_numeric_error(
                    stored_metrics, replayed_metrics
                ),
                "episode_replay_max_abs_error": max(episode_errors.values()),
                "episode_replay_max_abs_error_by_field": episode_errors,
                "shuffle_design_exact": stored_metrics["shuffle_design"]
                == replayed_metrics["shuffle_design"],
                "training_diagnostics": {
                    "epochs_ran": len(history),
                    "epoch_cap": training_config.epochs,
                    "best_epoch": manifest["best_epoch"],
                    "early_stopping_triggered": len(history) < training_config.epochs,
                    "response_gradient_norm": _quantiles(
                        item["response_gradient_norm"] for item in history
                    ),
                    "first_response_gradient_norm": history[0][
                        "response_gradient_norm"
                    ],
                    "last_response_gradient_norm": history[-1][
                        "response_gradient_norm"
                    ],
                },
                "reported_metrics": stored_metrics,
                "episode_descriptives": _episode_summary(stored_episodes),
                "day_diagnostics": day_report,
                "checkpoint_replayed_day_diagnostics": replayed_day_report,
                "frozen_primary_ci_replay": primary_ci_replay,
                "standardized_step_diagnostics": (
                    standardized_step_diagnostics(
                        model, caches[side], anchors, feature_columns
                    )
                    if run["mode"] == "joint_total"
                    else None
                ),
            }
            run_outputs.append(output)
            records[(side, run["candidate_id"], seed)] = {
                "episodes": stored_episodes,
                "metrics": stored_metrics,
            }

    side_outputs: dict[str, Any] = {}
    for side in ("A", "B"):
        joint = {
            seed: records[(side, "ms3_joint_total", seed)]
            for seed in matrix["seeds"]
        }
        free = {
            seed: records[(side, "ms3_free_only", seed)]
            for seed in matrix["seeds"]
        }
        side_outputs[side] = {
            "joint_to_free_logged_mae_ratio": [
                joint[seed]["metrics"]["logged_mae_c"]
                / free[seed]["metrics"]["logged_mae_c"]
                for seed in matrix["seeds"]
            ],
            "joint_minus_free_logged_mae_c": [
                joint[seed]["metrics"]["logged_mae_c"]
                - free[seed]["metrics"]["logged_mae_c"]
                for seed in matrix["seeds"]
            ],
            "dynamic_mean_abs_effect_c": [
                joint[seed]["metrics"]["dynamic_mean_abs_effect_c"]
                for seed in matrix["seeds"]
            ],
            "validation_anchor_overlap_across_seeds": _pairwise_anchor_overlap(joint),
            "joint_free_anchor_identity": {
                str(seed): joint[seed]["episodes"]["anchors"]
                == free[seed]["episodes"]["anchors"]
                for seed in matrix["seeds"]
            },
        }
    effect_ratios = [
        side_outputs["B"]["dynamic_mean_abs_effect_c"][index]
        / side_outputs["A"]["dynamic_mean_abs_effect_c"][index]
        for index in range(len(matrix["seeds"]))
    ]
    metric_replay_error = max(
        item["metric_replay_max_abs_error"] for item in run_outputs
    )
    episode_replay_error = max(
        item["episode_replay_max_abs_error"] for item in run_outputs
    )
    cross_side_anchor_identity = {
        str(seed): records[("A", "ms3_joint_total", seed)]["episodes"]["anchors"]
        == records[("B", "ms3_joint_total", seed)]["episodes"]["anchors"]
        for seed in matrix["seeds"]
    }
    cross_side_dose_median_ratios = []
    for seed in matrix["seeds"]:
        medians = {}
        for side in ("A", "B"):
            episodes = records[(side, "ms3_joint_total", seed)]["episodes"]
            dynamic = np.asarray(episodes["dynamic_mask"], dtype=bool)
            medians[side] = float(
                np.median(np.asarray(episodes["action_dose_pct"], dtype=float)[dynamic])
            )
        cross_side_dose_median_ratios.append(medians["B"] / medians["A"])
    return {
        "protocol_version": matrix["protocol_version"],
        "audit_scope": "validation_only_checkpoint_episode_and_day_level_replay",
        "matrix_sha256": matrix_sha,
        "execution_git_sha": summary["execution_git_sha"],
        "result_artifact_commit": __import__("subprocess")
        .check_output(
            [
                "git",
                "log",
                "-1",
                "--format=%H",
                "--",
                str((results_root / "summary_validation.json").relative_to(ROOT)),
            ],
            cwd=ROOT,
            text=True,
        )
        .strip(),
        "archive": {
            "path": str(archive_path.relative_to(ROOT)).replace("\\", "/"),
            "reported_sha256": summary["checkpoint_archive"]["sha256"],
            "replayed_sha256": archive_sha,
            "member_count": len(expected_members),
            "passes": archive_sha == summary["checkpoint_archive"]["sha256"],
        },
        "cache_contracts": {
            side: {
                "cache_id": cache_paths[side].name,
                "location_scope": "external_nonversioned_local_cache",
                "source_sha256": caches[side].metadata["source"]["sha256"],
                "grid_rows": len(caches[side].timestamps_ns),
                "test_split_accessed": False,
            }
            for side in ("A", "B")
        },
        "runs": run_outputs,
        "side_diagnostics": side_outputs,
        "cross_side_diagnostics": {
            "matched_seed_B_to_A_dynamic_effect_ratio": effect_ratios,
            "ratio_quantiles": _quantiles(effect_ratios),
            "matched_seed_B_to_A_dynamic_dose_median_ratio": (
                cross_side_dose_median_ratios
            ),
            "validation_anchor_identity_by_seed": cross_side_anchor_identity,
        },
        "replay_gate": {
            "all_archived_checkpoints_verified": True,
            "all_anchor_and_trajectory_contracts_exact": all(
                item["trajectory_contract_exact"] for item in run_outputs
            ),
            "all_shuffle_designs_exact": all(
                item["shuffle_design_exact"] for item in run_outputs
            ),
            "max_metric_replay_abs_error": metric_replay_error,
            "max_episode_replay_abs_error": episode_replay_error,
            "metric_tolerance": 2e-5,
            "episode_tolerance": 1e-3,
            "tolerance_basis": (
                "cross_platform_aarch64_cuda_to_x86_cpu_float_replay; "
                "anchors_and_trajectory_bytes_remain_exact"
            ),
            "passes": bool(
                all(item["trajectory_contract_exact"] for item in run_outputs)
                and all(item["shuffle_design_exact"] for item in run_outputs)
                and metric_replay_error <= 2e-5
                and episode_replay_error <= 1e-3
            ),
        },
        "independent_unit": "UTC_day_for_primary_bootstrap",
        "seed_interpretation": (
            "optimization_and_validation_anchor_subsampling_variation; "
            "seeds_are_not_independent_experimental_units"
        ),
        "test_accessed": False,
        "claim_boundary": (
            "This audit replays observational validation only. Standardized checkpoint "
            "steps are model diagnostics, not do(valve) or field-response evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cache-a", required=True)
    parser.add_argument("--cache-b", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--output")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1000 or args.torch_threads < 1:
        raise SystemExit("MS3 audit bootstrap/thread settings are invalid")
    torch.set_num_threads(args.torch_threads)
    results_root = Path(args.results_root).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else results_root / DEFAULT_OUTPUT_NAME
    )
    audit = run_audit(
        matrix_path=Path(args.matrix).resolve(),
        results_root=results_root,
        cache_paths={
            "A": Path(args.cache_a).resolve(),
            "B": Path(args.cache_b).resolve(),
        },
        bootstrap_samples=args.bootstrap_samples,
    )
    _json_dump(output, audit)
    print(
        json.dumps(
            {
                "output": str(output),
                "replay_gate": audit["replay_gate"],
                "test_accessed": audit["test_accessed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not audit["replay_gate"]["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
