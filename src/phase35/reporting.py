"""Seed-aware Phase 3.5 aggregation and preregistered gate evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .matrix import load_matrix


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else key
            out.update(_flatten_numeric(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None:
        if np.isfinite(float(value)):
            out[prefix] = float(value)
    return out


def collect_runs(run_root: str | Path, split: str = "validation") -> list[dict]:
    root = Path(run_root)
    records = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = _read_json(manifest_path) or {}
        forecast = _read_json(run_dir / f"metrics_{split}.json") or {}
        events = _read_json(run_dir / f"event_metrics_{split}.json") or {}
        record = {
            "run_id": manifest.get("run_id", run_dir.name),
            "side": manifest.get("side"),
            "seed": manifest.get("seed"),
            "config_id": (manifest.get("config") or {}).get("config_id"),
            "metrics": {
                **{f"forecast.{k}": v for k, v in _flatten_numeric(forecast).items()},
                **{f"event.{k}": v for k, v in _flatten_numeric(events).items()},
            },
        }
        records.append(record)
    return records


def aggregate_runs(records: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["side"] and record["config_id"]:
            grouped[(record["side"], record["config_id"])].append(record)
    result = {}
    for (side, config_id), runs in grouped.items():
        metric_names = sorted(set().union(*(run["metrics"].keys() for run in runs)))
        metrics = {}
        for name in metric_names:
            values = [run["metrics"][name] for run in runs if name in run["metrics"]]
            metrics[name] = {
                "n": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        result.setdefault(side, {})[config_id] = {
            "n_seeds": len(runs),
            "seeds": sorted(run["seed"] for run in runs if run["seed"] is not None),
            "metrics": metrics,
        }
    return result


def _metric(aggregate: dict, side: str, config: str, name: str) -> float | None:
    try:
        return aggregate[side][config]["metrics"][name]["mean"]
    except KeyError:
        return None


def _mean_direction(aggregate: dict, side: str, config: str, kind: str) -> float | None:
    values = [
        _metric(aggregate, side, config, f"event.main_temperature.{direction}.{kind}")
        for direction in ("open", "close")
    ]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _mean_irf(aggregate: dict, side: str, config: str) -> float | None:
    values = [
        _metric(aggregate, side, config, f"event.main_temperature.{direction}.irf_wmae")
        for direction in ("open", "close")
    ]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _mean_event_metric(aggregate: dict, side: str, config: str, name: str) -> float | None:
    values = [
        _metric(aggregate, side, config, f"event.main_temperature.{direction}.{name}")
        for direction in ("open", "close")
    ]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _gate(status: str, detail: str) -> dict:
    return {"status": status, "detail": detail}


def evaluate_gates(aggregate: dict, matrix: dict) -> dict:
    thresholds = matrix["gates"]
    out = {}
    for side in matrix["sides"]:
        side_gates = {}
        delta = _metric(aggregate, side, "delta_no_baseline", "forecast.integrated_mae")
        reconstructed = _metric(
            aggregate, side, "delta_with_baseline", "forecast.integrated_mae"
        )
        absolute = _metric(aggregate, side, "absolute_identity", "forecast.integrated_mae")
        if delta is None or reconstructed is None or absolute is None:
            side_gates["E1_action_representation"] = _gate(
                "INCONCLUSIVE", "missing naive-delta/reconstructed/absolute forecast metrics"
            )
        else:
            reconstructed_ratio = reconstructed / max(delta, 1e-12)
            absolute_ratio = absolute / max(reconstructed, 1e-12)
            threshold = thresholds["forecast_noninferiority_ratio"]
            status = "PASS" if max(reconstructed_ratio, absolute_ratio) <= threshold else "FAIL"
            side_gates["E1_action_representation"] = _gate(
                status,
                f"reconstructed/naive-delta ratio={reconstructed_ratio:.4f}; "
                f"absolute/reconstructed ratio={absolute_ratio:.4f}",
            )

        fixed = _metric(
            aggregate, side, "absolute_equal_percentage_r50", "forecast.integrated_mae"
        )
        nonlinear = _metric(aggregate, side, "absolute_nonlinear", "forecast.integrated_mae")
        irf_identity, irf_nonlinear = _mean_irf(aggregate, side, "absolute_identity"), _mean_irf(aggregate, side, "absolute_nonlinear")
        irf_fixed = _mean_irf(
            aggregate, side, "absolute_equal_percentage_r50"
        )
        dose_identity = _mean_event_metric(
            aggregate, side, "absolute_identity", "model_dose_monotonicity"
        )
        dose_nonlinear = _mean_event_metric(
            aggregate, side, "absolute_nonlinear", "model_dose_monotonicity"
        )
        dose_fixed = _mean_event_metric(
            aggregate,
            side,
            "absolute_equal_percentage_r50",
            "model_dose_monotonicity",
        )
        if None in (
            absolute, fixed, nonlinear, irf_identity, irf_fixed, irf_nonlinear,
            dose_identity, dose_fixed, dose_nonlinear,
        ):
            side_gates["E2_nonlinear_opening"] = _gate("INCONCLUSIVE", "missing forecast or event IRF metrics")
        else:
            threshold = thresholds["forecast_noninferiority_ratio"]
            fixed_pred_ok = fixed <= absolute * threshold
            learned_pred_ok = nonlinear <= absolute * threshold
            fixed_irf_ok = irf_fixed <= irf_identity * thresholds["irf_improvement_ratio"]
            learned_irf_ok = irf_nonlinear <= irf_identity * thresholds["irf_improvement_ratio"]
            fixed_dose_ok = dose_fixed >= dose_identity + thresholds["minimum_dose_monotonicity_gain"]
            learned_dose_ok = dose_nonlinear >= dose_identity + thresholds["minimum_dose_monotonicity_gain"]
            fixed_pass = fixed_pred_ok and (fixed_irf_ok or fixed_dose_ok)
            learned_pass = learned_pred_ok and (learned_irf_ok or learned_dose_ok)
            status = "PASS" if (fixed_pass or learned_pass) else (
                "FAIL" if not (fixed_pred_ok or learned_pred_ok) else "INCONCLUSIVE"
            )
            side_gates["E2_nonlinear_opening"] = _gate(
                status,
                f"fixed: forecast={fixed/absolute:.4f}, IRF={irf_fixed/irf_identity:.4f}, "
                f"dose gain={dose_fixed-dose_identity:.3f}; learned: forecast={nonlinear/absolute:.4f}, "
                f"IRF={irf_nonlinear/irf_identity:.4f}, dose gain={dose_nonlinear-dose_identity:.3f}",
            )

        n_events = _metric(aggregate, side, "absolute_identity", "event.valve_events_matched")
        n_blocks = _metric(
            aggregate,
            side,
            "absolute_identity",
            "event.main_temperature.all_oriented.n_clusters",
        )
        n_open = _metric(
            aggregate, side, "absolute_identity", "event.main_temperature.open.n_events"
        )
        n_close = _metric(
            aggregate, side, "absolute_identity", "event.main_temperature.close.n_events"
        )
        oriented_ci_high = _metric(
            aggregate,
            side,
            "absolute_identity",
            "event.main_temperature.all_oriented.empirical_ci_high_h60",
        )
        empirical_dir = _mean_direction(aggregate, side, "absolute_identity", "empirical_direction_rate")
        max_smd = _metric(aggregate, side, "absolute_identity", "event.matching_balance.max_abs_smd")
        pretrend = _metric(
            aggregate,
            side,
            "absolute_identity",
            "event.matching_balance.main_temperature_pretrend_difference_c",
        )
        if any(value is None for value in (
            n_events, n_blocks, n_open, n_close, oriented_ci_high,
            empirical_dir, max_smd, pretrend,
        )):
            side_gates["E3_empirical_response"] = _gate(
                "INCONCLUSIVE", "missing matched-event, balance, or pretrend metrics"
            )
        else:
            enough = (
                n_events >= thresholds["minimum_matched_events"]
                and n_blocks >= thresholds["minimum_time_blocks"]
                and n_open >= thresholds["minimum_events_per_direction"]
                and n_close >= thresholds["minimum_events_per_direction"]
            )
            balanced = max_smd <= thresholds["maximum_matching_smd"]
            pretrend_ok = abs(pretrend) <= thresholds["maximum_pretrend_difference_c"]
            response_nonzero = oriented_ci_high < 0.0
            detail = (
                f"events={n_events:.0f} (open={n_open:.0f}, close={n_close:.0f}); "
                f"day blocks={n_blocks:.0f}; direction={empirical_dir:.3f}; "
                f"oriented H60 CI upper={oriented_ci_high:.3f} C; "
                f"max|SMD|={max_smd:.3f}; pretrend diff={pretrend:.3f} C"
            )
            if not enough:
                side_gates["E3_empirical_response"] = _gate("INCONCLUSIVE", detail + "; insufficient common support")
            elif not balanced or not pretrend_ok:
                side_gates["E3_empirical_response"] = _gate("INCONCLUSIVE", detail + "; matching gate not met")
            else:
                ok = response_nonzero and empirical_dir >= thresholds["minimum_direction_rate"]
                side_gates["E3_empirical_response"] = _gate("PASS" if ok else "FAIL", detail)

        model_dir = _mean_direction(aggregate, side, "absolute_identity", "model_direction_rate")
        model_irf = _mean_irf(aggregate, side, "absolute_identity")
        if side_gates["E3_empirical_response"]["status"] != "PASS":
            side_gates["E4_model_response"] = _gate(
                "BLOCKED", "E3 empirical reference did not pass; model-response comparison is not identifiable"
            )
        elif model_dir is None or model_irf is None:
            side_gates["E4_model_response"] = _gate("INCONCLUSIVE", "missing model response metrics")
        else:
            ok = model_dir >= thresholds["minimum_direction_rate"] and model_irf <= thresholds["maximum_irf_wmae_c"]
            side_gates["E4_model_response"] = _gate(
                "PASS" if ok else "FAIL", f"model direction={model_dir:.3f}; IRF-WMAE={model_irf:.3f} C"
            )

        noexec = _metric(
            aggregate, side, "absolute_identity",
            "event.sp_negative_control.no_execution.mean_abs_model_effect_h6",
        )
        executed = _metric(
            aggregate, side, "absolute_identity",
            "event.sp_negative_control.executed.mean_abs_model_effect_h6",
        )
        n_noexec = _metric(
            aggregate, side, "absolute_identity", "event.sp_negative_control.no_execution.n_events"
        )
        n_executed = _metric(
            aggregate, side, "absolute_identity", "event.sp_negative_control.executed.n_events"
        )
        if None in (noexec, executed, n_noexec, n_executed):
            side_gates["E5_sp_negative_control"] = _gate("INCONCLUSIVE", "missing executed/no-execution groups")
        else:
            ratio = executed / max(noexec, 1e-12)
            ratio_text = "undefined (no-execution≈0)" if noexec <= 1e-6 else f"{ratio:.2f}"
            enough = min(n_noexec, n_executed) >= thresholds["minimum_sp_events_per_group"]
            ok = noexec <= thresholds["maximum_no_execution_effect_c"] and ratio >= thresholds["minimum_executed_to_no_execution_ratio"]
            side_gates["E5_sp_negative_control"] = _gate(
                "INCONCLUSIVE" if not enough else ("PASS" if ok else "FAIL"),
                f"n(no-execution/executed)={n_noexec:.0f}/{n_executed:.0f}; "
                f"no-execution={noexec:.4f} C; executed/no-execution={ratio_text}",
            )
        out[side] = side_gates
    return out


def render_markdown(aggregate: dict, gates: dict, split: str) -> str:
    lines = [
        "# Phase 3.5 Result Summary",
        "",
        f"> Split: `{split}`. Seed variation and event-level uncertainty are reported separately.",
        "",
        "## Forecast aggregation",
        "",
        "| Side | Config | Seeds | Integrated MAE mean±SD |",
        "|---|---|---:|---:|",
    ]
    for side in sorted(aggregate):
        for config in sorted(aggregate[side]):
            item = aggregate[side][config]
            metric = item["metrics"].get("forecast.integrated_mae")
            display = "—" if metric is None else f"{metric['mean']:.4f} ± {metric['sd']:.4f}"
            lines.append(f"| {side} | {config} | {item['n_seeds']} | {display} |")
    lines += ["", "## Preregistered gates", "", "| Side | Gate | Status | Evidence |", "|---|---|---|---|"]
    for side in sorted(gates):
        for name, result in gates[side].items():
            lines.append(f"| {side} | {name} | {result['status']} | {result['detail']} |")
    lines += [
        "",
        "PASS denotes a preregistered operational gate, not randomized causal proof. "
        "INCONCLUSIVE is retained when required runs/events are absent.",
    ]
    return "\n".join(lines) + "\n"


def summarize(run_root: str | Path, matrix_path: str | Path, split: str = "validation") -> dict:
    matrix = load_matrix(matrix_path)
    records = collect_runs(run_root, split)
    aggregate = aggregate_runs(records)
    gates = evaluate_gates(aggregate, matrix)
    return {
        "protocol_version": matrix["protocol_version"],
        "split": split,
        "n_runs": len(records),
        "aggregate": aggregate,
        "gates": gates,
        "markdown": render_markdown(aggregate, gates, split),
    }
