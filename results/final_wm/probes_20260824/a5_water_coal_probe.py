"""A5 true DCS water-coal-ratio mechanism probe (frozen 2026-08-28).

Protocol source: PREREG_load_scheduling_20260826.md, A5.  The model receives
the v2.2 7+2 oracle view and injects one bounded, sign-free metal-power term.
The quadratic reference is fitted on valid train samples only.  Seed 0 is
exploratory and cannot create a route verdict.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.final_wm import matrix_spec as ms
from src.final_wm.analysis import STEAM_FLOW_INDEX, WindowErrors, binning_stats
from src.final_wm.contracts import BOUNDARY_ELEMENTS
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL, sample_windows
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from src.final_wm.water_coal import WaterCoalRecord, promote_water_coal_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = Path(__file__).resolve().parent
OUT = P / "a5_water_coal_probe"
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_RECORD = ROOT / "artifacts/final_wm/canonical_sideA_v2.npz"

N_WIN = 256
EVAL_SEED = 50_000
BASELINE_H18 = 0.4840
BASELINE_SPREAD = 1.13


def probe_spec(arm: str):
    return ms._base(
        "t1", arm, 0,
        boundary_mode="oracle",
        initial_state_mode="hybrid",
        closure_mode="conservative_norew",
        epochs=120,
        patience=20,
        batch_size=32,
        batches_per_epoch=200,
    )


def build_a5(spec, props, reference):
    return promote_water_coal_model(build_world_model(spec, props), reference)


class BaseRecordView:
    """Seven-channel control view with exactly the A5 record's filtered split."""

    def __init__(self, record: WaterCoalRecord) -> None:
        self.boundary = record.base_boundary
        self.actions = record.actions
        self.obs = record.obs
        self.timestamps = record.timestamps
        self.split = record.split
        self.n = record.n
        self._record = record

    def split_runs(self, split_id: int):
        return self._record.split_runs(split_id)


def _base_history(history: HistoryWindow) -> HistoryWindow:
    return HistoryWindow(
        history.obs, history.actions, history.boundary[..., : len(BOUNDARY_ELEMENTS)]
    )


def identity_gate(record, props, reference) -> dict:
    torch.manual_seed(0)
    spec = probe_spec("a5_water_coal")
    base = build_world_model(spec, props).to(DEVICE).eval()
    a5 = build_a5(spec, props, reference).to(DEVICE).eval()
    a5.load_state_dict(base.state_dict(), strict=False)
    batch = sample_windows(record, SPLIT_TRAIN, 8, 96, 18, torch.Generator().manual_seed(7))
    history_a5 = HistoryWindow(
        batch.history.obs.to(DEVICE),
        batch.history.actions.to(DEVICE),
        batch.history.boundary.to(DEVICE),
    )
    history_base = _base_history(history_a5)
    actions = batch.future_actions.to(DEVICE)
    future_a5 = batch.future_boundary.to(DEVICE)
    with torch.no_grad():
        r0 = base.forecast(
            history_base, actions, boundary_mode="oracle",
            true_future_boundary=future_a5[..., : len(BOUNDARY_ELEMENTS)],
        )
        r1 = a5.forecast(
            history_a5, actions, boundary_mode="oracle", true_future_boundary=future_a5,
        )
    max_diff = float((r0.temps_mu - r1.temps_mu).abs().max())
    exact = bool(torch.equal(r0.temps_mu, r1.temps_mu))
    return {"exact": exact, "max_abs_temp_diff_c": max_diff}


def reference_diagnostics(record, reference) -> dict:
    mask = (record.split == SPLIT_TRAIN) & record.operating_mask
    load = record.unit_load[mask].to(torch.float64)
    ratio = record.water_coal_ratio[mask].to(torch.float64)
    x = (load - reference.load_center) / reference.load_scale
    c = reference.coefficients
    pred = c[0] + c[1] * x + c[2] * x.square()
    residual = ratio - pred
    denom = ((ratio - ratio.mean()) ** 2).sum()
    r2 = 1.0 - float((residual.square().sum() / denom))
    return {
        "train_r2": r2,
        "train_residual_mean": float(residual.mean()),
        "operating_fraction": float(record.operating_mask.float().mean()),
        "n_total": record.n,
        "n_operating": int(record.operating_mask.sum()),
    }


def _error_summary(abs_err: torch.Tensor, load: torch.Tensor, day_ids: torch.Tensor) -> dict:
    stats = binning_stats(WindowErrors(abs_err=abs_err, load=load, day_ids=day_ids))
    ch4 = stats["H18"]["final_outlet_temp"]
    bins = [float(v) for v in ch4["bin_means"]]
    return {
        "overall_h18_mae": float(np.mean(bins)),
        "bins_q1q5": bins,
        "spread_ratio": float(max(bins) / min(bins)),
    }


@torch.no_grad()
def paired_validation_metrics(model, control, record) -> dict:
    gen = torch.Generator().manual_seed(EVAL_SEED)
    errors_a5, errors_control, loads, days = [], [], [], []
    done = 0
    while done < N_WIN:
        bsz = min(32, N_WIN - done)
        batch = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
        history = HistoryWindow(
            batch.history.obs.to(DEVICE),
            batch.history.actions.to(DEVICE),
            batch.history.boundary.to(DEVICE),
        )
        result = model.forecast(
            history,
            batch.future_actions.to(DEVICE),
            boundary_mode="oracle",
            true_future_boundary=batch.future_boundary.to(DEVICE),
        )
        base_history = _base_history(history)
        result_control = control.forecast(
            base_history,
            batch.future_actions.to(DEVICE),
            boundary_mode="oracle",
            true_future_boundary=batch.future_boundary[
                ..., : len(BOUNDARY_ELEMENTS)
            ].to(DEVICE),
        )
        target = batch.future_obs.to(DEVICE)
        errors_a5.append((target - result.temps_mu).abs().cpu())
        errors_control.append((target - result_control.temps_mu).abs().cpu())
        loads.append(batch.future_boundary[:, 0, STEAM_FLOW_INDEX])
        days.append(batch.day_ids)
        done += bsz
    abs_err_a5 = torch.cat(errors_a5)
    abs_err_control = torch.cat(errors_control)
    load = torch.cat(loads)
    day_ids = torch.cat(days)
    a5 = _error_summary(abs_err_a5, load, day_ids)
    matched = _error_summary(abs_err_control, load, day_ids)
    return {
        "a5": a5,
        "matched_filtered_control": matched,
        "relative_h18_change": (
            a5["overall_h18_mae"] - matched["overall_h18_mae"]
        ) / matched["overall_h18_mae"],
        "spread_change": (a5["spread_ratio"] - matched["spread_ratio"]) / matched["spread_ratio"],
        "a5_change_vs_old_0p484_anchor": (
            a5["overall_h18_mae"] - BASELINE_H18
        ) / BASELINE_H18,
    }


def direction_metrics(model, record) -> dict:
    path = P / "direction_gate_v03.py"
    module_spec = importlib.util.spec_from_file_location("a5_direction_v03", path)
    gate_module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(gate_module)
    report = {}
    for valve, name in ((0, "valve1"), (1, "valve2")):
        report[name] = {}
        for horizon, tail, hname in ((18, 3, "H18"), (60, 10, "H60")):
            report[name][hname] = gate_module.direction_gate(
                model, record, valve_index=valve, horizon=horizon, tail=tail
            )
    return report


def direction_passed(report: dict) -> bool:
    cells = [report[v][h]["in_support_only"] for v in report for h in report[v]]
    return all(cell is not None and cell["gate_pass"] for cell in cells)


@torch.no_grad()
def power_summary(model, record) -> dict:
    batch = sample_windows(
        record, SPLIT_VAL, N_WIN, 96, 18, torch.Generator().manual_seed(EVAL_SEED)
    )
    power = model.transition.water_coal_total_power(
        batch.future_boundary.to(DEVICE)
    ).abs().flatten().cpu()
    return {
        "w_raw": float(model.transition.w_raw),
        "w_bounded": float(torch.tanh(model.transition.w_raw)),
        "abs_power_kw_p50": float(torch.quantile(power, 0.50)),
        "abs_power_kw_p95": float(torch.quantile(power, 0.95)),
        "abs_power_kw_max": float(power.max()),
        "hard_bound_kw": float(model.transition.wc_power_bound_kw),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()

    record = WaterCoalRecord(args.record)
    reference = record.fit_reference()
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
    identity = identity_gate(record, props, reference)
    diagnostics = reference_diagnostics(record, reference)
    print(f"[A5 identity] exact={identity['exact']} max_diff={identity['max_abs_temp_diff_c']:.3e}")
    print(f"[A5 data] operating={diagnostics['operating_fraction']:.3%} "
          f"n={diagnostics['n_operating']:,}/{diagnostics['n_total']:,} "
          f"train_R2={diagnostics['train_r2']:.4f} residual_std={reference.residual_scale:.6f}")
    if not identity["exact"]:
        raise RuntimeError("A5 nested identity gate failed; refusing to train")
    if args.sanity:
        print(json.dumps({"identity": identity, "reference": asdict(reference),
                          "data": diagnostics}, ensure_ascii=False, indent=2))
        return

    sys.path.insert(0, str(P))
    from probe_guard import assert_grid, verify_ledger_properties
    assert_grid(props)
    import src.final_wm.training as training

    control_spec = probe_spec("a5_filtered_control")
    a5_spec = probe_spec("a5_water_coal")
    control_out = OUT / "control"
    a5_out = OUT / "a5"

    control_final = train_arm(
        control_spec, BaseRecordView(record), control_out,
        device=DEVICE, properties=props, compile_substep=False,
    )
    verify_ledger_properties(control_out)

    original_builder = training.build_world_model

    def builder(spec, properties=None):
        return promote_water_coal_model(original_builder(spec, properties), reference)

    training.build_world_model = builder
    try:
        a5_final = train_arm(
            a5_spec, record, a5_out, device=DEVICE, properties=props, compile_substep=False
        )
    finally:
        training.build_world_model = original_builder
    verify_ledger_properties(a5_out)

    control = build_world_model(control_spec, props).to(DEVICE)
    control_checkpoint = control_out / "checkpoints" / f"{control_final['run_id']}.pt"
    control.load_state_dict(torch.load(
        control_checkpoint, map_location=DEVICE, weights_only=False
    )["state_dict"])
    control.eval()

    model = build_a5(a5_spec, props, reference).to(DEVICE)
    checkpoint = a5_out / "checkpoints" / f"{a5_final['run_id']}.pt"
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    evaluation = paired_validation_metrics(model, control, record)
    directions = direction_metrics(model, record)
    powers = power_summary(model, record)

    a5_eval = evaluation["a5"]
    matched_eval = evaluation["matched_filtered_control"]
    accuracy_pass = a5_eval["overall_h18_mae"] <= matched_eval["overall_h18_mae"] * 0.95
    spread_pass = a5_eval["spread_ratio"] <= matched_eval["spread_ratio"] * 1.10
    direction_pass = direction_passed(directions)
    if a5_eval["overall_h18_mae"] >= matched_eval["overall_h18_mae"] * 1.05 or not direction_pass:
        outcome = "REJECT_EXPLORATORY_SEED0"
    elif accuracy_pass and spread_pass:
        outcome = "PROMOTE_TO_FIXED_SEEDS_1_2"
    else:
        outcome = "INCONCLUSIVE_EXPLORATORY_SEED0"
    report = {
        "arm": "a5_true_water_coal_ratio",
        "status": outcome,
        "single_seed_exploratory": True,
        "record": str(args.record),
        "reference_train_only": asdict(reference),
        "reference_diagnostics": diagnostics,
        "identity_gate": identity,
        "train": {
            "matched_filtered_control": {k: control_final[k] for k in (
                "run_id", "commit", "best_val_nll", "best_epoch", "epochs_run",
                "stop_reason", "converged", "flags", "timing"
            )},
            "a5": {k: a5_final[k] for k in (
                "run_id", "commit", "best_val_nll", "best_epoch", "epochs_run",
                "stop_reason", "converged", "flags", "timing"
            )},
        },
        "evaluation": evaluation,
        "water_coal_power": powers,
        "direction_v03": directions,
        "preregistered_gates": {
            "accuracy_at_least_5pct": bool(accuracy_pass),
            "spread_no_more_than_10pct_worse": bool(spread_pass),
            "both_valves_h18_h60_direction": bool(direction_pass),
            "all_pass": bool(accuracy_pass and spread_pass and direction_pass),
        },
        "baseline": {
            "old_unfiltered_h18_mae": BASELINE_H18,
            "old_unfiltered_spread_ratio": BASELINE_SPREAD,
            "source": "PREREG_load_scheduling_20260826.md",
            "decision_comparator": "matched_filtered_control",
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[A5 result] {outcome} H18={a5_eval['overall_h18_mae']:.4f} "
          f"vs matched={matched_eval['overall_h18_mae']:.4f} "
          f"spread={a5_eval['spread_ratio']:.3f} vs {matched_eval['spread_ratio']:.3f} "
          f"w={powers['w_bounded']:+.4f}")
    print(f"written {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
