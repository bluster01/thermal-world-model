"""Protocol-fix rerun driver (PREREG_protocol_fix_20260829, execution-side).

Reruns the scheduling-campaign arms that were adjudicated on the OLD record
(canonical_sideA_v1fixed.npz, valve1 错侧缺陷) under the FIXED protocol:
v2.2 record + A5 quality gates + matched filtered control as denominator +
frozen arm spec + 256-window decision metric + full-val OOF readout.

Arms (mechanism classes imported from the original probe modules, no src change):
  a4         KScheduledTransition          (k load-schedule)
  a6         FlowPressureScheduledTransition (flow + pressure tau/UA)
  a46        JointScheduledTransition      (A4+A6)
  a1         FlowScheduled + alphas pinned (0, 0.8)  (Dittus-Boelter physics)
  a9         FlowScheduled, mix-only tau   (schedule_tau_b=False)
  lpv_free   FlowScheduled, free alphas
  a2         initial_state_mode=steady     (config-level ablation)

Usage:
  python protocol_rerun.py --sanity            # identity/mechanism gates only
  python protocol_rerun.py --arm a4            # run one arm (train + eval)
  python protocol_rerun.py --queue             # run all pending arms sequentially
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))

from experiments.final_wm import matrix_spec as ms
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL, sample_windows
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from src.final_wm.water_coal import WaterCoalRecord

from a5_water_coal_probe import (  # noqa: E402
    BaseRecordView, direction_metrics, direction_passed, probe_spec,
)
from zcond_oof_sweep import build_windows, sweep  # noqa: E402

import a4_k_schedule_probe as a4_mod  # noqa: E402
import a6_pressure_schedule_probe as a6_mod  # noqa: E402
import a46_joint_schedule_probe as a46_mod  # noqa: E402
import lpv_schedule_probe as lpv_mod  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = P / "protocol_rerun"
OUT.mkdir(parents=True, exist_ok=True)
CONTROL_CKPT = P / "a5_water_coal_probe" / "control" / "checkpoints" / "t1_a5_filtered_control_seed0.pt"
N_WIN, EVAL_SEED = 256, 50_000

ARMS = {
    "a4": {"builder": lambda spec, props: a4_mod.build(spec, props),
           "gate_builder": lambda spec, props: a4_mod.build(spec, props),
           "name": "a4_k_rerun"},
    "a6": {"builder": lambda spec, props: a6_mod.build(spec, props, alpha_scale=1.0),
           "gate_builder": lambda spec, props: a6_mod.build(spec, props, alpha_scale=0.0),
           "name": "a6_pressure_rerun"},
    "a46": {"builder": lambda spec, props: a46_mod.build(spec, props, alpha_scale=1.0),
            "gate_builder": lambda spec, props: a46_mod.build(spec, props, alpha_scale=0.0),
            "name": "a46_joint_rerun"},
    "a1": {"builder": lambda spec, props: lpv_mod.fix_alphas_to_physics(
               lpv_mod.build(spec, props, alpha_scale=1.0, schedule_tau_b=True),
               a_tau=0.0, a_ua=0.8),
           "gate_builder": lambda spec, props: lpv_mod.build(spec, props, alpha_scale=0.0,
                                                             schedule_tau_b=True),
           "name": "a1_uaphys_rerun"},
    "a9": {"builder": lambda spec, props: lpv_mod.build(spec, props, alpha_scale=1.0,
                                                        schedule_tau_b=False),
           "gate_builder": lambda spec, props: lpv_mod.build(spec, props, alpha_scale=0.0,
                                                             schedule_tau_b=False),
           "name": "a9_mixonly_rerun"},
    "lpv_free": {"builder": lambda spec, props: lpv_mod.build(spec, props, alpha_scale=1.0,
                                                              schedule_tau_b=True),
                 "gate_builder": lambda spec, props: lpv_mod.build(spec, props, alpha_scale=0.0,
                                                                   schedule_tau_b=True),
                 "name": "lpv_free_rerun"},
    "a2": {"builder": None, "gate_builder": None,
           "name": "a2_steady_rerun", "steady": True},
}


def arm_spec(arm: str):
    if ARMS[arm].get("steady"):
        return ms._base("t1", ARMS[arm]["name"], 0, boundary_mode="oracle",
                        initial_state_mode="steady", closure_mode="conservative_norew",
                        epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
    return probe_spec(ARMS[arm]["name"])


@torch.no_grad()
def identity_gate(arm: str, control_sd, view, props) -> dict:
    """Mechanism-OFF must reproduce the control bit-for-bit.

    alpha-scaled arms (a6/a46/a1/a9/lpv_free) gate via alpha_scale=0.0 (the LPV
    sanity convention; raw=0 gives sigmoid 0.5 -> half-open, NOT identity).
    a4 gates via k_w_raw=0 (tanh form -> factor 1).  a2 is a config-level
    ablation (steady initial_state_mode), no gate."""
    spec = arm_spec(arm)
    gate_builder = ARMS[arm].get("gate_builder")
    if gate_builder is None:
        return {"exact": True, "max_abs_temp_diff_c": 0.0,
                "note": "config-level ablation, no mechanism gate"}
    torch.manual_seed(0)
    base = build_world_model(spec, props).to(DEVICE).eval()
    base.load_state_dict(control_sd, strict=False)
    arm_m = gate_builder(spec, props).to(DEVICE).eval()
    arm_m.load_state_dict(control_sd, strict=False)
    # a4: k_w_raw is tanh-form, init 0 -> factor 1 (already identity); belt&braces
    for name in ("k_w_raw",):
        if name in arm_m.transition._parameters:
            arm_m.transition._parameters[name].data.zero_()
    batch = sample_windows(view, SPLIT_TRAIN, 8, 96, 18,
                           torch.Generator().manual_seed(7))
    h = HistoryWindow(batch.history.obs.to(DEVICE), batch.history.actions.to(DEVICE),
                      batch.history.boundary.to(DEVICE))
    r0 = base.forecast(h, batch.future_actions.to(DEVICE), boundary_mode="oracle",
                       true_future_boundary=batch.future_boundary.to(DEVICE))
    r1 = arm_m.forecast(h, batch.future_actions.to(DEVICE), boundary_mode="oracle",
                        true_future_boundary=batch.future_boundary.to(DEVICE))
    diff = float((r0.temps_mu - r1.temps_mu).abs().max())
    return {"exact": bool(torch.equal(r0.temps_mu, r1.temps_mu)),
            "max_abs_temp_diff_c": diff}


@torch.no_grad()
def paired_eval(arm_model, control, view) -> dict:
    gen = torch.Generator().manual_seed(EVAL_SEED)
    err_a, err_c, loads, days = [], [], [], []
    done = 0
    while done < N_WIN:
        bsz = min(32, N_WIN - done)
        batch = sample_windows(view, SPLIT_VAL, bsz, 96, 18, gen)
        h = HistoryWindow(batch.history.obs.to(DEVICE), batch.history.actions.to(DEVICE),
                          batch.history.boundary.to(DEVICE))
        ra = arm_model.forecast(h, batch.future_actions.to(DEVICE), boundary_mode="oracle",
                                true_future_boundary=batch.future_boundary.to(DEVICE))
        rc = control.forecast(h, batch.future_actions.to(DEVICE), boundary_mode="oracle",
                              true_future_boundary=batch.future_boundary.to(DEVICE))
        tgt = batch.future_obs.to(DEVICE)
        err_a.append((tgt - ra.temps_mu).abs().cpu())
        err_c.append((tgt - rc.temps_mu).abs().cpu())
        loads.append(batch.future_boundary[:, 0, 0])
        days.append(batch.day_ids)
        done += bsz
    abs_a = torch.cat(err_a)
    abs_c = torch.cat(err_c)
    load = torch.cat(loads)
    day_ids = torch.cat(days)
    from a5_water_coal_probe import _error_summary
    sa = _error_summary(abs_a, load, day_ids)
    sc = _error_summary(abs_c, load, day_ids)
    return {
        "arm": sa, "matched_filtered_control": sc,
        "relative_h18_change": (sa["overall_h18_mae"] - sc["overall_h18_mae"]) / sc["overall_h18_mae"],
        "spread_change": (sa["spread_ratio"] - sc["spread_ratio"]) / sc["spread_ratio"],
    }


def run_arm(arm: str, record, view, props, control_sd, wins) -> dict:
    cfg = ARMS[arm]
    arm_out = OUT / f"arm_{arm}"
    if (arm_out / "report.json").exists():
        print(f"[{arm}] report exists, skip (resume safe)")
        return json.loads((arm_out / "report.json").read_text(encoding="utf-8"))

    from probe_guard import assert_grid, verify_ledger_properties
    assert_grid(props)
    import src.final_wm.training as training
    original_builder = training.build_world_model
    if cfg["builder"] is not None:
        def builder(spec, properties=None):
            return cfg["builder"](spec, properties)
        training.build_world_model = builder
    try:
        final = train_arm(arm_spec(arm), view, arm_out, device=DEVICE,
                          properties=props, compile_substep=False)
    finally:
        training.build_world_model = original_builder
    verify_ledger_properties(arm_out)

    spec = arm_spec(arm)
    if cfg["builder"] is not None:
        arm_model = cfg["builder"](spec, props).to(DEVICE)
    else:
        arm_model = build_world_model(spec, props).to(DEVICE)
    ckpt = arm_out / "checkpoints" / f"{final['run_id']}.pt"
    arm_model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                         weights_only=False)["state_dict"])
    arm_model.eval()

    control = build_world_model(probe_spec("a5_filtered_control"), props).to(DEVICE)
    control.load_state_dict(control_sd)
    control.eval()

    ev = paired_eval(arm_model, control, view)
    oof_arm = sweep(arm_model, wins)
    oof_ctl = sweep(control, wins)
    directions = direction_metrics(arm_model, view)
    direction_pass = direction_passed(directions)

    a, c = ev["arm"], ev["matched_filtered_control"]
    accuracy_pass = a["overall_h18_mae"] <= c["overall_h18_mae"] * 0.95
    spread_pass = a["spread_ratio"] <= c["spread_ratio"] * 1.10
    if a["overall_h18_mae"] >= c["overall_h18_mae"] * 1.05 or not direction_pass:
        outcome = "REJECT_EXPLORATORY_SEED0"
    elif accuracy_pass and spread_pass:
        outcome = "PROMOTE_TO_FIXED_SEEDS_1_2"
    else:
        outcome = "INCONCLUSIVE_EXPLORATORY_SEED0"

    report = {
        "arm": cfg["name"], "status": outcome, "single_seed_exploratory": True,
        "protocol": "PREREG_protocol_fix_20260829.md (v2.2 record + A5 gates)",
        "train": {k: final[k] for k in ("run_id", "commit", "best_val_nll", "best_epoch",
                                        "epochs_run", "stop_reason", "converged",
                                        "flags", "timing")},
        "evaluation_256": ev, "oof": {"arm": oof_arm, "control": oof_ctl,
                                      "relative_h18_change": (oof_arm["overall_h18_mae"]
                                                              - oof_ctl["overall_h18_mae"])
                                      / oof_ctl["overall_h18_mae"],
                                      "spread_change": (oof_arm["spread_ratio"]
                                                        - oof_ctl["spread_ratio"])
                                      / oof_ctl["spread_ratio"]},
        "direction_v03": directions,
        "preregistered_gates": {"accuracy_at_least_5pct": bool(accuracy_pass),
                                "spread_no_more_than_10pct_worse": bool(spread_pass),
                                "both_valves_h18_h60_direction": bool(direction_pass),
                                "all_pass": bool(accuracy_pass and spread_pass and direction_pass)},
        "decision_comparator": "reused A5 matched filtered control",
    }
    (arm_out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"[RESULT {arm}] {outcome} 256:{a['overall_h18_mae']:.4f} vs {c['overall_h18_mae']:.4f} "
          f"({ev['relative_h18_change']*100:+.1f}%) OOF:{oof_arm['overall_h18_mae']:.4f} "
          f"vs {oof_ctl['overall_h18_mae']:.4f} ({report['oof']['relative_h18_change']*100:+.1f}%)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity", action="store_true")
    mode.add_argument("--arm", choices=sorted(ARMS))
    mode.add_argument("--queue", action="store_true")
    args = parser.parse_args()

    record = WaterCoalRecord(ROOT / "artifacts/final_wm/canonical_sideA_v2.npz")
    view = BaseRecordView(record)
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz",
                                 device=DEVICE)
    control_sd = torch.load(CONTROL_CKPT, map_location="cpu",
                            weights_only=False)["state_dict"]
    wins = build_windows(record, view)

    if args.sanity:
        for arm in sorted(ARMS):
            try:
                g = identity_gate(arm, control_sd, view, props)
                print(f"[sanity {arm}] exact={g['exact']} max_diff={g['max_abs_temp_diff_c']:.3e}")
            except Exception as e:
                print(f"[sanity {arm}] FAILED: {type(e).__name__}: {str(e)[:150]}")
        return

    arms = [args.arm] if args.arm else ["a4", "a46", "a6", "a1", "a9", "lpv_free", "a2"]
    for arm in arms:
        try:
            run_arm(arm, record, view, props, control_sd, wins)
        except Exception:
            print(f"[{arm}] FAILED:\n{traceback.format_exc()}")
    print("queue finished")


if __name__ == "__main__":
    main()
