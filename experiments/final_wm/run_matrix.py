"""Unified Linux execution entry for the final world model matrix.

Phases (each idempotent, artifacts appended read-only):

    python experiments/final_wm/run_matrix.py --phase discover --data-root /home/bluster/Desktop/AI --out artifacts/final_wm
    python experiments/final_wm/run_matrix.py --phase build    --data-root ... --mapping configs/final_wm/channel_mapping.json --out artifacts/final_wm
    python experiments/final_wm/run_matrix.py --phase dsyn     --out artifacts/final_wm
    python experiments/final_wm/run_matrix.py --phase matrix   --record artifacts/final_wm/canonical.npz --out artifacts/final_wm

`--quick` is a dry-run/smoke mode (tiny sizes, no verdicts).  Execution
must not edit code or thresholds; failures are reported as-is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import (
    SPLIT_VAL,
    CanonicalRecord,
    build_canonical,
    discover_dataset,
    import_dual_canonical,
)
from src.final_wm.diagnostics import leakage_probe
from src.final_wm.evaluation import (
    WindowMetrics,
    boundary_forecast_metrics,
    evaluate_windows,
    horizon_summary,
    persistence_boundary_metrics,
    relative_improvement_ci,
    residual_quantiles,
    step_response_direction,
)
from src.final_wm.properties import AnalyticThermoProperties, load_grid_properties
from src.final_wm.synthetic import synthetic_canonical_arrays
from src.final_wm.training import build_world_model, config_fingerprint, train_arm

from experiments.final_wm import matrix_spec as ms


def closure_blindness_check(model, device) -> dict:
    """Runtime action/W blindness check for the closure (R1).

    The W channel (spray_flow_total, excluded from the closure whitelist) must
    not move closure output.  Extracted as a top-level function so the import
    path is covered by tests (the original inline version referenced a
    non-existent CHANNEL_INDEX and crashed the first Linux run).
    """
    from src.final_wm.contracts import BOUNDARY_ELEMENTS, CLOSURE_BOUNDARY_CHANNELS

    if "spray_flow_total" in CLOSURE_BOUNDARY_CHANNELS:
        raise FinalWMProtocolError("closure whitelist must exclude spray_flow_total")
    batch_state = torch.zeros(2, model.layout.dim, device=device)
    boundary = torch.zeros(2, len(BOUNDARY_ELEMENTS), device=device)
    w_idx = BOUNDARY_ELEMENTS.index("spray_flow_total")
    boundary[:, w_idx] = 5.0
    with torch.no_grad():
        out_w = model.closure(batch_state, boundary)
        boundary[:, w_idx] = 0.0
        out_wo = model.closure(batch_state, boundary)
    blind = (
        torch.allclose(out_w.steam_power, out_wo.steam_power)
        and torch.allclose(out_w.metal_power, out_wo.metal_power)
    )
    if out_w.latent_step is not None or out_wo.latent_step is not None:
        blind = blind and (
            out_w.latent_step is not None
            and out_wo.latent_step is not None
            and torch.allclose(out_w.latent_step, out_wo.latent_step)
        )
    return {"runtime_blind_ok": bool(blind)}


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _properties(path: str | None):
    if path:
        return load_grid_properties(path)
    return AnalyticThermoProperties()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase: discover / build
# ---------------------------------------------------------------------------

def phase_discover(args) -> None:
    report = discover_dataset(args.data_root)
    out = Path(args.out)
    _write_json(out / "d0_discovery_report.json", report)
    print(f"[d0] discovery report written: {out / 'd0_discovery_report.json'} "
          f"({len(report['files'])} files)")


def phase_build(args) -> None:
    if not args.mapping:
        raise FinalWMProtocolError("--mapping is required for the build phase")
    out = Path(args.out)
    record_path = args.record or str(out / "canonical.npz")
    report = build_canonical(args.data_root, args.mapping, record_path)
    _write_json(out / "d0_quality_report.json", {
        "gap_ratio": report.gap_ratio,
        "stuck_ratio": report.stuck_ratio,
        "valve_active_ratio": report.valve_active_ratio,
        "days": report.days,
    })
    print(f"[d0] canonical record built: {record_path} (gates passed)")


def phase_split_sides(args) -> None:
    """Bridge the D0 dual-side record into per-side registry-schema records."""
    if not args.record:
        raise FinalWMProtocolError("--record (dual canonical npz) required for split-sides")
    written = import_dual_canonical(args.record, args.out)
    for side, path in written.items():
        print(f"[d0] side {side} record: {path} (gates passed)")
    _write_json(Path(args.out) / "split_sides_report.json", {"records": written})


# ---------------------------------------------------------------------------
# Phase: D-SYN same-type solvability gate
# ---------------------------------------------------------------------------

def run_dsyn(args) -> dict:
    device = _device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = (0,) if args.quick else ms.SEEDS
    total_steps = 2_000 if args.quick else 20_000
    results = []
    for seed in seeds:
        torch.manual_seed(10_000 + seed)
        teacher_spec = ms._base("dsyn", "teacher", seed, closure_mode="conservative")
        teacher = build_world_model(teacher_spec, AnalyticThermoProperties()).to(device)
        with torch.no_grad():  # deterministic same-type perturbation away from priors
            gen = torch.Generator().manual_seed(20_000 + seed)
            for name, p in teacher.transition.named_parameters():
                if name.startswith("raw_"):
                    p.add_(0.15 * torch.randn(p.shape, generator=gen).to(device))
        arrays = synthetic_canonical_arrays(total_steps, seed=30_000 + seed, teacher=teacher.transition)
        record_path = out / f"dsyn_record_seed{seed}.npz"
        np.savez_compressed(record_path, **arrays)
        record = CanonicalRecord(record_path)

        skeleton = build_world_model(
            ms._base("dsyn", "skeleton", seed), AnalyticThermoProperties()
        ).to(device)
        skeleton_val = float(evaluate_windows(
            skeleton, record, SPLIT_VAL, n_windows=32 if args.quick else 128, batch_size=16,
            history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON, boundary_mode="oracle",
            seed=seed, device=device,
        ).nll.mean())

        student_spec = ms._base(
            "dsyn", "student", seed, boundary_mode="oracle",
            initial_state_mode="learned", closure_mode="conservative",
        )
        if args.quick:
            student_spec = ms.quicken(student_spec)
        final = train_arm(student_spec, record, out, device=device, properties=AnalyticThermoProperties())
        # Matrix rule: student must cut the skeleton's validation NLL by >=30%
        # of its magnitude (sign-agnostic gap criterion).
        improvement = skeleton_val - final["best_val_nll"]
        passed = bool(improvement >= 0.3 * abs(skeleton_val))
        results.append({
            "seed": seed,
            "skeleton_val_nll": skeleton_val,
            "student_val_nll": final["best_val_nll"],
            "improvement": improvement,
            "pass": passed,
        })
        print(f"[dsyn] seed={seed} skeleton={skeleton_val:.3f} student={final['best_val_nll']:.3f} pass={passed}")
    verdict = {
        "unit": "dsyn",
        "per_seed": results,
        "passes": sum(r["pass"] for r in results),
        "verdict": "PASS" if sum(r["pass"] for r in results) >= ms.MIN_SEED_PASSES or args.quick else "FAIL",
        "quick": bool(args.quick),
    }
    _write_json(out / "dsyn_verdict.json", verdict)
    return verdict


# ---------------------------------------------------------------------------
# Phase: matrix units
# ---------------------------------------------------------------------------

def _save_metrics(out: Path, run_id: str, metrics, spec, final) -> Path:
    path = out / "metrics" / f"{run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metrics": {
            "nll": metrics.nll, "mae": metrics.mae, "crps": metrics.crps,
            "day_ids": metrics.day_ids,
        },
        "final": final,
        "fingerprint": config_fingerprint(spec),
    }, path)
    return path


def _save_eval_metrics(out: Path, run_id: str, metrics) -> Path:
    """Flat storage for secondary evaluation metrics (B1 boundary, J1 staged).
    These are audit artifacts, not resume keys."""
    path = out / "metrics" / f"{run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "nll": metrics.nll, "mae": metrics.mae, "crps": metrics.crps, "day_ids": metrics.day_ids,
    }, path)
    return path


def _spec_matches_ledger(out: Path, run_id: str, spec) -> bool:
    """Legacy-resume check: the ledger's final entry for run_id (last occurrence
    wins, per the duplicate-block convention) must carry the identical spec."""
    from dataclasses import asdict

    ledger = Path(out) / "ledger.jsonl"
    if not ledger.exists():
        return False
    want = asdict(spec)
    found = None
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("run_id") == run_id and entry.get("final"):
            found = entry.get("spec")
    return found == want


def _try_resume(spec, out) -> tuple[dict, WindowMetrics] | None:
    """Skip-if-artifacts-exist resume: a run is reused iff its checkpoint and
    metrics exist AND the stored fingerprint matches (new format) or the
    ledger's final-entry spec matches (legacy artifacts from the first run)."""
    run_id = f"{spec.unit}_{spec.arm}_seed{spec.seed}"
    ckpt = Path(out) / "checkpoints" / f"{run_id}.pt"
    mpath = Path(out) / "metrics" / f"{run_id}.pt"
    if not (ckpt.exists() and mpath.exists()):
        return None
    blob = torch.load(mpath, map_location="cpu", weights_only=False)
    if "metrics" in blob:  # new format
        if blob.get("fingerprint") != config_fingerprint(spec):
            return None
        m, final = blob["metrics"], blob.get("final") or {}
    else:  # legacy flat format from the first Linux run
        if not _spec_matches_ledger(out, run_id, spec):
            return None
        m, final = blob, {}
    metrics = WindowMetrics(**m)
    final = dict(final, run_id=run_id, resumed=True)
    final.setdefault("best_val_nll", float("nan"))
    return final, metrics


def _train_and_eval(spec, record, out, device, properties, quick):
    if quick:
        spec = ms.quicken(spec)
    resumed = _try_resume(spec, out)
    if resumed is not None:
        final, metrics = resumed
        print(f"[{spec.unit}] {spec.arm} seed={spec.seed} RESUMED (artifacts match spec) "
              f"eval={horizon_summary(metrics)}")
        return final, metrics
    final = train_arm(spec, record, out, device=device, properties=properties)
    run_id = final["run_id"]
    ckpt = out / "checkpoints" / f"{run_id}.pt"
    model = build_world_model(spec, properties).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    n_eval = 32 if quick else 256
    metrics = evaluate_windows(
        model, record, SPLIT_VAL, n_windows=n_eval, batch_size=32,
        history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
        boundary_mode=spec.boundary_mode, seed=50_000 + spec.seed, device=device,
    )
    _save_metrics(out, run_id, metrics, spec, final)
    print(f"[{spec.unit}] {spec.arm} seed={spec.seed} best_val={final['best_val_nll']:.3f} "
          f"eval={horizon_summary(metrics)}")
    return final, metrics


def _seed_passes(pairs, threshold, metric="nll"):
    """pairs: list of (baseline_metrics, arm_metrics) per seed."""
    passes, details = 0, []
    for base, arm in pairs:
        ci = relative_improvement_ci(base, arm, horizon=ms.HORIZON, metric=metric)
        ok = ci.point >= threshold and ci.ci_lo > 0.0
        passes += int(ok)
        details.append({"point": ci.point, "ci_lo": ci.ci_lo, "ci_hi": ci.ci_hi,
                        "n_days": ci.n_days, "pass": bool(ok)})
    return passes, details


def _verdict(passes, n_seeds):
    if passes >= ms.MIN_SEED_PASSES:
        return "SUPPORTED"
    if passes == 0:
        return "REJECTED"
    return "MIXED"


# ---------------------------------------------------------------------------
# auditpack: protocolized evidence-chain analyses (audit 2026-08-20 §P3)
# ---------------------------------------------------------------------------

def run_auditpack(args) -> dict:
    """Record-only analyses always run; model-based probes run when
    --checkpoint is given (model rebuilt from the matching T1 spec)."""
    from src.final_wm.analysis import (
        binning_stats,
        error_floor_anchors,
        event_study_summary,
        mixing_cooling_reference,
        persistence_increment_mae,
        rewetting_ablation,
        spray_sensitivity,
        valve_step_events,
        window_abs_errors,
    )

    device = _device(args.device)
    record = CanonicalRecord(args.record)
    out = Path(args.out)
    report: dict = {"record": str(args.record), "matrix_version": ms.MATRIX_VERSION}
    sensitivity = spray_sensitivity(record, SPLIT_VAL)
    report["spray_sensitivity"] = sensitivity
    report["mixing_reference"] = {
        "v1": mixing_cooling_reference(sensitivity["dW_dv1_kgs_per_2pct"]),
        "v2": mixing_cooling_reference(sensitivity["dW_dv2_kgs_per_2pct"]),
    }
    report["persistence_increment_mae"] = persistence_increment_mae(record, SPLIT_VAL)
    report["error_floor"] = error_floor_anchors(record, SPLIT_VAL)
    report["event_study"] = {
        f"v{v + 1}": event_study_summary(valve_step_events(record, SPLIT_VAL, v))
        for v in (0, 1)
    }
    if args.checkpoint:
        arm, seed = args.arm, int(args.seed)
        spec = next(s for s in ms.t1_specs((seed,)) if s.arm == arm)
        properties = _properties(args.properties_npz)
        report["properties_probes"] = type(properties).__name__
        model = build_world_model(spec, properties).to(device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=False)["state_dict"])
        errors = window_abs_errors(
            model, record, SPLIT_VAL, n_windows=64 if args.quick else 512, batch_size=32,
            history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
            boundary_mode="oracle", seed=110_000 + seed, device=device,
        )
        report["residual_binning"] = binning_stats(errors)
        report["rewetting_ablation"] = rewetting_ablation(
            model, record, SPLIT_VAL, n_windows=16 if args.quick else 64,
            history_steps=ms.HISTORY_STEPS, seed=120_000 + seed, device=device,
        )
    name = f"auditpack{('_' + args.side) if args.side else ''}.json"
    _write_json(out / name, report)
    print(f"[auditpack] written: {out / name}")
    return report


def run_matrix(args) -> dict:
    device = _device(args.device)
    properties = _properties(args.properties_npz)
    out = Path(args.out)
    record = CanonicalRecord(args.record)
    units = args.units.split(",") if args.units else ["o1", "t1", "b1", "j1", "r1"]
    quick = bool(args.quick)
    seeds = (0,) if quick else ms.SEEDS
    summary: dict = {
        "quick": quick,
        "matrix_version": ms.MATRIX_VERSION,
        "side": args.side,
        "record": str(args.record),
        "properties": type(properties).__name__,
        "units": {},
    }
    summary_name = "matrix_summary.json" if not args.side else f"matrix_summary_side{args.side}.json"

    def dump_summary() -> None:
        # Incremental verdict persistence: a crash in a later unit must not
        # lose verdicts already computed (the first Linux run lost O1..J1 to
        # the R1 crash).  Rewritten after every unit.
        _write_json(out / summary_name, summary)

    metrics_store: dict[str, object] = {}

    if "o1" in units:
        for spec in ms.o1_specs(seeds):
            final, metrics = _train_and_eval(spec, record, out, device, properties, quick)
            metrics_store[final["run_id"]] = metrics
        if not quick:
            unit_verdicts = {}
            for arm in ("learned", "hybrid"):
                pairs = [
                    (metrics_store[f"o1_steady_seed{s}"], metrics_store[f"o1_{arm}_seed{s}"])
                    for s in seeds
                ]
                passes, details = _seed_passes(pairs, ms.THRESH_O1_NLL)
                unit_verdicts[arm] = {"verdict": _verdict(passes, len(seeds)), "per_seed": details}
            summary["units"]["o1"] = unit_verdicts
        dump_summary()

    if "t1" in units:
        for spec in ms.t1_specs(seeds):
            final, metrics = _train_and_eval(spec, record, out, device, properties, quick)
            metrics_store[final["run_id"]] = metrics
        if not quick:
            unit_verdicts = {}
            nested = [
                ("closure_cons", "physics_only"),
                ("closure_steam", "closure_cons"),
                ("latent4", "closure_cons"),
            ]
            for arm, base_arm in nested:
                pairs = [
                    (metrics_store[f"t1_{base_arm}_seed{s}"], metrics_store[f"t1_{arm}_seed{s}"])
                    for s in seeds
                ]
                passes, details = _seed_passes(pairs, ms.THRESH_T1_NLL)
                unit_verdicts[f"{arm}_vs_{base_arm}"] = {
                    "verdict": _verdict(passes, len(seeds)), "per_seed": details
                }
            summary["units"]["t1"] = unit_verdicts
        dump_summary()

    if "b1" in units:
        b_metrics = {}
        for spec in ms.b1_specs(seeds):
            final, _ = _train_and_eval(spec, record, out, device, properties, quick)
            model = build_world_model(spec, properties).to(device)
            ckpt = out / "checkpoints" / f"{final['run_id']}.pt"
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
            b_metrics[spec.seed] = boundary_forecast_metrics(
                model, record, SPLIT_VAL, n_windows=32 if quick else 256, batch_size=32,
                history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON, seed=60_000 + spec.seed, device=device,
            )
            _save_eval_metrics(out, f"{final['run_id']}_boundary", b_metrics[spec.seed])
        if not quick:
            pairs = []
            for s in seeds:
                base = persistence_boundary_metrics(
                    record, SPLIT_VAL, n_windows=256, batch_size=32,
                    history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON, seed=60_000 + s,
                )
                pairs.append((base, b_metrics[s]))
            passes, details = _seed_passes(pairs, ms.THRESH_B1_CRPS, metric="crps")
            summary["units"]["b1"] = {"verdict": _verdict(passes, len(seeds)), "per_seed": details}
        dump_summary()

    if "j1" in units:
        joint_metrics, staged_metrics = {}, {}
        for spec in ms.j1_specs(seeds):
            final, metrics = _train_and_eval(spec, record, out, device, properties, quick)
            metrics_store[final["run_id"]] = metrics
            if spec.arm == "joint":
                joint_metrics[spec.seed] = metrics
        for seed in seeds:
            main_ckpt = out / "checkpoints" / f"j1_staged_main_seed{seed}.pt"
            bnd_spec = ms.j1_staged_boundary_spec(seed, str(main_ckpt))
            if quick:
                bnd_spec = ms.quicken(bnd_spec)
            bnd_run_id = f"{bnd_spec.unit}_{bnd_spec.arm}_seed{bnd_spec.seed}"
            bnd_ckpt = out / "checkpoints" / f"{bnd_run_id}.pt"
            if bnd_ckpt.exists() and _spec_matches_ledger(out, bnd_run_id, bnd_spec):
                print(f"[j1] staged_boundary seed={seed} RESUMED (artifacts match spec)")
            else:
                final = train_arm(bnd_spec, record, out, device=device, properties=properties)
                bnd_run_id = final["run_id"]
            model = build_world_model(
                ms._base("j1", "staged", seed, boundary_mode="forecast", train_boundary=True,
                         initial_state_mode="hybrid", closure_mode="conservative"),
                properties,
            ).to(device)
            model.load_state_dict(torch.load(out / "checkpoints" / f"{bnd_run_id}.pt",
                                             map_location=device, weights_only=False)["state_dict"])
            staged_metrics[seed] = evaluate_windows(
                model, record, SPLIT_VAL, n_windows=32 if quick else 256, batch_size=32,
                history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON, boundary_mode="forecast",
                seed=70_000 + seed, device=device,
            )
            _save_eval_metrics(out, f"j1_staged_seed{seed}", staged_metrics[seed])
        if not quick:
            pairs = [(staged_metrics[s], joint_metrics[s]) for s in seeds]
            passes, details = _seed_passes(pairs, ms.THRESH_J1_NLL)
            summary["units"]["j1"] = {"verdict": _verdict(passes, len(seeds)), "per_seed": details}
        dump_summary()

    if "r1" in units:
        r1_reports = []
        for seed in seeds:
            ckpt = out / "checkpoints" / f"t1_closure_cons_seed{seed}.pt"
            if not ckpt.exists():
                r1_reports.append({"seed": seed, "error": "missing t1 closure_cons checkpoint"})
                continue
            spec = ms._base("t1", "closure_cons", seed, boundary_mode="oracle",
                            initial_state_mode="hybrid", closure_mode="conservative")
            model = build_world_model(spec, properties).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
            blind_ok = closure_blindness_check(model, device)["runtime_blind_ok"]
            direction_on = step_response_direction(
                model, record, SPLIT_VAL, n_windows=16 if quick else 32,
                history_steps=ms.HISTORY_STEPS, seed=80_000 + seed, device=device,
            )
            leak = leakage_probe(
                model, record, n_windows=64 if quick else 512,
                history_steps=ms.HISTORY_STEPS, epochs=3 if quick else 20,
                seed=90_000 + seed, device=device,
            )
            quant = residual_quantiles(
                model, record, SPLIT_VAL, n_windows=16 if quick else 64,
                history_steps=ms.HISTORY_STEPS, seed=seed, device=device,
            )
            r1_reports.append({
                "seed": seed,
                "runtime_blind_ok": blind_ok,
                "direction": direction_on,
                "leakage": leak,
                "residual_quantiles": quant,
            })
        verdict = "SUPPORTED"
        for rep in r1_reports:
            if "error" in rep:
                verdict = "MIXED"
                continue
            if not rep["runtime_blind_ok"] or rep["leakage"]["leakage_suspected"]:
                verdict = "REJECTED"
            if rep["direction"]["frac_negative"] < 1.0:
                verdict = "REJECTED"
        summary["units"]["r1"] = {"verdict": verdict, "reports": r1_reports}
        _write_json(out / "r1_report.json", summary["units"]["r1"])
        dump_summary()

    dump_summary()
    print(f"[matrix] summary written: {out / summary_name}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=["discover", "build", "split-sides", "dsyn", "matrix", "auditpack"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--record", default=None)
    parser.add_argument("--side", default=None, choices=["A", "B"],
                        help="label for per-side matrix runs (recorded in the summary)")
    parser.add_argument("--out", default="artifacts/final_wm")
    parser.add_argument("--units", default=None, help="comma-separated subset: o1,t1,b1,j1,r1")
    parser.add_argument("--properties-npz", default=None, help="real IAPWS grid (else analytic fallback)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true", help="dry-run sizes; no verdicts")
    parser.add_argument("--checkpoint", default=None, help="auditpack: trained model checkpoint")
    parser.add_argument("--arm", default="closure_cons", help="auditpack: T1 arm of --checkpoint")
    parser.add_argument("--seed", default=0, help="auditpack: seed of --checkpoint")
    args = parser.parse_args()

    if args.phase == "discover":
        if not args.data_root:
            raise FinalWMProtocolError("--data-root required for discover")
        phase_discover(args)
    elif args.phase == "build":
        if not args.data_root:
            raise FinalWMProtocolError("--data-root required for build")
        phase_build(args)
    elif args.phase == "split-sides":
        phase_split_sides(args)
    elif args.phase == "dsyn":
        run_dsyn(args)
    elif args.phase == "matrix":
        if not args.record:
            raise FinalWMProtocolError("--record required for matrix")
        run_matrix(args)
    elif args.phase == "auditpack":
        if not args.record:
            raise FinalWMProtocolError("--record required for auditpack")
        run_auditpack(args)


if __name__ == "__main__":
    main()
