#!/usr/bin/env python3
"""Frozen Linux runner for JEPA-B v1.

Examples:
  python experiments/final_wm/run_jepa_b.py --sanity
  python experiments/final_wm/run_jepa_b.py --queue

Full execution is registry-gated, single-GPU, sequential, seed0-only and
never reads test.  A completed arm may be reused only when its commit and
matrix hash match; partial arms are not retried automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.final_wm.jepa_b_spec import (
    ORDERED_ARMS,
    load_matrix,
    matrix_sha256,
    require_linux_authorization,
)
from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL
from src.final_wm.jepa import (
    JepaBModel,
    JepaBRecord,
    JepaWindowBatch,
    PrivilegedNormalizer,
    build_jepa_model,
    fit_privileged_normalizer,
    sample_jepa_windows,
)
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import AnalyticThermoProperties, load_grid_properties

DEFAULT_MATRIX = ROOT / "configs/final_wm/jepa_b_series_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/phase3_5/experiment_registry.json"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _device_batch(batch: JepaWindowBatch, device: torch.device) -> JepaWindowBatch:
    return JepaWindowBatch(*[
        HistoryWindow(
            batch.history.obs.to(device), batch.history.actions.to(device),
            batch.history.boundary.to(device),
        ),
        batch.future_boundary.to(device), batch.future_actions.to(device),
        batch.future_obs.to(device), batch.history_privileged.to(device),
        batch.future_privileged.to(device), batch.partner_history_privileged.to(device),
        batch.partner_future_privileged.to(device), batch.future_indices,
        batch.partner_future_indices, batch.day_ids, batch.unit_load,
    ])


def _fixed_indices(
    record: JepaBRecord, split_id: int, history_steps: int, horizon: int, n: int, seed: int
) -> torch.Tensor:
    pool = record.valid_window_starts(split_id, history_steps, horizon)
    gen = torch.Generator().manual_seed(seed)
    return pool[torch.randint(len(pool), (n,), generator=gen)]


def _anchor_hash(indices: torch.Tensor) -> str:
    return hashlib.sha256(indices.to(torch.int64).numpy().tobytes()).hexdigest()


def _loss(
    model: JepaBModel, batch: JepaWindowBatch, result, weights: dict
) -> tuple[torch.Tensor, dict[str, float]]:
    main = model.observation_nll(result.temps_mu, result.temps_sigma, batch.future_obs)
    terms = model.auxiliary_terms(batch, result)
    total = weights["observation_nll"] * main
    mapping = {
        "prediction": "jepa_prediction",
        "gaussian_cf": "gaussian_cf",
        "static": "b4_static",
        "dynamic": "b4_dynamic",
    }
    for name, value in terms.items():
        total = total + weights[mapping[name]] * value
    scalars = {"observation_nll": float(main.detach())}
    scalars.update({name: float(value.detach()) for name, value in terms.items()})
    scalars["total"] = float(total.detach())
    return total, scalars


@torch.no_grad()
def _validation_nll(
    model: JepaBModel,
    record: JepaBRecord,
    indices: torch.Tensor,
    history_steps: int,
    horizon: int,
    device: torch.device,
) -> float:
    model.eval()
    values = []
    for start in range(0, len(indices), 32):
        chunk = indices[start:start + 32]
        batch = sample_jepa_windows(
            record, SPLIT_VAL, len(chunk), history_steps, horizon,
            torch.Generator().manual_seed(0), fixed_indices=chunk,
        )
        batch = _device_batch(batch, device)
        result = model.forecast(
            batch.history, batch.future_actions, boundary_mode="oracle",
            true_future_boundary=batch.future_boundary,
        )
        values.append(model.observation_nll(
            result.temps_mu, result.temps_sigma, batch.future_obs
        ).detach().cpu())
    return float(torch.stack(values).mean())


def _arm_paths(out_root: Path, arm: str) -> tuple[Path, Path, Path]:
    arm_dir = out_root / arm
    return arm_dir, arm_dir / "ledger.jsonl", arm_dir / "checkpoints" / f"jepa_b_{arm}_seed0.pt"


def _verified_existing_train(
    arm: str,
    report_path: Path,
    ledger_path: Path,
    checkpoint_path: Path,
    commit: str,
    matrix_hash: str,
) -> dict:
    """Return a reusable train summary only when every bound artifact agrees."""
    if not ledger_path.exists() or not checkpoint_path.exists():
        raise FinalWMProtocolError(f"{arm} report exists without complete ledger/checkpoint")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ledger_lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    if not ledger_lines:
        raise FinalWMProtocolError(f"{arm} ledger is empty")
    final = json.loads(ledger_lines[-1])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    artifacts = (report, final, checkpoint)
    if any(item.get("arm") != arm for item in artifacts):
        raise FinalWMProtocolError(f"{arm} reusable artifacts have an arm mismatch")
    if any(item.get("commit") != commit for item in artifacts):
        raise FinalWMProtocolError(f"{arm} reusable artifacts have a commit mismatch")
    if any(item.get("matrix_sha256") != matrix_hash for item in artifacts):
        raise FinalWMProtocolError(f"{arm} reusable artifacts have a matrix mismatch")
    if final.get("final") is not True or not isinstance(report.get("train"), dict):
        raise FinalWMProtocolError(f"{arm} reusable artifacts are not final")
    return report["train"]


def train_arm(
    arm: str,
    matrix: dict,
    matrix_hash: str,
    record: JepaBRecord,
    normalizer: PrivilegedNormalizer,
    properties,
    out_root: Path,
    device: torch.device,
    *,
    quick: bool,
) -> dict:
    arm_dir, ledger_path, checkpoint_path = _arm_paths(out_root, arm)
    report_path = arm_dir / "report.json"
    commit = _git_commit()
    if report_path.exists():
        existing = _verified_existing_train(
            arm, report_path, ledger_path, checkpoint_path, commit, matrix_hash
        )
        print(f"[{arm}] complete artifact set matches; reuse")
        return existing
    if ledger_path.exists() or checkpoint_path.exists():
        raise FinalWMProtocolError(f"{arm} has partial artifacts; automatic retry is forbidden")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = build_jepa_model(
        arm, history_steps=matrix["data_contract"]["history_steps"],
        properties=properties, normalizer=normalizer,
    ).to(device)
    train_cfg = matrix["training"]
    epochs = 2 if quick else train_cfg["epochs"]
    batches_per_epoch = 2 if quick else train_cfg["batches_per_epoch"]
    batch_size = 4 if quick else train_cfg["batch_size"]
    patience = 2 if quick else train_cfg["patience"]
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    generator = torch.Generator().manual_seed(0)
    history_steps = matrix["data_contract"]["history_steps"]
    horizon = matrix["data_contract"]["training_horizon"]
    val_indices = _fixed_indices(
        record, SPLIT_VAL, history_steps, horizon, 8 if quick else 64, 10_000
    )
    best = float("inf")
    best_epoch = -1
    since_best = 0
    started = time.time()
    stop_reason = "cap"
    with ledger_path.open("x", encoding="utf-8") as ledger:
        for epoch in range(epochs):
            model.train()
            aggregate: dict[str, float] = {}
            for _ in range(batches_per_epoch):
                raw = sample_jepa_windows(
                    record, SPLIT_TRAIN, batch_size, history_steps, horizon, generator
                )
                batch = _device_batch(raw, device)
                result = model.forecast(
                    batch.history, batch.future_actions, boundary_mode="oracle",
                    true_future_boundary=batch.future_boundary,
                )
                loss, scalars = _loss(model, batch, result, matrix["losses"])
                if not bool(torch.isfinite(loss)):
                    raise FinalWMProtocolError(f"{arm} produced non-finite loss")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                opt.step()
                for key, value in scalars.items():
                    aggregate[key] = aggregate.get(key, 0.0) + value
            aggregate = {key: value / batches_per_epoch for key, value in aggregate.items()}
            val_nll = _validation_nll(
                model, record, val_indices, history_steps, horizon, device
            )
            entry = {
                "arm": arm, "seed": 0, "epoch": epoch, "commit": commit,
                "matrix_sha256": matrix_hash, "validation_anchor_sha256": _anchor_hash(val_indices),
                "train": aggregate, "val_nll": val_nll,
                "wall_seconds": time.time() - started,
            }
            ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")
            ledger.flush()
            if val_nll < best - 1e-4:
                best, best_epoch, since_best = val_nll, epoch, 0
                torch.save({
                    "state_dict": model.state_dict(), "arm": arm, "seed": 0,
                    "commit": commit, "matrix_sha256": matrix_hash,
                    "validation_anchor_sha256": _anchor_hash(val_indices),
                }, checkpoint_path)
            else:
                since_best += 1
                if since_best >= patience:
                    stop_reason = "patience"
                    break
        final = {
            "arm": arm, "seed": 0, "final": True, "commit": commit,
            "matrix_sha256": matrix_hash, "best_val_nll": best,
            "best_epoch": best_epoch, "epochs_run": epoch + 1,
            "stop_reason": stop_reason, "wall_seconds": time.time() - started,
            "validation_anchor_sha256": _anchor_hash(val_indices),
        }
        ledger.write(json.dumps(final, ensure_ascii=False) + "\n")
    return final


def _load_arm(
    arm: str, matrix: dict, matrix_hash: str, normalizer: PrivilegedNormalizer, properties,
    out_root: Path, device: torch.device,
) -> JepaBModel:
    _arm_dir, _ledger, checkpoint = _arm_paths(out_root, arm)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("arm") != arm or payload.get("commit") != _git_commit():
        raise FinalWMProtocolError(f"{arm} checkpoint identity mismatch")
    if payload.get("matrix_sha256") != matrix_hash:
        raise FinalWMProtocolError(f"{arm} checkpoint matrix mismatch")
    model = build_jepa_model(
        arm, history_steps=matrix["data_contract"]["history_steps"],
        properties=properties, normalizer=normalizer,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


@torch.no_grad()
def evaluate_horizon(
    model: JepaBModel,
    record: JepaBRecord,
    indices: torch.Tensor,
    history_steps: int,
    horizon: int,
    device: torch.device,
    load_edges: list[float],
) -> dict:
    signed_all, load_all, day_all, persistence_all = [], [], [], []
    for start in range(0, len(indices), 32):
        chunk = indices[start:start + 32]
        raw = sample_jepa_windows(
            record, SPLIT_VAL, len(chunk), history_steps, horizon,
            torch.Generator().manual_seed(0), fixed_indices=chunk,
        )
        batch = _device_batch(raw, device)
        result = model.forecast(
            batch.history, batch.future_actions, boundary_mode="oracle",
            true_future_boundary=batch.future_boundary,
        )
        signed_all.append((result.temps_mu - batch.future_obs).cpu())
        persistence_all.append(
            (batch.history.obs[:, -1:, :].expand_as(batch.future_obs) - batch.future_obs).cpu()
        )
        load_all.append(raw.unit_load)
        day_all.append(raw.day_ids)
    signed = torch.cat(signed_all)
    persistence = torch.cat(persistence_all)
    load = torch.cat(load_all)
    days = torch.cat(day_all)
    terminal = signed[:, -1, -1]
    terminal_persistence = persistence[:, -1, -1]
    bin_mae: list[float | None] = []
    for lo, hi in zip(load_edges[:-1], load_edges[1:]):
        mask = (load >= lo) & (load < hi)
        bin_mae.append(float(terminal[mask].abs().mean()) if bool(mask.any()) else None)
    usable = [value for value in bin_mae if value is not None]
    spread = float(max(usable) / min(usable)) if len(usable) == len(bin_mae) else None
    unique_days = torch.unique(days)
    by_day = torch.stack([terminal[days == day].mean() for day in unique_days])
    return {
        "n_windows": len(indices),
        "anchor_sha256": _anchor_hash(indices),
        "all_channels_all_steps_mae": float(signed.abs().mean()),
        "final_outlet_terminal_mae": float(terminal.abs().mean()),
        "final_outlet_terminal_bias": float(terminal.mean()),
        "persistence_terminal_mae": float(terminal_persistence.abs().mean()),
        "relative_rollout_drift_vs_persistence": float(
            terminal.abs().mean() / terminal_persistence.abs().mean().clamp_min(1e-8)
        ),
        "mean_abs_utc_day_bias": float(by_day.abs().mean()),
        "n_days": int(len(unique_days)),
        "load_bin_terminal_mae": bin_mae,
        "load_spread_ratio": spread,
    }


def _day_bootstrap_ci(values: torch.Tensor, days: torch.Tensor, n_boot: int, seed: int) -> dict | None:
    unique = torch.unique(days)
    if len(unique) < 2:
        return None
    by_day = torch.stack([values[days == day].mean() for day in unique])
    gen = torch.Generator().manual_seed(seed)
    boot = torch.stack([
        by_day[torch.randint(len(by_day), (len(by_day),), generator=gen)].mean()
        for _ in range(n_boot)
    ])
    return {
        "point": float(by_day.mean()), "ci_lo": float(torch.quantile(boot, 0.025)),
        "ci_hi": float(torch.quantile(boot, 0.975)), "n_days": int(len(unique)),
    }


@torch.no_grad()
def direction_gate(
    model: JepaBModel,
    record: JepaBRecord,
    indices: torch.Tensor,
    history_steps: int,
    horizon: int,
    valve: int,
    delta: float,
    n_boot: int,
    device: torch.device,
) -> dict:
    values, masks, days_all = [], [], []
    tail = 3 if horizon == 18 else 10
    for start in range(0, len(indices), 32):
        chunk = indices[start:start + 32]
        raw = sample_jepa_windows(
            record, SPLIT_VAL, len(chunk), history_steps, 1,
            torch.Generator().manual_seed(0), fixed_indices=chunk,
        )
        batch = _device_batch(raw, device)
        b0 = batch.future_boundary[:, 0]
        a0 = batch.future_actions[:, 0]
        boundary = b0[:, None].repeat(1, horizon, 1)
        base_actions = a0[:, None].repeat(1, horizon, 1)
        step_actions = base_actions.clone()
        step_actions[..., valve] = (step_actions[..., valve] + delta).clamp(max=1.0)
        base = model.counterfactual(
            batch.history, base_actions, boundary_mode="oracle",
            true_future_boundary=boundary, allow_extrapolation=True,
        )
        step = model.counterfactual(
            batch.history, step_actions, boundary_mode="oracle",
            true_future_boundary=boundary, allow_extrapolation=True,
        )
        values.append((step.temps_mu[:, -tail:, -1] - base.temps_mu[:, -tail:, -1]).mean(1).cpu())
        masks.append(step.in_support.all(1).cpu())
        days_all.append(raw.day_ids)
    delta_t = torch.cat(values)
    in_support = torch.cat(masks)
    days = torch.cat(days_all)
    selected = delta_t[in_support]
    selected_days = days[in_support]
    ci = _day_bootstrap_ci(selected, selected_days, n_boot, seed=60_000 + 100 * valve + horizon)
    frac = float((selected < 0).float().mean()) if len(selected) else float("nan")
    mean = float(selected.mean()) if len(selected) else float("nan")
    passed = bool(
        len(selected) and mean < 0 and ci is not None and ci["ci_hi"] < 0 and frac >= 0.60
    )
    return {
        "n_all": len(delta_t), "n_in_support": int(in_support.sum()),
        "in_support_fraction": float(in_support.float().mean()),
        "mean_delta_c": mean, "frac_negative": frac, "legacy_frac_negative_1": frac == 1.0,
        "day_bootstrap_ci": ci, "gate_pass_v03": passed,
    }


@torch.no_grad()
def sanity_report(
    matrix: dict, record: JepaBRecord, normalizer: PrivilegedNormalizer, properties, device
) -> dict:
    history_steps = matrix["data_contract"]["history_steps"]
    horizon = matrix["data_contract"]["training_horizon"]
    raw = sample_jepa_windows(
        record, SPLIT_TRAIN, 4, history_steps, horizon, torch.Generator().manual_seed(7)
    )
    batch = _device_batch(raw, device)
    torch.manual_seed(0)
    control = build_jepa_model(
        "c0", history_steps=history_steps, properties=properties, normalizer=normalizer
    ).to(device).eval()
    r0 = control.forecast(
        batch.history, batch.future_actions, boundary_mode="oracle",
        true_future_boundary=batch.future_boundary,
    )
    identities = {}
    for arm in ORDERED_ARMS[1:]:
        torch.manual_seed(0)
        model = build_jepa_model(
            arm, history_steps=history_steps, properties=properties, normalizer=normalizer
        ).to(device).eval()
        if arm != "b4":
            model.base.load_state_dict(control.base.state_dict())
        if arm == "b2":
            model.slow_mechanism_scale = 0.0
        result = model.forecast(
            batch.history, batch.future_actions, boundary_mode="oracle",
            true_future_boundary=batch.future_boundary,
        )
        diff = float((result.temps_mu - r0.temps_mu).abs().max())
        identities[arm] = {"exact": bool(torch.equal(result.temps_mu, r0.temps_mu)),
                           "max_abs_temp_diff_c": diff}
    return {
        "identities": identities,
        "target_action_permissions": {
            "b1": "future_obs_only",
            "b3": "representation_targets_no_action",
            "b4": "residual_target_no_action; physical_anchor_uses_logged_same_instant_action",
        },
        "privileged_dim": int(record.privileged.shape[1]),
        "test_locked": True,
    }


def run_arm_and_report(
    arm: str, matrix: dict, matrix_hash: str, record: JepaBRecord,
    normalizer: PrivilegedNormalizer, properties, out_root: Path, device: torch.device,
    *, quick: bool,
) -> dict:
    train = train_arm(
        arm, matrix, matrix_hash, record, normalizer, properties, out_root, device, quick=quick
    )
    model = _load_arm(arm, matrix, matrix_hash, normalizer, properties, out_root, device)
    evaluation = {}
    n_eval = 16 if quick else matrix["evaluation"]["paired_windows"]
    for horizon in matrix["evaluation"]["horizons"]:
        indices = _fixed_indices(
            record, SPLIT_VAL, matrix["data_contract"]["history_steps"], horizon,
            n_eval, matrix["evaluation"]["paired_seed"],
        )
        evaluation[str(horizon)] = evaluate_horizon(
            model, record, indices, matrix["data_contract"]["history_steps"], horizon,
            device, matrix["evaluation"]["load_bins_mw"],
        )
    direction = {}
    direction_indices = _fixed_indices(
        record, SPLIT_VAL, matrix["data_contract"]["history_steps"], 1,
        n_eval, matrix["evaluation"]["paired_seed"],
    )
    for valve in matrix["evaluation"]["direction"]["valves"]:
        direction[f"valve{valve + 1}"] = {}
        for horizon in matrix["evaluation"]["direction"]["horizons"]:
            direction[f"valve{valve + 1}"][f"H{horizon}"] = direction_gate(
                model, record, direction_indices, matrix["data_contract"]["history_steps"],
                horizon, valve, matrix["evaluation"]["direction"]["delta_valve"],
                20 if quick else matrix["evaluation"]["direction"]["bootstrap_replicates"],
                device,
            )
    arm_dir, _ledger, _checkpoint = _arm_paths(out_root, arm)
    report = {
        "arm": arm, "status": "SMOKE" if quick else "PENDING_CONTROL_COMPARISON",
        "single_seed_exploratory": True, "quick": quick,
        "commit": _git_commit(), "matrix_sha256": matrix_hash,
        "train": train, "evaluation": evaluation, "direction_v03": direction,
        "paper_verdict_upgraded": False,
    }
    _write_json(arm_dir / "report.json", report)
    return report


def adjudicate(report: dict, control: dict, matrix: dict) -> dict:
    if report["arm"] == "c0":
        return {"status": "MATCHED_CONTROL"}
    if report["arm"] == "b3_shuffle":
        return {"status": "NEGATIVE_CONTROL_ONLY", "paper_verdict_upgraded": False}
    metric = matrix["evaluation"]["decision_metric"]
    arm_h18 = report["evaluation"]["18"][metric]
    control_h18 = control["evaluation"]["18"][metric]
    relative_change = (arm_h18 - control_h18) / control_h18
    arm_spread = report["evaluation"]["18"]["load_spread_ratio"]
    control_spread = control["evaluation"]["18"]["load_spread_ratio"]
    spread_change = None if arm_spread is None or control_spread is None else (
        arm_spread - control_spread
    ) / control_spread
    direction_pass = all(
        cell["gate_pass_v03"]
        for valve in report["direction_v03"].values()
        for cell in valve.values()
    )
    accuracy_pass = relative_change <= -matrix["evaluation"]["promotion"][
        "h18_mae_relative_improvement_min"
    ]
    spread_pass = spread_change is not None and spread_change <= matrix["evaluation"][
        "promotion"
    ]["load_spread_relative_worsening_max"]
    if relative_change >= 0.05 or not direction_pass:
        status = "REJECT_EXPLORATORY_SEED0"
    elif accuracy_pass and spread_pass:
        status = "PROMOTE_TO_FIXED_SEEDS_1_2"
    else:
        status = "INCONCLUSIVE_EXPLORATORY_SEED0"
    return {
        "status": status, "relative_h18_change": relative_change,
        "load_spread_change": spread_change,
        "gates": {"accuracy_5pct": accuracy_pass, "spread_10pct": spread_pass,
                  "direction_all_cells": direction_pass},
        "paper_verdict_upgraded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity", action="store_true")
    mode.add_argument("--queue", action="store_true")
    mode.add_argument("--arm", choices=ORDERED_ARMS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--properties", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--analytic-properties", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_matrix(args.matrix)
    matrix_hash = matrix_sha256(args.matrix)
    if not args.sanity:
        require_linux_authorization(args.registry)
    if args.analytic_properties and not args.quick and not args.sanity:
        raise FinalWMProtocolError("analytic properties are forbidden for full JEPA-B execution")
    record_path = args.record or ROOT / matrix["record"]
    properties_path = args.properties or ROOT / matrix["properties"]
    out_root = args.out or ROOT / matrix["result_root"]
    if args.quick and args.out is None:
        out_root = Path(str(out_root) + "_quick")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    record = JepaBRecord(record_path)
    normalizer = fit_privileged_normalizer(record)
    properties = AnalyticThermoProperties() if args.analytic_properties else load_grid_properties(
        properties_path, device=device
    )
    if args.sanity:
        report = sanity_report(matrix, record, normalizer, properties, device)
        report.update({
            "commit": _git_commit(), "matrix_sha256": matrix_hash,
            "record_sha256": _sha256(record_path),
            "properties_sha256": None if args.analytic_properties else _sha256(properties_path),
        })
        _write_json(out_root / "sanity_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not all(item["exact"] for item in report["identities"].values()):
            raise SystemExit("JEPA-B identity gate failed")
        return
    arms = list(ORDERED_ARMS) if args.queue else [args.arm]
    if arms != list(ORDERED_ARMS) and "c0" not in arms:
        control_report = out_root / "c0" / "report.json"
        if not control_report.exists():
            raise FinalWMProtocolError("run c0 before a single non-control arm")
    reports: dict[str, dict] = {}
    for arm in arms:
        print(f"[{arm}] start", flush=True)
        reports[arm] = run_arm_and_report(
            arm, matrix, matrix_hash, record, normalizer, properties, out_root, device,
            quick=args.quick,
        )
        print(f"[{arm}] training/evaluation complete", flush=True)
    if "c0" not in reports:
        reports["c0"] = json.loads((out_root / "c0" / "report.json").read_text(encoding="utf-8"))
    control = reports["c0"]
    for arm, report in reports.items():
        decision = {"status": "SMOKE"} if args.quick else adjudicate(report, control, matrix)
        report["decision"] = decision
        report["status"] = decision["status"]
        _write_json(out_root / arm / "report.json", report)
    root_report = {
        "protocol_version": matrix["protocol_version"], "commit": _git_commit(),
        "matrix_sha256": matrix_hash, "record_sha256": _sha256(record_path),
        "properties_sha256": None if args.analytic_properties else _sha256(properties_path),
        "quick": args.quick, "single_seed_exploratory": True,
        "arms": {arm: report["status"] for arm, report in reports.items()},
        "paper_verdict_upgraded": False,
    }
    _write_json(out_root / "report.json", root_report)
    print(json.dumps(root_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
