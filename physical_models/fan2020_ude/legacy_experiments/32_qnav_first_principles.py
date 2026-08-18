#!/usr/bin/env python3
"""Q32: minimal first-principles closure for the qnav architecture.

This driver changes only two things around the frozen evaporation grey-box:
1. where the learned residual enters the energy balance;
2. whether the learned residual can read measured total spray flow W.

Linux executes the frozen matrix once and returns raw development-fold artifacts.
The script deliberately emits no scientific winner or PASS/FAIL verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
os.chdir(BASE)


def _imp(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, BASE / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


t02 = _imp("02_train.py", "q32_t02")
r09 = _imp("09_residual.py", "q32_r09")
r26 = _imp("26_fix_evap.py", "q32_r26")

import numpy as np
import pandas as pd
import torch

DEVICE = t02.DEVICE
P_CRIT = t02.P_CRIT
E0_COLS = r09.E0_COLS
OUTPUTS = t02.OUTPUTS
PHYSICAL_ORDER_PAIRS = r09.PAIRS_PHYS
POWER_COLUMN = "机组负荷"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
    ).strip()


def load_matrix(path: Path) -> dict[str, Any]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    required = {"evap_only", "double_w", "h_w", "h_now", "conservative_now"}
    if set(matrix["candidates"]) != required:
        raise ValueError("Q32 candidate set changed")
    if set(matrix["folds"]) != {"F0", "F1"}:
        raise ValueError("Q32 fold set changed")
    if int(matrix["seed"]) != 0:
        raise ValueError("Q32 is frozen to seed 0")
    for fold_id, fold in matrix["folds"].items():
        tr, va, ev = fold["train"], fold["validation"], fold["evaluation"]
        if not (tr[0] == 0 <= tr[1] <= va[0] < va[1] <= ev[0] < ev[1] <= 40000):
            raise ValueError(f"invalid blocked fold {fold_id}: {fold}")
    return matrix


def expand_units(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate": candidate,
            "fold": fold,
            "seed": int(matrix["seed"]),
            **config,
        }
        for candidate, config in matrix["candidates"].items()
        for fold in matrix["folds"]
    ]


def residual_fluxes(injection: str, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return residual contributions to (metal, steam) energy equations."""
    zero = torch.zeros_like(z)
    if injection == "none":
        return zero, zero
    if injection == "double":
        return z, z
    if injection == "h_only":
        return zero, z
    if injection == "conservative":
        return -z, z
    raise ValueError(f"unknown injection: {injection}")


def residual_features(
    ts: torch.Tensor,
    metal_t: torch.Tensor,
    pm: torch.Tensor,
    steam_flow: torch.Tensor,
    coal: torch.Tensor,
    coal_state: torch.Tensor,
    valve1: torch.Tensor,
    valve2: torch.Tensor,
    spray_w: torch.Tensor,
    reads_w: bool,
) -> torch.Tensor:
    """Build the historical qnaw features, optionally removing W as well as valves."""
    return r09.build_feats(
        ts,
        metal_t,
        pm,
        steam_flow,
        coal,
        coal_state,
        valve1,
        valve2,
        spray_w,
        None,
        no_act=not reads_w,
        no_v12=reads_w,
    )


def integrate(
    model,
    residual,
    exo: torch.Tensor,
    h: torch.Tensor,
    metal_t: torch.Tensor,
    coal_state: torch.Tensor,
    liquid1: torch.Tensor,
    liquid2: torch.Tensor,
    steps: int,
    injection: str,
    residual_reads_w: bool,
    residual_enabled: bool = True,
):
    """Evaporation grey-box with a single explicit residual-placement switch."""
    mass = model.tri("M")[:, None]
    ua = model.tri("UA")[:, None]
    metal_capacity = model.tri("Cm")[:, None]
    tau_coal = model.val("tauB")
    tau_evap = model.val("tau_evap")
    wall1 = model.val("aW1")
    wall2 = model.val("aW2")
    dry_mass = model.val("m_dry0")
    steam_flow, coal, pm, sep_t, feed_t, valve1, valve2, p_out, spray_w = [
        exo[:, :, index] for index in range(9)
    ]
    spray_h = t02.hliq_of_T(feed_t)
    p0 = pm + (p_out - pm) / 3.0
    p1 = pm + 2.0 * (p_out - pm) / 3.0
    sep_h = t02.h_sep_of(pm, sep_t)
    outputs = []

    gain1, gain2 = model.th_of(pm[:, 0])
    denominator = gain1 * valve1[:, 0] + gain2 * valve2[:, 0] + 1e-6
    spray0 = spray_w[:, 0].clamp(min=0.0)
    spray1 = t02.KAPPA * spray0 * gain1 * valve1[:, 0] / denominator
    spray2 = t02.KAPPA * spray0 * gain2 * valve2[:, 0] / denominator
    mix1 = (steam_flow[:, 0] * h[0] + spray1 * spray_h[:, 0]) / (
        steam_flow[:, 0] + spray1 + 1e-6
    )
    mix2 = (steam_flow[:, 0] * h[1] + spray2 * spray_h[:, 0]) / (
        steam_flow[:, 0] + spray2 + 1e-6
    )

    for step in range(steps):
        heat_gain = model.k_of(pm[:, step])
        gain1, gain2 = model.th_of(pm[:, step])
        denominator = gain1 * valve1[:, step] + gain2 * valve2[:, step] + 1e-6
        current_w = spray_w[:, step].clamp(min=0.0)
        for _ in range(t02.N_SUB):
            steam_t = t02.T_of_ph(
                torch.stack([p0[:, step], p1[:, step], p_out[:, step]]), h
            )
            dry1 = torch.sigmoid(3.0 * (dry_mass - liquid1) / dry_mass)
            dry2 = torch.sigmoid(3.0 * (dry_mass - liquid2) / dry_mass)
            saturation0 = t02.tsat_poly(p0[:, step])
            saturation1 = t02.tsat_poly(p1[:, step])
            q_wall1 = wall1 * (metal_t[0] - saturation0) * (1.0 - dry1)
            q_wall2 = wall2 * (metal_t[1] - saturation1) * (1.0 - dry2)
            q_metal_to_steam = ua * (metal_t - steam_t)

            if residual is None or not residual_enabled or injection == "none":
                z = torch.zeros_like(steam_t)
            else:
                features = residual_features(
                    steam_t,
                    metal_t,
                    pm[:, step],
                    steam_flow[:, step],
                    coal[:, step],
                    coal_state,
                    valve1[:, step],
                    valve2[:, step],
                    current_w,
                    residual_reads_w,
                )
                z = residual(features).permute(1, 0)
            z_metal, z_steam = residual_fluxes(injection, z)

            metal_input = torch.stack(
                [
                    heat_gain[0] * coal_state / 3600.0
                    + ua[0] * steam_t[0]
                    - q_wall1
                    + z_metal[0],
                    heat_gain[1] * coal_state / 3600.0
                    + ua[1] * steam_t[1]
                    - q_wall2
                    + z_metal[1],
                    heat_gain[2] * coal_state / 3600.0
                    + ua[2] * steam_t[2]
                    + z_metal[2],
                ]
            ) / metal_capacity
            metal_t = (metal_t + t02.DT_SUB * metal_input) / (
                1.0 + t02.DT_SUB * ua / metal_capacity
            )

            inlet1 = mix1 + q_wall1 / (steam_flow[:, step] + 1e-6)
            inlet2 = mix2 + q_wall2 / (steam_flow[:, step] + 1e-6)
            inlet_h = torch.stack([sep_h[:, step], inlet1, inlet2])
            h = (
                h
                + t02.DT_SUB
                * (
                    steam_flow[:, step][None, :] * inlet_h
                    + q_metal_to_steam
                    + z_steam
                )
                / mass
            ) / (1.0 + t02.DT_SUB * steam_flow[:, step][None, :] / mass)
            h = t02._ste_clamp(h, t02.H_LO, t02.H_HI)

            spray1 = (
                t02.KAPPA
                * current_w
                * gain1
                * valve1[:, step]
                / denominator
            )
            spray2 = (
                t02.KAPPA
                * current_w
                * gain2
                * valve2[:, step]
                / denominator
            )
            mix1 = (steam_flow[:, step] * h[0] + spray1 * spray_h[:, step]) / (
                steam_flow[:, step] + spray1 + 1e-6
            )
            mix2 = (steam_flow[:, step] * h[1] + spray2 * spray_h[:, step]) / (
                steam_flow[:, step] + spray2 + 1e-6
            )
            liquid1 = (
                liquid1 + t02.DT_SUB * (spray1 - liquid1 / tau_evap)
            ).clamp(min=0.0)
            liquid2 = (
                liquid2 + t02.DT_SUB * (spray2 - liquid2 / tau_evap)
            ).clamp(min=0.0)
            coal_state = coal_state + t02.DT_SUB * (
                coal[:, step] - coal_state
            ) / tau_coal

        dry1 = torch.sigmoid(3.0 * (dry_mass - liquid1) / dry_mass)
        dry2 = torch.sigmoid(3.0 * (dry_mass - liquid2) / dry_mass)
        saturation0 = t02.tsat_poly(p0[:, step])
        saturation1 = t02.tsat_poly(p1[:, step])
        q_wall1 = wall1 * (metal_t[0] - saturation0) * (1.0 - dry1)
        q_wall2 = wall2 * (metal_t[1] - saturation1) * (1.0 - dry2)
        out_h1 = mix1 + q_wall1 / (steam_flow[:, step] + 1e-6)
        out_h2 = mix2 + q_wall2 / (steam_flow[:, step] + 1e-6)
        pressure = torch.stack(
            [p0[:, step], p0[:, step], p1[:, step], p1[:, step], p_out[:, step]]
        )
        enthalpy = torch.stack([h[0], out_h1, h[1], out_h2, h[2]])
        raw_t = t02.T_of_ph(pressure, enthalpy)
        outputs.append(
            torch.stack(
                [
                    raw_t[0],
                    saturation0 + dry1 * (raw_t[1] - saturation0),
                    raw_t[2],
                    saturation1 + dry2 * (raw_t[3] - saturation1),
                    raw_t[4],
                ]
            )
        )

    prediction = torch.stack(outputs, dim=2).permute(1, 2, 0)
    return prediction, h, metal_t, coal_state, liquid1, liquid2


def prepare_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    exo = df[E0_COLS].copy()
    exo["主蒸汽流量"] = exo["主蒸汽流量"] / 3.6
    exo["一级减温调节门阀位"] = exo["一级减温调节门阀位"].clip(lower=0) / 100.0
    exo["二级减温调节门阀位"] = exo["二级减温调节门阀位"].clip(lower=0) / 100.0
    return exo.to_numpy(np.float32), df[OUTPUTS].to_numpy(np.float32)


def build_windows(
    exo: np.ndarray,
    targets: np.ndarray,
    lo: int,
    hi: int,
    stride: int,
    sequence_steps: int,
):
    starts = np.arange(lo, hi - sequence_steps, stride)
    x = np.stack([exo[start : start + sequence_steps] for start in starts])
    y = np.stack([targets[start + 1 : start + sequence_steps + 1] for start in starts])
    return x, y, exo[starts], targets[starts]


def load_evap_model(checkpoint: Path):
    warm = torch.load(BASE / "out" / "model_e0_seed0.pt", map_location=DEVICE, weights_only=True)
    model = r26.E0Evap(warm).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def initialize(model, row: np.ndarray, observed: np.ndarray):
    return r26.init_states_evap(
        model,
        torch.tensor(row, device=DEVICE)[None, :],
        torch.tensor(observed, device=DEVICE)[None, :],
    )


def train_residual(
    model,
    candidate: dict[str, Any],
    fold: dict[str, Any],
    exo: np.ndarray,
    targets: np.ndarray,
    training: dict[str, Any],
    sequence_steps: int,
    seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_count = 11 if candidate["residual_reads_w"] else 10
    residual = r09.ResMLP(feature_count, r09.Q_SCALE).to(DEVICE)
    optimizer = torch.optim.Adam(
        residual.parameters(), lr=float(training["learning_rate"])
    )
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=DEVICE)
    train = build_windows(
        exo,
        targets,
        *fold["train"],
        int(training["train_stride"]),
        sequence_steps,
    )
    validation = build_windows(
        exo,
        targets,
        *fold["validation"],
        int(training["validation_stride"]),
        sequence_steps,
    )
    train_tensors = [torch.from_numpy(array).to(DEVICE) for array in train]
    validation_tensors = [torch.from_numpy(array).to(DEVICE) for array in validation]

    def forward(batch):
        x, _, initial_exo, initial_t = batch
        state = r26.init_states_evap(model, initial_exo, initial_t)
        return integrate(
            model,
            residual,
            x,
            *state,
            x.shape[1],
            candidate["injection"],
            bool(candidate["residual_reads_w"]),
        )[0]

    best_loss = float("inf")
    best_state = None
    patience = 0
    batch_size = int(training["batch_size"])
    epochs_done = 0
    for epoch in range(int(training["max_epochs"])):
        epochs_done = epoch + 1
        residual.train()
        order = torch.randperm(len(train_tensors[0]), device=DEVICE)
        for offset in range(0, len(order), batch_size):
            index = order[offset : offset + batch_size]
            if len(index) == 0:
                continue
            batch = [tensor[index] for tensor in train_tensors]
            prediction = forward(batch)
            loss = (((prediction - batch[1]) ** 2) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual.parameters(), 10.0)
            optimizer.step()
        residual.eval()
        with torch.no_grad():
            prediction = forward(validation_tensors)
            validation_loss = float(
                (((prediction - validation_tensors[1]) ** 2) * weights).mean().item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().clone() for name, value in residual.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError("residual training produced no checkpoint")
    residual.load_state_dict(best_state)
    residual.eval()
    return residual, best_loss, epochs_done, best_state


def conditional_rollout(
    model,
    residual,
    candidate: dict[str, Any],
    exo: np.ndarray,
    targets: np.ndarray,
    start: int,
    steps: int,
):
    predictions = np.empty((steps, len(OUTPUTS)), dtype=np.float32)
    state = None
    with torch.no_grad():
        for index in range(steps):
            row = exo[start + index]
            if state is None:
                state = initialize(model, row, targets[start])
            output = integrate(
                model,
                residual,
                torch.tensor(row, device=DEVICE)[None, None, :],
                *state,
                1,
                candidate["injection"],
                bool(candidate["residual_reads_w"]),
            )
            predictions[index] = output[0][0, 0].cpu().numpy()
            state = output[1:]
    truth = targets[start : start + steps]
    errors = predictions - truth
    pm = exo[start : start + steps, 2]
    order_violation = np.zeros(steps, dtype=bool)
    for lower, upper in PHYSICAL_ORDER_PAIRS:
        order_violation |= predictions[:, lower] >= predictions[:, upper]
    metrics = {
        "mode": "logged_future_boundary_conditional_rollout",
        "steps": steps,
        "rmse_main": float(np.sqrt(np.mean(errors[:, 4] ** 2))),
        "rmse_outputs": [float(np.sqrt(np.mean(errors[:, i] ** 2))) for i in range(5)],
        "bias_outputs": [float(np.mean(errors[:, i])) for i in range(5)],
        "rmse_wet_main": float(np.sqrt(np.mean(errors[pm <= P_CRIT, 4] ** 2)))
        if np.any(pm <= P_CRIT)
        else None,
        "rmse_dry_main": float(np.sqrt(np.mean(errors[pm > P_CRIT, 4] ** 2)))
        if np.any(pm > P_CRIT)
        else None,
        "band_violation_fraction": float(
            np.mean(
                (predictions[:, 4] < t02.T_BAND[0])
                | (predictions[:, 4] > t02.T_BAND[1])
            )
        ),
        "physical_order_violation_fraction": float(order_violation.mean()),
    }
    return metrics, predictions, truth


def select_operating_point(
    exo: np.ndarray,
    targets: np.ndarray,
    lo: int,
    hi: int,
    state: str,
) -> int | None:
    pm = exo[:, 2]
    mask = pm <= P_CRIT if state == "wet" else pm > P_CRIT
    candidates = np.where(mask[lo : hi - 61])[0] + lo
    if len(candidates) == 0:
        return None
    volatility = [float(np.std(np.diff(targets[index : index + 61, 4]))) for index in candidates]
    return int(candidates[int(np.argmin(volatility))])


def estimate_w_coupling(exo: np.ndarray, train_range: list[int]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    train = exo[train_range[0] : train_range[1]]
    for state, mask in (("wet", train[:, 2] <= P_CRIT), ("dry", train[:, 2] > P_CRIT)):
        subset = train[mask]
        if len(subset) < 20 or float(np.mean(subset[:, 8])) <= 1e-6:
            result[state] = None
            continue
        design = np.stack([subset[:, 5], subset[:, 6], np.ones(len(subset))], axis=1)
        coefficient, *_ = np.linalg.lstsq(design, subset[:, 8], rcond=None)
        result[state] = float(np.clip(coefficient[1] / np.mean(subset[:, 8]), 0.5, 4.0))
    return result


def run_constant(
    model,
    residual,
    candidate: dict[str, Any],
    row: np.ndarray,
    observed: np.ndarray,
    steps: int,
    valve_delta: float,
    w_multiplier: float,
    residual_enabled: bool,
):
    scenario = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, steps, 1)
    scenario[0, :, 6] += valve_delta
    scenario[0, :, 8] *= w_multiplier
    state = initialize(model, row, observed)
    with torch.no_grad():
        output = integrate(
            model,
            residual,
            scenario,
            *state,
            steps,
            candidate["injection"],
            bool(candidate["residual_reads_w"]),
            residual_enabled=residual_enabled,
        )[0]
    return output[0].cpu().numpy()


def response_metrics(delta: np.ndarray, sample_seconds: int) -> dict[str, Any]:
    main = delta[:, 4]
    steady = float(np.mean(main[-60:]))
    threshold = 0.63 * steady
    crossing = np.where(main <= threshold)[0] if steady < 0 else np.where(main >= threshold)[0]
    return {
        "steady_main_c": steady,
        "tau63_seconds": int(crossing[0] * sample_seconds) if len(crossing) else None,
        "anchors_main_c": [
            float(main[min(len(main) - 1, seconds // sample_seconds - 1)])
            for seconds in (60, 120, 180, 300, 420, 600)
        ],
    }


def action_path_probes(
    model,
    residual,
    candidate: dict[str, Any],
    row: np.ndarray,
    observed: np.ndarray,
    kw: float,
    power_mw: float,
    interventions: dict[str, Any],
    sample_seconds: int,
):
    steps = int(interventions["steps"])
    valve_step = float(interventions["valve_step_fraction"])
    w_multiplier = 1.0 + kw * valve_step
    baseline = run_constant(model, residual, candidate, row, observed, steps, 0.0, 1.0, True)
    paths = {
        "valve_only": run_constant(
            model, residual, candidate, row, observed, steps, valve_step, 1.0, True
        ),
        "w_only": run_constant(
            model, residual, candidate, row, observed, steps, 0.0, w_multiplier, True
        ),
        "coupled": run_constant(
            model,
            residual,
            candidate,
            row,
            observed,
            steps,
            valve_step,
            w_multiplier,
            True,
        ),
    }
    baseline_physical = run_constant(
        model, residual, candidate, row, observed, steps, 0.0, 1.0, False
    )
    paths["coupled_residual_off"] = run_constant(
        model,
        residual,
        candidate,
        row,
        observed,
        steps,
        valve_step,
        w_multiplier,
        False,
    )
    result = {
        name: response_metrics(
            prediction - (baseline_physical if name == "coupled_residual_off" else baseline),
            sample_seconds,
        )
        for name, prediction in paths.items()
    }
    result["w_multiplier"] = w_multiplier
    return result


def pi_parameters(deviation: float, power: float) -> tuple[float, float]:
    deviation_x = np.array([-12, -10, -8, -5, -3, 3, 5, 8, 10, 12])
    kp_y = np.array([0.6, 0.6, 0.8, 1.0, 1.2, 1.2, 1.0, 0.8, 0.6, 0.6])
    ti_y = np.array([800, 650, 550, 450, 350, 350, 450, 550, 650, 800])
    power_x = np.array([150, 200, 300, 400, 500, 600])
    power_kp = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.5])
    power_ti = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.6])
    kp = float(np.interp(abs(deviation), deviation_x, kp_y)) * float(
        np.interp(power, power_x, power_kp)
    )
    ti = float(np.interp(abs(deviation), deviation_x, ti_y)) * float(
        np.interp(power, power_x, power_ti)
    )
    return kp, ti


def clone_state(state):
    return tuple(value.clone() for value in state)


def closed_loop_probe(
    model,
    residual,
    candidate: dict[str, Any],
    row: np.ndarray,
    observed: np.ndarray,
    kw: float,
    power_mw: float,
    interventions: dict[str, Any],
    sample_seconds: int,
):
    steps = int(interventions["steps"])
    warm_steps = steps
    rate = float(interventions["valve_rate_per_step"])
    setpoint_delta = float(interventions["setpoint_step_c"])
    constant = torch.tensor(row, device=DEVICE)[None, None, :]
    state = initialize(model, row, observed)
    with torch.no_grad():
        for _ in range(warm_steps):
            output = integrate(
                model,
                residual,
                constant,
                *state,
                1,
                candidate["injection"],
                bool(candidate["residual_reads_w"]),
            )
            state = output[1:]
    base_state = clone_state(state)
    loop_state = clone_state(state)
    base = np.empty(steps, dtype=np.float64)
    controlled = np.empty(steps, dtype=np.float64)
    valve_history = np.empty(steps, dtype=np.float64)
    valve0 = float(row[6])
    w0 = float(row[8])
    valve = valve0
    command = valve0
    integral = 0.0
    baseline_initial = float(output[0][0, 0, 4])
    setpoint = baseline_initial + setpoint_delta
    power = float(power_mw)

    with torch.no_grad():
        for index in range(steps):
            base_output = integrate(
                model,
                residual,
                constant,
                *base_state,
                1,
                candidate["injection"],
                bool(candidate["residual_reads_w"]),
            )
            base[index] = float(base_output[0][0, 0, 4])
            base_state = base_output[1:]

            scenario = constant.clone()
            scenario[0, 0, 6] = valve
            scenario[0, 0, 8] = max(0.0, w0 * (1.0 + kw * (valve - valve0)))
            loop_output = integrate(
                model,
                residual,
                scenario,
                *loop_state,
                1,
                candidate["injection"],
                bool(candidate["residual_reads_w"]),
            )
            controlled[index] = float(loop_output[0][0, 0, 4])
            loop_state = loop_output[1:]
            error = controlled[index] - setpoint
            kp, ti = pi_parameters(-error, power)
            integral += error * sample_seconds
            command = float(np.clip(valve0 + kp * error + kp / ti * integral, 0.0, 1.0))
            valve = float(np.clip(valve + np.clip(command - valve, -rate, rate), 0.0, 1.0))
            valve_history[index] = valve

    delta = controlled - base
    achieved = float(np.mean(delta[-60:]))
    valve_diff = np.diff(valve_history)
    nonzero = valve_diff[np.abs(valve_diff) > 1e-8]
    reversals = int(np.sum(nonzero[1:] * nonzero[:-1] < 0)) if len(nonzero) > 1 else 0
    return {
        "setpoint_delta_c": setpoint_delta,
        "achieved_delta_c": achieved,
        "tracking_error_c": float(abs(achieved - setpoint_delta)),
        "tail_std_delta_c": float(np.std(delta[-120:])),
        "valve_min": float(np.min(valve_history)),
        "valve_max": float(np.max(valve_history)),
        "valve_saturation_fraction": float(
            np.mean((valve_history <= 1e-6) | (valve_history >= 1.0 - 1e-6))
        ),
        "valve_reversals": reversals,
        "max_valve_move_per_step": float(np.max(np.abs(valve_diff))) if len(valve_diff) else 0.0,
        "controller_power_mw": power,
        "actuator_lag_seconds": None,
    }


def json_dump(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def execute(args, matrix: dict[str, Any], matrix_path: Path):
    output_root = Path(args.output).resolve()
    summary_path = output_root / "summary_development.json"
    if summary_path.exists():
        raise RuntimeError(f"Q32 already has a summary; refusing implicit retry: {summary_path}")
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    checkpoint = BASE / "out" / "model_e0_evap_seed0.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    use_columns = list(dict.fromkeys(E0_COLS + OUTPUTS + [POWER_COLUMN]))
    df = (
        pd.read_csv(csv_path, usecols=use_columns, dtype=np.float32)
        .iloc[
            int(matrix["data"]["window_start"]) : int(matrix["data"]["window_start"])
            + int(matrix["data"]["window_rows"])
        ]
        .ffill()
        .bfill()
        .reset_index(drop=True)
    )
    if len(df) != int(matrix["data"]["window_rows"]):
        raise ValueError(f"expected 50000 rows, got {len(df)}")
    power_mw = df[POWER_COLUMN].to_numpy(np.float32)
    exo, targets = prepare_arrays(df)
    model = load_evap_model(checkpoint)
    units = []
    start_time = time.time()
    sample_seconds = int(matrix["data"]["sample_seconds"])

    for unit in expand_units(matrix):
        candidate_id = unit["candidate"]
        fold_id = unit["fold"]
        candidate = matrix["candidates"][candidate_id]
        fold = matrix["folds"][fold_id]
        unit_dir = output_root / f"{candidate_id}_{fold_id}_s0"
        unit_dir.mkdir(parents=True, exist_ok=False)
        print(f"[Q32] {candidate_id} {fold_id}", flush=True)
        if candidate["train_residual"]:
            residual, validation_loss, epochs, state = train_residual(
                model,
                candidate,
                fold,
                exo,
                targets,
                matrix["training"],
                int(matrix["data"]["sequence_steps"]),
                int(matrix["seed"]),
            )
            checkpoint_path = unit_dir / "residual_best_validation.pt"
            torch.save(state, checkpoint_path)
            checkpoint_hash = sha256_file(checkpoint_path)
        else:
            residual, validation_loss, epochs, checkpoint_hash = None, None, 0, None

        evaluation_start, evaluation_end = fold["evaluation"]
        rollout_metrics, predictions, truth = conditional_rollout(
            model,
            residual,
            candidate,
            exo,
            targets,
            int(evaluation_start),
            int(matrix["data"]["conditional_rollout_steps"]),
        )
        np.savez_compressed(
            unit_dir / "conditional_rollout_development.npz",
            predictions=predictions,
            truth=truth,
        )
        kw_by_state = estimate_w_coupling(exo, fold["train"])
        diagnostics: dict[str, Any] = {
            "w_coupling_estimated_on": "fold_training_rows_only",
            "w_coupling": kw_by_state,
            "operating_points": {},
        }
        for state_name in ("wet", "dry"):
            row_index = select_operating_point(
                exo,
                targets,
                int(evaluation_start),
                int(evaluation_end),
                state_name,
            )
            kw = kw_by_state[state_name]
            if row_index is None or kw is None:
                diagnostics["operating_points"][state_name] = {
                    "available": False,
                    "reason": "missing operating point or training-only W coupling",
                }
                continue
            diagnostics["operating_points"][state_name] = {
                "available": True,
                "row": row_index,
                "pm_mpa": float(exo[row_index, 2]),
                "action_paths": action_path_probes(
                    model,
                    residual,
                    candidate,
                    exo[row_index],
                    targets[row_index],
                    float(kw),
                    float(power_mw[row_index]),
                    matrix["interventions"],
                    sample_seconds,
                ),
                "closed_loop": closed_loop_probe(
                    model,
                    residual,
                    candidate,
                    exo[row_index],
                    targets[row_index],
                    float(kw),
                    float(power_mw[row_index]),
                    matrix["interventions"],
                    sample_seconds,
                ),
            }

        metrics = {
            "experiment": matrix["experiment"],
            "candidate": candidate_id,
            "fold": fold_id,
            "seed": int(matrix["seed"]),
            "candidate_config": candidate,
            "validation_loss_c2": validation_loss,
            "epochs": epochs,
            "conditional_rollout": rollout_metrics,
            "diagnostics": diagnostics,
            "checkpoint_sha256": checkpoint_hash,
            "scientific_verdict": None,
        }
        json_dump(unit_dir / "metrics_development.json", metrics)
        units.append(metrics)

    summary = {
        "experiment": matrix["experiment"],
        "status": "raw_results_returned_for_local_audit",
        "unit_count": len(units),
        "units": units,
        "scientific_verdict": None,
    }
    json_dump(summary_path, summary)
    manifest = {
        "experiment": matrix["experiment"],
        "git_commit": git_head(),
        "matrix_path": str(matrix_path.relative_to(BASE)),
        "matrix_sha256": sha256_file(matrix_path),
        "script_sha256": sha256_file(Path(__file__)),
        "data_path": str(csv_path),
        "data_sha256": sha256_file(csv_path),
        "evap_checkpoint_sha256": sha256_file(checkpoint),
        "torch_version": torch.__version__,
        "device": str(DEVICE),
        "exact_command": " ".join(args.command),
        "elapsed_seconds": time.time() - start_time,
        "reserved_rows_accessed": False,
    }
    json_dump(output_root / "manifest.json", manifest)
    print(json.dumps({"summary": str(summary_path), "units": len(units)}, indent=2))


def dry_run(matrix: dict[str, Any], matrix_path: Path, args):
    units = expand_units(matrix)
    payload = {
        "experiment": matrix["experiment"],
        "matrix_sha256": sha256_file(matrix_path),
        "unit_count": len(units),
        "units": units,
        "reserved_rows": matrix["data"]["reserved_historical_rows"],
        "linux_command": (
            f"python 32_qnav_first_principles.py --execute --csv {args.csv} "
            f"--output {args.output}"
        ),
        "scientific_verdict": None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default=str(BASE / "configs" / "qnav_first_principles_matrix.json"),
    )
    parser.add_argument("--csv", default=t02.CSV)
    parser.add_argument(
        "--output", default=str(BASE / "out" / "qnav_first_principles")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.command = ["python", *os.sys.argv]
    return args


def main():
    args = parse_args()
    matrix_path = Path(args.matrix).resolve()
    matrix = load_matrix(matrix_path)
    if args.dry_run:
        dry_run(matrix, matrix_path, args)
    else:
        execute(args, matrix, matrix_path)


if __name__ == "__main__":
    main()
