"""Frozen matrix contract for JEPA-B5 (action-blind slow state, v1, 2026-08-31).

B5 is the targeted fix for the B2 causal break: B2's slow state reads the
physical state (a function of logged actions) in its update, so it absorbs
spray-valve cooling and breaks the valve2 direction certificate (H18 +0.010,
H60 +0.079 degC vs control -0.105). B5 keeps the B2 structure (4-D slow, stride
6, conserved power injection, Gaussian-CF) but makes the slow update input
action-blind: only slow + boundary (no physical state).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.final_wm.contracts import FinalWMProtocolError

FROZEN_MATRIX_SHA256 = "28dcb4b6f41aed44184c1ca785916b9e3b6d03499c22f78e7969cbc0ceeed884"
PROTOCOL_VERSION = "jepa-b5-series-v1"
ORDERED_ARMS = ("c0", "b5")

EXPECTED_TRAINING = {
    "seeds": [0], "epochs": 120, "patience": 20, "batch_size": 32,
    "batches_per_epoch": 200, "learning_rate": 0.001, "grad_clip": 10.0,
    "boundary_mode": "oracle", "initial_state_mode": "hybrid",
    "closure_mode": "conservative_norew", "automatic_retry": False,
}
EXPECTED_LOSSES = {
    "observation_nll": 1.0, "jepa_prediction": 0.0, "gaussian_cf": 0.01,
    "b4_static": 0.0, "b4_dynamic": 0.0,
    "gaussian_cf_slices": 16, "gaussian_cf_knots": 17, "gaussian_cf_seed": 260830,
}


def matrix_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_matrix(path: str | Path) -> dict:
    if matrix_sha256(path) != FROZEN_MATRIX_SHA256:
        raise FinalWMProtocolError("JEPA-B5 matrix SHA-256 is not the preregistered fingerprint")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise FinalWMProtocolError("unsupported JEPA-B5 protocol version")
    arms = tuple(arm.get("id") for arm in payload.get("arms", []))
    if arms != ORDERED_ARMS:
        raise FinalWMProtocolError("JEPA-B5 arm order or membership changed")
    if payload.get("training", {}) != EXPECTED_TRAINING:
        raise FinalWMProtocolError("JEPA-B5 training budget changed")
    if payload.get("losses", {}) != EXPECTED_LOSSES:
        raise FinalWMProtocolError("JEPA-B5 loss weights changed")
    data = payload.get("data_contract", {})
    if data.get("canonical_revision") != "2.2" or data.get("test_locked") is not True:
        raise FinalWMProtocolError("JEPA-B5 canonical/test-lock contract changed")
    if data.get("history_steps") != 96 or data.get("training_horizon") != 18:
        raise FinalWMProtocolError("JEPA-B5 temporal contract changed")
    evaluation = payload.get("evaluation", {})
    if evaluation.get("decision_metric") != "final_outlet_terminal_mae":
        raise FinalWMProtocolError("JEPA-B5 decision metric changed")
    if evaluation.get("direction", {}).get("semantics") != "original_trajectory_base":
        raise FinalWMProtocolError("JEPA-B5 direction semantics must be original-trajectory")
    execution = payload.get("execution_contract", {})
    if tuple(execution.get("ordered_arms", ())) != ORDERED_ARMS:
        raise FinalWMProtocolError("JEPA-B5 execution order changed")
    if execution.get("linux_authorized_batch") != "jepa_b5_series_v1":
        raise FinalWMProtocolError("JEPA-B5 Linux batch label changed")
    if execution.get("test_authorized") is not False:
        raise FinalWMProtocolError("JEPA-B5 test must remain locked")
    return payload


def require_linux_authorization(registry_path: str | Path) -> dict:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if registry.get("active_gate") != "jepa_b5":
        raise FinalWMProtocolError(
            f"registry active gate is {registry.get('active_gate')!r}, expected 'jepa_b5'"
        )
    if registry.get("linux_authorized_gate") != "jepa_b5":
        raise FinalWMProtocolError(
            f"registry Linux gate is {registry.get('linux_authorized_gate')!r}, expected 'jepa_b5'"
        )
    exp = registry.get("experiments", {}).get("jepa_b5", {})
    if exp.get("status") != "ready_for_linux":
        raise FinalWMProtocolError(f"jepa_b5 registry status is {exp.get('status')!r}")
    state = exp.get("protocol_state", {})
    if state.get("authorized_batch") != "jepa_b5_series_v1":
        raise FinalWMProtocolError("JEPA-B5 authorized batch mismatch")
    if state.get("seed_scope") != [0] or state.get("automatic_retry") is not False:
        raise FinalWMProtocolError("JEPA-B5 seed/retry authorization contract breached")
    if state.get("ready_for_linux") is not True or state.get("results_returned") is not False:
        raise FinalWMProtocolError("JEPA-B5 execution-state contract breached")
    if state.get("test_locked") is not True or state.get("paper_verdict_authorized") is not False:
        raise FinalWMProtocolError("JEPA-B5 test/paper authorization contract breached")
    return {
        "active_gate": "jepa_b5",
        "status": exp.get("status"),
        "protocol_state": state,
    }
