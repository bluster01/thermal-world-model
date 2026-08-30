"""Fail-closed loader for the frozen JEPA-B v1 matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.final_wm.contracts import FinalWMProtocolError


PROTOCOL_VERSION = "jepa-b-series-v1"
ORDERED_ARMS = ("c0", "b1", "b2", "b3", "b3_shuffle", "b4")
FROZEN_MATRIX_SHA256 = "b664c06272318775ad5aa89cc93c337c09a72806e5b16340552d536c66224751"


def matrix_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_matrix(path: str | Path) -> dict:
    if matrix_sha256(path) != FROZEN_MATRIX_SHA256:
        raise FinalWMProtocolError("JEPA-B matrix SHA-256 is not the preregistered fingerprint")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise FinalWMProtocolError("unsupported JEPA-B protocol version")
    arms = tuple(arm.get("id") for arm in payload.get("arms", []))
    if arms != ORDERED_ARMS:
        raise FinalWMProtocolError("JEPA-B arm order or membership changed")
    training = payload.get("training", {})
    expected_training = {
        "seeds": [0], "epochs": 120, "patience": 20, "batch_size": 32,
        "batches_per_epoch": 200, "learning_rate": 0.001, "grad_clip": 10.0,
        "boundary_mode": "oracle", "initial_state_mode": "hybrid",
        "closure_mode": "conservative_norew", "automatic_retry": False,
    }
    if training != expected_training:
        raise FinalWMProtocolError("JEPA-B training budget changed")
    losses = payload.get("losses", {})
    expected_losses = {
        "observation_nll": 1.0, "jepa_prediction": 0.1, "gaussian_cf": 0.01,
        "b4_static": 0.05, "b4_dynamic": 0.05,
        "gaussian_cf_slices": 16, "gaussian_cf_knots": 17,
        "gaussian_cf_seed": 260830,
    }
    if losses != expected_losses:
        raise FinalWMProtocolError("JEPA-B loss weights changed")
    data = payload.get("data_contract", {})
    if data.get("canonical_revision") != "2.2" or data.get("test_locked") is not True:
        raise FinalWMProtocolError("JEPA-B canonical/test-lock contract changed")
    if data.get("history_steps") != 96 or data.get("training_horizon") != 18:
        raise FinalWMProtocolError("JEPA-B temporal contract changed")
    if data.get("privileged_registry", {}).get("total") != 32:
        raise FinalWMProtocolError("JEPA-B privileged registry changed")
    evaluation = payload.get("evaluation", {})
    if evaluation.get("decision_metric") != "final_outlet_terminal_mae":
        raise FinalWMProtocolError("JEPA-B decision metric changed")
    execution = payload.get("execution_contract", {})
    if tuple(execution.get("ordered_arms", ())) != ORDERED_ARMS:
        raise FinalWMProtocolError("JEPA-B execution order changed")
    if execution.get("linux_authorized_batch") != "jepa_b_series_v1":
        raise FinalWMProtocolError("JEPA-B Linux batch label changed")
    if execution.get("test_authorized") is not False:
        raise FinalWMProtocolError("JEPA-B test must remain locked")
    if execution.get("paper_verdict_authorized") is not False:
        raise FinalWMProtocolError("JEPA-B cannot authorize a paper verdict")
    return payload


def require_linux_authorization(registry_path: str | Path) -> dict:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if registry.get("active_gate") != "jepa_b_series":
        raise FinalWMProtocolError("JEPA-B requires active_gate=jepa_b_series")
    if registry.get("linux_authorized_gate") != "jepa_b_series":
        raise FinalWMProtocolError("JEPA-B requires linux_authorized_gate=jepa_b_series")
    experiment = registry.get("experiments", {}).get("jepa_b_series", {})
    if experiment.get("status") != "ready_for_linux":
        raise FinalWMProtocolError("JEPA-B registry status must be ready_for_linux")
    state = experiment.get("protocol_state", {})
    if state.get("authorized_batch") != "jepa_b_series_v1":
        raise FinalWMProtocolError("JEPA-B authorized batch mismatch")
    if state.get("seed_scope") != [0] or state.get("automatic_retry") is not False:
        raise FinalWMProtocolError("JEPA-B seed/retry authorization contract breached")
    if state.get("ready_for_linux") is not True or state.get("results_returned") is not False:
        raise FinalWMProtocolError("JEPA-B execution-state contract breached")
    if state.get("test_locked") is not True or state.get("paper_verdict_authorized") is not False:
        raise FinalWMProtocolError("JEPA-B test/paper authorization contract breached")
    return experiment
