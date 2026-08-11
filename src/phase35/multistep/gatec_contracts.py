"""Immutable contracts for the MS3-R Gate C model framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..schema import Phase35ProtocolError


BOUNDARY_MODES = {"oracle_boundary", "forecast_boundary", "scenario_boundary"}
RESPONSE_ROUTES = {
    "none",
    "a1phys_three_pole",
    "stable_koopman_lpv",
    "pi_neural_ode",
    "deeponet_response",
}
RESIDUAL_CAPACITIES = {"small", "base", "large"}
RESPONSE_SCHEDULING = {"none", "additive", "scheduled"}


@dataclass(frozen=True)
class GateCRunSpec:
    candidate_id: str
    response_route: str
    residual_capacity: str
    local_supervision: bool
    response_scheduling: str
    split: str
    seed: int
    fold: int


@dataclass(frozen=True)
class GateCModelConfig:
    window: int
    horizon: int
    n_features: int
    d_model: int = 32
    latent_dim: int = 16
    local_state_dim: int = 8
    response_route: str = "a1phys_three_pole"
    residual_capacity: str = "base"
    response_scheduling: str = "scheduled"
    dropout: float = 0.1
    dt_seconds: float = 10.0
    tau_min_seconds: float = 20.0
    tau_max_seconds: float = 1200.0

    def validate(self) -> None:
        if min(self.window, self.horizon, self.n_features, self.d_model, self.latent_dim) < 1:
            raise Phase35ProtocolError("Gate C model dimensions must be positive")
        if self.local_state_dim != 6:
            raise Phase35ProtocolError(
                "Gate C route contract requires six local states: three bases per common/differential mode"
            )
        if self.response_route not in RESPONSE_ROUTES:
            raise Phase35ProtocolError("Gate C model response route is invalid")
        if self.residual_capacity not in RESIDUAL_CAPACITIES:
            raise Phase35ProtocolError("Gate C model residual capacity is invalid")
        if self.response_scheduling not in RESPONSE_SCHEDULING:
            raise Phase35ProtocolError("Gate C model response scheduling is invalid")
        if self.response_route == "none" and self.response_scheduling != "none":
            raise Phase35ProtocolError("Gate C no-response model cannot enable scheduling")
        if self.response_route != "none" and self.response_scheduling == "none":
            raise Phase35ProtocolError("Gate C response model requires additive or scheduled mode")
        if not 0.0 <= self.dropout < 1.0:
            raise Phase35ProtocolError("Gate C dropout must be in [0,1)")
        if not 0 < self.dt_seconds < self.tau_max_seconds or not 0 < self.tau_min_seconds < self.tau_max_seconds:
            raise Phase35ProtocolError("Gate C time constants are invalid")


def _validate_candidate(raw: Mapping[str, Any]) -> None:
    required = {
        "candidate_id",
        "response_route",
        "residual_capacity",
        "local_supervision",
        "response_scheduling",
    }
    if set(raw) != required:
        raise Phase35ProtocolError("Gate C candidate fields are not closed")
    if not isinstance(raw["candidate_id"], str) or not raw["candidate_id"]:
        raise Phase35ProtocolError("Gate C candidate_id must be non-empty")
    if raw["response_route"] not in RESPONSE_ROUTES:
        raise Phase35ProtocolError("Gate C candidate has an unknown response route")
    if raw["residual_capacity"] not in RESIDUAL_CAPACITIES:
        raise Phase35ProtocolError("Gate C residual capacity is invalid")
    if not isinstance(raw["local_supervision"], bool):
        raise Phase35ProtocolError("Gate C local_supervision must be boolean")
    if raw["response_scheduling"] not in RESPONSE_SCHEDULING:
        raise Phase35ProtocolError("Gate C response scheduling is invalid")
    if raw["response_route"] == "none" and raw["response_scheduling"] != "none":
        raise Phase35ProtocolError("Gate C free control cannot schedule a response")
    if raw["response_route"] != "none" and raw["response_scheduling"] == "none":
        raise Phase35ProtocolError("Gate C response route requires additive or scheduled semantics")


def validate_gatec_matrix(matrix: Mapping[str, Any]) -> None:
    if matrix.get("protocol_version") != "phase3.5-ms3r-gatec-v1":
        raise Phase35ProtocolError("unsupported Gate C protocol version")
    data = matrix.get("data_contract", {})
    if data.get("split") != "validation":
        raise Phase35ProtocolError("Gate C is validation-only")
    if data.get("test_allowed") is not False:
        raise Phase35ProtocolError("Gate C must prohibit test access")
    if list(data.get("seeds", [])) != [0] or int(data.get("rolling_fold", -1)) != 0:
        raise Phase35ProtocolError("Gate C RM1 screen is frozen to seed 0 and fold 0")
    flow = matrix.get("information_flow", {})
    if set(flow.get("boundary_modes", [])) != BOUNDARY_MODES:
        raise Phase35ProtocolError("Gate C boundary modes changed")
    if flow.get("forecast_reads_future_tin_truth") is not False:
        raise Phase35ProtocolError("Gate C forecast mode must not read future Tin truth")
    if flow.get("forecast_reads_future_terminal_truth") is not False:
        raise Phase35ProtocolError("Gate C forecast mode must not read future terminal truth")
    if flow.get("residual_reads_future_logged_valve") is not False:
        raise Phase35ProtocolError("Gate C residual branch must not read future logged valve")
    selector = matrix.get("selector", {})
    if selector.get("primary_boundary_mode") == "oracle_boundary":
        raise Phase35ProtocolError("Gate C oracle boundary cannot select checkpoints")
    if selector.get("primary_boundary_mode") != "forecast_boundary":
        raise Phase35ProtocolError("Gate C primary selector must use forecast boundary")
    weights = selector.get("weights", {})
    if set(weights) != {"valve", "tin", "local", "terminal", "rollout", "structure"}:
        raise Phase35ProtocolError("Gate C selector weight fields changed")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-12:
        raise Phase35ProtocolError("Gate C selector weights must sum to one")
    claims = matrix.get("claim_contract", {})
    forbidden = {
        "open_loop_plant_identification": "open-loop plant",
        "arbitrary_do_valve": "arbitrary do(valve)",
        "valid_sp_instrument": "valid SP instrument",
        "measured_spray_flow_physics": "measured spray-flow physics",
        "independent_test": "independent test",
        "ms4_release": "MS4 release",
    }
    for key, label in forbidden.items():
        if claims.get(key) is not False:
            raise Phase35ProtocolError(f"Gate C cannot claim {label}")
    execution = matrix.get("execution_contract", {})
    if execution.get("linux_authorized") is not False or execution.get("real_training_authorized") is not False:
        raise Phase35ProtocolError("Gate C real/Linux execution is not authorized")
    candidates = [*matrix.get("rm1_attribution", []), *matrix.get("rm1_operator", [])]
    if len(matrix.get("rm1_attribution", [])) != 6 or len(matrix.get("rm1_operator", [])) != 4:
        raise Phase35ProtocolError("Gate C RM1 candidate counts changed")
    identifiers: list[str] = []
    for candidate in candidates:
        _validate_candidate(candidate)
        identifiers.append(candidate["candidate_id"])
    if len(identifiers) != len(set(identifiers)):
        raise Phase35ProtocolError("Gate C has a duplicate candidate id")
    expected_routes = {
        item["response_route"] for item in matrix["rm1_operator"]
    }
    if expected_routes != RESPONSE_ROUTES - {"none"}:
        raise Phase35ProtocolError("Gate C operator response route matrix changed")


def gatec_run_specs(matrix: Mapping[str, Any], stage: str) -> list[GateCRunSpec]:
    validate_gatec_matrix(matrix)
    if stage not in {"rm1_attribution", "rm1_operator"}:
        raise Phase35ProtocolError(f"unknown Gate C stage={stage!r}")
    data = matrix["data_contract"]
    return [
        GateCRunSpec(
            candidate_id=raw["candidate_id"],
            response_route=raw["response_route"],
            residual_capacity=raw["residual_capacity"],
            local_supervision=raw["local_supervision"],
            response_scheduling=raw["response_scheduling"],
            split=data["split"],
            seed=int(data["seeds"][0]),
            fold=int(data["rolling_fold"]),
        )
        for raw in matrix[stage]
    ]
