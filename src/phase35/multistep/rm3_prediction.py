"""Fair H60 paired prediction adapters for RM3 architecture families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from ..schema import Phase35ProtocolError
from .gatec_contracts import GateCModelConfig
from .gatec_model import build_gatec_model, build_local_response_operator
from .rm3_joint_model import JointLatentPhysicalInterfaces, RM3JointConfig


PREDICTION_CANDIDATES = {
    "P0_m7_oracle_valve",
    "P1_m7_predicted_valve",
    "P2_m9_future_sp",
    "P3_gatec_paired_free",
    "P4_gatec_a1_scheduled",
    "P5_hybrid_joint_latent",
}


@dataclass(frozen=True)
class RM3PredictionConfig:
    candidate_id: str
    window: int
    horizon: int
    n_features: int
    d_model: int = 64
    latent_dim: int = 32
    dropout: float = 0.1

    def validate(self) -> None:
        if self.candidate_id not in PREDICTION_CANDIDATES:
            raise Phase35ProtocolError("RM3 prediction candidate is invalid")
        if min(self.window, self.horizon, self.n_features, self.d_model, self.latent_dim) < 1:
            raise Phase35ProtocolError("RM3 prediction dimensions must be positive")
        if self.horizon != 60:
            raise Phase35ProtocolError("RM3 fair prediction candidates must use H60")
        if not 0 <= self.dropout < 1:
            raise Phase35ProtocolError("RM3 prediction dropout is invalid")


class PairedHistoryBackbone(nn.Module):
    """High-capacity per-variable temporal encoder plus variable attention."""

    def __init__(self, n_features: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.n_features = n_features
        self.temporal = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.variable_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, normalized_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, window, features = normalized_history.shape
        if features != self.n_features:
            raise Phase35ProtocolError("RM3 high-capacity history feature count changed")
        values = normalized_history.permute(0, 2, 1).reshape(batch * features, 1, window)
        tokens = self.temporal(values).mean(dim=-1).reshape(batch, features, -1)
        attended, _ = self.variable_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        return tokens, tokens.reshape(batch, -1)


class CausalValveDecoder(nn.Module):
    def __init__(self, context_dim: int, dropout: float) -> None:
        super().__init__()
        self.cell = nn.GRUCell(4, context_dim)
        self.delta = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(context_dim, 2)
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self,
        context: torch.Tensor,
        future_sp: torch.Tensor,
        baseline_valve: torch.Tensor,
        baseline_temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del baseline_temperature
        state = context
        valve = baseline_valve
        outputs = []
        for step in range(future_sp.shape[1]):
            state = self.cell(torch.cat((future_sp[:, step], valve), dim=1), state)
            valve = valve + self.delta(state)
            outputs.append(valve)
        return torch.stack(outputs, dim=1)


class M7StylePairedPredictor(nn.Module):
    """M7-family dense action injection on the paired RM3 contract."""

    def __init__(self, config: RM3PredictionConfig, *, oracle_action: bool) -> None:
        super().__init__()
        self.config = config
        self.oracle_action = oracle_action
        self.register_buffer("history_center", torch.zeros(config.n_features))
        self.register_buffer("history_scale", torch.ones(config.n_features))
        self.backbone = PairedHistoryBackbone(config.n_features, config.d_model, config.dropout)
        context_dim = config.n_features * config.d_model
        self.context_projection = nn.Linear(context_dim, config.d_model)
        self.valve_policy = CausalValveDecoder(config.d_model, config.dropout)
        self.action_encoder = nn.Sequential(
            nn.Linear(config.horizon * 2, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.terminal_head = nn.Sequential(
            nn.Linear(context_dim + config.d_model * 2, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.horizon * 2),
        )
        nn.init.zeros_(self.terminal_head[-1].weight)
        nn.init.zeros_(self.terminal_head[-1].bias)

    def forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        baseline_valve: torch.Tensor,
        baseline_terminal: torch.Tensor,
        logged_future_valve: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str | bool]:
        normalized = (history - self.history_center) / self.history_scale
        _, flat = self.backbone(normalized)
        context = torch.tanh(self.context_projection(flat))
        predicted_valve = self.valve_policy(context, future_sp, baseline_valve)
        if self.oracle_action:
            if logged_future_valve is None or logged_future_valve.shape != future_sp.shape:
                raise Phase35ProtocolError("RM3 M7 oracle requires logged future valve")
            action = logged_future_valve
            access = "logged_future_valve_oracle"
        else:
            if logged_future_valve is not None:
                raise Phase35ProtocolError("RM3 deployable M7 must not receive logged future valve")
            action = predicted_valve
            access = "future_sp_to_predicted_valve"
        action_feature = self.action_encoder((action - baseline_valve[:, None]).reshape(len(history), -1))
        delta = self.terminal_head(torch.cat((flat, action_feature), dim=1)).reshape(
            -1, self.config.horizon, 2
        )
        return {
            "terminal_prediction": baseline_terminal[:, None] + delta,
            "valve_prediction": predicted_valve,
            "action_used": action,
            "action_access": access,
            "deployable": not self.oracle_action,
            "prefix_causal_action_path": False,
        }


class M9StylePairedPredictor(nn.Module):
    """M9-family future-SP cross-attention without the old test-selected script."""

    def __init__(self, config: RM3PredictionConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("history_center", torch.zeros(config.n_features))
        self.register_buffer("history_scale", torch.ones(config.n_features))
        self.backbone = PairedHistoryBackbone(config.n_features, config.d_model, config.dropout)
        self.sp_tokens = nn.Linear(2, config.d_model)
        self.time_queries = nn.Parameter(torch.randn(1, config.horizon, config.d_model) * 0.02)
        self.state_attention = nn.MultiheadAttention(
            config.d_model, 4, config.dropout, batch_first=True
        )
        self.action_attention = nn.MultiheadAttention(
            config.d_model, 4, config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.terminal_head = nn.Linear(config.d_model, 2)
        nn.init.zeros_(self.terminal_head.weight)
        nn.init.zeros_(self.terminal_head.bias)

    def forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        baseline_terminal: torch.Tensor,
    ) -> dict[str, torch.Tensor | str | bool]:
        normalized = (history - self.history_center) / self.history_scale
        state, _ = self.backbone(normalized)
        action = self.sp_tokens(future_sp)
        queries = self.time_queries.expand(len(history), -1, -1)
        queries = queries + self.state_attention(
            queries, state, state, need_weights=False
        )[0]
        causal_mask = torch.triu(
            torch.ones(
                self.config.horizon,
                self.config.horizon,
                dtype=torch.bool,
                device=history.device,
            ),
            diagonal=1,
        )
        queries = queries + self.action_attention(
            queries, action, action, attn_mask=causal_mask, need_weights=False
        )[0]
        delta = self.terminal_head(self.norm(queries))
        return {
            "terminal_prediction": baseline_terminal[:, None] + delta,
            "action_used": future_sp,
            "action_access": "future_sp_cross_attention",
            "deployable": True,
            "prefix_causal_action_path": True,
        }


class HybridJointLatentPredictor(nn.Module):
    """High-capacity context with explicit local response and shared latent readouts."""

    def __init__(self, config: RM3PredictionConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("history_center", torch.zeros(config.n_features))
        self.register_buffer("history_scale", torch.ones(config.n_features))
        self.backbone = PairedHistoryBackbone(config.n_features, config.d_model, config.dropout)
        context_dim = config.n_features * config.d_model
        self.context_projection = nn.Linear(context_dim, config.d_model)
        self.valve_policy = CausalValveDecoder(config.d_model, config.dropout)
        self.tin_head = nn.Linear(config.d_model, config.horizon * 2)
        self.local_response = build_local_response_operator(
            route="a1phys_three_pole",
            context_dim=config.d_model,
            state_dim=6,
            horizon=config.horizon,
            dt_seconds=10.0,
            scheduled=True,
            tau_min_seconds=20.0,
            tau_max_seconds=1200.0,
        )
        self.joint = JointLatentPhysicalInterfaces(
            RM3JointConfig(
                context_dim=config.d_model,
                latent_dim=config.latent_dim,
                horizon=config.horizon,
                terminal_bypass_hidden=min(16, config.latent_dim),
            )
        )

    def forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        baseline_valve: torch.Tensor,
        baseline_tin: torch.Tensor,
        baseline_local: torch.Tensor,
        baseline_terminal: torch.Tensor,
        initial_local_state: torch.Tensor | None = None,
        initial_latent_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str | bool]:
        normalized = (history - self.history_center) / self.history_scale
        _, flat = self.backbone(normalized)
        context = torch.tanh(self.context_projection(flat))
        valve = self.valve_policy(
            context,
            future_sp,
            baseline_valve,
            baseline_temperature=baseline_terminal,
        )
        tin = baseline_tin[:, None] + self.tin_head(context).reshape(-1, self.config.horizon, 2)
        response = self.local_response(
            context,
            valve,
            baseline_valve,
            initial_state=initial_local_state,
        )
        output = self.joint(
            context,
            tin,
            response["effect"],
            baseline_valve=baseline_valve,
            baseline_tin=baseline_tin,
            baseline_local=baseline_local,
            baseline_terminal=baseline_terminal,
            initial_state=initial_latent_state,
        )
        return {
            **output,
            "valve_prediction": valve,
            "tin_prediction": tin,
            "local_stable_poles": response["stable_poles"],
            "local_state": response["state"],
            "action_used": valve,
            "action_access": "future_sp_to_predicted_valve",
            "deployable": True,
            "prefix_causal_action_path": True,
        }


def build_rm3_prediction_model(
    config: RM3PredictionConfig, feature_names: Sequence[str]
) -> nn.Module:
    config.validate()
    if len(feature_names) != config.n_features:
        raise Phase35ProtocolError("RM3 prediction feature contract changed")
    if config.candidate_id == "P0_m7_oracle_valve":
        return M7StylePairedPredictor(config, oracle_action=True)
    if config.candidate_id == "P1_m7_predicted_valve":
        return M7StylePairedPredictor(config, oracle_action=False)
    if config.candidate_id == "P2_m9_future_sp":
        return M9StylePairedPredictor(config)
    if config.candidate_id in {"P3_gatec_paired_free", "P4_gatec_a1_scheduled"}:
        response = config.candidate_id == "P4_gatec_a1_scheduled"
        return build_gatec_model(
            GateCModelConfig(
                window=config.window,
                horizon=config.horizon,
                n_features=config.n_features,
                d_model=config.d_model,
                latent_dim=config.latent_dim,
                local_state_dim=6,
                response_route="a1phys_three_pole" if response else "none",
                response_scheduling="scheduled" if response else "none",
                response_coordinate_mode="full_mimo",
                downstream_mode="latent_mimo",
                dropout=config.dropout,
            ),
            feature_names,
        )
    return HybridJointLatentPredictor(config)


class RM3FairPredictionAdapter(nn.Module):
    """One closed forward signature for every fair-prediction candidate."""

    def __init__(self, config: RM3PredictionConfig, feature_names: Sequence[str]) -> None:
        super().__init__()
        self.config = config
        self.feature_names = tuple(feature_names)
        required = {
            *(f"{side}::二级减温调节门阀位" for side in ("A", "B")),
            *(f"{side}::二级减温器入口温度" for side in ("A", "B")),
            *(f"{side}::二级减温器出口温度" for side in ("A", "B")),
            *(f"{side}::末级过热器出口汽温" for side in ("A", "B")),
        }
        if required - set(self.feature_names):
            raise Phase35ProtocolError("RM3 fair adapter is missing measured interface features")
        self.model = build_rm3_prediction_model(config, self.feature_names)
        self.valve_indices = [
            self.feature_names.index(f"{side}::二级减温调节门阀位") for side in ("A", "B")
        ]
        self.tin_indices = [
            self.feature_names.index(f"{side}::二级减温器入口温度") for side in ("A", "B")
        ]
        self.tout_indices = [
            self.feature_names.index(f"{side}::二级减温器出口温度") for side in ("A", "B")
        ]
        self.terminal_indices = [
            self.feature_names.index(f"{side}::末级过热器出口汽温") for side in ("A", "B")
        ]

    def set_history_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        target = self.model
        if hasattr(target, "set_history_normalization"):
            target.set_history_normalization(center, scale)
            return
        if center.shape != target.history_center.shape or scale.shape != target.history_scale.shape:
            raise Phase35ProtocolError("RM3 fair normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise Phase35ProtocolError("RM3 fair normalization values are invalid")
        target.history_center.copy_(center.detach().to(target.history_center))
        target.history_scale.copy_(scale.detach().to(target.history_scale))

    def forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        logged_future_valve: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str | bool]:
        baseline_valve = history[:, -1, self.valve_indices]
        baseline_tin = history[:, -1, self.tin_indices]
        baseline_tout = history[:, -1, self.tout_indices]
        baseline_local = baseline_tin - baseline_tout
        baseline_terminal = history[:, -1, self.terminal_indices]
        candidate = self.config.candidate_id
        if candidate in {"P0_m7_oracle_valve", "P1_m7_predicted_valve"}:
            return self.model(
                history,
                future_sp,
                baseline_valve=baseline_valve,
                baseline_terminal=baseline_terminal,
                logged_future_valve=logged_future_valve,
            )
        if logged_future_valve is not None:
            raise Phase35ProtocolError(
                "RM3 logged future valve may enter only the declared M7 oracle candidate"
            )
        if candidate == "P2_m9_future_sp":
            return self.model(history, future_sp, baseline_terminal=baseline_terminal)
        if candidate in {"P3_gatec_paired_free", "P4_gatec_a1_scheduled"}:
            output = self.model(history, future_sp, boundary_mode="forecast_boundary")
            return {
                **output,
                "action_used": output["valve_prediction"],
                "action_access": "future_sp_to_predicted_valve",
                "deployable": True,
                "prefix_causal_action_path": True,
            }
        return self.model(
            history,
            future_sp,
            baseline_valve=baseline_valve,
            baseline_tin=baseline_tin,
            baseline_local=baseline_local,
            baseline_terminal=baseline_terminal,
        )
