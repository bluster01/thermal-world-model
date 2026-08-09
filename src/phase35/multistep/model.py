"""A1phys-MS composition: action-blind free forecast plus response operator."""

from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import ActionResponseOperator


class ContextFreePredictor(nn.Module):
    """Small action-blind baseline used by the synthetic feasibility benchmark."""

    def __init__(self, context_dim: int, horizon: int, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class A1PhysMultiStep(nn.Module):
    """Thin additive wrapper that prevents future action from entering the free head."""

    def __init__(self, free_predictor: nn.Module, response_operator: ActionResponseOperator):
        super().__init__()
        self.free_predictor = free_predictor
        self.response_operator = response_operator

    def forward(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
        initial_response_state: torch.Tensor | None = None,
    ) -> dict[str, object]:
        free = self.free_predictor(context)
        response = self.response_operator(context, action, reference, initial_response_state)
        if free.shape != response.effect.shape:
            raise ValueError(
                f"free prediction shape {tuple(free.shape)} differs from effect {tuple(response.effect.shape)}"
            )
        return {
            "prediction": free + response.effect,
            "free_prediction": free,
            "effect": response.effect,
            "response_state": response.state_trajectory,
            "response_diagnostics": response.diagnostics,
        }

    def intervention_effect(
        self, context: torch.Tensor, action: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        return self.response_operator(context, action, reference).effect
