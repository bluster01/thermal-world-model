"""Explicit state and scenario records for the vNext research prototype."""
from dataclasses import dataclass
from typing import Literal

import torch


ScenarioOrigin = Literal['specified_scenario', 'forecast_from_history', 'oracle']
SCENARIO_ORIGINS = ('specified_scenario', 'forecast_from_history', 'oracle')


@dataclass(frozen=True)
class PointBelief:
    """Physical point estimate plus recurrent memory; no uncertainty claim."""

    physical: torch.Tensor  # B,S, physical units
    memory: torch.Tensor  # B,D
    steps: int = 0
    observed_at_step: torch.Tensor | None = None  # B boolean; None means no update yet


@dataclass(frozen=True)
class BoundaryScenario:
    """Declared boundary provenance; the label does not verify its provenance.

    A caller must construct history forecasts without future measurements.
    Oracle scenarios remain usable for explicitly retrospective diagnostics.
    """

    values: torch.Tensor  # B,H,F
    origin: ScenarioOrigin

    def __post_init__(self):
        if self.origin not in SCENARIO_ORIGINS:
            raise ValueError(f'Unknown boundary origin: {self.origin}')
        if self.values.ndim != 3 or self.values.shape[1] < 1:
            raise ValueError('Boundary scenarios require shape (B,H,F), H >= 1')
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError('Boundary scenarios must be finite')


@dataclass(frozen=True)
class ImaginedRollout:
    physical: torch.Tensor  # B,H,S
    observations: torch.Tensor  # B,H,O
    final_belief: PointBelief
    boundary_origin: ScenarioOrigin
