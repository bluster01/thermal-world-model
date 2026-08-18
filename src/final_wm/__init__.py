"""Final probabilistic physics-state world-model package.

This package implements the frozen interface layer of the final pipeline
design (`docs/plans/2026-08-18-final-world-model-pipeline-design.md`):

- `contracts`: fail-closed registries, layouts, and configuration validation;
- `properties`: injectable differentiable thermodynamic property providers;
- `transition`: the Fan2020-inspired UDE state transition (physics core);
- `closure`: action-blind residual closure with fixed injection positions;
- `observer`: probabilistic initial-state posterior q(x0|H);
- `boundary`: forecast/oracle dual-mode future boundary model;
- `observation`: probabilistic multi-measurement output model;
- `controller`: layered SP -> controller -> actuator -> valve chain;
- `model`: the assembled world model sharing one transition across natural
  forecasting, action replacement, and closed-loop rollout.

Nothing in this package is authorized for long training.  Local interfaces,
unit tests, and micro-smoke only.
"""

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    BOUNDARY_ELEMENTS,
    BOUNDARY_MODES,
    OBSERVATION_ELEMENTS,
    PHYSICAL_STATE_ELEMENTS,
    FinalWMProtocolError,
    StateLayout,
)

__all__ = [
    "ACTION_ELEMENTS",
    "BOUNDARY_ELEMENTS",
    "BOUNDARY_MODES",
    "OBSERVATION_ELEMENTS",
    "PHYSICAL_STATE_ELEMENTS",
    "FinalWMProtocolError",
    "StateLayout",
]
