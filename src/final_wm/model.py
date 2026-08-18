"""Assembled final world model.

One `Fan2020UDETransition` instance is shared by natural forecasting, action
replacement (counterfactual), and closed-loop rollout -- the design forbids
separate response heads for counterfactuals.

Information permissions are fail-closed:

- `forecast` mode never accepts true future boundary data;
- `oracle` mode requires it and labels the result accordingly;
- `counterfactual` actions must stay inside the history-estimated support
  box unless the caller explicitly opts into extrapolation (the result then
  carries the in-support mask and must be reported with it).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from src.final_wm.boundary import BoundaryModel, BoundarySequence, require_mode
from src.final_wm.closure import ActionBlindClosure
from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    FinalWMProtocolError,
    OBSERVATION_ELEMENTS,
    StateLayout,
    WorldModelConfig,
    action_support_from_history,
    validate_world_model_config,
)
from src.final_wm.observation import ObservationModel
from src.final_wm.observer import ProbabilisticObserver
from src.final_wm.controller import CascadePIController
from src.final_wm.properties import ThermoProperties
from src.final_wm.transition import Fan2020UDETransition


class HistoryWindow(NamedTuple):
    obs: torch.Tensor        # (B, L, 5) past measurements, degC
    actions: torch.Tensor    # (B, L, 2) past valve positions, 0..1
    boundary: torch.Tensor   # (B, L, 7) past boundary channels


class RolloutResult(NamedTuple):
    states: torch.Tensor              # (B, H, dim)
    temps_mu: torch.Tensor            # (B, H, 5)
    temps_sigma: torch.Tensor         # (B, H, 5)
    boundary: BoundarySequence
    mode: str                         # "forecast" | "oracle"
    in_support: torch.Tensor | None   # (B, H) bool, counterfactual only


class FinalWorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig, properties: ThermoProperties) -> None:
        super().__init__()
        layout = validate_world_model_config(config)
        self.config = config
        self.layout: StateLayout = layout
        self.transition = Fan2020UDETransition(config.transition, properties)
        self.closure = ActionBlindClosure(config.closure, layout)
        self.observer = ProbabilisticObserver(config.observer, layout)
        self.boundary_model = BoundaryModel(config.boundary)
        self.observation = ObservationModel(layout, config.observation)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def initial_state_posterior(self, history: HistoryWindow) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observer.posterior(history.obs, history.actions, history.boundary)

    def _steady_initial_state(self, history: HistoryWindow) -> torch.Tensor:
        """Observation-anchored steady init (legacy Step 8 pattern); latent
        block is zero, so this mode ignores learned latent content."""
        return self.transition.initial_steady_state(
            history.boundary[:, -1], history.actions[:, -1], history.obs[:, -1]
        )

    def _initial_state(self, history: HistoryWindow, sample_posterior: bool = False) -> torch.Tensor:
        """Dispatch on the declared initial_state_mode (O1 arms)."""
        mode = self.config.initial_state_mode
        if mode == "steady":
            return self._steady_initial_state(history)
        mu, sigma = self.initial_state_posterior(history)
        if mode == "learned":
            return self.observer.sample(mu, sigma) if sample_posterior else mu
        # hybrid: precision-weighted fusion of the steady anchor (fixed
        # sigma = 0.3 x state scale) with the learned posterior.
        steady = self._steady_initial_state(history)
        steady_sigma = 0.3 * self.observer.state_scale
        w_steady = 1.0 / steady_sigma**2
        w_learned = 1.0 / sigma**2
        fused = (steady * w_steady + mu * w_learned) / (w_steady + w_learned)
        if sample_posterior:
            fused_sigma = (w_steady + w_learned).rsqrt()
            fused = fused + fused_sigma * torch.randn_like(fused)
        return fused

    def _check_history(self, history: HistoryWindow) -> None:
        steps = self.config.observer.history_steps
        if history.obs.shape != (history.obs.shape[0], steps, len(OBSERVATION_ELEMENTS)):
            raise FinalWMProtocolError("history obs shape mismatch")
        if history.actions.shape[:2] != history.obs.shape[:2] or history.actions.shape[2] != len(ACTION_ELEMENTS):
            raise FinalWMProtocolError("history actions shape mismatch")
        if history.boundary.shape[:2] != history.obs.shape[:2]:
            raise FinalWMProtocolError("history boundary shape mismatch")

    def _boundary_sequence(
        self,
        history: HistoryWindow,
        horizon: int,
        boundary_mode: str,
        true_future_boundary: torch.Tensor | None,
        scenario: torch.Tensor | None,
    ) -> BoundarySequence:
        if boundary_mode == "forecast":
            if true_future_boundary is not None:
                raise FinalWMProtocolError("forecast mode must not receive true future boundary data")
            return self.boundary_model.forecast(
                history.boundary, history.actions, scenario=scenario, horizon=horizon
            )
        if boundary_mode == "oracle":
            if true_future_boundary is None:
                raise FinalWMProtocolError("oracle mode requires true future boundary data")
            if true_future_boundary.shape[1] < horizon:
                raise FinalWMProtocolError("oracle boundary shorter than the horizon")
            return self.boundary_model.oracle(true_future_boundary[:, :horizon])
        raise FinalWMProtocolError(f"unknown boundary mode: {boundary_mode}")

    def _rollout(
        self,
        state_0: torch.Tensor,
        boundary: BoundarySequence,
        action_seq: torch.Tensor,
        *,
        mode: str,
        in_support: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> RolloutResult:
        closure = self.closure if self.config.closure.injection_mode != "none" else None
        states, temps = self.transition.integrate(
            state_0, boundary.mu, action_seq, closure=closure, noise=noise
        )
        batch, horizon = temps.shape[:2]
        sigma = self.observation.sigma(states.reshape(batch * horizon, -1)).reshape(batch, horizon, -1)
        return RolloutResult(
            states=states, temps_mu=temps, temps_sigma=sigma,
            boundary=boundary, mode=mode, in_support=in_support,
        )

    # ------------------------------------------------------------------
    # Public capabilities
    # ------------------------------------------------------------------

    def forecast(
        self,
        history: HistoryWindow,
        action_seq: torch.Tensor,
        *,
        boundary_mode: str | None = None,
        true_future_boundary: torch.Tensor | None = None,
        scenario: torch.Tensor | None = None,
        sample_posterior: bool = False,
        noise: torch.Tensor | None = None,
    ) -> RolloutResult:
        """Natural conditional prediction given a future action sequence.

        The action sequence is the logged or planned valve trajectory; no
        counterfactual claim attaches to this path.
        """
        self._check_history(history)
        mode = boundary_mode or self.config.boundary_mode
        horizon = action_seq.shape[1]
        boundary = self._boundary_sequence(history, horizon, mode, true_future_boundary, scenario)
        state_0 = self._initial_state(history, sample_posterior=sample_posterior)
        return self._rollout(state_0, boundary, action_seq, mode=mode, noise=noise)

    def counterfactual(
        self,
        history: HistoryWindow,
        action_seq: torch.Tensor,
        *,
        boundary_mode: str | None = None,
        true_future_boundary: torch.Tensor | None = None,
        scenario: torch.Tensor | None = None,
        allow_extrapolation: bool = False,
    ) -> RolloutResult:
        """Support-domain action replacement through the shared transition.

        Actions outside the history-estimated support box are rejected
        unless `allow_extrapolation` is set; extrapolated steps are then
        flagged in the result and must be reported as unsupported.
        """
        self._check_history(history)
        support = action_support_from_history(history.actions, self.config.support_margin)
        in_support = support.contains(action_seq)
        if not allow_extrapolation and not bool(in_support.all()):
            raise FinalWMProtocolError(
                "counterfactual actions leave the history support box; "
                "refusing to extrapolate (set allow_extrapolation to override and report)"
            )
        mode = boundary_mode or self.config.boundary_mode
        boundary = self._boundary_sequence(
            history, action_seq.shape[1], mode, true_future_boundary, scenario
        )
        state_0 = self._initial_state(history)
        return self._rollout(state_0, boundary, action_seq, mode=mode, in_support=in_support)

    def closed_loop(
        self,
        history: HistoryWindow,
        sp_seq: torch.Tensor,
        controller: CascadePIController,
        *,
        boundary_mode: str | None = None,
        true_future_boundary: torch.Tensor | None = None,
        scenario: torch.Tensor | None = None,
        controlled_valve_index: int = 1,
    ) -> RolloutResult:
        """Step-by-step interaction with a controller-in-the-loop.

        The controlled valve (default: secondary attemperator valve) is
        driven by the controller acting on the predicted terminal
        temperature; the other valve is held at its last logged position.
        """
        self._check_history(history)
        if controlled_valve_index not in (0, 1):
            raise FinalWMProtocolError("controlled_valve_index must be 0 or 1")
        mode = boundary_mode or self.config.boundary_mode
        horizon = sp_seq.shape[1]
        boundary = self._boundary_sequence(history, horizon, mode, true_future_boundary, scenario)
        state_0 = self._initial_state(history)
        controller.reset(history.actions[:, -1, controlled_valve_index])
        held_valve = history.actions[:, -1, 1 - controlled_valve_index]

        closure = self.closure if self.config.closure.injection_mode != "none" else None
        state = state_0
        states, temps = [], []
        support_masks = []
        support = action_support_from_history(history.actions, self.config.support_margin)
        for t in range(horizon):
            boundary_t = boundary.mu[:, t]
            action_t = torch.stack([
                held_valve if controlled_valve_index == 1 else controller.valve,
                controller.valve if controlled_valve_index == 1 else held_valve,
            ], dim=-1)
            residual = closure(state, boundary_t) if closure is not None else None
            step = self.transition.step(state, boundary_t, action_t, residual)
            state = step.state
            temp_t = self.transition.output_temperatures(state, boundary_t, action_t)
            states.append(state)
            temps.append(temp_t)
            support_masks.append(support.contains(action_t))
            measured = temp_t[:, -1]  # terminal temperature anchor
            controller.step(sp_seq[:, t], measured)
        states_t = torch.stack(states, dim=1)
        temps_t = torch.stack(temps, dim=1)
        batch = temps_t.shape[0]
        sigma = self.observation.sigma(states_t.reshape(batch * horizon, -1)).reshape(batch, horizon, -1)
        return RolloutResult(
            states=states_t, temps_mu=temps_t, temps_sigma=sigma,
            boundary=boundary, mode=mode, in_support=torch.stack(support_masks, dim=1),
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def observation_nll(
        temps_mu: torch.Tensor,
        temps_sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Mean Gaussian NLL over batch/horizon/channels (degC units)."""
        if target.shape != temps_mu.shape:
            raise FinalWMProtocolError("NLL target shape mismatch")
        var = temps_sigma ** 2
        return (0.5 * (target - temps_mu) ** 2 / var + torch.log(temps_sigma)).mean()

    def state_continuity(
        self,
        history: HistoryWindow,
        gap_boundary: BoundarySequence,
        gap_actions: torch.Tensor,
        next_history: HistoryWindow,
    ) -> torch.Tensor:
        """Roll the posterior to the adjacent window and compare with the
        next window's posterior mean (the S1/O1 state-continuity check)."""
        mu, _sigma = self.initial_state_posterior(history)
        rolled = self._rollout(mu, gap_boundary, gap_actions, mode=gap_boundary.mode)
        mu_next, _ = self.initial_state_posterior(next_history)
        return self.observer.state_continuity_error(rolled.states[:, -1], mu_next)
