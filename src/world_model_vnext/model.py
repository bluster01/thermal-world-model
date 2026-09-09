"""Persistent point-belief prototype with one action-conditioned physical path.

No training results, calibrated uncertainty, causal identification, adaptive
plant transfer or planning capability are implied by these interfaces.
"""
import torch
from torch import nn

from .backends import PhysicsBackend
from .contracts import BoundaryScenario, ImaginedRollout, PointBelief
from .encoder import CausalHistoryEncoder
from .gru_encoder import CausalGRUHistoryEncoder


class PersistentPhysicsWorldModel(nn.Module):
    def __init__(self, backend: PhysicsBackend, *, hidden_dim=64, heads=4, layers=2,
                 correction_fraction=.1, encoder_kind='attention', use_neural_residual=True):
        super().__init__()
        if not 0 < correction_fraction <= 1:
            raise ValueError('correction_fraction must be in (0,1]')
        self.backend = backend
        self.hidden_dim = hidden_dim
        self.correction_fraction = correction_fraction
        if not isinstance(use_neural_residual, bool):
            raise ValueError('use_neural_residual must be boolean')
        self.use_neural_residual = use_neural_residual
        s, o, a = backend.state_dim, backend.observation_dim, backend.action_dim
        c = backend.allowed_boundary_indices.numel()
        encoders = {'attention': CausalHistoryEncoder, 'gru': CausalGRUHistoryEncoder}
        if encoder_kind not in encoders:
            raise ValueError('encoder_kind must be attention or gru')
        self.encoder_kind = encoder_kind
        # Only history initialization differs. Both arms retain identical online
        # GRU update mechanisms; this is not an online attention ablation.
        self.encoder = encoders[encoder_kind](backend.feature_loc.numel(), hidden_dim, heads, layers)
        self.initial_correction = nn.Linear(hidden_dim, s)
        self.observation_memory = nn.GRUCell(s + 2 * o + c, hidden_dim)
        self.observation_correction = nn.Linear(hidden_dim + s + 2 * o, s)
        self.transition_memory = nn.GRUCell(s + a + c, hidden_dim)
        self.power_head = nn.Sequential(nn.Linear(s + hidden_dim + c, hidden_dim), nn.Tanh(),
                                        nn.Linear(hidden_dim, backend.residual_dim))
        for head in (self.initial_correction, self.observation_correction, self.power_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if not use_neural_residual:
            self.power_head.requires_grad_(False)

    def _numeric_contract(self, value, label):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f'{label} requires a floating-point tensor')
        if value.device != self.backend.state_loc.device or value.dtype != self.backend.state_loc.dtype:
            raise ValueError(f'{label} must match the model device and dtype')

    def _finite_matrix(self, value, width, label, batch=None):
        self._numeric_contract(value, label)
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != width or (batch is not None and value.shape[0] != batch):
            raise ValueError(f'{label} must have shape (B,{width}) with matching batch')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{label} must be finite')

    def _belief(self, belief):
        self._finite_matrix(belief.physical, self.backend.state_dim, 'physical state')
        self._finite_matrix(belief.memory, self.hidden_dim, 'memory', belief.physical.shape[0])
        if isinstance(belief.steps, bool) or not isinstance(belief.steps, int) or belief.steps < 0:
            raise ValueError('belief.steps must be a nonnegative integer')
        if belief.observed_at_step is not None and (
            belief.observed_at_step.shape != (belief.physical.shape[0],) or
            belief.observed_at_step.dtype != torch.bool or
            belief.observed_at_step.device != belief.physical.device
        ):
            raise ValueError('observed_at_step requires a boolean per batch item on the state device')

    def _state_features(self, state):
        return (state - self.backend.state_loc) / self.backend.state_scale

    def _correction(self, head_value):
        return (self.correction_fraction * self.backend.state_scale *
                self.backend.correction_mask * torch.tanh(head_value))

    def initialize(self, physical_anchor, history_values, history_mask, elapsed_seconds):
        """Start a new episode from a caller-supplied, past-only physical anchor.

        History feature order is observations, actions, boundaries. The caller
        supplies its masks and actual timestamps. This is the only history reset.
        """
        self._finite_matrix(physical_anchor, self.backend.state_dim, 'physical anchor')
        self._numeric_contract(history_values, 'history values')
        if history_values.shape != history_mask.shape or history_values.ndim != 3 or history_mask.dtype != torch.bool:
            raise ValueError('History values and masks require equal (B,T,F) shapes')
        if history_mask.device != history_values.device:
            raise ValueError('History mask must match the history device')
        if history_values.shape[0] != physical_anchor.shape[0] or history_values.shape[2] != self.backend.feature_loc.numel():
            raise ValueError('History batch/feature dimensions do not match the backend')
        clean = torch.where(history_mask, history_values, self.backend.feature_loc)
        normalized = (clean - self.backend.feature_loc) / self.backend.feature_scale
        memory = self.encoder(normalized, history_mask, elapsed_seconds)[:, -1]
        state = physical_anchor + self._correction(self.initial_correction(memory))
        seen = history_mask[:, -1, :self.backend.observation_dim].any(-1)
        result = PointBelief(state, memory, observed_at_step=seen)
        self._belief(result)
        return result

    def observe(self, prior, observation, observed_mask, boundary, *, step_index: int):
        """Assimilate only newly arrived measurements; missing values are ignored.

        No observations means exactly the unchanged prior, including its memory.
        This is a learned deterministic innovation update, not Bayesian filtering.
        """
        self._belief(prior)
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index != prior.steps:
            raise ValueError('Observation step must match the prior step index')
        b = prior.physical.shape[0]
        o = self.backend.observation_dim
        self._numeric_contract(observation, 'observation')
        if observation.shape != (b, o) or observed_mask.shape != observation.shape or observed_mask.dtype != torch.bool:
            raise ValueError('Observation and boolean mask must have shape (B,O)')
        if observed_mask.device != observation.device:
            raise ValueError('Observation mask must match the observation device')
        if not bool(torch.isfinite(observation[observed_mask]).all()):
            raise ValueError('Observed measurements must be finite')
        has_observation = observed_mask.any(-1, keepdim=True)
        seen = (torch.zeros(b, device=prior.physical.device, dtype=torch.bool)
                if prior.observed_at_step is None else prior.observed_at_step)
        if bool((seen & has_observation[:, 0]).any()):
            raise ValueError('A measurement update was already applied at this step')
        self._finite_matrix(boundary, self.backend.boundary_dim, 'boundary', b)
        predicted = self.backend.decode(prior.physical, boundary)
        self._finite_matrix(predicted, o, 'decoded observation', b)
        clean = torch.where(observed_mask, observation, predicted)
        innovation = (clean - predicted) / self.backend.observation_scale
        state_features = self._state_features(prior.physical)
        update_input = torch.cat([state_features, innovation, observed_mask.to(innovation.dtype),
                                  self.backend.allowed_boundary(boundary)], -1)
        memory = self.observation_memory(update_input, prior.memory)
        correction_input = torch.cat([memory, state_features, innovation, observed_mask.to(innovation.dtype)], -1)
        state = prior.physical + self._correction(self.observation_correction(correction_input))
        result = PointBelief(torch.where(has_observation, state, prior.physical),
                             torch.where(has_observation, memory, prior.memory), prior.steps,
                             seen | has_observation[:, 0])
        self._belief(result)
        return result

    def residual_power(self, belief, boundary):
        """Declared power ports, including exact zeros in the residual ablation."""
        self._belief(belief)
        self._finite_matrix(boundary, self.backend.boundary_dim, 'boundary', belief.physical.shape[0])
        if not self.use_neural_residual:
            return belief.physical.new_zeros((belief.physical.shape[0], self.backend.residual_dim))
        power_input = torch.cat([self._state_features(belief.physical), belief.memory,
                                 self.backend.allowed_boundary(boundary)], -1)
        return self.backend.residual_scale * torch.tanh(self.power_head(power_input))

    def advance(self, belief, action, boundary):
        """Predict one step without observations; actions enter physical dynamics.

        The residual reads existing state/memory and whitelisted boundary, so
        past actions can affect it through those states. New actions also update
        recurrent memory. There is no claim of complete action independence.
        """
        self._belief(belief)
        b = belief.physical.shape[0]
        self._finite_matrix(action, self.backend.action_dim, 'action', b)
        self._finite_matrix(boundary, self.backend.boundary_dim, 'boundary', b)
        if bool(((action < 0) | (action > 1)).any()):
            raise ValueError('These backends use valve/heater fractions in [0,1]')
        allowed = self.backend.allowed_boundary(boundary)
        residual = self.residual_power(belief, boundary)
        state = self.backend.advance(belief.physical, action, boundary, residual)
        action_features = (action - self.backend.action_loc) / self.backend.action_scale
        memory = self.transition_memory(torch.cat([self._state_features(state), action_features, allowed], -1), belief.memory)
        result = PointBelief(state, memory, belief.steps + 1,
                             torch.zeros(b, device=state.device, dtype=torch.bool))
        self._belief(result)
        return result

    def imagine(self, belief, actions, scenario: BoundaryScenario):
        """Cloneable action branches through advance; no future observation input."""
        self._belief(belief)
        if actions.ndim != 3 or actions.shape[0] != belief.physical.shape[0] or actions.shape[2] != self.backend.action_dim:
            raise ValueError('Actions require matching (B,H,A) shape')
        if scenario.values.shape != (*actions.shape[:2], self.backend.boundary_dim):
            raise ValueError('Scenario and action horizons/batches must match')
        states, observations = [], []
        current = belief
        for step in range(actions.shape[1]):
            boundary = scenario.values[:, step]
            current = self.advance(current, actions[:, step], boundary)
            observation = self.backend.decode(current.physical, boundary)
            self._finite_matrix(observation, self.backend.observation_dim, 'decoded observation', actions.shape[0])
            states.append(current.physical)
            observations.append(observation)
        return ImaginedRollout(torch.stack(states, 1), torch.stack(observations, 1), current, scenario.origin)
