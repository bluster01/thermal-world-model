"""Keep actuator dependencies separate from the packed neural thermal state.

The frozen wrapper's actuator physics and the Legacy zero-opening derivative
guard are reused unchanged. A separate ``actual`` tensor retains dependencies
on the original anchor and commands; it never acquires a dependency merely
because thermal coordinates were corrected by a neural network.
"""

from dataclasses import dataclass

import torch

from src.world_model_streaming.actuators import ActuatorLagBackend
from src.world_model_streaming.memory import EventMemory, empty_memory, select_memory
from src.world_model_streaming.model import (
    HISTORY, OBSERVATION, PREDICTION, StreamingBelief, StreamingPhysicsWorldModel,
)


@dataclass(frozen=True)
class ActuatedStreamingBelief(StreamingBelief):
    """Packed physical view plus the authoritative actuator dependency node.

    ``physical[:, inner_state_dim:]`` must equal ``actual`` in value. Use the
    model's constructors and copy helpers to preserve their graph provenance;
    equality of externally forged tensors cannot certify graph equivalence.
    """

    actual: torch.Tensor | None = None  # B,A; only genuine anchor/command paths


class ActuatedStreamingPhysicsWorldModel(StreamingPhysicsWorldModel):
    """Same numerical model and weights, with an independent actuator graph.

    Only an existing ActuatorLagBackend is accepted. No action or position is
    automatically detached, and v**gamma is unchanged. A differentiable zero
    effective opening with gamma < 1 still reaches the Legacy guard.
    """

    def __init__(self, backend, **model_kwargs):
        if not isinstance(backend, ActuatorLagBackend):
            raise ValueError('Actuated streaming requires an ActuatorLagBackend')
        super().__init__(backend, **model_kwargs)
        self.inner_state_dim = backend.inner_state_dim
        self._actuator_contract()

    def _actuator_contract(self):
        # Wrapper buffers remain mutable even when its original constructor
        # validated them. A later nonzero correction mask would change the
        # meaning of the independently carried actuator coordinates.
        if (self.backend.inner_state_dim != self.inner_state_dim or
                self.backend.inner.state_dim != self.inner_state_dim or
                self.backend.state_dim != self.inner_state_dim + self.backend.action_dim):
            raise ValueError('Actuator and packed-state dimensions changed')
        if bool((self.backend.correction_mask[self.inner_state_dim:] != 0).any()):
            raise ValueError('Actual actuator positions must have zero neural correction mask')

    def _belief(self, belief):
        if not isinstance(belief, ActuatedStreamingBelief):
            raise ValueError('A complete ActuatedStreamingBelief with independent actual positions is required')
        self._actuator_contract()
        super()._belief(belief)
        self.backend._physical(belief.physical)
        self._finite_matrix(belief.actual, self.backend.action_dim, 'independent actual positions',
                            belief.physical.shape[0])
        if not torch.equal(belief.physical[:, self.inner_state_dim:], belief.actual):
            raise ValueError('Packed actuator positions and independent actual positions disagree')

    def initialize(self, physical_anchor, history_values, history_mask, elapsed_seconds):
        """Initialize once; preserve actual-position dependencies from the anchor.

        A constant anchor yields an independent constant actual node even when
        history/neural thermal corrections require gradients. An anchor that
        requires gradients retains that dependence, including its zero-point
        singularity when passed to the unchanged Legacy valve mapping. If a
        differentiable thermal anchor and constant positions are separate,
        use initialize_from_parts before concatenating their dependency graphs.
        """
        self._actuator_contract()
        self._finite_matrix(physical_anchor, self.backend.state_dim, 'physical anchor')
        self.backend._physical(physical_anchor)
        return self.initialize_from_parts(
            physical_anchor[:, :self.inner_state_dim], physical_anchor[:, self.inner_state_dim:],
            history_values, history_mask, elapsed_seconds,
        )

    def initialize_from_parts(self, thermal_anchor, actual_positions, history_values,
                              history_mask, elapsed_seconds):
        """Preserve separately supplied thermal and actual-position dependencies.

        ``thermal_anchor`` covers the inner backend's complete physical state,
        which can include fluid inventories and lags as well as temperatures.
        Either input may require gradients. Constant actual positions remain
        constant even when the thermal anchor depends on learned parameters;
        no coordinate is detached or re-extracted from a concatenated tensor.
        """
        self._actuator_contract()
        self._finite_matrix(thermal_anchor, self.inner_state_dim, 'thermal anchor')
        self._finite_matrix(actual_positions, self.backend.action_dim, 'initial actual positions',
                            thermal_anchor.shape[0])
        # Validation may use a packed view, but it never supplies the actual
        # node below: that must come from the original, separate argument.
        self.backend._physical(torch.cat([thermal_anchor, actual_positions], -1))
        self._numeric_contract(history_values, 'history values')
        if (not isinstance(history_mask, torch.Tensor) or history_mask.shape != history_values.shape or
                history_values.ndim != 3 or history_mask.dtype != torch.bool or
                history_mask.device != history_values.device):
            raise ValueError('History and mask require matching (B,T,F) shapes/device')
        b, length, features = history_values.shape
        if b != thermal_anchor.shape[0] or features != self.backend.feature_loc.numel():
            raise ValueError('History dimensions do not match the physical backend')
        clean = torch.where(history_mask, history_values, self.backend.feature_loc)
        tokens = self.encoder((clean - self.backend.feature_loc) / self.backend.feature_scale,
                              history_mask, elapsed_seconds)
        memory = tokens[:, -1]
        correction = self._correction(self.initial_correction(memory))
        actual = actual_positions.clone()  # never detach or extract from a packed graph
        thermal = thermal_anchor + correction[:, :self.inner_state_dim]
        state = torch.cat([thermal, actual], -1)
        seen = history_mask[:, -1, :self.backend.observation_dim].any(-1)
        keep = min(length, self.cache_capacity)
        cache = empty_memory(b, self.cache_capacity, self.hidden_dim, thermal_anchor)
        values = tokens[:, -keep:] + self.event_type_embedding.weight[HISTORY]
        time64 = elapsed_seconds.to(torch.float64)
        relative_time = time64[:, -keep:] - time64[:, -1:]
        pad = self.cache_capacity - keep
        cache = EventMemory(torch.cat([cache.tokens[:, :pad], values], 1),
                            torch.cat([cache.times[:, :pad], relative_time], 1),
                            torch.cat([cache.valid[:, :pad], torch.ones(b, keep, device=state.device, dtype=torch.bool)], 1))
        kinds = torch.cat([torch.full((b, pad), -1, dtype=torch.long, device=state.device),
                           torch.full((b, keep), HISTORY, dtype=torch.long, device=state.device)], 1)
        result = ActuatedStreamingBelief(state, memory, 0, seen, cache, kinds, actual)
        self._belief(result)
        return result

    def advance(self, belief, action, boundary):
        """Read old memory, evolve independent actual, then append one PRED."""
        self._belief(belief)
        b = belief.physical.shape[0]
        self._finite_matrix(action, self.backend.action_dim, 'command', b)
        self._finite_matrix(boundary, self.backend.boundary_dim, 'boundary', b)
        if bool(((action < 0) | (action > 1)).any()):
            raise ValueError('Command fractions must be in [0,1]')
        memory = self._retrieved_memory(belief, boundary)
        residual = self._power_with_memory(belief.physical, memory, boundary)
        self._finite_matrix(residual, self.backend.residual_dim, 'residual power', b)
        actual, mean_actual = self.backend.actuator_step(belief.actual, action)
        # Do not call the wrapper with the packed tensor: that would recreate
        # the artificial thermal-to-actuator autograd dependency. The inner
        # call is otherwise identical and retains its genuine zero-point guard.
        thermal = self.backend.inner.advance(belief.physical[:, :self.inner_state_dim],
                                             mean_actual, boundary, residual)
        self._finite_matrix(thermal, self.inner_state_dim, 'inner next state', b)
        state = torch.cat([thermal, actual], -1)
        action_features = (action - self.backend.action_loc) / self.backend.action_scale
        memory = self.transition_memory(torch.cat([self._state_features(state), action_features,
                                                   self.backend.allowed_boundary(boundary)], -1), memory)
        cache = EventMemory(belief.events.tokens,
                            torch.where(belief.events.valid, belief.events.times - self.dt_seconds, belief.events.times),
                            belief.events.valid)
        kinds = belief.event_kinds
        if self.store_predictions:
            zero = state.new_zeros((b, self.backend.observation_dim))
            token = self._event_token(state, action, boundary, zero, zero.bool(), PREDICTION)
            cache, kinds = self._append(cache, kinds, token, torch.ones(b, dtype=torch.bool, device=state.device), PREDICTION)
        result = ActuatedStreamingBelief(state, memory, belief.steps + 1,
                                        torch.zeros(b, dtype=torch.bool, device=state.device), cache, kinds, actual)
        self._belief(result)
        return result

    def observe(self, prior, observation, observed_mask, boundary, *, step_index):
        """Assimilate once at this step, without changing the actual node."""
        self._belief(prior)
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index != prior.steps:
            raise ValueError('Observation step must match the prior step index')
        b, o = prior.physical.shape[0], self.backend.observation_dim
        self._numeric_contract(observation, 'observation')
        if (observation.shape != (b, o) or not isinstance(observed_mask, torch.Tensor) or
                observed_mask.shape != observation.shape or observed_mask.dtype != torch.bool or
                observed_mask.device != observation.device):
            raise ValueError('Observation and mask require matching (B,O) shapes/device')
        if not bool(torch.isfinite(observation[observed_mask]).all()):
            raise ValueError('Observed measurements must be finite')
        self._finite_matrix(boundary, self.backend.boundary_dim, 'boundary', b)
        active = observed_mask.any(-1)
        seen = torch.zeros_like(active) if prior.observed_at_step is None else prior.observed_at_step
        if bool((active & seen).any()):
            raise ValueError('A measurement update was already applied at this step')
        if not bool(active.any()):
            return prior
        prediction = self.backend.decode(prior.physical, boundary)
        self._finite_matrix(prediction, o, 'decoded prior observation', b)
        innovation = (torch.where(observed_mask, observation, prediction) - prediction) / self.backend.observation_scale
        features = self._state_features(prior.physical)
        memory = self.observation_memory(torch.cat([features, innovation, observed_mask.to(features.dtype),
                                                    self.backend.allowed_boundary(boundary)], -1),
                                          self._retrieved_memory(prior, boundary))
        correction = self._correction(self.observation_correction(
            torch.cat([memory, features, innovation, observed_mask.to(features.dtype)], -1)))
        thermal = prior.physical[:, :self.inner_state_dim]
        thermal = torch.where(active[:, None], thermal + correction[:, :self.inner_state_dim], thermal)
        actual = prior.actual.clone()  # preserve values and genuine gradients
        state = torch.cat([thermal, actual], -1)
        memory = torch.where(active[:, None], memory, prior.memory)
        token = self._event_token(state, None, boundary, innovation, observed_mask, OBSERVATION)
        cache, kinds = self._append(prior.events, prior.event_kinds, token, active, OBSERVATION)
        result = ActuatedStreamingBelief(state, memory, prior.steps, seen | active, cache, kinds, actual)
        self._belief(result)
        return result

    def select_belief(self, belief, indices):
        """Clone selected rows of every field, including independent actual."""
        self._belief(belief)
        cache = select_memory(belief.events, indices)
        result = ActuatedStreamingBelief(
            belief.physical.index_select(0, indices), belief.memory.index_select(0, indices), belief.steps,
            None if belief.observed_at_step is None else belief.observed_at_step.index_select(0, indices),
            cache, belief.event_kinds.index_select(0, indices), belief.actual.index_select(0, indices),
        )
        self._belief(result)
        return result

    # Inherited clone/repeat dispatch through select_belief; inherited imagine
    # dispatches through advance, so their returned beliefs keep the new field.
    def detach_belief(self, belief):
        """Explicit TBPTT boundary: detach and copy the entire state together."""
        self._belief(belief)
        clone = lambda value: value.detach().clone()
        result = ActuatedStreamingBelief(
            clone(belief.physical), clone(belief.memory), belief.steps,
            None if belief.observed_at_step is None else clone(belief.observed_at_step),
            EventMemory(clone(belief.events.tokens), clone(belief.events.times), clone(belief.events.valid)),
            clone(belief.event_kinds), clone(belief.actual),
        )
        self._belief(result)
        return result
