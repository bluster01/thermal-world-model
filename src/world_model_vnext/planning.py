"""Auditable finite-candidate screening using a world model's own predictions.

This is not CEM, globally optimal MPC, a safety certificate, or evidence of
control benefit on an independent plant. Bounds and weights are caller inputs.
Empirical action support describes coverage only; it does not establish safety.
"""

from dataclasses import dataclass
import math
from numbers import Real

import torch

from .contracts import BoundaryScenario, PointBelief
from .model import PersistentPhysicsWorldModel


def _vector(value, label):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or
            value.ndim != 1 or value.numel() == 0 or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{label} must be a finite, nonempty floating-point vector')


@dataclass(frozen=True)
class ActionLimits:
    """Action bounds and maximum absolute change per model step, in action units."""

    lower: torch.Tensor
    upper: torch.Tensor
    rate_limit: torch.Tensor

    def __post_init__(self):
        for name in ('lower', 'upper', 'rate_limit'):
            _vector(getattr(self, name), f'action {name}')
        if self.lower.shape != self.upper.shape or self.lower.shape != self.rate_limit.shape:
            raise ValueError('Action limit vectors must have matching shapes')
        if any(value.device != self.lower.device or value.dtype != self.lower.dtype
               for value in (self.upper, self.rate_limit)):
            raise ValueError('Action limit vectors must share a device and dtype')
        if bool((self.lower > self.upper).any()) or bool((self.rate_limit <= 0).any()):
            raise ValueError('Action bounds must be ordered and rate limits positive')


@dataclass(frozen=True)
class EmpiricalActionSupport:
    """A marginal action coverage box, not a joint state-action support model."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self):
        _vector(self.lower, 'support lower')
        _vector(self.upper, 'support upper')
        if (self.lower.shape != self.upper.shape or self.lower.device != self.upper.device or
                self.lower.dtype != self.upper.dtype or bool((self.lower > self.upper).any())):
            raise ValueError('Support bounds must have matching shapes/device/dtype and be ordered')


@dataclass(frozen=True)
class CandidateEvaluation:
    """All arrays retain input candidate order; unevaluated model costs are inf.

    A finite score is not sufficient for selection: predicted observation limits
    also determine feasibility. NaN prediction rows indicate no valid rollout.
    """

    costs: torch.Tensor  # K: tracking MSE + weight * action-change MSE
    tracking_costs: torch.Tensor  # K: mean over horizon and observation channels
    action_change_costs: torch.Tensor  # K: mean over horizon and action channels
    action_change_weight: float
    evaluated: torch.Tensor  # K boolean: a finite model rollout was obtained
    feasible: torch.Tensor  # K boolean: eligible for selection under these checks
    violation_reasons: tuple[tuple[str, ...], ...]
    support_status: tuple[str, ...]  # not_provided / inside / outside / nonfinite
    predicted_observations: torch.Tensor  # K,H,O; NaN for unevaluated rows
    boundary_origin: str
    best_index: int | None
    first_action: torch.Tensor | None  # 1,A, detached copy
    status: str  # selected / infeasible
    batch_rollout_error: str | None  # any failed batched attempt is made visible


@torch.no_grad()
def evaluate_candidates(
    model: PersistentPhysicsWorldModel,
    belief: PointBelief,
    candidate_actions: torch.Tensor,
    scenario: BoundaryScenario,
    last_action: torch.Tensor,
    target: torch.Tensor,
    limits: ActionLimits,
    *,
    action_change_weight: float,
    observation_lower: torch.Tensor | None,
    observation_upper: torch.Tensor | None,
    empirical_support: EmpiricalActionSupport | None = None,
) -> CandidateEvaluation:
    """Screen K prescribed action sequences from the same single initial belief.

    The objective is temperature error squared, averaged over H,O, plus the
    explicit weight times action differences squared, averaged over H,A. The
    first difference is relative to last_action. Rates are per model step, not
    per second. The first minimum wins ties; no feasible candidate returns None.

    Oracle boundaries are rejected for decisions. Caller provenance labels are
    declarations, not verified forecasts. All work is CPU/no-grad. The existing
    core's [0,1] fraction domain is checked separately from caller action limits.
    Provided empirical support is enforced conservatively; its absence is
    reported rather than interpreted as evidence of coverage.
    """
    if not isinstance(model, PersistentPhysicsWorldModel) or model.backend.state_loc.device.type != 'cpu':
        raise ValueError('Candidate evaluation requires a CPU PersistentPhysicsWorldModel')
    if not isinstance(belief, PointBelief):
        raise ValueError('belief must be a PointBelief')
    model._belief(belief)
    if belief.physical.shape[0] != 1:
        raise ValueError('Candidate evaluation requires a single initial belief (B=1)')
    if not isinstance(scenario, BoundaryScenario) or scenario.origin not in ('specified_scenario', 'forecast_from_history'):
        raise ValueError('Decision boundaries require an explicit non-oracle scenario origin')
    if not isinstance(limits, ActionLimits):
        raise ValueError('limits must be explicit ActionLimits')
    # Dataclass tensors remain mutable; revalidate at the decision boundary.
    limits.__post_init__()
    if empirical_support is not None:
        if not isinstance(empirical_support, EmpiricalActionSupport):
            raise ValueError('empirical_support must be EmpiricalActionSupport or None')
        empirical_support.__post_init__()
    if (isinstance(action_change_weight, bool) or not isinstance(action_change_weight, Real) or
            not math.isfinite(action_change_weight) or action_change_weight < 0):
        raise ValueError('action_change_weight must be an explicit finite nonnegative scalar')

    a, o, f = model.backend.action_dim, model.backend.observation_dim, model.backend.boundary_dim
    model._numeric_contract(candidate_actions, 'candidate actions')
    if candidate_actions.ndim != 3 or min(candidate_actions.shape[:2]) < 1 or candidate_actions.shape[2] != a:
        raise ValueError('Candidate actions require shape (K,H,A), K >= 1 and H >= 1')
    k, h, _ = candidate_actions.shape
    model._finite_matrix(last_action, a, 'last action', 1)
    if bool(((last_action < 0) | (last_action > 1)).any()):
        raise ValueError('Last action must be in the core fraction domain [0,1]')
    model._numeric_contract(scenario.values, 'scenario values')
    if scenario.values.shape != (1, h, f) or not bool(torch.isfinite(scenario.values).all()):
        raise ValueError('Scenario must be finite and have shape (1,H,F) matching candidates')
    model._numeric_contract(target, 'target')
    if target.shape != (h, o) or not bool(torch.isfinite(target).all()):
        raise ValueError('Target must be finite and have shape (H,O)')
    for label, value, width in (
        ('action lower', limits.lower, a), ('action upper', limits.upper, a),
        ('action rate limit', limits.rate_limit, a),
        ('observation lower', observation_lower, o), ('observation upper', observation_upper, o),
        ('support lower', None if empirical_support is None else empirical_support.lower, a),
        ('support upper', None if empirical_support is None else empirical_support.upper, a),
    ):
        if value is not None:
            _vector(value, label)
            model._numeric_contract(value, label)
            if value.shape != (width,):
                raise ValueError(f'{label} must have shape ({width},)')
    if observation_lower is not None and observation_upper is not None and bool((observation_lower > observation_upper).any()):
        raise ValueError('Observation bounds must be ordered')

    reasons: list[list[str]] = [[] for _ in range(k)]
    support_status = ['not_provided'] * k
    finite = torch.isfinite(candidate_actions).all(dim=(1, 2))
    differences = torch.cat([candidate_actions[:, :1] - last_action,
                             candidate_actions[:, 1:] - candidate_actions[:, :-1]], dim=1)
    for index in range(k):
        if not bool(finite[index]):
            reasons[index].append('nonfinite_action')
            if empirical_support is not None:
                support_status[index] = 'nonfinite'
            continue
        actions = candidate_actions[index]
        if bool(((actions < 0) | (actions > 1)).any()):
            reasons[index].append('outside_model_action_domain')
        if bool(((actions < limits.lower) | (actions > limits.upper)).any()):
            reasons[index].append('action_bounds')
        if bool((differences[index].abs() > limits.rate_limit).any()):
            reasons[index].append('action_rate_limit')
        if empirical_support is not None:
            outside = bool(((actions < empirical_support.lower) | (actions > empirical_support.upper)).any())
            support_status[index] = 'outside' if outside else 'inside'
            if outside:
                reasons[index].append('outside_empirical_action_support')

    costs = candidate_actions.new_full((k,), float('inf'))
    tracking_costs = costs.clone()
    action_costs = costs.clone()
    predictions = candidate_actions.new_full((k, h, o), float('nan'))
    evaluated = torch.zeros(k, dtype=torch.bool)
    eligible = torch.tensor([not item for item in reasons], dtype=torch.bool)
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    # Clone expanded rows to prevent a branch from aliasing the caller's belief.
    branches = PointBelief(belief.physical.expand(k, -1).clone(), belief.memory.expand(k, -1).clone(),
                          belief.steps, None if belief.observed_at_step is None else
                          belief.observed_at_step.expand(k).clone())
    boundaries = scenario.values.expand(k, -1, -1).clone()
    batch_error = None

    def run(selected):
        branch = PointBelief(branches.physical[selected], branches.memory[selected], branches.steps,
                             None if branches.observed_at_step is None else branches.observed_at_step[selected])
        return model.imagine(branch, candidate_actions[selected].clone(),
                             BoundaryScenario(boundaries[selected], scenario.origin)).observations

    if indices.numel():
        try:
            predictions[indices] = run(indices)
            evaluated[indices] = True
        except (ValueError, RuntimeError) as error:
            # Isolate a failed numerical branch, retaining all original indices.
            # This is reported explicitly and never supplies a fallback action.
            batch_error = f'{type(error).__name__}: {error}'
            for index in indices.tolist():
                try:
                    predictions[index:index + 1] = run(torch.tensor([index]))
                    evaluated[index] = True
                except (ValueError, RuntimeError) as branch_error:
                    reasons[index].append(f'rollout_error: {type(branch_error).__name__}: {branch_error}')

    for index in torch.nonzero(evaluated, as_tuple=False).flatten().tolist():
        prediction = predictions[index]
        tracking_costs[index] = (prediction - target).square().mean()
        action_costs[index] = differences[index].square().mean()
        costs[index] = tracking_costs[index] + float(action_change_weight) * action_costs[index]
        if not bool(torch.isfinite(costs[index])):
            costs[index] = float('inf')
            reasons[index].append('nonfinite_cost')
        if observation_lower is not None and bool((prediction < observation_lower).any()):
            reasons[index].append('predicted_observation_lower_bound')
        if observation_upper is not None and bool((prediction > observation_upper).any()):
            reasons[index].append('predicted_observation_upper_bound')
    feasible = evaluated & torch.tensor([not item for item in reasons], dtype=torch.bool)
    best = int(torch.where(feasible, costs, float('inf')).argmin()) if bool(feasible.any()) else None
    return CandidateEvaluation(
        costs, tracking_costs, action_costs, float(action_change_weight), evaluated, feasible,
        tuple(tuple(item) for item in reasons), tuple(support_status), predictions, scenario.origin,
        best, None if best is None else candidate_actions[best, :1].clone(),
        'infeasible' if best is None else 'selected', batch_error,
    )
