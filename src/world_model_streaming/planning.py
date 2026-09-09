"""Finite-candidate screening that preserves the complete streaming belief.

Adapted from the frozen vNext planner with identical objective/constraints.
Only branch copying changes: tokens, relative clocks, provenance and flags
travel with physical state and recurrent memory. No control benefit is implied.
"""
import math
from numbers import Real
import torch

from src.world_model_vnext.contracts import BoundaryScenario
from src.world_model_vnext.planning import ActionLimits, EmpiricalActionSupport, CandidateEvaluation, _vector
from .model import StreamingBelief, StreamingPhysicsWorldModel

@torch.no_grad()
def evaluate_candidates(
    model: StreamingPhysicsWorldModel,
    belief: StreamingBelief,
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
    if not isinstance(model, StreamingPhysicsWorldModel) or model.backend.state_loc.device.type != 'cpu':
        raise ValueError('Candidate evaluation requires a CPU StreamingPhysicsWorldModel')
    if not isinstance(belief, StreamingBelief):
        raise ValueError('belief must be a StreamingBelief')
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
    branches = model.repeat_belief(belief, k)
    boundaries = scenario.values.expand(k, -1, -1).clone()
    batch_error = None

    def run(selected):
        branch = model.select_belief(branches, selected)
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

