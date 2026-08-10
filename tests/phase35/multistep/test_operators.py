import pytest
import torch

from src.phase35.multistep.contracts import OperatorConfig, Phase35MultiStepError
from src.phase35.multistep.operators import build_response_operator


def _config(route: str, **overrides) -> OperatorConfig:
    values = dict(
        route=route,
        horizon=12,
        context_dim=3,
        dt_seconds=10.0,
        hidden_dim=12,
        latent_dim=4,
        poles=2,
        tau_min_seconds=20.0,
        tau_max_seconds=600.0,
    )
    values.update(overrides)
    return OperatorConfig(**values)


def _paths(batch: int = 3, horizon: int = 12):
    torch.manual_seed(7)
    context = torch.randn(batch, 3)
    reference = torch.full((batch, horizon), 25.0)
    action = reference.clone()
    action[:, 2:7] += 4.0
    action[:, 7:] -= 2.0
    return context, action, reference


@pytest.mark.parametrize("route", ["graybox", "koopman", "pi_ode", "deeponet"])
def test_all_routes_obey_shape_and_exact_reference_identity(route):
    operator = build_response_operator(_config(route)).eval()
    context, _, reference = _paths()
    output = operator(context, reference, reference)
    assert output.effect.shape == (3, 12)
    assert output.state_trajectory.shape[:2] == (3, 12)
    assert torch.count_nonzero(output.effect).item() == 0
    assert output.diagnostics["reference_identity_max_error"].item() == 0.0


@pytest.mark.parametrize("route", ["graybox", "koopman", "pi_ode", "deeponet"])
def test_future_action_cannot_change_earlier_response(route):
    operator = build_response_operator(_config(route)).eval()
    context, action, reference = _paths()
    changed = action.clone()
    changed[:, 8:] += 11.0
    first = operator(context, action, reference).effect
    second = operator(context, changed, reference).effect
    torch.testing.assert_close(first[:, :8], second[:, :8], atol=1e-7, rtol=0)
    assert torch.any(torch.abs(first[:, 8:] - second[:, 8:]) > 1e-7)


@pytest.mark.parametrize("poles", [1, 2, 3])
def test_graybox_has_physical_direction_and_positive_time_constants(poles):
    operator = build_response_operator(_config("graybox", poles=poles)).eval()
    context, _, reference = _paths()
    opening = reference + 5.0
    output = operator(context, opening, reference)
    assert torch.all(output.effect <= 1e-8)
    assert torch.all(output.effect[:, -1] < 0)
    assert torch.all(output.diagnostics["tau_seconds"] > 0)
    assert output.diagnostics["spectral_radius"].item() < 1.0


def test_context_scheduled_graybox_varies_parameters_without_losing_direction_or_stability():
    operator = build_response_operator(
        _config("graybox", poles=2, context_scheduled=True, schedule_log_scale=0.5)
    ).eval()
    with torch.no_grad():
        operator.gain_schedule.weight[0, 0] = 1.0
        operator.tau_schedule.weight[0, 1] = 1.0
    context, _, reference = _paths()
    output = operator(context, reference + 5.0, reference)
    gain, tau = operator.physical_parameters(context)
    assert gain.shape == (3,)
    assert tau.shape == (3, 2)
    assert torch.unique(gain).numel() > 1
    assert torch.unique(tau[:, 0]).numel() > 1
    assert torch.all(output.effect <= 1e-8)
    assert output.diagnostics["spectral_radius"].item() < 1.0


def test_three_pole_scheduled_graybox_preserves_state_across_rollout_chunks():
    operator = build_response_operator(
        _config("graybox", poles=3, context_scheduled=True)
    ).eval()
    context, action, reference = _paths()
    full = operator(context, action, reference)
    first = operator(context, action[:, :5], reference[:, :5])
    second = operator(context, action[:, 5:], reference[:, 5:], first.final_state)
    assert first.final_state.shape == (3, 3)
    torch.testing.assert_close(
        torch.cat((first.effect, second.effect), dim=1),
        full.effect,
        atol=1e-6,
        rtol=1e-6,
    )


def test_fixed_delay_graybox_is_zero_until_the_frozen_delay_and_then_responds():
    operator = build_response_operator(
        _config(
            "graybox",
            delay_mode="fixed",
            fixed_delay_steps=2,
            max_delay_steps=4,
        )
    ).eval()
    context, _, reference = _paths()
    output = operator(context, reference + 5.0, reference)
    assert torch.count_nonzero(output.effect[:, :2]).item() == 0
    assert torch.all(output.effect[:, 2:] < 0)
    torch.testing.assert_close(
        output.diagnostics["delay_weights"],
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
        atol=0,
        rtol=0,
    )
    assert output.diagnostics["expected_delay_seconds"].item() == 20.0


def test_learned_delay_graybox_uses_a_causal_probability_simplex():
    operator = build_response_operator(
        _config("graybox", delay_mode="learned", max_delay_steps=4)
    ).eval()
    context, action, reference = _paths()
    output = operator(context, action, reference)
    weights = output.diagnostics["delay_weights"]
    assert weights.shape == (5,)
    assert torch.all(weights >= 0)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0), atol=1e-7, rtol=0)
    assert 0 < output.diagnostics["expected_delay_seconds"].item() < 10.0


@pytest.mark.parametrize("delay_mode", ["fixed", "learned"])
def test_delayed_graybox_preserves_state_across_rollout_chunks(delay_mode):
    operator = build_response_operator(
        _config(
            "graybox",
            delay_mode=delay_mode,
            fixed_delay_steps=2 if delay_mode == "fixed" else 0,
            max_delay_steps=4,
        )
    ).eval()
    context, action, reference = _paths()
    full = operator(context, action, reference)
    first = operator(context, action[:, :5], reference[:, :5])
    second = operator(context, action[:, 5:], reference[:, 5:], first.final_state)
    torch.testing.assert_close(
        torch.cat((first.effect, second.effect), dim=1),
        full.effect,
        atol=1e-6,
        rtol=1e-6,
    )


def test_controlled_koopman_is_stable_and_not_the_legacy_free_head():
    operator = build_response_operator(_config("koopman", latent_dim=6)).eval()
    context, action, reference = _paths()
    output = operator(context, action, reference)
    assert operator.__class__.__name__ == "ControlledKoopmanOperator"
    assert output.state_trajectory.shape == (3, 12, 6)
    assert 0.0 < output.diagnostics["spectral_radius"].item() < 1.0
    assert torch.isfinite(output.effect).all()


def test_pi_ode_reports_finite_neural_closure_penalty():
    operator = build_response_operator(_config("pi_ode", ode_substeps=2)).eval()
    context, action, reference = _paths()
    output = operator(context, action, reference)
    residual = output.diagnostics["physics_residual_mse"]
    assert residual.ndim == 0
    assert torch.isfinite(residual)
    assert residual.item() >= 0.0


@pytest.mark.parametrize("route", ["graybox", "koopman", "pi_ode"])
def test_recursive_routes_preserve_state_across_rollout_chunks(route):
    operator = build_response_operator(_config(route)).eval()
    context, action, reference = _paths()
    full = operator(context, action, reference)
    first = operator(context, action[:, :5], reference[:, :5])
    second = operator(context, action[:, 5:], reference[:, 5:], first.final_state)
    stitched = torch.cat((first.effect, second.effect), dim=1)
    torch.testing.assert_close(stitched, full.effect, atol=1e-6, rtol=1e-6)


def test_deeponet_rejects_a_different_horizon_instead_of_silently_extrapolating():
    operator = build_response_operator(_config("deeponet", horizon=12)).eval()
    context, action, reference = _paths(horizon=10)
    with pytest.raises(Phase35MultiStepError, match="horizon"):
        operator(context, action, reference)


def test_invalid_physical_config_fails_closed():
    with pytest.raises(Phase35MultiStepError):
        _config("graybox", dt_seconds=0).validate()
    with pytest.raises(Phase35MultiStepError):
        _config("unknown").validate()
    with pytest.raises(Phase35MultiStepError, match="only by graybox"):
        _config("koopman", context_scheduled=True).validate()
    with pytest.raises(Phase35MultiStepError, match="only supported by graybox"):
        _config("koopman", delay_mode="learned", max_delay_steps=4).validate()
    with pytest.raises(Phase35MultiStepError, match="within max_delay_steps"):
        _config(
            "graybox",
            delay_mode="fixed",
            fixed_delay_steps=5,
            max_delay_steps=4,
        ).validate()
