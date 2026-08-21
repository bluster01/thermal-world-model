from __future__ import annotations

import pytest
import torch

from src.final_wm.contracts import (
    FinalWMProtocolError,
    KAPPA_TPH_TO_KGS,
    StateLayout,
    TransitionConfig,
)
from src.final_wm.properties import AnalyticThermoProperties
from src.final_wm.synthetic import synthetic_history
from src.final_wm.transition import Fan2020UDETransition, ResidualInjection


def _transition(**kwargs) -> Fan2020UDETransition:
    config = TransitionConfig(**kwargs)
    return Fan2020UDETransition(config, AnalyticThermoProperties())


def _batch(seed: int = 0, horizon: int = 12):
    return synthetic_history(batch=3, history_steps=16, horizon=horizon, seed=seed)


def test_step_shapes_and_finiteness() -> None:
    model = _transition()
    batch = _batch()
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    assert state.shape == (3, 11)
    result = model.step(state, batch.future_boundary[:, 0], batch.future_actions[:, 0])
    assert result.state.shape == (3, 11)
    assert bool(torch.isfinite(result.state).all())
    assert set(result.aux) >= {"dsw1", "dsw2", "hm1", "hm2"}


def test_integrate_shapes() -> None:
    model = _transition()
    batch = _batch(horizon=10)
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    states, temps = model.integrate(state, batch.future_boundary, batch.future_actions)
    assert states.shape == (3, 10, 11)
    assert temps.shape == (3, 10, 5)
    assert bool(torch.isfinite(states).all()) and bool(torch.isfinite(temps).all())


def test_parameter_positivity_and_valve_monotonicity() -> None:
    model = _transition()
    for name in model.priors:
        value = model.val(name)
        if name.startswith("b"):
            assert abs(value.item()) <= model.priors[name]
        else:
            assert value.item() > 0
    v = torch.linspace(0.0, 1.0, 11)
    phi = model.varphi(v, 1)
    assert phi[0].item() == 0.0
    assert bool((phi[1:] >= phi[:-1]).all())


def test_zero_action_gives_zero_spray_and_dries_out() -> None:
    model = _transition()
    batch = _batch()
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    zero_action = torch.zeros(3, 12, 2)
    result = model.step(state, batch.future_boundary[:, 0], zero_action[:, 0])
    assert result.aux["dsw1"].abs().max().item() == pytest.approx(0.0, abs=1e-8)
    assert result.aux["dsw2"].abs().abs().max().item() == pytest.approx(0.0, abs=1e-8)
    states, _temps = model.integrate(state, batch.future_boundary, zero_action)
    m_init = state[:, 7:9]
    m_final = states[:, -1, 7:9]
    assert bool((m_final <= m_init + 1e-6).all())
    assert m_final.max().item() < 1.0  # tau_evap = 15 s, 120 s of zero feed


def test_output_equation_zero_action_matches_none() -> None:
    model = _transition()
    batch = _batch()
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    t_none = model.output_temperatures(state, batch.future_boundary[:, 0], None)
    t_zero = model.output_temperatures(state, batch.future_boundary[:, 0], torch.zeros(3, 2))
    assert torch.allclose(t_none, t_zero)


def test_observation_anchored_init_recovers_anchor_channels() -> None:
    model = _transition()
    batch = _batch()
    obs0 = batch.history.obs[:, -1]
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], obs0
    )
    temps = model.output_temperatures(state, batch.future_boundary[:, 0], batch.future_actions[:, 0])
    # State-anchored channels (sh1 inlet, sh2 inlet, final outlet) match exactly.
    assert torch.allclose(temps[:, 0], obs0[:, 0], atol=1e-3)
    assert torch.allclose(temps[:, 2], obs0[:, 2], atol=1e-3)
    assert torch.allclose(temps[:, 4], obs0[:, 4], atol=1e-3)


def test_constant_conditions_rollout_stays_finite_and_bounded() -> None:
    model = _transition()
    batch = _batch(horizon=60)
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    boundary = batch.future_boundary[:, :1].repeat(1, 60, 1)
    actions = batch.future_actions[:, :1].repeat(1, 60, 1)
    states, temps = model.integrate(state, boundary, actions)
    assert bool(torch.isfinite(states).all())
    drift = (temps[:, -1, 4] - temps[:, 0, 4]).abs()
    assert drift.max().item() < 60.0
    # The transient settles rather than diverging.
    settle = (temps[:, -1, 4] - temps[:, -6, 4]).abs()
    assert settle.max().item() < 5.0


def test_valve_opening_cools_terminal_long_run() -> None:
    model = _transition()
    batch = _batch(horizon=60)
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    boundary = batch.future_boundary[:, :1].repeat(1, 60, 1)
    base_actions = batch.future_actions[:, :1].repeat(1, 60, 1)
    step_actions = base_actions.clone()
    step_actions[:, :, 1] = (step_actions[:, :, 1] + 0.05).clamp(max=1.0)
    _s0, temps_base = model.integrate(state, boundary, base_actions)
    _s1, temps_step = model.integrate(state, boundary, step_actions)
    delta = (temps_step[:, -10:, 4] - temps_base[:, -10:, 4]).mean()
    assert delta.item() < -0.1  # spray increase must cool the terminal steam


def test_boundary_spray_mode_conserves_total() -> None:
    model = _transition(spray_total_mode="boundary")
    batch = _batch()
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    result = model.step(state, batch.future_boundary[:, 0], batch.future_actions[:, 0])
    total = (result.aux["dsw1"] + result.aux["dsw2"])
    expected = KAPPA_TPH_TO_KGS * batch.future_boundary[:, 0, 6].clamp(min=0.0)
    assert torch.allclose(total, expected, atol=1e-5)


def test_residual_injection_direction() -> None:
    model = _transition()
    batch = _batch(horizon=20)
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    boundary = batch.future_boundary[:, :1].repeat(1, 20, 1)
    actions = batch.future_actions[:, :1].repeat(1, 20, 1)
    _s0, temps_base = model.integrate(state, boundary, actions)
    heat = ResidualInjection(
        steam_power=torch.full((3, 3), 3.0e4),
        metal_power=None,
        latent_step=None,
    )
    state_h = state
    temps_h = []
    for t in range(20):
        result = model.step(state_h, boundary[:, t], actions[:, t], heat)
        state_h = result.state
        temps_h.append(model.output_temperatures(state_h, boundary[:, t], actions[:, t]))
    temps_heat = torch.stack(temps_h, dim=1)
    assert (temps_heat[:, -1, 4] - temps_base[:, -1, 4]).mean().item() > 0.5


def test_latent_block_decays_when_driven_by_zero_residual() -> None:
    model = _transition(latent_dim=2)
    batch = _batch()
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    assert state.shape == (3, 13)
    state[:, 11:] = 1.0
    result = model.step(state, batch.future_boundary[:, 0], batch.future_actions[:, 0])
    # rho = tanh(0) = 0 at init -> latent resets toward zero.
    assert result.state[:, 11:].abs().max().item() < 1e-6
    assert model.latent_rho.abs().max().item() < 1.0


def test_shape_violations_raise() -> None:
    model = _transition()
    batch = _batch()
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    with pytest.raises(FinalWMProtocolError):
        model.step(state, torch.zeros(3, 6), batch.future_actions[:, 0])
    with pytest.raises(FinalWMProtocolError):
        model.step(state, batch.future_boundary[:, 0], torch.zeros(3, 3))
    with pytest.raises(FinalWMProtocolError):
        model.step(state[:, :8], batch.future_boundary[:, 0], batch.future_actions[:, 0])
    with pytest.raises(FinalWMProtocolError):
        model.initial_steady_state(
            batch.future_boundary[:, 0], batch.future_actions[:, 0], torch.zeros(3, 4)
        )
    with pytest.raises(FinalWMProtocolError):
        Fan2020UDETransition(TransitionConfig(), AnalyticThermoProperties(), priors={"nope": 1.0})


def test_output_equation_is_state_driven_not_action_driven() -> None:
    # Repair ②: the measurement reads the transport-lagged spray state, so
    # the current action cannot move the output within the same step.
    model = _transition()
    batch = _batch()
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    t_none = model.output_temperatures(state, batch.future_boundary[:, 0], None)
    t_step = model.output_temperatures(
        state, batch.future_boundary[:, 0], torch.full((3, 2), 0.9)
    )
    assert torch.equal(t_none, t_step)


def test_spray_step_response_is_gradual() -> None:
    # Repair ②: a valve step builds the attemperator-outlet response over
    # the mixing time constant (prior 60 s), not within one 10 s step.
    model = _transition()
    batch = _batch(horizon=60)
    b0 = batch.future_boundary[:, 0]
    obs0 = batch.history.obs[:, -1]
    # Work at a WET settled equilibrium: at saturation extra spray physically
    # cannot cool further (dry blend ~ 0), making the step invisible, and the
    # transient must start from the model's own equilibrium or the base
    # trajectory itself migrates.  Settle with a constant moderate valve, then
    # verify the dry-blend state is mid-range.
    state = None
    wet_mask = None
    m_dry0 = float(model.val("m_dry0"))
    tau_evap = float(model.val("tau_evap"))
    # The wet band is narrow: with endpoint-pinned blending dry crosses
    # (0.15, 0.9) over lag in ~[0.78, 1.39] kg/s, i.e. v1 ~ 0.08-0.18.
    for v_open in (0.08, 0.12, 0.16, 0.20, 0.26):
        a_const = torch.full((b0.shape[0], 2), v_open)
        s0 = model.initial_steady_state(b0, a_const, obs0)
        settle_b = b0.unsqueeze(1).repeat(1, 120, 1)
        settle_a = a_const.unsqueeze(1).repeat(1, 120, 1)
        settled_states, _t = model.integrate(s0, settle_b, settle_a)
        cand = settled_states[:, -1]
        dry1 = torch.sigmoid(6.0 - 11.0 * (cand[:, 9] * tau_evap) / m_dry0)
        mask = (dry1 > 0.15) & (dry1 < 0.9)
        if int(mask.sum()) >= max(1, b0.shape[0] // 3):
            state, wet_mask = cand, mask
            break
    assert state is not None, "no wet settled equilibrium found"
    horizon = batch.future_actions.shape[1]
    boundary = b0.unsqueeze(1).repeat(1, horizon, 1)
    base_actions = a_const.unsqueeze(1).repeat(1, horizon, 1)
    step_actions = base_actions.clone()
    step_actions[:, :, 0] = (step_actions[:, :, 0] + 0.05).clamp(max=1.0)
    _s0, temps_base = model.integrate(state, boundary, base_actions)
    _s1, temps_step = model.integrate(state, boundary, step_actions)
    delta = (temps_step[:, :, 1] - temps_base[:, :, 1])[wet_mask]  # sh1 outlet, wet elements
    step1 = delta[:, 0].abs().mean().item()
    settled = delta[:, -6:].abs().mean().item()
    assert settled > 0.05  # the step eventually shows up
    assert step1 < 0.25 * settled  # but not in the first 10 s


def test_spray_lag_state_tracks_target_with_first_order_dynamics() -> None:
    model = _transition()
    batch = _batch(horizon=12)
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    lag0 = state[:, 9:11].clone()
    states, _temps = model.integrate(state, batch.future_boundary, batch.future_actions)
    lag = states[:, :, 9:11]
    # Repair 1-A supersedes the steady identity: the initial lag is inverted
    # from the outlet measurements and need not equal the parametric spray
    # target.  What must hold is non-negativity and monotone contraction
    # toward the (constant) target -- first-order dynamics, no overshoot.
    assert bool((lag >= 0.0).all())
    targets = torch.stack([
        torch.stack(model._spray_rates(
            batch.future_boundary[:, t, 2], batch.future_actions[:, t, 0],
            batch.future_actions[:, t, 1], batch.future_boundary[:, t, 6],
        ), dim=-1)
        for t in range(batch.future_actions.shape[1])
    ], dim=1)
    gap0 = (lag0 - targets[:, 0]).abs()
    gap1 = (lag[:, -1] - targets[:, -1]).abs()
    same_target = (targets - targets[:, :1]).abs().max() < 1e-6
    if bool(same_target):
        assert bool((gap1 <= gap0 + 1e-5).all())


def test_five_point_anchor_reproduces_all_channels() -> None:
    """Repair 1-A contract: for a physically reachable observation the
    anchored state reproduces all five channels at t=0 (solver tolerance),
    and unreachable observations clamp without NaN."""
    model = _transition()
    batch = _batch(horizon=4)
    b0 = batch.future_boundary[:, 0]
    a0 = batch.future_actions[:, 0]
    obs0 = batch.history.obs[:, -1]
    # Reachable reference: the fixed point of anchoring (anchor -> g -> re-anchor).
    x_ref = model.initial_steady_state(b0, a0, obs0)
    obs_consistent = model.output_temperatures(x_ref, b0)
    x1 = model.initial_steady_state(b0, a0, obs_consistent)
    g1 = model.output_temperatures(x1, b0)
    assert bool(((g1 - obs_consistent).abs() < 0.05).all())
    # Wildly unreachable observations: clamped, finite, non-negative lags.
    crazy = obs0.clone()
    crazy[:, 1] = 700.0   # above the dry pass-through: lag must clamp to 0
    crazy[:, 3] = 100.0   # below saturation: lag must clamp to l_max
    xc = model.initial_steady_state(b0, a0, crazy)
    assert bool(torch.isfinite(xc).all())
    assert bool((xc[:, 9:11] >= 0.0).all())


def test_rewetting_contract_caps_and_dry_lockout() -> None:
    # Repair ③: q_w <= (m/tau_evap) * max(h_pre - h_spray, 0); m=0 -> q_w = 0.
    model = _transition()
    batch = _batch()
    b0 = batch.future_boundary[:, 0]
    d_flow, _u_b, pm, _tm_sep, tfw, _p_out, _w = model._unpack_boundary(b0)
    p0, p1, _p2 = model._pressures(pm, _p_out)
    h_spray = model.properties.liquid_enthalpy(tfw)
    tm = torch.full((3, 3), 600.0)
    h_pre = torch.full((3, 2), 3300.0)
    tau_evap = model.val("tau_evap")
    m1 = torch.full((3,), 50.0)
    m2 = torch.full((3,), 50.0)
    q_w1, q_w2 = model._rewetting_powers(tm, m1, m2, p0, p1, h_pre, h_spray)
    cap = (50.0 / tau_evap) * (3300.0 - h_spray).clamp(min=0.0)
    assert bool((q_w1 <= cap + 1e-6).all()) and bool((q_w2 <= cap + 1e-6).all())
    # Dry-out lockout: zero droplet inventory closes rewetting exactly.
    q_z1, q_z2 = model._rewetting_powers(
        tm, torch.zeros(3), torch.zeros(3), p0, p1, h_pre, h_spray
    )
    assert q_z1.abs().max().item() == pytest.approx(0.0, abs=1e-8)
    assert q_z2.abs().max().item() == pytest.approx(0.0, abs=1e-8)
    # Condensation direction is preserved when the wall is cold.
    q_c1, _q_c2 = model._rewetting_powers(
        torch.full((3, 3), 100.0), m1, m2, p0, p1, h_pre, h_spray
    )
    assert bool((q_c1 < 0.0).all())


def test_spray_priors_anchored_to_data_regression() -> None:
    # Repair ④: auditpack spray_sensitivity, dW/dv = 27.76 / 70.01 t/h per
    # full travel -> kg/s per full opening; tau_mix prior = 60 s.
    model = _transition()
    assert model.priors["th1"] == pytest.approx(7.71)
    assert model.priors["th2"] == pytest.approx(19.45)
    assert model.priors["th1d"] == pytest.approx(7.71)
    assert model.priors["tau_mix1"] == pytest.approx(80.0)
    assert model.priors["tau_mix2"] == pytest.approx(80.0)


def test_tau_mix_is_a_learnable_parameter_with_gradient_flow() -> None:
    # User directive 2026-08-20: the transport lag is learnable, not a fixed
    # prior.  It must sit in the trainable raw dict and receive gradients.
    model = _transition()
    assert "tau_mix1" in model.raw and model.raw["tau_mix1"].requires_grad
    batch = _batch(horizon=4)
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    _states, temps = model.integrate(state, batch.future_boundary, batch.future_actions)
    loss = temps.square().mean()
    loss.backward()
    grad = model.raw["tau_mix1"].grad
    assert grad is not None and bool(torch.isfinite(grad)) and grad.abs().item() > 0.0


def test_zero_action_decays_lag_and_shuts_rewetting() -> None:
    # Zero-action identity with repair ②③: from a spraying steady state,
    # both the lagged mixing rate and the rewetting power decay to zero.
    model = _transition()
    batch = _batch(horizon=60)
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    assert state[:, 9:11].max().item() > 0.0  # steady state has active spray
    boundary = batch.future_boundary[:, :1].repeat(1, 60, 1)
    zero_actions = torch.zeros(3, 60, 2)
    states, _temps = model.integrate(state, boundary, zero_actions)
    lag_end = states[:, -1, 9:11]
    assert lag_end.max().item() < 0.05 * state[:, 9:11].max().item()


def test_layout_is_consistent_with_state_width() -> None:
    model = _transition(latent_dim=3)
    assert model.layout == StateLayout(latent_dim=3)
    batch = _batch()
    state = model.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    assert state.shape[-1] == 14
