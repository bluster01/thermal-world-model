"""Fan2020-inspired UDE state transition (physics core of the final WM).

This module rewrites the recovered legacy E0/E0Evap dynamics
(`physical_models/fan2020_ude/legacy_experiments/02_train.py` and
`26_fix_evap.py`) as a clean, testable interface.  The scientific content is
unchanged; the software contract is new:

- packed state vector (`StateLayout`): h[3], Tm[3], rB, m_liq[2], latent[L];
- exogenous inputs split into boundary (7 channels) and action (2 channels);
- actions enter *only* through the monotone valve mapping varphi(u); the
  measured total spray flow W is usable solely in oracle diagnostics mode;
- residual corrections are injected at fixed positions through
  `ResidualInjection`; the transition itself never computes them;
- stability-relevant parameters are positive by softplus parameterisation.

Units (kept identical to the legacy snapshot): h in kJ/kg, temperatures in
degC, pressures in MPa, D in kg/s, uB/rB in t/h, m_liq in kg, W in t/h,
valve positions in 0..1, powers in kW.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    BLEND_HALF_WIDTH_MPA,
    BOUNDARY_ELEMENTS,
    FinalWMProtocolError,
    KAPPA_TPH_TO_KGS,
    OBSERVATION_ELEMENTS,
    PRESSURE_REF_MPA,
    StateLayout,
    TransitionConfig,
    validate_transition_config,
)
from src.final_wm.properties import ThermoProperties, ste_clamp


# Physical magnitude priors (660 MW supercritical unit; identical values to
# the recovered legacy snapshot so evidence numbers stay comparable).
TRANSITION_PARAM_PRIORS: dict[str, float] = {
    "M0": 5000.0, "M1": 5000.0, "M2": 5000.0,          # kg steam inventory
    "UA0": 600.0, "UA1": 600.0, "UA2": 600.0,          # kW/K metal-steam
    "Cm0": 60000.0, "Cm1": 60000.0, "Cm2": 60000.0,    # kJ/K metal capacity
    "k0": 1.2e6, "k0d": 1.2e6,                          # kJ/t fuel->metal gain
    "k1": 1.2e6, "k1d": 1.2e6,
    "k2": 1.2e6, "k2d": 1.2e6,
    "b0": 0.1e6, "b1": 0.1e6, "b2": 0.1e6,             # kJ/t per MPa slope
    "tauB": 120.0,                                      # s fuel lag
    "th1": 10.0, "th2": 20.0,                           # kg/s per full opening
    "th1d": 10.0, "th2d": 20.0,                         # dry-mode gains
    "dTm0": 1.0, "dTm1": 1.0, "dTm2": 1.0,             # K init correction
    "tau_evap": 15.0,                                   # s droplet evaporation
    "aW1": 150.0, "aW2": 150.0,                         # kW/K wall rewetting
    "m_dry0": 30.0,                                     # kg dry-out threshold
    "gamma1": 1.0, "gamma2": 1.0,                       # valve map exponents
}
_SIGNED_PARAMS = frozenset({"b0", "b1", "b2"})
_TRI_GROUPS = {
    "M": ("M0", "M1", "M2"),
    "UA": ("UA0", "UA1", "UA2"),
    "Cm": ("Cm0", "Cm1", "Cm2"),
    "k": ("k0", "k1", "k2"),
    "dTm": ("dTm0", "dTm1", "dTm2"),
}
_EPS = 1e-6


def _softplus_inverse(target: float) -> float:
    if target > 30.0:
        return float(target)
    return float(np.log(np.expm1(target)))


class ResidualInjection(NamedTuple):
    """Residual corrections applied inside the transition at fixed positions.

    `steam_power` / `metal_power` are per-stage power corrections (kW) with
    signs already resolved by the closure's injection mode.  `latent_step`
    is a per-second increment for the latent block.
    """

    steam_power: torch.Tensor | None   # (B, 3) kW
    metal_power: torch.Tensor | None   # (B, 3) kW
    latent_step: torch.Tensor | None   # (B, L) 1/s


class TransitionStep(NamedTuple):
    state: torch.Tensor                # (B, dim)
    aux: dict[str, torch.Tensor]


class Fan2020UDETransition(nn.Module):
    """Physics-embedded neural state-space transition candidate.

    This is a Fan2020-inspired UDE *candidate*, not a white-box plant truth:
    the spray mass flow, wall temperatures and mixing details are not
    reliably measured and remain parameterised/latent.
    """

    def __init__(
        self,
        config: TransitionConfig,
        properties: ThermoProperties,
        *,
        priors: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        validate_transition_config(config)
        self.config = config
        self.properties = properties
        self.layout = StateLayout(latent_dim=config.latent_dim)
        merged = dict(TRANSITION_PARAM_PRIORS)
        if priors is not None:
            unknown = set(priors) - set(merged)
            if unknown:
                raise FinalWMProtocolError(f"unknown transition parameters: {sorted(unknown)}")
            merged.update(priors)
        self.priors = merged
        self.raw = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(
                0.0 if name in _SIGNED_PARAMS else _softplus_inverse(1.0),
                dtype=torch.float32,
            ))
            for name in merged
        })
        if self.layout.latent_dim > 0:
            self.latent_rho_raw = nn.Parameter(torch.zeros(self.layout.latent_dim))

    # ------------------------------------------------------------------
    # Parameters (positive-by-construction or bounded signed)
    # ------------------------------------------------------------------

    def val(self, name: str) -> torch.Tensor:
        prior = float(self.priors[name])
        if name in _SIGNED_PARAMS:
            return prior * torch.tanh(self.raw[name])
        return prior * F.softplus(self.raw[name])

    def tri(self, group: str) -> torch.Tensor:
        return torch.stack([self.val(name) for name in _TRI_GROUPS[group]], dim=0)

    @property
    def latent_rho(self) -> torch.Tensor:
        """Latent drift, contractually bounded to (-1, 1) per step."""
        if self.layout.latent_dim == 0:
            raise FinalWMProtocolError("transition has no latent block")
        return torch.tanh(self.latent_rho_raw)

    def k_of(self, pm: torch.Tensor) -> torch.Tensor:
        """Fuel->metal heating gains (B, 3); wet/dry blend + pressure slope."""
        a = torch.sigmoid((self.properties.critical_pressure - pm) / BLEND_HALF_WIDTH_MPA)
        dpm = pm - PRESSURE_REF_MPA
        cols = []
        for index in range(3):
            wet = self.val(f"k{index}")
            dry = self.val(f"k{index}d")
            slope = self.val(f"b{index}")
            cols.append(a * wet + (1.0 - a) * dry + slope * dpm)
        return torch.stack(cols, dim=1)

    def th_of(self, pm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Spray gains (kg/s per unit effective opening), wet/dry blended."""
        a = torch.sigmoid((self.properties.critical_pressure - pm) / BLEND_HALF_WIDTH_MPA)
        th1 = a * self.val("th1") + (1.0 - a) * self.val("th1d")
        th2 = a * self.val("th2") + (1.0 - a) * self.val("th2d")
        return th1, th2

    def varphi(self, valve: torch.Tensor, index: int) -> torch.Tensor:
        """Monotone effective-opening map: varphi(v) = v**gamma, gamma > 0.

        varphi(0) = 0 gives the constant-action zero-spray identity;
        monotonicity on [0, 1] is guaranteed by gamma > 0.
        """
        gamma = self.val(f"gamma{index}")
        return valve.clamp(0.0, 1.0) ** gamma

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_boundary(self, boundary: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if boundary.shape[-1] != len(BOUNDARY_ELEMENTS):
            raise FinalWMProtocolError("boundary last dim must be 7 (BOUNDARY_ELEMENTS order)")
        cols = boundary.unbind(dim=-1)
        return cols  # D, uB, pm, Tm_sep, Tfw, p_out, W

    def _unpack_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action.shape[-1] != len(ACTION_ELEMENTS):
            raise FinalWMProtocolError("action last dim must be 2 (ACTION_ELEMENTS order)")
        v1, v2 = action.unbind(dim=-1)
        return v1.clamp(0.0, 1.0), v2.clamp(0.0, 1.0)

    def _pressures(self, pm: torch.Tensor, p_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0 = pm + (p_out - pm) / 3.0
        p1 = pm + 2.0 * (p_out - pm) / 3.0
        return p0, p1, p_out

    def _spray_rates(
        self,
        pm: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        w_total: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-attemperator spray rates (kg/s).

        `action` mode: Dsw_i = th_i(pm) * varphi(v_i); the measured total W
        is never read.  `boundary` mode (oracle diagnostics only, enforced
        upstream by the world-model config): hard mass conservation
        Dsw1 + Dsw2 = KAPPA * W with th*varphi determining the split.
        """
        th1, th2 = self.th_of(pm)
        phi1 = self.varphi(v1, 1)
        phi2 = self.varphi(v2, 2)
        if self.config.spray_total_mode == "boundary":
            total = KAPPA_TPH_TO_KGS * w_total.clamp(min=0.0)
            denom = th1 * phi1 + th2 * phi2 + _EPS
            return total * (th1 * phi1) / denom, total * (th2 * phi2) / denom
        return th1 * phi1, th2 * phi2

    def _mix_enthalpies(
        self,
        h: torch.Tensor,
        d_sw: torch.Tensor,
        d_flow: torch.Tensor,
        h_spray: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hm1 = (d_flow * h[:, 0] + d_sw[0] * h_spray) / (d_flow + d_sw[0] + _EPS)
        hm2 = (d_flow * h[:, 1] + d_sw[1] * h_spray) / (d_flow + d_sw[1] + _EPS)
        return hm1, hm2

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initial_steady_state(
        self,
        boundary_0: torch.Tensor,
        action_0: torch.Tensor,
        obs_0: torch.Tensor,
    ) -> torch.Tensor:
        """Observation-anchored steady initial state (the O1 'steady' arm).

        h is anchored to the three downstream observed temperatures; metal
        temperatures sit at the steady heat-balance offset; droplet masses
        start at the spray feed-rate equilibrium m = Dsw * tau_evap.
        """
        if obs_0.shape[-1] != len(OBSERVATION_ELEMENTS):
            raise FinalWMProtocolError("obs_0 last dim must be 5 (OBSERVATION_ELEMENTS order)")
        d_flow, u_b, pm, _tm_sep, _tfw, p_out, w_total = self._unpack_boundary(boundary_0)
        v1, v2 = self._unpack_action(action_0)
        p0, p1, p2 = self._pressures(pm, p_out)
        h = torch.stack([
            self.properties.enthalpy_of_pt(p0, obs_0[..., 0]),
            self.properties.enthalpy_of_pt(p1, obs_0[..., 2]),
            self.properties.enthalpy_of_pt(p2, obs_0[..., 4]),
        ], dim=-1)
        p_stack = torch.stack([p0, p1, p2], dim=-1)
        ts = self.properties.temperature_of_ph(p_stack, h)
        rb = u_b.clone()
        tm = ts + self.k_of(pm) * rb.unsqueeze(-1) / 3600.0 / self.tri("UA").unsqueeze(0)
        if self.config.init_metal_offset:
            tm = tm + self.tri("dTm").unsqueeze(0)
        dsw1, dsw2 = self._spray_rates(pm, v1, v2, w_total)
        tau_evap = self.val("tau_evap")
        m1 = dsw1 * tau_evap
        m2 = dsw2 * tau_evap
        parts = [h, tm, rb.unsqueeze(-1), m1.unsqueeze(-1), m2.unsqueeze(-1)]
        if self.layout.latent_dim > 0:
            parts.append(torch.zeros(h.shape[0], self.layout.latent_dim, dtype=h.dtype, device=h.device))
        return torch.cat(parts, dim=-1)

    def step(
        self,
        state: torch.Tensor,
        boundary: torch.Tensor,
        action: torch.Tensor,
        residual: ResidualInjection | None = None,
    ) -> TransitionStep:
        """One dt-second transition with semi-implicit substeps."""
        if state.shape[-1] != self.layout.dim:
            raise FinalWMProtocolError("state last dim does not match the state layout")
        layout = self.layout
        h = state[..., layout.h_slice]
        tm = state[..., layout.tm_slice]
        rb = state[..., layout.rb_index]
        m1 = state[..., layout.m_liq_slice.start]
        m2 = state[..., layout.m_liq_slice.stop - 1]
        latent = state[..., layout.latent_slice] if layout.latent_dim > 0 else None

        d_flow, u_b, pm, tm_sep, tfw, p_out, w_total = self._unpack_boundary(boundary)
        v1, v2 = self._unpack_action(action)
        p0, p1, p2 = self._pressures(pm, p_out)
        p_stack = torch.stack([p0, p1, p2], dim=-1)

        m_cap = self.tri("M").unsqueeze(0)
        ua = self.tri("UA").unsqueeze(0)
        cm = self.tri("Cm").unsqueeze(0)
        tau_b = self.val("tauB")
        tau_evap = self.val("tau_evap")
        a_w1 = self.val("aW1")
        a_w2 = self.val("aW2")
        m_dry0 = self.val("m_dry0")
        k_t = self.k_of(pm)
        h_spray = self.properties.liquid_enthalpy(tfw)
        h_sep = self.properties.separator_enthalpy(pm, tm_sep)
        dsw1, dsw2 = self._spray_rates(pm, v1, v2, w_total)
        hm1, hm2 = self._mix_enthalpies(h, (dsw1, dsw2), d_flow, h_spray)

        steam_power = None
        metal_power = None
        latent_step = None
        if residual is not None:
            steam_power = residual.steam_power
            metal_power = residual.metal_power
            latent_step = residual.latent_step
            for name, tensor, width in (
                ("steam_power", steam_power, 3),
                ("metal_power", metal_power, 3),
                ("latent_step", latent_step, layout.latent_dim),
            ):
                if tensor is not None and tensor.shape != (state.shape[0], width):
                    raise FinalWMProtocolError(f"residual {name} shape must be (B, {width})")

        dt_sub = float(self.config.substep_seconds)
        h_lo = self.properties.bounds.h_lo
        h_hi = self.properties.bounds.h_hi
        for _ in range(self.config.n_substeps):
            ts = self.properties.temperature_of_ph(p_stack, h)
            dry1 = torch.sigmoid(3.0 * (m_dry0 - m1) / m_dry0)
            dry2 = torch.sigmoid(3.0 * (m_dry0 - m2) / m_dry0)
            tsat0 = self.properties.saturation_temperature(p0)
            tsat1 = self.properties.saturation_temperature(p1)
            q_w1 = a_w1 * (tm[:, 0] - tsat0) * (1.0 - dry1)
            q_w2 = a_w2 * (tm[:, 1] - tsat1) * (1.0 - dry2)
            q_wall = torch.stack([q_w1, q_w2, torch.zeros_like(q_w1)], dim=-1)
            q = ua * (tm - ts)
            tm_in = (k_t * rb.unsqueeze(-1) / 3600.0 + ua * ts - q_wall) / cm
            if metal_power is not None:
                tm_in = tm_in + metal_power / cm
            tm = (tm + dt_sub * tm_in) / (1.0 + dt_sub * ua / cm)
            h_in = torch.stack([
                h_sep,
                hm1 + q_w1 / (d_flow + _EPS),
                hm2 + q_w2 / (d_flow + _EPS),
            ], dim=-1)
            h_flux = d_flow.unsqueeze(-1) * h_in + q
            if steam_power is not None:
                h_flux = h_flux + steam_power
            h = (h + dt_sub * h_flux / m_cap) / (1.0 + dt_sub * d_flow.unsqueeze(-1) / m_cap)
            h = ste_clamp(h, h_lo, h_hi)
            hm1, hm2 = self._mix_enthalpies(h, (dsw1, dsw2), d_flow, h_spray)
            m1 = (m1 + dt_sub * (dsw1 - m1 / tau_evap)).clamp(min=0.0)
            m2 = (m2 + dt_sub * (dsw2 - m2 / tau_evap)).clamp(min=0.0)
            rb = rb + dt_sub * (u_b - rb) / tau_b

        parts = [h, tm, rb.unsqueeze(-1), m1.unsqueeze(-1), m2.unsqueeze(-1)]
        if latent is not None:
            increment = latent_step if latent_step is not None else torch.zeros_like(latent)
            latent_next = self.latent_rho.unsqueeze(0) * latent + increment * float(self.config.dt_seconds)
            parts.append(latent_next)
        next_state = torch.cat(parts, dim=-1)
        aux = {
            "dsw1": dsw1, "dsw2": dsw2,
            "hm1": hm1, "hm2": hm2,
            "p0": p0, "p1": p1, "p_out": p2,
        }
        return TransitionStep(state=next_state, aux=aux)

    def output_temperatures(
        self,
        state: torch.Tensor,
        boundary: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Physical output equation g(x, b, u): five measurement temperatures.

        The attemperator outlet temperatures depend on the current spray
        rate, hence on the current action; pass `action=None` to evaluate at
        zero spray (constant-action identity reference).
        """
        layout = self.layout
        h = state[..., layout.h_slice]
        tm = state[..., layout.tm_slice]
        m1 = state[..., layout.m_liq_slice.start]
        m2 = state[..., layout.m_liq_slice.stop - 1]
        d_flow, _u_b, pm, _tm_sep, tfw, p_out, w_total = self._unpack_boundary(boundary)
        if action is None:
            v1 = torch.zeros_like(d_flow)
            v2 = torch.zeros_like(d_flow)
        else:
            v1, v2 = self._unpack_action(action)
        p0, p1, p2 = self._pressures(pm, p_out)
        h_spray = self.properties.liquid_enthalpy(tfw)
        dsw1, dsw2 = self._spray_rates(pm, v1, v2, w_total)
        hm1, hm2 = self._mix_enthalpies(h, (dsw1, dsw2), d_flow, h_spray)
        m_dry0 = self.val("m_dry0")
        dry1 = torch.sigmoid(3.0 * (m_dry0 - m1) / m_dry0)
        dry2 = torch.sigmoid(3.0 * (m_dry0 - m2) / m_dry0)
        tsat0 = self.properties.saturation_temperature(p0)
        tsat1 = self.properties.saturation_temperature(p1)
        q_w1 = self.val("aW1") * (tm[:, 0] - tsat0) * (1.0 - dry1)
        q_w2 = self.val("aW2") * (tm[:, 1] - tsat1) * (1.0 - dry2)
        h_o1 = hm1 + q_w1 / (d_flow + _EPS)
        h_o2 = hm2 + q_w2 / (d_flow + _EPS)
        p5 = torch.stack([p0, p0, p1, p1, p2], dim=-1)
        h5 = torch.stack([h[:, 0], h_o1, h[:, 1], h_o2, h[:, 2]], dim=-1)
        t_all = self.properties.temperature_of_ph(p5, h5)
        return torch.stack([
            t_all[:, 0],
            tsat0 + dry1 * (t_all[:, 1] - tsat0),
            t_all[:, 2],
            tsat1 + dry2 * (t_all[:, 3] - tsat1),
            t_all[:, 4],
        ], dim=-1)

    def integrate(
        self,
        state_0: torch.Tensor,
        boundary_seq: torch.Tensor,
        action_seq: torch.Tensor,
        *,
        closure: nn.Module | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out H steps.  Returns (states (B, H, dim), temperatures (B, H, 5)).

        When a closure module is supplied it is evaluated on the *current*
        state and boundary only (causal, action-blind by contract).
        """
        if boundary_seq.ndim != 3 or action_seq.shape[:2] != boundary_seq.shape[:2]:
            raise FinalWMProtocolError("boundary/action sequences must be (B, H, channels)")
        horizon = boundary_seq.shape[1]
        state = state_0
        states = []
        temps = []
        for t in range(horizon):
            boundary_t = boundary_seq[:, t]
            action_t = action_seq[:, t]
            residual = None
            if closure is not None:
                eps_t = noise[:, t] if noise is not None else None
                residual = closure(state, boundary_t, epsilon=eps_t)
            result = self.step(state, boundary_t, action_t, residual)
            state = result.state
            states.append(state)
            temps.append(self.output_temperatures(state, boundary_t, action_t))
        return torch.stack(states, dim=1), torch.stack(temps, dim=1)
