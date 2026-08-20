"""Fan2020-inspired UDE state transition (physics core of the final WM).

This module rewrites the recovered legacy E0/E0Evap dynamics
(`physical_models/fan2020_ude/legacy_experiments/02_train.py` and
`26_fix_evap.py`) as a clean, testable interface.  The scientific content is
unchanged; the software contract is new:

- packed state vector (`StateLayout`): h[3], Tm[3], rB, m_liq[2],
  dsw_lag[2], latent[L];
- exogenous inputs split into boundary (7 channels) and action (2 channels);
- actions enter *only* through the monotone valve mapping varphi(u); the
  measured total spray flow W is usable solely in oracle diagnostics mode;
- spray affects measurements through first-order transport-lag states
  (repair ②, design 2026-08-20): the output equation is state-driven and
  never responds to the current action instantaneously;
- rewetting powers are hard-bounded by the evaporation mass-balance
  contract (repair ③): q_w <= (m/tau_evap) * max(h_pre - h_spray, 0);
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
    # Repair ④: spray gains anchored to the auditpack data regression
    # dW/dv1 = 27.76, dW/dv2 = 70.01 t/h per full travel (side A val).
    "th1": 7.71, "th2": 19.45,                          # kg/s per full opening
    "th1d": 7.71, "th2d": 19.45,                        # dry-mode gains (same anchor)
    "dTm0": 1.0, "dTm1": 1.0, "dTm2": 1.0,             # K init correction
    "tau_evap": 15.0,                                   # s droplet evaporation
    # Repair ② lag priors anchored to the adhoc2 learned-lag evidence
    # (fix3_learnlag: tau_sw = 73-86 s learned from data); learnable params.
    "tau_mix1": 80.0, "tau_mix2": 80.0,                 # s spray->mixing lag (repair ②)
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

    # executor-side fix (2026-08-18, per user instruction; Supervisor review
    # required): move injected thermo properties together with the module for
    # GPU execution.  properties is a plain attribute, so nn.Module._apply
    # does not touch it; a to() override on a child module is never invoked
    # by the parent's .to(), which recurses via _apply only.
    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        if hasattr(self.properties, "_apply"):
            self.properties = self.properties._apply(fn)
        return result

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
        d_sw: tuple[torch.Tensor, torch.Tensor],
        d_flow: torch.Tensor,
        h_spray: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hm1 = (d_flow * h[:, 0] + d_sw[0] * h_spray) / (d_flow + d_sw[0] + _EPS)
        hm2 = (d_flow * h[:, 1] + d_sw[1] * h_spray) / (d_flow + d_sw[1] + _EPS)
        return hm1, hm2

    def _rewetting_powers(
        self,
        tm: torch.Tensor,
        m1: torch.Tensor,
        m2: torch.Tensor,
        p0: torch.Tensor,
        p1: torch.Tensor,
        h_pre: torch.Tensor,
        h_spray: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall rewetting powers (kW) with the mass-balance hard contract.

        Repair ③ (design 2026-08-20): rewetting heat is carried by the
        evaporating deposited droplets (mass flux m/tau_evap); each kg of
        evaporated droplet can transfer at most (h_pre - h_spray) of net
        enthalpy to the steam phase.  The raw aW*(Tm - Tsat) term is capped
        by that flux-limited bound, which closes the spray -> m -> (1-dry)
        -> q_w positive-feedback loop (aW=0 ablation: 0.27 -> 1.00 correct
        direction) while preserving the condensation direction when the
        wall is colder than saturation (min() passes negatives through).
        """
        a_w1 = self.val("aW1")
        a_w2 = self.val("aW2")
        m_dry0 = self.val("m_dry0")
        tau_evap = self.val("tau_evap")
        dry1 = torch.sigmoid(3.0 * (m_dry0 - m1) / m_dry0)
        dry2 = torch.sigmoid(3.0 * (m_dry0 - m2) / m_dry0)
        tsat0 = self.properties.saturation_temperature(p0)
        tsat1 = self.properties.saturation_temperature(p1)
        raw1 = a_w1 * (tm[:, 0] - tsat0) * (1.0 - dry1)
        raw2 = a_w2 * (tm[:, 1] - tsat1) * (1.0 - dry2)
        cap1 = (m1 / tau_evap) * (h_pre[:, 0] - h_spray).clamp(min=0.0)
        cap2 = (m2 / tau_evap) * (h_pre[:, 1] - h_spray).clamp(min=0.0)
        return torch.minimum(raw1, cap1), torch.minimum(raw2, cap2)

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
        # Repair ②: at steady state the transport-lag state equals the target rate.
        lag1 = dsw1.clone()
        lag2 = dsw2.clone()
        parts = [h, tm, rb.unsqueeze(-1), m1.unsqueeze(-1), m2.unsqueeze(-1),
                 lag1.unsqueeze(-1), lag2.unsqueeze(-1)]
        if self.layout.latent_dim > 0:
            parts.append(torch.zeros(h.shape[0], self.layout.latent_dim, dtype=h.dtype, device=h.device))
        return torch.cat(parts, dim=-1)

    def _substep(
        self,
        h: torch.Tensor, tm: torch.Tensor, rb: torch.Tensor,
        m1: torch.Tensor, m2: torch.Tensor,
        lag1: torch.Tensor, lag2: torch.Tensor,
        dsw1: torch.Tensor, dsw2: torch.Tensor,
        d_flow: torch.Tensor, u_b: torch.Tensor,
        p_stack: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor,
        h_spray: torch.Tensor, h_sep: torch.Tensor,
        m_cap: torch.Tensor, ua: torch.Tensor, cm: torch.Tensor,
        k_t: torch.Tensor,
        tau_b: torch.Tensor, tau_evap: torch.Tensor,
        tau_mix1: torch.Tensor, tau_mix2: torch.Tensor,
        dt_sub: float, h_lo: float, h_hi: float,
        steam_power: torch.Tensor | None,
        metal_power: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """One semi-implicit Euler substep.

        Extracted from `step` (2026-08-21) so the runner can wrap it with
        torch.compile: the rollout is launch-bound (n_substeps x horizon
        sequential small kernels), fusing this body is the speed lever.
        Numerics are identical to the pre-extraction inline loop.
        """
        # Repair ②: spray reaches the mixing chambers through first-order
        # transport lags; deposition into m_liq still reads the target rate.
        lag1 = (lag1 + dt_sub * (dsw1 - lag1) / tau_mix1).clamp(min=0.0)
        lag2 = (lag2 + dt_sub * (dsw2 - lag2) / tau_mix2).clamp(min=0.0)
        hm1, hm2 = self._mix_enthalpies(h, (lag1, lag2), d_flow, h_spray)
        ts = self.properties.temperature_of_ph(p_stack, h)
        q_w1, q_w2 = self._rewetting_powers(
            tm, m1, m2, p0, p1, h[:, :2], h_spray
        )
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
        m1 = (m1 + dt_sub * (dsw1 - m1 / tau_evap)).clamp(min=0.0)
        m2 = (m2 + dt_sub * (dsw2 - m2 / tau_evap)).clamp(min=0.0)
        rb = rb + dt_sub * (u_b - rb) / tau_b
        return h, tm, rb, m1, m2, lag1, lag2

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
        lag1 = state[..., layout.dsw_lag_slice.start]
        lag2 = state[..., layout.dsw_lag_slice.stop - 1]
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
        tau_mix1 = self.val("tau_mix1")
        tau_mix2 = self.val("tau_mix2")
        k_t = self.k_of(pm)
        h_spray = self.properties.liquid_enthalpy(tfw)
        h_sep = self.properties.separator_enthalpy(pm, tm_sep)
        dsw1, dsw2 = self._spray_rates(pm, v1, v2, w_total)

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
            h, tm, rb, m1, m2, lag1, lag2 = self._substep(
                h, tm, rb, m1, m2, lag1, lag2, dsw1, dsw2, d_flow, u_b,
                p_stack, p0, p1, h_spray, h_sep, m_cap, ua, cm, k_t,
                tau_b, tau_evap, tau_mix1, tau_mix2, dt_sub, h_lo, h_hi,
                steam_power, metal_power,
            )

        hm1, hm2 = self._mix_enthalpies(h, (lag1, lag2), d_flow, h_spray)
        parts = [h, tm, rb.unsqueeze(-1), m1.unsqueeze(-1), m2.unsqueeze(-1),
                 lag1.unsqueeze(-1), lag2.unsqueeze(-1)]
        if latent is not None:
            increment = latent_step if latent_step is not None else torch.zeros_like(latent)
            latent_next = self.latent_rho.unsqueeze(0) * latent + increment * float(self.config.dt_seconds)
            parts.append(latent_next)
        next_state = torch.cat(parts, dim=-1)
        aux = {
            "dsw1": dsw1, "dsw2": dsw2,                    # instantaneous targets
            "dsw_lag1": lag1, "dsw_lag2": lag2,            # transport-lagged rates
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
        """Physical output equation g(x, b): five measurement temperatures.

        Repair ② (design 2026-08-20): the attemperator outlet temperatures
        read the transport-lagged spray rates from the state vector, so the
        measurement is fully state-driven and never responds to the current
        action within the same step.  The dry-out blend follows the
        lagged-rate-equivalent wetness (the sensor sits downstream of the
        transport path), not the wall droplet inventory.  `action` is
        accepted for interface compatibility and ignored.
        """
        layout = self.layout
        h = state[..., layout.h_slice]
        tm = state[..., layout.tm_slice]
        lag1 = state[..., layout.dsw_lag_slice.start]
        lag2 = state[..., layout.dsw_lag_slice.stop - 1]
        d_flow, _u_b, pm, _tm_sep, tfw, p_out, _w_total = self._unpack_boundary(boundary)
        p0, p1, p2 = self._pressures(pm, p_out)
        h_spray = self.properties.liquid_enthalpy(tfw)
        hm1, hm2 = self._mix_enthalpies(h, (lag1, lag2), d_flow, h_spray)
        m1 = state[..., layout.m_liq_slice.start]
        m2 = state[..., layout.m_liq_slice.stop - 1]
        m_dry0 = self.val("m_dry0")
        tau_evap = self.val("tau_evap")
        dry1 = torch.sigmoid(3.0 * (m_dry0 - lag1 * tau_evap) / m_dry0)
        dry2 = torch.sigmoid(3.0 * (m_dry0 - lag2 * tau_evap) / m_dry0)
        tsat0 = self.properties.saturation_temperature(p0)
        tsat1 = self.properties.saturation_temperature(p1)
        q_w1, q_w2 = self._rewetting_powers(tm, m1, m2, p0, p1, h[:, :2], h_spray)
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
