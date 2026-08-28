"""Pre-registered A5 true water-coal-ratio mechanism arm.

This module is deliberately isolated from the frozen seven-channel world-model
contract.  It exposes a 7+2 oracle view (base boundary + true DCS ratio + unit
load), while the observer, closure and output equation keep reading base seven.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.final_wm.boundary import BoundarySequence
from src.final_wm.contracts import BOUNDARY_ELEMENTS, FinalWMProtocolError
from src.final_wm.data import SPLIT_TRAIN
from src.final_wm.data_v2 import BOUNDARY_EXT_ELEMENTS, CanonicalV2Record
from src.final_wm.model import FinalWorldModel, HistoryWindow
from src.final_wm.transition import Fan2020UDETransition, ResidualInjection, TransitionStep

A5_BOUNDARY_WIDTH = len(BOUNDARY_ELEMENTS) + 2
WATER_COAL_INDEX = len(BOUNDARY_ELEMENTS)
UNIT_LOAD_INDEX = WATER_COAL_INDEX + 1


@dataclass(frozen=True)
class WaterCoalReference:
    """Frozen train-split quadratic reference in normalized load coordinates."""

    coefficients: tuple[float, float, float]
    load_center: float
    load_scale: float
    residual_scale: float
    n_train: int


def fit_water_coal_reference(
    unit_load: torch.Tensor,
    water_coal_ratio: torch.Tensor,
    split: torch.Tensor,
    valid: torch.Tensor,
) -> WaterCoalReference:
    """Fit ``wc_ref(L)`` on valid train samples only; validation is unread."""
    tensors = tuple(torch.as_tensor(x).flatten().cpu() for x in (
        unit_load, water_coal_ratio, split, valid
    ))
    load, ratio, split_t, valid_t = tensors
    if not (load.shape == ratio.shape == split_t.shape == valid_t.shape):
        raise FinalWMProtocolError("A5 reference inputs must be aligned vectors")
    mask = (split_t == SPLIT_TRAIN) & valid_t.bool() & torch.isfinite(load) & torch.isfinite(ratio)
    if int(mask.sum()) < 3:
        raise FinalWMProtocolError("A5 reference needs at least three valid train samples")

    load_d = load[mask].to(torch.float64)
    ratio_d = ratio[mask].to(torch.float64)
    center = load_d.mean()
    scale = load_d.std(unbiased=False)
    if not bool(torch.isfinite(scale)) or float(scale) <= 1e-6:
        raise FinalWMProtocolError("A5 train load has no usable variation")
    x = (load_d - center) / scale
    design = torch.stack([torch.ones_like(x), x, x.square()], dim=1)
    coeff = torch.linalg.lstsq(design, ratio_d.unsqueeze(1)).solution[:, 0]
    residual = ratio_d - design @ coeff
    residual_scale = residual.std(unbiased=False)
    if not bool(torch.isfinite(residual_scale)) or float(residual_scale) <= 1e-6:
        raise FinalWMProtocolError("A5 train water-coal residual has no usable variation")
    return WaterCoalReference(
        coefficients=tuple(float(v) for v in coeff),
        load_center=float(center),
        load_scale=float(scale),
        residual_scale=float(residual_scale),
        n_train=int(mask.sum()),
    )


class WaterCoalRecord(CanonicalV2Record):
    """A5-only 7+2 view with the pre-registered operating gate applied."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        if "water_coal_ratio" not in self.boundary_ext_elements or "unit_load" not in self.boundary_ext_elements:
            raise FinalWMProtocolError(f"A5 requires a canonical v2.2 record: {path}")
        arrays = np.load(path)
        if "valid" not in arrays:
            raise FinalWMProtocolError(f"A5 record lacks valid mask: {path}")
        fuel = self.boundary_ext[:, BOUNDARY_EXT_ELEMENTS.index("fuel_corrected")]
        ratio = self.boundary_ext[:, BOUNDARY_EXT_ELEMENTS.index("water_coal_ratio")]
        load = self.boundary_ext[:, BOUNDARY_EXT_ELEMENTS.index("unit_load")]
        source_valid = torch.from_numpy(arrays["valid"].astype(bool))
        self.operating_mask = (
            source_valid
            & (load > 160.0)
            & (ratio > 1.0)
            & (ratio < 8.0)
            & (fuel > 50.0)
        )
        self.base_boundary = self.boundary
        self.water_coal_ratio = ratio
        self.unit_load = load
        self.boundary = torch.cat(
            [self.base_boundary, ratio.unsqueeze(1), load.unsqueeze(1)], dim=1
        )
        self.split = self.split.clone()
        self.split[~self.operating_mask] = -1
        self._split_runs_cache = {}

    def fit_reference(self) -> WaterCoalReference:
        return fit_water_coal_reference(
            self.unit_load, self.water_coal_ratio, self.split, self.operating_mask
        )

    def boundary_full(self) -> torch.Tensor:
        """Preserve the canonical v2 full-view meaning despite A5 sampling view."""
        return torch.cat([self.base_boundary, self.boundary_ext], dim=1)


class WaterCoalTransition(Fan2020UDETransition):
    """Base transition plus one bounded, sign-free water-coal power term."""

    def _base_boundary(self, boundary: torch.Tensor) -> torch.Tensor:
        if boundary.shape[-1] == A5_BOUNDARY_WIDTH:
            return boundary[..., : len(BOUNDARY_ELEMENTS)]
        # Parent initialisation calls the virtual output equation after the
        # A5 wrapper has already stripped its boundary.  Accept that internal
        # base view; public model entry points still require 9 channels.
        if boundary.shape[-1] == len(BOUNDARY_ELEMENTS):
            return boundary
        raise FinalWMProtocolError("A5 boundary must have 9 channels (base 7 + ratio + load)")

    def water_coal_total_power(self, boundary: torch.Tensor) -> torch.Tensor:
        if boundary.shape[-1] != A5_BOUNDARY_WIDTH:
            raise FinalWMProtocolError("A5 boundary must have 9 channels (base 7 + ratio + load)")
        load = boundary[..., UNIT_LOAD_INDEX]
        ratio = boundary[..., WATER_COAL_INDEX]
        x = (load - self.wc_load_center) / self.wc_load_scale
        ref = self.wc_coefficients[0] + self.wc_coefficients[1] * x + self.wc_coefficients[2] * x.square()
        z = (ratio - ref) / self.wc_residual_scale
        gain = torch.tanh(self.w_raw)
        return self.wc_power_bound_kw * torch.tanh(gain * z)

    def initial_steady_state(
        self, boundary_0: torch.Tensor, action_0: torch.Tensor, obs_0: torch.Tensor
    ) -> torch.Tensor:
        return super().initial_steady_state(self._base_boundary(boundary_0), action_0, obs_0)

    def step(
        self,
        state: torch.Tensor,
        boundary: torch.Tensor,
        action: torch.Tensor,
        residual: ResidualInjection | None = None,
    ) -> TransitionStep:
        total = self.water_coal_total_power(boundary)
        correction = total.unsqueeze(-1).expand(-1, 3) / 3.0
        if residual is None:
            residual = ResidualInjection(None, correction, None)
        else:
            metal = correction if residual.metal_power is None else residual.metal_power + correction
            residual = ResidualInjection(residual.steam_power, metal, residual.latent_step)
        return super().step(state, self._base_boundary(boundary), action, residual)

    def output_temperatures(
        self,
        state: torch.Tensor,
        boundary: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().output_temperatures(state, self._base_boundary(boundary), action)


class WaterCoalWorldModel(FinalWorldModel):
    """Oracle-only A5 adapter; learned components retain the seven-channel view."""

    @staticmethod
    def _base_history(history: HistoryWindow) -> HistoryWindow:
        return HistoryWindow(history.obs, history.actions, history.boundary[..., : len(BOUNDARY_ELEMENTS)])

    def _check_history(self, history: HistoryWindow) -> None:
        if history.boundary.shape[-1] != A5_BOUNDARY_WIDTH:
            raise FinalWMProtocolError("A5 boundary must have 9 channels (base 7 + ratio + load)")
        super()._check_history(self._base_history(history))

    def initial_state_posterior(self, history: HistoryWindow) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = self._steady_initial_state(history)
        base = self._base_history(history)
        return self.observer.posterior(base.obs, base.actions, base.boundary, anchor)

    def _initial_state(self, history: HistoryWindow, sample_posterior: bool = False) -> torch.Tensor:
        mode = self.config.initial_state_mode
        anchor = self._steady_initial_state(history)
        if mode == "steady":
            return anchor
        base = self._base_history(history)
        mu, sigma = self.observer.posterior(base.obs, base.actions, base.boundary, anchor)
        mask = self._correction_mask(mode, mu.device, mu.dtype)
        fused = anchor + (mu - anchor) * mask
        if sample_posterior:
            fused = fused + sigma * mask * torch.randn_like(fused)
        return fused

    def _boundary_sequence(
        self,
        history: HistoryWindow,
        horizon: int,
        boundary_mode: str,
        true_future_boundary: torch.Tensor | None,
        scenario: torch.Tensor | None,
    ) -> BoundarySequence:
        if boundary_mode != "oracle":
            raise FinalWMProtocolError("A5 true water-coal probe is oracle-only")
        if scenario is not None:
            raise FinalWMProtocolError("A5 oracle probe does not accept a scenario vector")
        if true_future_boundary is None:
            raise FinalWMProtocolError("A5 oracle mode requires true future boundary")
        if true_future_boundary.ndim != 3 or true_future_boundary.shape[-1] != A5_BOUNDARY_WIDTH:
            raise FinalWMProtocolError("A5 oracle boundary must have 9 channels")
        if true_future_boundary.shape[1] < horizon:
            raise FinalWMProtocolError("A5 oracle boundary shorter than the horizon")
        mu = true_future_boundary[:, :horizon]
        return BoundarySequence(
            mu=mu,
            logvar=torch.full_like(mu, 2.0 * math.log(1e-3)),
            mode="oracle",
        )


def _install_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value)


def promote_water_coal_model(
    model: FinalWorldModel,
    reference: WaterCoalReference,
) -> WaterCoalWorldModel:
    """Promote an already-built baseline model without reinitializing weights."""
    transition = model.transition
    transition.__class__ = WaterCoalTransition
    device = next(transition.parameters()).device
    dtype = next(transition.parameters()).dtype
    if "w_raw" not in transition._parameters:
        transition.register_parameter("w_raw", nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype)))
    _install_buffer(transition, "wc_coefficients", torch.tensor(reference.coefficients, device=device, dtype=dtype))
    _install_buffer(transition, "wc_load_center", torch.tensor(reference.load_center, device=device, dtype=dtype))
    _install_buffer(transition, "wc_load_scale", torch.tensor(reference.load_scale, device=device, dtype=dtype))
    _install_buffer(transition, "wc_residual_scale", torch.tensor(reference.residual_scale, device=device, dtype=dtype))
    _install_buffer(
        transition,
        "wc_power_bound_kw",
        torch.tensor(model.config.closure.residual_scale_kw, device=device, dtype=dtype),
    )
    model.__class__ = WaterCoalWorldModel
    return model
