"""Deterministic GRU history baseline with the attention encoder's input contract.

This module is a unidirectional GRU, not an RSSM: it has no stochastic latent,
dynamics prior, or variational posterior. ``heads`` is accepted only for API
compatibility and does not affect the GRU. Matching hidden width and depth does
not match the number of parameters or computational cost of attention.
"""

import math

import torch
from torch import Tensor, nn


def _history_tokens(
    values: Tensor, observed_mask: Tensor, elapsed_seconds: Tensor,
    projection: nn.Linear, frequencies: Tensor, input_dim: int, hidden_dim: int,
) -> Tensor:
    """Validate and encode observations without sequence-wide statistics.

    Kept local because the attention encoder has no public preprocessing helper.
    A cross-encoder test checks parity at the temporal modules' inputs.
    """
    if not all(isinstance(x, Tensor) for x in (values, observed_mask, elapsed_seconds)):
        raise TypeError("values, observed_mask, and elapsed_seconds must be tensors")
    if values.ndim != 3 or values.shape[-1] != input_dim:
        raise ValueError("values must have shape (batch, time, input_dim)")
    batch, length, _ = values.shape
    if not batch or not length:
        raise ValueError("batch and time dimensions must be nonempty")
    if not values.is_floating_point():
        raise TypeError("values must be a real floating-point tensor")
    if observed_mask.shape != values.shape:
        raise ValueError("observed_mask must have the same shape as values")
    if observed_mask.dtype != torch.bool:
        raise TypeError("observed_mask must be Boolean")
    if elapsed_seconds.shape != (batch, length):
        raise ValueError("elapsed_seconds must have shape (batch, time)")
    if elapsed_seconds.dtype == torch.bool or elapsed_seconds.is_complex():
        raise TypeError("elapsed_seconds must be a real numeric tensor")
    if not (values.device == observed_mask.device == elapsed_seconds.device == projection.weight.device):
        raise ValueError("inputs and encoder must be on the same device")
    if not torch.isfinite(values[observed_mask]).all():
        raise ValueError("observed values must be finite")
    if not torch.isfinite(elapsed_seconds).all():
        raise ValueError("elapsed_seconds must be finite")
    if length > 1 and not (elapsed_seconds[:, 1:] > elapsed_seconds[:, :-1]).all():
        raise ValueError("elapsed_seconds must be strictly increasing")

    # Mask before arithmetic/casting: multiplying missing NaNs by zero is unsafe.
    clean = torch.where(observed_mask, values, torch.zeros_like(values)).to(projection.weight.dtype)
    if not torch.isfinite(clean).all():
        raise ValueError("observed values exceed the encoder dtype's finite range")
    tokens = projection(torch.cat((clean, observed_mask.to(clean.dtype)), dim=-1))
    # Exactly the attention baseline's pointwise signed-log sinusoidal seconds.
    time64 = elapsed_seconds.to(torch.float64)
    compressed = torch.sign(time64) * torch.log1p(torch.abs(time64))
    angles = compressed.to(tokens.dtype).unsqueeze(-1) * frequencies
    encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
    return tokens + encoding[..., :hidden_dim]


class CausalGRUHistoryEncoder(nn.Module):
    """Map normalized ``B,T,F`` histories to deterministic ``B,T,H`` memory.

    Missing rows remain valid observation-mask/time tokens. Each forward starts
    from zero recurrent memory; callers supply the complete initialization
    history. ``heads`` must be positive but is otherwise ignored, so hidden
    width need not be divisible by it. Dropout is zero at every depth.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, heads: int = 4, layers: int = 2,
    ) -> None:
        super().__init__()
        for name, value in (('input_dim', input_dim), ('hidden_dim', hidden_dim),
                            ('heads', heads), ('layers', layers)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(2 * input_dim, hidden_dim)
        self.recurrent = nn.GRU(
            hidden_dim, hidden_dim, num_layers=layers, batch_first=True,
            bidirectional=False, dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer(
            "time_frequencies",
            torch.exp(-math.log(10000.0)
                      * torch.arange(0, hidden_dim, 2, dtype=torch.float32) / hidden_dim),
        )

    def forward(
        self, values: Tensor, observed_mask: Tensor, elapsed_seconds: Tensor,
    ) -> Tensor:
        tokens = _history_tokens(
            values, observed_mask, elapsed_seconds, self.input_projection,
            self.time_frequencies, self.input_dim, self.hidden_dim,
        )
        output, _ = self.recurrent(tokens)
        return self.output_norm(output)
