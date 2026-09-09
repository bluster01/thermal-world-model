"""Small causal Transformer baseline for normalized process histories.

This is an ordinary causal Transformer, not PatchTST or Mamba. Its predictive
or control effectiveness has not been established. Every time step, including
an entirely missing row, remains a token with explicit observation-mask and
elapsed-time features. No future observations or sequence-wide time statistics
are used to construct a prefix representation.
"""

import math

import torch
from torch import Tensor, nn


class CausalHistoryEncoder(nn.Module):
    """Encode ``(batch, time, input_dim)`` histories into per-step memory.

    ``values`` must already be normalized by the caller. ``observed_mask`` is
    Boolean and marks actual observations, not padding. Unobserved values may
    contain arbitrary fill values, including NaN or infinity. Times are finite
    elapsed seconds and must increase strictly within each batch member.

    Outputs have shape ``(batch, time, hidden_dim)`` and the module's floating
    dtype. All attention layers use dropout=0; both training and evaluation
    preserve causal prefix independence.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, heads: int = 4, layers: int = 2
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("heads", heads),
            ("layers", layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(2 * input_dim, hidden_dim)
        # Construct layers separately so their initial parameters are independent.
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=2 * hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer(
            "time_frequencies",
            torch.exp(
                -math.log(10000.0)
                * torch.arange(0, hidden_dim, 2, dtype=torch.float32)
                / hidden_dim
            ),
        )

    def forward(
        self, values: Tensor, observed_mask: Tensor, elapsed_seconds: Tensor
    ) -> Tensor:
        self._validate_inputs(values, observed_mask, elapsed_seconds)

        # Multiplication by a zero mask would retain NaNs. Remove missing values
        # before casting, projecting, or performing any other arithmetic.
        clean_values = torch.where(observed_mask, values, torch.zeros_like(values))
        clean_values = clean_values.to(dtype=self.input_projection.weight.dtype)
        if not torch.isfinite(clean_values).all():
            raise ValueError("observed values exceed the encoder dtype's finite range")
        features = torch.cat(
            (clean_values, observed_mask.to(dtype=clean_values.dtype)), dim=-1
        )
        tokens = self.input_projection(features)

        # Pointwise signed-log seconds keeps long histories numerically bounded.
        # In particular, neither the final timestamp nor a full-window mean/scale
        # can affect a prefix. Double precision avoids overflow when converting
        # a finite large timestamp to the module's (usually float32) dtype.
        time64 = elapsed_seconds.to(dtype=torch.float64)
        compressed_time = torch.sign(time64) * torch.log1p(torch.abs(time64))
        angles = compressed_time.to(dtype=tokens.dtype).unsqueeze(-1)
        angles = angles * self.time_frequencies
        time_encoding = torch.stack((angles.sin(), angles.cos()), dim=-1)
        time_encoding = time_encoding.flatten(-2)[..., : self.hidden_dim]
        tokens = tokens + time_encoding

        length = values.shape[1]
        future_mask = torch.ones(
            length, length, dtype=torch.bool, device=values.device
        ).triu(diagonal=1)
        # No key-padding mask: an all-missing row is still a valid time token.
        # Each query can at least attend to itself, so no attention row is empty.
        for layer in self.layers:
            tokens = layer(tokens, src_mask=future_mask)
        return self.output_norm(tokens)

    def _validate_inputs(
        self, values: Tensor, observed_mask: Tensor, elapsed_seconds: Tensor
    ) -> None:
        if not all(
            isinstance(item, Tensor) for item in (values, observed_mask, elapsed_seconds)
        ):
            raise TypeError("values, observed_mask, and elapsed_seconds must be tensors")
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError("values must have shape (batch, time, input_dim)")
        batch, length, _ = values.shape
        if batch == 0 or length == 0:
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
        if not (
            values.device
            == observed_mask.device
            == elapsed_seconds.device
            == self.input_projection.weight.device
        ):
            raise ValueError("inputs and encoder must be on the same device")
        if not torch.isfinite(values[observed_mask]).all():
            raise ValueError("observed values must be finite")
        if not torch.isfinite(elapsed_seconds).all():
            raise ValueError("elapsed_seconds must be finite")
        if length > 1 and not (
            elapsed_seconds[:, 1:] > elapsed_seconds[:, :-1]
        ).all():
            raise ValueError("elapsed_seconds must be strictly increasing")
