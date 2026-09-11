"""Explicit rich-history adapters; future extension tensors are not accepted."""
from dataclasses import replace
import torch
from torch import nn
from src.final_wm.observer import ProbabilisticObserver
from src.final_wm.training import build_world_model
from src.final_wm.contracts import FinalWMProtocolError


class RichFusion(nn.Module):
    def __init__(self, spec, mean, std, properties=None):
        super().__init__()
        self.base = build_world_model(spec, properties)
        self.register_buffer("history_mean", mean.clone())
        self.register_buffer("history_std", std.clone())
        if spec.initial_state_mode != "steady":
            config = replace(self.base.config.observer, history_extension_dim=9)
            self.base.observer = ProbabilisticObserver(
                config, self.base.layout)
            self.base.config = replace(self.base.config, observer=config)

    def forward(self, history, extension, actions, boundary):
        if extension.shape != (*history.obs.shape[:2], 9):
            raise FinalWMProtocolError("expected nine past extension channels")
        anchor = self.base._steady_initial_state(history)
        if self.base.config.initial_state_mode == "steady":
            initial = anchor
        else:
            ext = (extension - self.history_mean[-9:]) / self.history_std[-9:]
            mu, _ = self.base.observer.posterior(history.obs, history.actions,
                torch.cat([history.boundary, ext], -1), anchor)
            mask = self.base._correction_mask("hybrid", mu.device, mu.dtype)
            initial = anchor + (mu - anchor) * mask
        return self.base._rollout(initial, self.base.boundary_model.oracle(boundary),
                                  actions, mode="oracle")


class RichBlackbox(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("history_mean", mean.clone())
        self.register_buffer("history_std", std.clone())
        self.hist_tok = nn.Linear(96, 64)
        self.fut_tok = nn.Linear(18, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1,
                                          batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear((23 + 9) * 64, 18)

    def forward(self, history, extension, actions, boundary):
        hist = torch.cat([history.obs, history.actions, history.boundary, extension], -1)
        if hist.shape[1:] != (96, 23) or boundary.shape[1:] != (18, 7):
            raise FinalWMProtocolError("rich blackbox input contract mismatch")
        normalized = (hist - self.history_mean) / self.history_std
        tokens = torch.cat([self.hist_tok(normalized.transpose(1, 2)),
            self.fut_tok(torch.cat([actions, boundary], -1).transpose(1, 2))], 1)
        deviation = self.head(self.enc(tokens).flatten(1))
        return deviation + history.obs[:, :, 4].mean(1, keepdim=True)
