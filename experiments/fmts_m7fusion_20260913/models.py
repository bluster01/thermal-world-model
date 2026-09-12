"""M7-family history encoding, unchanged independent physical-state posterior.

No temperature residual head and no new action path. The parent's GRU config is
a construction/dispatch template; effective implementation is saved explicitly.
"""
from dataclasses import asdict
import torch
from torch import nn
from src.world_model import RevIN, PerVariableTCN, VariableAttention
from src.final_wm.observer import ProbabilisticObserver
from src.final_wm.contracts import FinalWMProtocolError
from experiments.fmts_mainsteam_20260911.models import RichFusion
from .spec import MODEL


class FixedPatchEmbedding(nn.Module):
    """Identical M7 patch math, with dimensions owned by the frozen local spec."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 64)
        self.norm = nn.LayerNorm(64)
        self.pos_embed = nn.Parameter(torch.randn(1, 11, 64)*.02)

    def forward(self, variable_history):
        patches = variable_history.unfold(-1, 16, 8)
        return self.norm(self.proj(patches)) + self.pos_embed


class M7HistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.revin = RevIN(23)
        self.patch = FixedPatchEmbedding()
        self.tcn = PerVariableTCN(11, 64, n_layers=2, dropout=.1)
        self.varattn = VariableAttention(64, n_heads=4, dropout=.1)
        self.context = nn.Sequential(
            nn.Linear(23*64+2*23, 256), nn.GELU(), nn.Dropout(.1),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128, elementwise_affine=False))

    def forward(self, normalized_history):
        if normalized_history.shape[1:] != (96, 23):
            raise FinalWMProtocolError('M7 encoder requires exactly 96 past x 23 variables')
        normalized = self.revin(normalized_history, mode='norm')
        # Capture each call's detached past-only statistics before another call.
        mean, std = self.revin._mean[:, 0], self.revin._std[:, 0]
        batch = len(normalized)
        patches = self.patch(normalized.transpose(1, 2))
        variables = self.tcn(patches.reshape(batch*23, 11, 64)).reshape(batch, 23, 64)
        variables, _ = self.varattn(variables)
        return self.context(torch.cat([variables.flatten(1), mean, std], -1))


class M7Observer(ProbabilisticObserver):
    """Reuse the tested vector posterior, replacing only its encode method/module.

The `gru` value in inherited ObserverConfig selects a vector-context posterior
interface, not the effective network. model_metadata() records M7Observer_v1.
This avoids mutating the frozen parent implementation/registry for a supplement.
"""
    def __init__(self, config, layout):
        if config.encoder_type != 'gru' or config.history_extension_dim != 9 or config.d_hidden != 128:
            raise FinalWMProtocolError('M7 requires the registered rich GRU posterior template')
        super().__init__(config, layout)
        self.encoder = M7HistoryEncoder()

    def encode(self, obs, actions, boundary):
        self._check_history(obs, actions, boundary)
        features = torch.cat([(obs-self.obs_loc)/self.obs_scale,
                              (actions-self.action_loc)/self.action_scale,
                              (boundary-self.boundary_loc)/self.boundary_scale], -1)
        return self.encoder(features)


class M7Fusion(RichFusion):
    def __init__(self, spec, mean, std, properties=None):
        super().__init__(spec, mean, std, properties)
        if spec.initial_state_mode != 'hybrid' or spec.closure_mode != 'conservative_norew':
            raise FinalWMProtocolError('M7 fusion requires the frozen no-rewet hybrid template')
        self.base.observer = M7Observer(self.base.config.observer, self.base.layout)

    def model_metadata(self):
        return dict(physical_and_posterior_template=asdict(self.base.config),
                    effective_observer=MODEL, observer_class='M7Observer',
                    vector_posterior_dispatch='gru interface only; NOT a GRU encoder')
