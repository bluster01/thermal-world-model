"""Synthetic engineering checks, not evidence of plant prediction improvement."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
import pytest
from experiments.fmts_mainsteam_20260911.models import RichFusion
from experiments.fmts_mainsteam_20260911.spec import structured_specs
from experiments.fmts_m7fusion_20260913.spec import specs, MODEL, ARM
from experiments.fmts_m7fusion_20260913.models import M7Fusion
from src.world_model import PatchEmbedding
from test_fmts_greybox_norew import parent_bundle
from src.final_wm.contracts import FinalWMProtocolError


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_parent_training_template_and_active_physics_unchanged(seed):
    old = next(s for s in structured_specs((seed,)) if s.arm == 'fusion_gru_norew')
    new = specs((seed,))[0]
    assert {k for k in asdict(old) if asdict(old)[k] != asdict(new)[k]} == {'arm'}
    torch.manual_seed(seed)
    a = RichFusion(old, torch.zeros(23), torch.ones(23))
    torch.manual_seed(seed)
    b = M7Fusion(new, torch.zeros(23), torch.ones(23))
    for key, value in a.state_dict().items():
        if not key.startswith('base.observer.'):
            assert torch.equal(value, b.state_dict()[key]), key
    assert b.model_metadata()['effective_observer'] == MODEL


def inputs():
    torch.manual_seed(811)
    model = M7Fusion(specs((0,))[0], torch.zeros(23), torch.ones(23)).eval()
    observer = model.base.observer
    obs = observer.obs_loc + torch.randn(3, 96, 5)*observer.obs_scale*.2
    act = observer.action_loc + torch.randn(3, 96, 2)*observer.action_scale*.1
    boundary = observer.boundary_loc + torch.randn(3, 96, 16)*observer.boundary_scale*.1
    anchor = torch.randn(3, 11)
    return model, obs, act, boundary, anchor


def test_zero_head_anchor_and_independent_state_rows():
    model, obs, act, b, anchor = inputs()
    o = model.base.observer
    mu, _ = o.posterior(obs, act, b, anchor)
    assert torch.equal(mu, anchor)
    assert o.mu_head.weight.shape == (11, 131)
    with torch.no_grad():
        o.mu_head.bias[3] = .2
    after, _ = o.posterior(obs, act, b, anchor)
    assert torch.all(after[:, 3] != anchor[:, 3])
    assert torch.equal(after[:, [0,1,2,4,5,6,7,8,9,10]], anchor[:, [0,1,2,4,5,6,7,8,9,10]])
    assert torch.all((after-anchor).abs() <= .1*o.state_scale+1e-6)


def test_patch_and_m7_temporal_variable_path_match_legacy_modules():
    model, *_ = inputs()
    e = model.base.observer.encoder
    old = PatchEmbedding(96, 16, 8, 64).eval()
    old.load_state_dict(e.patch.state_dict())
    x = torch.randn(3, 96, 23)
    actual = e.patch(x.transpose(1, 2))
    expected = torch.stack([old(x[:, :, i]) for i in range(23)], 1)
    assert torch.allclose(actual, expected, atol=1e-6)
    from src.world_model import PerVariableTCN, VariableAttention, RevIN
    assert isinstance(e.tcn, PerVariableTCN)
    assert isinstance(e.varattn, VariableAttention)
    assert isinstance(e.revin, RevIN)


def test_all_history_extensions_receive_gradients_and_levels_are_retained():
    model, obs, act, b, anchor = inputs()
    o = model.base.observer
    with torch.no_grad():
        o.mu_head.weight.normal_(0, .001)
    b.requires_grad_()
    mu, _ = o.posterior(obs, act, b, anchor)
    mu[:, 3:7].sum().backward()
    assert torch.isfinite(b.grad).all()
    assert (b.grad[:, :, -9:].abs().sum((0,1)) > 0).all()
    with torch.no_grad():
        old = o.encode(obs, act, b)
        shifted = o.encode(obs+10, act, b)
    assert not torch.allclose(old, shifted)
    assert not hasattr(o, 'state_queries')


def test_full_synthetic_run_replay_and_tamper_detection(parent_bundle):
    from experiments.fmts_m7fusion_20260913.run import execute
    from experiments.fmts_m7fusion_20260913.audit import audit
    root, path = parent_bundle
    out = root/'m7_supplement'
    reports = execute(root/'parent',path,'configs/final_wm/channel_mapping_v2.json',None,out,smoke=True)
    assert len(reports)==1 and reports[0]['arm']==ARM and reports[0]['updates']==2
    result = audit(out,root/'parent',path)
    assert result['complete'] and result['array_sets_replayed']==2
    assert result['health_array_sets_replayed']==3
    assert (out/'indices.npz').read_bytes()==(root/'parent'/'indices.npz').read_bytes()
    assert len(json.loads((out/'comparison.json').read_text())['pairs'])==4
    with pytest.raises(FinalWMProtocolError,match='output exists'):
        execute(root/'parent',path,'configs/final_wm/channel_mapping_v2.json',None,out,smoke=True)
    artifact = out/f'{ARM}_seed0'/'observer_health.npz'
    original = artifact.read_bytes()
    try:
        artifact.write_bytes(original+b'tamper')
        with pytest.raises(FinalWMProtocolError,match='artifact hash mismatch'):
            audit(out,root/'parent',path)
    finally:
        artifact.write_bytes(original)
