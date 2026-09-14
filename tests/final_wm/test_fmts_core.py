"""Synthetic, no heldout plant access and no training for CORE1 contracts."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from experiments.fmts_mainsteam_20260911.models import RichFusion
from experiments.fmts_mainsteam_20260911.spec import structured_specs
from src.final_wm.model import HistoryWindow
from experiments.fmts_core_20260914.run import CoreView, replay_check, collect
from experiments.fmts_core_20260914.preflight import inspect_metadata
from test_fmts_greybox_norew import parent_bundle


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    spec=next(s for s in structured_specs((0,)) if s.arm=='fusion_gru_norew')
    model=RichFusion(spec,torch.zeros(23),torch.ones(23)).eval()
    obs=torch.tensor([420.,410.,500.,490.,540.]).repeat(2,96,1)
    actions=torch.tensor([.2,.3]).repeat(2,96,1)
    boundary=torch.tensor([450.,180.,24.,400.,310.,22.,30.]).repeat(2,96,1)
    h=HistoryWindow(obs,actions,boundary)
    return model,h,torch.zeros(2,96,9),actions[:,:18],boundary[:,:18]


def test_c11_equal_and_no_physical_mutation(fixture):
    m,h,e,a,b=fixture
    before={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad(): original=m(h,e,a,b).temps_mu[:,:,4]
    view=CoreView(m,'C11')
    assert torch.equal(view(h,e,a,b),original)
    assert all(not p.requires_grad for p in view.parameters())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('variant',['C00','C10','C01','C11'])
def test_network_call_contract(fixture,variant):
    m,h,e,a,b=fixture
    calls={'observer':0,'closure':0}
    hook=m.base.observer.mu_head.register_forward_hook(lambda *args:calls.__setitem__('observer',calls['observer']+1))
    chook=m.base.closure.register_forward_hook(lambda *args:calls.__setitem__('closure',calls['closure']+1))
    out=CoreView(m,variant)(h,e,a,b)
    hook.remove();chook.remove()
    assert torch.isfinite(out).all()
    assert calls['observer']==int(variant[1]=='1')
    assert calls['closure']==(18 if variant[2]=='1' else 0)


def test_invalid_variant(fixture):
    with pytest.raises(ValueError): CoreView(fixture[0],'best_sign')


def test_difference_gate_not_only_absolute_temperature():
    saved={'base':np.full((2,18),567.,dtype=np.float32),
           'valve1_prediction':np.full((2,18),566.999,dtype=np.float32),
           'valve2_prediction':np.full((2,18),566.998,dtype=np.float32)}
    actual={k:v.copy() for k,v in saved.items()}
    actual['valve1_prediction']+=.002
    with pytest.raises(AssertionError,match='difference'):replay_check(saved,actual)


def test_collect_once_initial_and_roundtrip_identity(fixture):
    m,h,e,a,b=fixture
    class Record:
        def batch(self,idx,split,device):
            assert split==1
            return h,e,a,b,h.obs[:,:18],torch.zeros(2,dtype=torch.long)
    view=CoreView(m,'C11')
    calls=[]
    original=view.initial
    def count(*args):calls.append(1);return original(*args)
    view.initial=count
    raw,trace=collect(view,Record(),np.array([96,97]),'cpu',True)
    assert len(calls)==1
    assert trace['initial'].shape==(2,11)
    assert trace['base_states'].shape==(2,18,11)
    assert replay_check(raw,raw)['max_response_difference_error_c']==0


def test_preflight_does_not_open_arrays(tmp_path):
    # A nonexistent values path must not be opened by metadata-only inspection.
    p=tmp_path/'metadata.json'
    p.write_text(json.dumps({'source_sha256':'a'*64,'partition_start_epoch_s':{'extension':1773619210},
                            'values_path':'DO_NOT_OPEN.npz','history_variables':13,
                            'exposure_status':'unverified'}))
    result=inspect_metadata(p)
    assert not result['scoring_enabled'] and not result['ready_for_independent_scoring']
    assert 'FMTS_SCHEMA_NOT_CONFIRMED' in result['blockers']
    assert 'EXPOSURE_NOT_CERTIFIED' in result['blockers']


def test_end_to_end_inference_and_audit(parent_bundle):
    from experiments.fmts_greybox_norew_20260913.run import execute as make_gnr
    from experiments.fmts_core_20260914.run import execute
    from experiments.fmts_core_20260914.audit import audit
    root,record=parent_bundle
    mapping='configs/final_wm/channel_mapping_v2.json'
    # Reuse the established engineering-only two-update synthetic fixture.
    make_gnr(root/'parent',record,mapping,None,root/'core_gnr',smoke=True)
    result=execute(root/'parent',root/'core_gnr',record,None,root/'core',mapping,smoke=True)
    assert len(result['runs'])==5 and result['training_updates']==0
    partial=audit(root/'core',root/'parent',root/'core_gnr')
    assert partial['artifacts_complete'] and not partial['complete']
    replay=audit(root/'core',root/'parent',root/'core_gnr',record)
    assert replay['complete'] and replay['array_sets_replayed']==10 and replay['trace_sets_replayed']==5
    with pytest.raises(AssertionError,match='output exists'):
        execute(root/'parent',root/'core_gnr',record,None,root/'core',mapping,smoke=True)


def test_confirmed_exposure_cannot_become_independent_by_schema_freeze(tmp_path):
    p=tmp_path/'exposed.json'
    p.write_text(json.dumps({'history_variables':23,'fmts_alignment_verified':True,
                            'final_checkpoint_manifest_sha256':'a'*64,
                            'evaluation_window_manifest_sha256':'b'*64,
                            'exposure_status':'historically_exposed'}))
    result=inspect_metadata(p)
    assert result['blockers']==['HISTORICAL_EXPOSURE_CONFIRMED']
    assert not result['scoring_enabled'] and not result['ready_for_independent_scoring']
