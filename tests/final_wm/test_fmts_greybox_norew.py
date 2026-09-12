"""Engineering-only tests. Synthetic smoke must not become paper evidence."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from experiments.fmts_mainsteam_20260911.spec import structured_specs
from experiments.fmts_mainsteam_20260911.models import RichFusion
from experiments.fmts_mainsteam_20260911.run import main as parent_main
from experiments.fmts_greybox_norew_20260913.spec import specs, serialized_spec, ARM
from experiments.fmts_greybox_norew_20260913.run import execute, validate_parent
from experiments.fmts_greybox_norew_20260913.audit import audit
from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.synthetic import synthetic_canonical_arrays


@pytest.mark.parametrize('seed',[0,1,2])
def test_only_registered_scientific_change(seed):
    parent=asdict(next(s for s in structured_specs((seed,)) if s.arm=='greybox_steady_none'))
    new=asdict(specs((seed,))[0])
    assert {k for k in parent if parent[k]!=new[k]}=={'arm','closure_mode'}
    assert new['closure_mode']=='none_norew' and new['initial_state_mode']=='steady'
    assert new['epochs']*new['batches_per_epoch']==24000
    assert json.loads(json.dumps(serialized_spec()))==serialized_spec()


def test_only_rewetting_parameters_differ_at_initialization():
    parent=next(s for s in structured_specs((0,)) if s.arm=='greybox_steady_none')
    torch.manual_seed(0)
    old=RichFusion(parent,torch.zeros(23),torch.ones(23))
    torch.manual_seed(0)
    new=RichFusion(specs((0,))[0],torch.zeros(23),torch.ones(23))
    changes={k for k,v in old.state_dict().items() if not torch.equal(v,new.state_dict()[k])}
    assert changes=={'base.transition.raw.aW1','base.transition.raw.aW2'}
    assert new.base.config.transition.rewet_ablate
    assert new.base.config.closure.injection_mode=='none'
    for name in ['aW1','aW2']:
        assert not new.base.transition.raw[name].requires_grad
        assert new.base.transition.raw[name].item()==-30


@pytest.fixture(scope='module')
def parent_bundle(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp('gnr1_smoke')
    arrays=synthetic_canonical_arrays(total_steps=2200,seed=11)
    centers=np.array([250,4,400,5,600,500,500,4,400])
    arrays['boundary_ext']=(centers[None]*(1+np.random.default_rng(10).normal(0,.01,(2200,9)))).astype('float32')
    arrays['aux']=np.zeros((2200,15),dtype='float32')
    arrays['mill_on']=np.zeros((2200,8),dtype='uint8')
    path=root/'synthetic_v22.npz'
    np.savez_compressed(path,**arrays)
    original=sys.argv
    try:
        sys.argv=['run','--record',str(path),'--out',str(root/'parent'),'--smoke']
        parent_main()
    finally:
        sys.argv=original
    return root,path


def test_synthetic_end_to_end_and_checkpoint_replay(parent_bundle):
    root,path=parent_bundle
    mapping='configs/final_wm/channel_mapping_v2.json'
    reports=execute(root/'parent',path,mapping,None,root/'supplement',smoke=True)
    assert len(reports)==1 and reports[0]['arm']==ARM and reports[0]['updates']==2
    assert (root/'parent'/'indices.npz').read_bytes()==(root/'supplement'/'indices.npz').read_bytes()
    result=audit(root/'supplement',root/'parent',path)
    assert result['complete'] and result['runs_checked']==1 and result['array_sets_replayed']==2
    comparison=json.loads((root/'supplement'/'comparison.json').read_text())
    assert len(comparison['pairs'])==3
    assert not comparison['locked_test_evaluated']
    with pytest.raises(FinalWMProtocolError,match='output exists'):
        execute(root/'parent',path,mapping,None,root/'supplement',smoke=True)


def test_formal_run_rejects_smoke_parent(parent_bundle):
    root,path=parent_bundle
    with pytest.raises(FinalWMProtocolError,match='smoke/test'):
        validate_parent(root/'parent',path,'configs/final_wm/channel_mapping_v2.json',None,False)


def test_changed_record_rejected(parent_bundle,tmp_path):
    root,path=parent_bundle
    modified=tmp_path/'changed.npz'
    modified.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(FinalWMProtocolError,match='record hash'):
        validate_parent(root/'parent',modified,'configs/final_wm/channel_mapping_v2.json',None,True)
