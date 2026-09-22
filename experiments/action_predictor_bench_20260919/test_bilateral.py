"""Information isolation, routing, two-parameter calibration and exact restart tests."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .bilateral import PARENT,objective,select,unpack
from .bilateral_data import OUT_DEFAULT
from .bilateral_models import ARMS,MODES,SIDE_RAW,make_model
from .bilateral_events import features,find_events,responses,unique_train_rows
from .full_baselines import fit

torch.set_num_threads(1)


@pytest.fixture(scope='module')
def data():
    with np.load(OUT_DEFAULT) as z:
        data={k:z[k] for k in ('mean','scale')}
        for split in ('train','selector'):
            b=np.zeros((4,192,30),np.float32)
            b[:,:64]=z[f'hist30_{split}'][:4]
            b[:,64:,:10]=z[f'future_temp_{split}'][:4]
            b[:,64:,10:14]=z[f'future_act_{split}'][:4,1:]
            b[:,64:,14:21]=z[f'future_bnd_{split}'][:4,1:]
            data[split]=b
    return data


def model(data,arm,seed=11):
    result=make_model(arm,data['mean'],data['scale'],seed,PARENT)
    result.arm_name=arm
    return result


@pytest.mark.parametrize('arm',ARMS)
def test_all_arms_train_both_horizon_sections(data,arm):
    m=model(data,arm)
    losses=objective(m,torch.tensor(data['train'][:2]))
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    gradients=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if arm in ('S3','R3','P3'): torch.testing.assert_close(losses['loss'],losses['loss_A']+losses['loss_B'],rtol=0,atol=0)
    if arm=='G2': assert sum(p.numel() for p in m.parameters() if p.requires_grad)==2


@pytest.mark.parametrize('arm',('S0_A','S0_B','S1'))
@pytest.mark.parametrize('mode',MODES)
def test_unavailable_future_valves_never_leak_through_blocks(data,arm,mode):
    m=model(data,arm).double()
    h,u,d,_=[x.double() for x in unpack(data['train'][:2],'cpu')]
    changed=u.clone()
    excluded=sorted(set(range(4))-set(m.action_indices))
    changed[:,:,excluded]+=1234
    with torch.no_grad():
        before=m.forecast(h,u,d,mode);after=m.forecast(h,changed,d,mode)
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_future_temperature_poison_does_not_change_forecast(data):
    m=model(data,'S3')
    x=torch.tensor(data['train'][:2]);poison=x.clone();poison[:,64:,:10]=1e8
    for mode in MODES:
        h,u,d,_=unpack(x,'cpu');p=m.forecast(h,u,d,mode)
        h,u,d,_=unpack(poison,'cpu');q=m.forecast(h,u,d,mode)
        torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_common_initialization_and_fixed_context_comparison(data):
    s0,s1,s2,s3=[model(data,a) for a in ('S0_A','S1','S2','S3')]
    for name in ('encoder.weight_ih_l0','drive.0.weight','readout.weight','log_tau'):
        assert all(torch.equal(s0.state_dict()[name],m.state_dict()[name]) for m in (s1,s2,s3))
    # Exercise context use instead of relying on the initial zero adapter.
    with torch.no_grad():
        s2.adapter[-1].weight.fill_(.01)
        s3.adapter[-1].weight.copy_(s2.adapter[-1].weight)
    h,u,d,_=unpack(data['train'][:2],'cpu')
    for mode in ('native','block_context_fixed'):
        torch.testing.assert_close(s2.forecast(h,u,d,mode),s3.forecast(h,u,d,mode)[:,:,:5],rtol=0,atol=0)
    assert not torch.equal(s3.forecast(h,u,d,'block_joint_refresh')[:,32:,:5],s3.forecast(h,u,d,'block_context_fixed')[:,32:,:5])


@pytest.mark.parametrize('arm',('S3','R3','P3'))
def test_prefix_causality_and_protected_routes(data,arm):
    m=model(data,arm).double()
    h,u,d,_=[x.double() for x in unpack(data['train'][:2],'cpu')]
    u=h[:,-1:,10:14].expand_as(u).clone();candidate=u.clone();candidate[:,64:,3]+=.03
    with torch.no_grad():
        base=m.forecast(h,u,d,'native');p=m.forecast(h,candidate,d,'native')
        short=m.forecast(h,u[:,:65],d[:,:65],'native')
        torch.testing.assert_close(base[:,:64],short,rtol=0,atol=0)
        torch.testing.assert_close(base[:,:64],p[:,:64],rtol=0,atol=0)
        if arm in ('R3','P3'):
            change=p-base
            torch.testing.assert_close(change[:,:,:3],torch.zeros_like(change[:,:,:3]),rtol=0,atol=1e-12)
            torch.testing.assert_close(change[:,:,5:],torch.zeros_like(change[:,:,5:]),rtol=0,atol=1e-12)
            assert change[:,65:,3].min()<0
        if arm=='P3':
            for mode in MODES:
                torch.testing.assert_close(m.nominal(h,u,d,mode),m.nominal(h,candidate,d,mode),rtol=0,atol=0)
                diff=m.forecast(h,candidate,d,mode)-m.forecast(h,u,d,mode)
                torch.testing.assert_close(diff,p-base,rtol=0,atol=1e-12)


def test_g2_changes_only_terminal_readout_and_keeps_carrier(data):
    m=model(data,'G2').double()
    h,u,d,_=[x.double() for x in unpack(data['train'][:2],'cpu')]
    core=m.old.response.core
    raw=h[:,:,SIDE_RAW[0]];actions=u[:,:-1,[0,3]];boundary=d[:,:-1,:6]
    with torch.no_grad():
        before=core(raw,actions,boundary)
        whole=m.forecast(h,u,d,'block_context_fixed')
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.1)
    opt.zero_grad();m.forecast(h,u,d,'block_context_fixed')[:,:,4].sum().backward();opt.step()
    with torch.no_grad():
        after=core(raw,actions,boundary);new=m.forecast(h,u,d,'block_context_fixed')
    for key in ('transport_state','reference_prediction'):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    torch.testing.assert_close(whole[:,:,:4],new[:,:,:4],rtol=0,atol=0)
    assert not torch.equal(whole[:,:,4],new[:,:,4])


@pytest.mark.parametrize('arm',('G2','S3','R3','P3'))
def test_exact_optimizer_and_selector_resume(data,arm,tmp_path):
    small={**data,'train':data['train'][:2],'selector':data['selector'][:2]}
    args=SimpleNamespace(device='cpu',batch_size=2,learning_rate=.001,min_lr=.0001,min_delta=.002,
        min_epochs=12,max_epochs=2,lr_patience=4,stop_patience=6,smoke=False)
    joint=arm!='G2';choices=('short','balanced','AB_short','AB_balanced') if joint else ('short','balanced')
    budget=('AB_short','AB_balanced') if joint else ('short','balanced')
    def run(folder,epochs,resume=False):
        folder.mkdir(exist_ok=True);args.max_epochs=epochs
        return fit(arm,11,small,args,folder,model=model(data,arm),training_objective=objective,
            supervision_horizon=128,resume=resume,selector_fn=select,checkpoint_choices=choices,budget_choices=budget)
    run(tmp_path/'full',2);run(tmp_path/'split',1);run(tmp_path/'split',2,True)
    a=torch.load(tmp_path/'full/last.pt',weights_only=True);b=torch.load(tmp_path/'split/last.pt',weights_only=True)
    for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    for key in ('best','best_epochs','milestones','stale','updates'): assert a[key]==b[key]
    for parameter,state in a['optimizer']['state'].items():
        for key,value in state.items(): torch.testing.assert_close(value,b['optimizer']['state'][parameter][key],rtol=0,atol=0)


def test_sixty_second_event_detection_and_other_valve_excursion():
    time=np.arange(500)*10;v=np.zeros((500,30))
    # A 0.5pp per10s ramp qualifies at60s, but no single10s increment is2pp.
    v[101:107,10]=np.arange(1,7)*.005;v[107:,10]=.03
    events,_,_=find_events(time,v)
    assert any(e['valve']==0 for e in events)
    v[103,11]=.01  # brief excursion inside every qualifying ramp interval
    events,_,_=find_events(time,v)
    assert not any(e['valve']==0 for e in events)


def test_matching_features_are_strictly_pre_action():
    v=np.random.default_rng(11).normal(size=(300,30));poison=v.copy();poison[101:]+=10000
    np.testing.assert_array_equal(features(v,np.array([100]),0),features(poison,np.array([100]),0))


def test_control_responses_use_their_own_matching_anchor():
    values=np.zeros((600,30));values[:,0]=np.arange(600)**2
    event=dict(valve=0,index=100,epoch=1000,dose=.03,controls=[300,450],pre_balance_standardized=[0]*17)
    arrays,_=responses(np.arange(600)*10,values,[event])
    assert arrays['u1A_control_delta'][0,0,1,0]==301**2-300**2
    assert arrays['u1A_event_delta'][0,1,0]==101**2-100**2


def test_duplicate_train_windows_do_not_duplicate_timestamp_rows():
    h=np.tile(np.arange(64)[None,:,None],(2,1,30))
    times,values=unique_train_rows(dict(hist30_train=h,train_time=np.array([630,630])))
    assert len(times)==64 and values.shape==(64,30)
