"""Small bilateral ablations; future targets are never part of this interface."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .focused_models import Focused
from .models import R4_PLAN
from .r4_transport import TransportWorldModel

ARMS = ('G2', 'S0_A', 'S0_B', 'S1', 'S2', 'S3', 'R3', 'P3')
MODES = ('native', 'block_context_fixed', 'block_joint_refresh')
SIDE_VALVES = ((0, 3), (1, 2))
SIDE_RAW = (tuple(range(5))+(10,13,14,15,16,17,18,19),
            tuple(range(5,10))+(11,12,14,15,16,17,18,20))
OLD_AUX = tuple(range(21,29))+(5,9,29)


class BilateralSSM(nn.Module):
    def __init__(self, mean, scale, arm):
        super().__init__()
        self.arm, self.side = arm, int(arm == 'S0_B')
        self.joint = arm == 'S3'
        self.output_indices = tuple(range(10)) if self.joint else tuple(range(5*self.side,5*self.side+5))
        self.action_indices = tuple(range(4)) if arm in ('S2','S3') else SIDE_VALVES[self.side]
        self.register_buffer('mean', torch.as_tensor(mean).float().clone())
        self.register_buffer('scale', torch.as_tensor(scale).float().clone())
        self.core_indices = tuple(range(5*self.side,5*self.side+5))+tuple(10+i for i in SIDE_VALVES[self.side])+tuple(range(14,21))
        if arm == 'S0_A': self.extra_indices = (5,9)+tuple(range(21,30))
        elif arm == 'S0_B': self.extra_indices = (0,4)+tuple(range(21,30))
        else: self.extra_indices = tuple(range(5,10))+(11,12)+tuple(range(21,30))
        self.encoder = nn.GRU(14,48,batch_first=True)
        self.drive = nn.Sequential(nn.Linear(48+5+4+7,64),nn.SiLU(),nn.Linear(64,48),nn.Tanh())
        self.log_tau = nn.Parameter(torch.linspace(math.log(30),math.log(900),48))
        self.readout = nn.Linear(48,5)
        nn.init.normal_(self.readout.weight,std=.01); nn.init.zeros_(self.readout.bias)
        # Initialize common modules BEFORE the different-size context adapter.
        # S0_A/S1 and S2/S3 therefore have paired core weights at a given seed.
        self.adapter = nn.Sequential(nn.Linear(len(self.extra_indices)*8,48),nn.SiLU(),nn.Linear(48,48))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        if self.joint:
            self.readout_b = nn.Linear(48,5)
            nn.init.normal_(self.readout_b.weight,std=.01); nn.init.zeros_(self.readout_b.bias)

    def context(self, history, original, refresh):
        # Only observed/predicted bilateral state may refresh. Nine auxiliary
        # channels remain the ORIGINAL history, including load and spray total.
        source = history.clone() if refresh else original
        if refresh: source[:,:,21:] = original[:,:,21:]
        z = ((source-self.mean)/self.scale)[:,:,self.extra_indices]
        pooled = F.adaptive_avg_pool1d(z.transpose(1,2),8).flatten(1)
        return .1*torch.tanh(self.adapter(pooled))

    def segment(self, history, actions, boundaries, context):
        z = (history-self.mean)/self.scale
        _, encoded = self.encoder(z[:,:,self.core_indices])
        hidden = encoded[0]+context
        initial = hidden
        side = slice(self.side*5,self.side*5+5)
        anchor = z[:,-1,side]
        state = anchor
        u = (actions-self.mean[10:14])/self.scale[10:14]
        mask = torch.zeros(4,device=u.device,dtype=u.dtype)
        mask[list(self.action_indices)] = 1
        u = u*mask  # excluded future valves are never consulted, even at t=0
        d = (boundaries-self.mean[14:21])/self.scale[14:21]
        alpha = -torch.expm1(-10/self.log_tau.exp())
        output = []
        for t in range(actions.shape[1]):
            hidden = hidden+alpha*(self.drive(torch.cat((hidden,state,u[:,t],d[:,t]),-1))-hidden)
            state = anchor+self.readout(hidden-initial)
            a = state*self.scale[side]+self.mean[side]
            if self.joint:
                b = history[:,-1,5:10]+self.readout_b(hidden-initial)*self.scale[5:10]
                a = torch.cat((a,b),-1)
            output.append(a)
        return torch.stack(output,1)

    def forecast(self, history, actions, boundaries, mode='native'):
        if mode not in MODES: raise ValueError(mode)
        horizon = actions.shape[1]-1
        if boundaries.shape[1] != horizon+1 or horizon < 1: raise ValueError('H+1 controls required')
        if mode == 'native':
            return self.segment(history,actions[:,:-1],boundaries[:,:-1],self.context(history,history,False))
        current, result = history, []
        for start in range(0,horizon,32):
            stop = min(start+32,horizon)
            context = self.context(current,history,mode=='block_joint_refresh' and self.joint)
            p = self.segment(current,actions[:,start:stop],boundaries[:,start:stop],context)
            result.append(p)
            rows = history[:,-1:].expand(-1,stop-start,-1).clone()
            # Only supplied actions enter synthetic history; unavailable ones
            # stay at the ORIGINAL endpoint and cannot leak in from a bank.
            for j in self.action_indices: rows[:,:,10+j] = actions[:,start+1:stop+1,j]
            rows[:,:,14:21] = boundaries[:,start+1:stop+1]
            if self.joint and mode=='block_joint_refresh': rows[:,:,:10] = p
            else: rows[:,:,self.side*5:self.side*5+5] = p[:,:,:5]
            current = torch.cat((current,rows),1)[:,-64:]
        return torch.cat(result,1)

    forward = forecast


class BilateralR4(nn.Module):
    """Two uninterrupted R4 carriers, with common observed context and 7 boundaries."""
    output_indices = tuple(range(10))
    action_indices = tuple(range(4))

    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean).float().clone())
        self.register_buffer('scale',torch.as_tensor(scale).float().clone())
        self.context = nn.Sequential(nn.Linear(30*8,48),nn.SiLU(),nn.Linear(48,54))
        nn.init.zeros_(self.context[-1].weight); nn.init.zeros_(self.context[-1].bias)
        self.chains = nn.ModuleList()
        for indices in SIDE_RAW:
            core = TransportWorldModel(self.mean[list(indices)],self.scale[list(indices)],R4_PLAN)
            # Same widths/nonlinearity, explicit full held-reference valves and
            # seven future boundaries rather than silently ignoring right P.
            core.global_conditioner[0] = nn.Linear(27+4+7,32)
            core.conditioner[0] = nn.Linear(core.conditioner[0].in_features+1,16)
            self.chains.append(core)

    def decompose(self, history, actions, boundaries):
        context = .1*torch.tanh(self.context(F.adaptive_avg_pool1d(
            ((history-self.mean)/self.scale).transpose(1,2),8).flatten(1)))
        reference_action = (history[:,-1,10:14]-self.mean[10:14])/self.scale[10:14]
        d = (boundaries[:,:-1]-self.mean[14:21])/self.scale[14:21]
        predictions, responses, traces = [], [], []
        for side, core in enumerate(self.chains):
            raw = history[:,:,SIDE_RAW[side]]
            initial,_ = core.encoder(core.normalize(raw))
            h0 = initial[:,5:]+context[:,27*side:27*(side+1)]
            hidden = h0
            carrier = history.new_zeros((len(history),2,5))
            memory = history.new_zeros((len(history),5,core.local_size))
            modes = history.new_zeros((len(history),2,4))
            u = actions[:,:-1,SIDE_VALVES[side]]
            delta = (u-raw[:,-1:,5:7])/core.scale[5:7]
            refs, rs, cs, ks = [],[],[],[]
            for t in range(d.shape[1]):
                hidden = core.reference_transition(hidden,reference_action,d[:,t])
                ref = raw[:,-1,:5]+core.global_temperature(hidden-h0)*core.scale[:5]
                carrier,memory,modes,response,gain = core.transport_transition(
                    carrier,memory,modes,delta[:,t],d[:,t],(ref-core.mean[:5])/core.scale[:5],hidden)
                refs.append(ref); rs.append(response); cs.append(carrier); ks.append(gain)
            response = torch.stack(rs,1)
            predictions.append(torch.stack(refs,1)+response); responses.append(response)
            traces.append(dict(carrier=torch.stack(cs,1),gain=torch.stack(ks,1)))
        return dict(prediction=torch.cat(predictions,-1),response=torch.cat(responses,-1),traces=traces)

    def forecast(self, history, actions, boundaries, mode='native'):
        if mode not in MODES: raise ValueError(mode)
        # R3 never resets its reference/carrier. Its three columns are the same
        # protocol, explicitly identified in the run manifest.
        return self.decompose(history,actions,boundaries)['prediction']

    forward = forecast


class ProtectedBilateral(nn.Module):
    output_indices = tuple(range(10))
    action_indices = tuple(range(4))

    def __init__(self, mean, scale, seed):
        super().__init__()
        self.background = BilateralSSM(mean,scale,'S3')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100)
            self.response = BilateralR4(mean,scale)

    @property
    def mean(self): return self.background.mean

    @property
    def scale(self): return self.background.scale

    def nominal(self, history, actions, boundaries, mode):
        held = history[:,-1:,10:14].expand_as(actions)
        return self.background.forecast(history,held,boundaries,mode)

    def forecast(self, history, actions, boundaries, mode='native'):
        return self.nominal(history,actions,boundaries,mode)+self.response.decompose(history,actions,boundaries)['response']

    forward = forecast


class TerminalGains(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.value = nn.Parameter(initial[[3,5]].detach().clone())
        self.register_buffer('indices',torch.tensor([3,5]))

    def forward(self, original):
        return original.index_copy(0,self.indices,self.value)


class GainCalibration(nn.Module):
    output_indices = tuple(range(5))
    action_indices = (0,3)

    def __init__(self, mean, scale, parent):
        super().__init__()
        state = torch.load(Path(parent),weights_only=True,map_location='cpu')['model']
        self.old = Focused(state['mean'],state['scale'],state['aux_mean'],state['aux_scale'],'D')
        self.old.load_state_dict(state); self.old.requires_grad_(False)
        core = self.old.response.core
        parametrize.register_parametrization(core,'path_gain_raw',TerminalGains(core.path_gain_raw))
        self.register_buffer('mean',torch.as_tensor(mean).float().clone())
        self.register_buffer('scale',torch.as_tensor(scale).float().clone())
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==2

    def forecast(self, history, actions, boundaries, mode='native'):
        raw = torch.cat((history[:,:,SIDE_RAW[0]],history[:,:,OLD_AUX]),-1)
        return self.old.forecast_plan(raw,actions[:,:,SIDE_VALVES[0]],boundaries[:,:,:6],
                                     mode='native' if mode=='native' else 'block')

    forward = forecast


def make_model(arm, mean, scale, seed=11, parent=None):
    torch.manual_seed(seed)
    if arm == 'G2': return GainCalibration(mean,scale,parent)
    if arm == 'P3': return ProtectedBilateral(mean,scale,seed)
    if arm == 'R3':
        torch.manual_seed(seed+100)
        return BilateralR4(mean,scale)
    if arm in ARMS: return BilateralSSM(mean,scale,arm)
    raise ValueError(arm)
