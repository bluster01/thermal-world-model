#!/usr/bin/env python3
"""
exp_075_sac_wm.py — P4-2: M7 内 SAC (世界模型当仿真器, MBPO协议)
==================================================================
环境: M7 预测步进 + 回填窗口 + 扰动注入(训练时模拟扰动, 解决评测OOD)
SAC: actor(tanh) + 双Q + 自动熵 | 奖励 r = −|e| − 0.5·|Δa|
评测: 复用 exp_073 roll_policy (策略接口一致)
用法: python exp_075_sac_wm.py [--smoke] [--seed 42]
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv
from experiments.phase1_dynamics.exp_025_unified_benchmark import train_raw as TRAIN_RAW

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/exp_075_sac_wm'
os.makedirs(f'{OUT}/checkpoints', exist_ok=True)
SMOKE = '--smoke' in sys.argv
STEPS = 3000 if SMOKE else 100000

# ============ SAC 组件 ============
def mlp(din, dout, hidden=256):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, dout))

class Actor(nn.Module):
    def __init__(self, din=40, dout=2):
        super().__init__()
        self.net = mlp(din, dout * 2)
    def forward(self, s):
        mu, logstd = self.net(s).chunk(2, -1)
        logstd = logstd.clamp(-5, 2)
        std = logstd.exp()
        z = mu + std * torch.randn_like(mu)
        a = torch.tanh(z)
        log_pi = (logstd + 0.5 * (z ** 2)).sum(-1)
        return a, log_pi
    def act_det(self, s):
        mu, _ = self.net(s).chunk(2, -1)
        return torch.tanh(mu)

class TwinQ(nn.Module):
    def __init__(self, din=42):
        super().__init__()
        self.q1 = mlp(din, 1)
        self.q2 = mlp(din, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x), self.q2(x)

# ============ M7 环境 ============
class WMEnv:
    """M7 当仿真器: 窗口状态 → 动作 → M7预测1步(+扰动) → 回填"""
    def __init__(self, seed=0, dist_amp=0.3):
        d = np.load('results/exp_071_rl_data/train.npz')
        self.sm = torch.FloatTensor(d['s'].mean(0)).to(DEVICE)        # [40] (勿unsqueeze: 广播会变[1,40])
        self.ss = torch.FloatTensor(d['s'].std(0) + 1e-6).to(DEVICE)
        self.am = torch.FloatTensor(TRAIN_RAW[:, M.VALVE_IDX].mean(0)).to(DEVICE)
        self.astd = torch.FloatTensor(TRAIN_RAW[:, M.VALVE_IDX].std(0) + 1e-6).to(DEVICE)
        self.amin = torch.FloatTensor(TRAIN_RAW[:, M.VALVE_IDX].min(0)).to(DEVICE)
        self.amax = torch.FloatTensor(TRAIN_RAW[:, M.VALVE_IDX].max(0)).to(DEVICE)
        self.wm = M.load_wm()
        self.dist_amp = dist_amp
        self.starts = np.random.RandomState(seed).choice(
            range(len(TRAIN_RAW) - M.W - 300), 200, replace=False)
        self.reset()
    def reset(self):
        self.i = int(self.starts[np.random.randint(len(self.starts))])
        self.t = 0
        self.win = torch.FloatTensor(TRAIN_RAW[self.i:self.i + M.W]).unsqueeze(0).to(DEVICE)
        self.rng = np.random.RandomState(np.random.randint(100000))
        self.d_state = 0.0
        self.a_prev = None
        return (self.win[0, -1, :40] - self.sm) / self.ss
    def step(self, a_norm):
        a_raw = a_norm * self.astd + self.am
        a_raw = a_raw.clamp(self.amin, self.amax)
        a_full = a_raw.unsqueeze(0).repeat(M.H_OUT, 1).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.wm(self.win, a_full.reshape(1, -1))
        y = mu[0, 0].item()
        self.d_state = 0.9 * self.d_state + self.rng.normal(0, self.dist_amp)
        y += self.d_state
        gi = self.i + self.t
        nr = torch.FloatTensor(TRAIN_RAW[gi + M.W]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        nr[0, 0, M.TARGET_IDX] = y
        self.win = torch.cat([self.win[:, 1:, :], nr], 1)
        self.t += 1
        done = self.t >= 500
        e = y - TRAIN_RAW[gi + M.W, M.SP_IDX]
        da = float((a_raw - self.a_prev).abs().sum()) if self.a_prev is not None else 0.0
        self.a_prev = a_raw
        r = -abs(e) - 0.5 * da
        s_next = (self.win[0, -1, :40] - self.sm) / self.ss
        return s_next, float(r), done

# ============ SAC 训练 ============
def train(seed=42, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    env = WMEnv(seed)
    actor = Actor().to(DEVICE)
    q = TwinQ().to(DEVICE)
    qt = TwinQ().to(DEVICE); qt.load_state_dict(q.state_dict())
    oa = torch.optim.Adam(actor.parameters(), lr=3e-4)
    oq = torch.optim.Adam(q.parameters(), lr=3e-4)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    oa_alpha = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -2.0
    buf = {'s': [], 'a': [], 'r': [], 's2': [], 'd': []}
    t0 = time.time()
    s = env.reset()
    for it in range(STEPS):
        with torch.no_grad():
            a_norm, _ = actor(s.unsqueeze(0))
        a_norm = a_norm.squeeze(0)
        s2, r, done = env.step(a_norm)
        buf['s'].append(s.cpu()); buf['a'].append(a_norm.cpu())
        buf['r'].append(r); buf['s2'].append(s2.cpu()); buf['d'].append(done)
        s = env.reset() if done else s2
        if len(buf['s']) >= bs:
            idx = np.random.randint(0, len(buf['s']), bs)
            sb = torch.stack([buf['s'][i] for i in idx]).to(DEVICE)
            ab = torch.stack([buf['a'][i] for i in idx]).to(DEVICE)
            rb = torch.FloatTensor([buf['r'][i] for i in idx]).to(DEVICE)
            s2b = torch.stack([buf['s2'][i] for i in idx]).to(DEVICE)
            db = torch.FloatTensor([buf['d'][i] for i in idx]).to(DEVICE)
            with torch.no_grad():
                a2, lp2 = actor(s2b)
                q1t, q2t = qt(s2b, a2)
                qtgt = torch.min(q1t, q2t) - log_alpha.exp() * lp2
                y = rb + 0.99 * (1 - db) * qtgt
            q1, q2 = q(sb, ab)
            lq = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            oq.zero_grad(); lq.backward(); oq.step()
            a2, lp2 = actor(sb)
            q1a, q2a = q(sb, a2)
            qa = torch.min(q1a, q2a)
            la = (log_alpha.exp() * lp2 - qa).mean()
            oa.zero_grad(); la.backward(); oa.step()
            lalpha = -(log_alpha.exp() * (lp2 + target_entropy).detach()).mean()
            oa_alpha.zero_grad(); lalpha.backward(); oa_alpha.step()
            with torch.no_grad():
                for p, pt in zip(q.parameters(), qt.parameters()):
                    pt.data.mul_(0.995).add_(0.005 * p.data)
        if it > 0 and (it % 20000 == 0 or it == STEPS - 1):
            print(f"  SAC seed{seed} it{it} | Q {lq.item():.3f} | α {log_alpha.exp().item():.3f} | "
                  f"r 均值 {np.mean(buf['r'][-1000:]):.3f} | lenbuf {len(buf['s'])} | {time.time()-t0:.0f}s", flush=True)
    torch.save({'actor': actor.state_dict()}, f'{OUT}/checkpoints/sac_seed{seed}.pth')
    print(f"  SAC seed{seed} done ({time.time()-t0:.0f}s)", flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    train(args.seed)
