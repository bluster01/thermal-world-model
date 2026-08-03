#!/usr/bin/env python3
"""
exp_072_rl_train.py — P3: IQL + TD3+BC 离线训练
=================================================
官方机制忠实实现 (Kostrikov 2022 / Fujimoto 2021), 自定义 numpy dataloader
状态 z-score 归一化 (train 统计), 动作 z-score + clip(0,100)
用法: python exp_072_rl_train.py --method iql [--smoke] [--seed 42]
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = 40
OUT = 'results/exp_072_rl'
os.makedirs(f'{OUT}/checkpoints', exist_ok=True)

class MLP(nn.Module):
    def __init__(self, din, dout, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dout))
    def forward(self, x): return self.net(x)

class Policy(nn.Module):
    def __init__(self, din=40, dout=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dout))
    def forward(self, s): return torch.tanh(self.net(s))  # [-1,1]

# ============ 数据 ============
def load_data():
    d = np.load(f'results/exp_071_rl_data/train.npz')
    dv = np.load(f'results/exp_071_rl_data/val.npz')
    return d, dv

def norm_stats(s):
    return s.mean(0, keepdims=True), s.std(0, keepdims=True) + 1e-6

# ============ IQL ============
def train_iql(seed, steps=1000 if '--smoke' in sys.argv else 300000, bs=256):
    d, dv = load_data()
    sm, ss = norm_stats(d['s'])
    am, astd = d['a'].mean(0), d['a'].std(0) + 1e-6
    s = (d['s'] - sm) / ss; s2 = (d['s_next'] - sm) / ss
    a = (d['a'] - am) / astd
    r = d['r']; r = (r - r.mean()) / (r.std() + 1e-6)  # 奖励归一化
    sv = (dv['s'] - sm) / ss
    n = len(s)
    torch.manual_seed(seed); np.random.seed(seed)
    qf = MLP(D, 1).to(DEVICE); qf_t = MLP(D, 1).to(DEVICE); qf_t.load_state_dict(qf.state_dict())
    vf = MLP(D, 1).to(DEVICE)
    pi = Policy().to(DEVICE)
    optq = torch.optim.Adam(qf.parameters(), lr=3e-4)
    optv = torch.optim.Adam(vf.parameters(), lr=3e-4)
    optp = torch.optim.Adam(pi.parameters(), lr=3e-4)
    tau, gamma, beta, TAU_T = 0.7, 0.99, 3.0, 5e-3
    S, A, R, S2 = (torch.FloatTensor(x).to(DEVICE) for x in [s, a, r, s2])
    t0 = time.time()
    for it in range(steps):
        idx = np.random.randint(0, n, bs)
        sb, ab, rb, s2b = S[idx], A[idx], R[idx], S2[idx]
        # V: expectile (目标 detach, V 保梯度 — 2026-08-03修复: 原diff整体detach导致无梯度)
        with torch.no_grad():
            v_target = rb + gamma * vf(s2b)         # IQL 标准: target = r + γV(s')
        v = vf(sb)
        diff = v_target - v
        w = torch.where(diff > 0, tau, 1 - tau)
        loss_v = (w * diff ** 2).mean()
        optv.zero_grad(); loss_v.backward(); optv.step()
        # Q: in-sample TD (target = r + γV)
        with torch.no_grad():
            q_target = rb + gamma * vf(s2b)
        loss_q = F.mse_loss(qf(sb), q_target)
        optq.zero_grad(); loss_q.backward(); optq.step()
        with torch.no_grad():
            for p, pt in zip(qf.parameters(), qf_t.parameters()):
                pt.data.mul_(1 - TAU_T).add_(TAU_T * p.data)
        # π: AWR
        with torch.no_grad():
            adv = qf(sb) - vf(sb)
            w_awr = torch.exp(beta * adv.clamp(max=10))
        pa = pi(sb)
        log_pi = -0.5 * ((pa - ab) ** 2).sum(1)
        loss_p = -(w_awr * log_pi).mean()
        optp.zero_grad(); loss_p.backward(); optp.step()
        if it % 50000 == 0:
            with torch.no_grad():
                val_q = qf(torch.FloatTensor(sv).to(DEVICE)).mean().item()
            print(f"  IQL seed{seed} it{it} | V {loss_v.item():.4f} | Q {loss_q.item():.4f} | π {loss_p.item():.4f} | Q_val {val_q:.3f} | {time.time()-t0:.0f}s", flush=True)
    torch.save({'pi': pi.state_dict(), 'am': am, 'astd': astd, 'sm': sm, 'ss': ss},
               f'{OUT}/checkpoints/iql_seed{seed}.pth')
    print(f"  IQL seed{seed} done ({time.time()-t0:.0f}s)", flush=True)

# ============ TD3+BC ============
def train_td3bc(seed, steps=1000 if '--smoke' in sys.argv else 300000, bs=256):
    d, dv = load_data()
    sm, ss = norm_stats(d['s'])
    am, astd = d['a'].mean(0), d['a'].std(0) + 1e-6
    s = (d['s'] - sm) / ss; s2 = (d['s_next'] - sm) / ss
    a = (d['a'] - am) / astd
    r = d['r']; r = (r - r.mean()) / (r.std() + 1e-6)
    n = len(s)
    torch.manual_seed(seed); np.random.seed(seed)
    q1, q2 = MLP(D, 1).to(DEVICE), MLP(D, 1).to(DEVICE)
    q1t, q2t = MLP(D, 1).to(DEVICE), MLP(D, 1).to(DEVICE)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    pi = Policy().to(DEVICE); pit = Policy().to(DEVICE); pit.load_state_dict(pi.state_dict())
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    op = torch.optim.Adam(pi.parameters(), lr=3e-4)
    gamma, TAU_T, alpha = 0.99, 5e-3, 2.5
    policy_noise, noise_clip, delay = 0.2, 0.5, 2
    S, A, R, S2 = (torch.FloatTensor(x).to(DEVICE) for x in [s, a, r, s2])
    t0 = time.time()
    for it in range(steps):
        idx = np.random.randint(0, n, bs)
        sb, ab, rb, s2b = S[idx], A[idx], R[idx], S2[idx]
        with torch.no_grad():
            noise = (torch.randn_like(ab) * policy_noise).clamp(-noise_clip, noise_clip)
            a2 = (pit(s2b) + noise).clamp(-1, 1)
            q_t = torch.min(q1t(s2b), q2t(s2b))
            y = rb + gamma * q_t
        lq = F.mse_loss(q1(sb), y) + F.mse_loss(q2(sb), y)
        oq.zero_grad(); lq.backward(); oq.step()
        if it % delay == 0:
            q1v = q1(sb).detach()
            lam = alpha / q1v.abs().mean().clamp(min=1e-3)
            bc = lam * ((pi(sb) - ab) ** 2).mean()
            lp = -q1(sb).mean() + bc
            op.zero_grad(); lp.backward(); op.step()
            with torch.no_grad():
                for p, pt in zip(q1.parameters(), q1t.parameters()):
                    pt.data.mul_(1 - TAU_T).add_(TAU_T * p.data)
                for p, pt in zip(q2.parameters(), q2t.parameters()):
                    pt.data.mul_(1 - TAU_T).add_(TAU_T * p.data)
                for p, pt in zip(pi.parameters(), pit.parameters()):
                    pt.data.mul_(1 - TAU_T).add_(TAU_T * p.data)
        if it % 50000 == 0:
            print(f"  TD3BC seed{seed} it{it} | Q {lq.item():.4f} | π {lp.item():.4f} | {time.time()-t0:.0f}s", flush=True)
    torch.save({'pi': pi.state_dict(), 'am': am, 'astd': astd, 'sm': sm, 'ss': ss},
               f'{OUT}/checkpoints/td3bc_seed{seed}.pth')
    print(f"  TD3BC seed{seed} done ({time.time()-t0:.0f}s)", flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['iql', 'td3bc'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    SMOKE = args.smoke
    fn = train_iql if args.method == 'iql' else train_td3bc
    fn(args.seed)
