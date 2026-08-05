#!/usr/bin/env python3
"""debug: 定位 M7-DSP H=60 训练 nan 源头 (detect_anomaly)"""
import os, sys
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

torch.autograd.set_detect_anomaly(True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E.cfg.WINDOW_SIZE
H = 60
E.H_OUT = H
n_train = 495407

raw = E.data_all
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40
train_raw = raw41[:n_train]

class SafeBetaNLL(E.BetaNLLLoss):
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -6., 20.)
        v = torch.exp(lv) + 1e-4
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0:
            nll = v.detach()**self.beta * nll
        return nll.mean()

class M7DSP(E.DirectWM):
    def __init__(self):
        super().__init__(use_action=True, use_patch=True, per_variable=True,
                         use_varattn=True, beta_mode='fixed')
        d = E.cfg.D_MODEL
        self.action_enc = nn.Sequential(
            nn.Linear(H, d * 2), nn.GELU(), nn.Dropout(E.cfg.DROPOUT))

model = M7DSP().to(DEVICE)
crit = SafeBetaNLL(beta=-0.3)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
BS, STEPS = 256, 500

for ep in range(1, 9):
    model.train()
    tot = 0.0
    for bi in range(STEPS):
        idxs = np.random.randint(0, len(train_raw) - W - H, size=BS)
        X = [train_raw[i:i+W, :40] for i in idxs]
        A = [train_raw[i+W:i+W+H, I_DSP] for i in idxs]
        Y = [train_raw[i+W:i+W+H, E.TARGET_IDX] for i in idxs]
        x = torch.FloatTensor(np.stack(X)).to(DEVICE)
        a = torch.FloatTensor(np.stack(A)).unsqueeze(-1).to(DEVICE)
        y = torch.FloatTensor(np.stack(Y)).to(DEVICE)
        try:
            mu, lv = model(x, a)
            w = torch.linspace(1.0, 0.6, H, device=DEVICE)
            loss = (w * crit(mu, lv, y).mean(dim=0)).sum() / H
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            tot += loss.item()
            if not np.isfinite(loss.item()):
                print(f"ep{ep} batch{bi} loss={loss.item()} (nan/inf 出现在 loss, 未抛异常)")
                sys.exit(1)
        except RuntimeError as e:
            print(f"ep{ep} batch{bi} RuntimeError: {e}")
            # 打印 mu/lv 统计
            print(f"  mu: nan={torch.isnan(mu).sum().item()} inf={torch.isinf(mu).sum().item()} | "
                  f"lv: nan={torch.isnan(lv).sum().item()} inf={torch.isinf(lv).sum().item()} | "
                  f"mu范围 [{mu.min().item():.2f}, {mu.max().item():.2f}] lv范围 [{lv.min().item():.2f}, {lv.max().item():.2f}]")
            sys.exit(1)
    print(f"ep {ep} loss {tot/STEPS:.4f}")
