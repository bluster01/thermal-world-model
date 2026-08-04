#!/usr/bin/env python3
"""
exp_083_m13_phys.py — 组合: M13 DeepONet 架构 + 配对扰动物理正则
================================================================
M13 (DeepONet 算子, 首步响应强 −0.78) + M7phys 配对正则 (持续阶跃方向修复)
期望: 全程负 + 幅度大 + 精度好 — 最优组合
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
_argv = sys.argv
sys.argv = ['exp_082_deeponet_wm.py']
from experiments.phase2_mpc.exp_082_deeponet_wm import DeepONetWM, check_causal
sys.argv = _argv
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
import config as cfg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
EPOCHS = 3 if SMOKE else 60
PATIENCE = 6 if SMOKE else 10
LAMBDA_H = float(os.environ.get('LAMBDA_H', '1.0'))
EXP_DIR = f'results/exp_025_M13phys_l{LAMBDA_H}'
os.makedirs(f'{EXP_DIR}/checkpoints', exist_ok=True)
W = cfg.WINDOW_SIZE; H = E.H_OUT; BS = 64

def train_epoch_phys13(model, raw, opt, crit):
    """M13 训练 + 配对扰动正则 (同 exp_081)"""
    model.train(); N = len(raw)
    total = 0.
    for _ in range(E.STEPS):
        idxs = np.random.randint(0, N - W - H, size=BS)
        xh, af, tt = [], [], []
        for i in idxs:
            xh.append(raw[i:i+W]); af.append(raw[i+W:i+W+H, E.VALVE_IDX])
            tt.append(raw[i+W:i+W+H, E.TARGET_IDX])
        x_hist = torch.FloatTensor(np.stack(xh)).to(DEVICE)
        a_fut = torch.FloatTensor(np.stack(af)).to(DEVICE)
        t_true = torch.FloatTensor(np.stack(tt)).to(DEVICE)
        aug = torch.rand(BS, device=DEVICE) < 0.5
        a_pos, a_neg = a_fut.clone(), a_fut.clone()
        if aug.any():
            first = aug & (torch.rand(BS, device=DEVICE) < 0.6)
            step = aug & ~first
            if first.any():
                a_pos[first, 0, 1] = (a_pos[first, 0, 1] + 5).clamp(0, 100)
                a_neg[first, 0, 1] = (a_neg[first, 0, 1] - 5).clamp(0, 100)
            if step.any():
                for j in step.nonzero()[:, 0]:
                    k = int(torch.randint(1, H, (1,)))
                    a_pos[j, k:, 1] = (a_pos[j, k:, 1] + 5).clamp(0, 100)
                    a_neg[j, k:, 1] = (a_neg[j, k:, 1] - 5).clamp(0, 100)
        opt.zero_grad()
        mu_aug, lv_aug = model(x_hist, a_pos)
        w = torch.linspace(1.0, 0.6, H, device=DEVICE)
        loss = (w * crit(mu_aug, lv_aug, t_true).mean(dim=0)).sum() / H
        if aug.any():
            mu_neg, _ = model(x_hist, a_neg)
            hinge = torch.relu(mu_aug[aug] - mu_neg[aug]).mean()
            loss = loss + LAMBDA_H * hinge
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); total += loss.item()
    return total / E.STEPS

if __name__ == '__main__':
    model = DeepONetWM().to(DEVICE)
    crit = E.BetaNLLLoss(beta=-0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    best_m, pc, be = float('inf'), 0, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        crit.beta = -0.3
        nll = train_epoch_phys13(model, E.train_raw, opt, crit)
        v0, v4 = E.validate(model, E.val_raw, True); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{nll:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}", flush=True)
        if v4 < best_m - 0.001:
            best_m, be, pc = v4, ep, 0
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict()},
                       f'{EXP_DIR}/checkpoints/best_model.pth')
        else:
            pc += 1
        if pc >= PATIENCE:
            print(f"  Stop@{ep} best@{be}"); break
    ck = torch.load(f'{EXP_DIR}/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    mae = E.eval_rollout(model, E.test_raw, True)
    print(f"  M13phys(λ={LAMBDA_H}) 训练完成 best@{be} ({time.time()-t0:.0f}s) | Rollout avg={mae.mean():.4f}")

    fp, sp = check_causal(model)
    m7 = E.build_model('M7').to(DEVICE).eval()
    ck7 = torch.load('results/exp_025_M7/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
    m7.load_state_dict(ck7['model_state_dict'])
    f7, s7 = check_causal(m7)
    print(f"\n  首步扰动 t1/t3/t8/t12:  M7      {[f'{f7[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"                           M13phys {[f'{fp[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"  持续阶跃 t1/t3/t8/t12:  M7      {[f'{s7[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"                           M13phys {[f'{sp[s]:+.3f}' for s in [1,3,8,12]]}")
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ts = [1, 3, 8, 12]
    axes[0].plot(ts, [f7[s] for s in ts], 'o-', color='#c0504d', label='M7')
    axes[0].plot(ts, [fp[s] for s in ts], 's--', color='#4f81bd', label=f'M13phys λ={LAMBDA_H}')
    axes[0].axhline(0, color='gray', lw=0.7); axes[0].set_title('(a) First-step pulse (+10% V2)')
    axes[0].set_xlabel('Prediction step t (×10s)'); axes[0].set_ylabel('ΔT (°C)'); axes[0].legend()
    axes[1].plot(ts, [s7[s] for s in ts], 'o-', color='#c0504d', label='M7')
    axes[1].plot(ts, [sp[s] for s in ts], 's--', color='#4f81bd', label=f'M13phys λ={LAMBDA_H}')
    axes[1].axhline(0, color='gray', lw=0.7)
    axes[1].set_title('(b) Sustained step from t=10 (+10% V2) — physical sign: negative')
    axes[1].set_xlabel('Prediction step t (×10s)'); axes[1].legend()
    fig.suptitle(f'DeepONet + paired phys-regularization (λ={LAMBDA_H})', fontsize=10.5)
    fig.tight_layout()
    fig.savefig('figures/fig_m13phys_effect.png', dpi=180, bbox_inches='tight')
    print('Saved: figures/fig_m13phys_effect.png')
