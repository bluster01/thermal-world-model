#!/usr/bin/env python3
"""
exp_079_aug_train.py — 方案2: 动作扰动增强训练 (M7aug) + 因果一致性验证
========================================================================
训练时对 a_fut 注入时域扰动 (首步脉冲/随机起点阶跃, ±2-10%), 目标温度不变:
  验证"扰动增强能否修复模型长程动作响应方向翻转" (exp_078 发现: 持续阶跃 t12 翻转 +0.65°C)
  朴素版预期: 目标-动作不匹配 → 模型稀释动作通道 (方向不改善) — 实证检验
训练完自动跑响应验证 + 画对比图 (M7 vs M7aug)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv
import config as cfg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv
EPOCHS = 3 if SMOKE else 60
PATIENCE = 6 if SMOKE else 10
AUG_AMP_MAX = float(os.environ.get('AUG_AMP_MAX', 10.0))   # 扰动幅度上限 (%)
EXP_DIR = 'results/exp_025_M7aug'
os.makedirs(f'{EXP_DIR}/checkpoints', exist_ok=True)

W = cfg.WINDOW_SIZE; H = E.H_OUT; BS = 64

def train_epoch_aug(model, raw, opt, crit):
    """train_epoch + 动作时域扰动增强 (目标不变)"""
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
        # === 动作扰动增强: 随机模式 ===
        aug = torch.rand(BS, device=DEVICE) < 0.5        # 50% 样本扰动
        if aug.any():
            delta = (torch.rand(BS, 2, device=DEVICE) * 2 - 1) * AUG_AMP_MAX  # ±AUG_AMP_MAX
            a_aug = a_fut.clone()
            # 模式A: 首步脉冲 (60% of perturbed)
            first = aug & (torch.rand(BS, device=DEVICE) < 0.6)
            if first.any():
                a_aug[first, 0, :] = (a_aug[first, 0, :] + delta[first]).clamp(0, 100)
            # 模式B: 随机起点持续阶跃 (40%)
            step = aug & ~first
            if step.any():
                k = torch.randint(1, H, (int(step.sum()),), device=DEVICE)
                for j, kk in enumerate(k):
                    a_aug[step.clone().nonzero()[j], kk:, :] = \
                        (a_aug[step.clone().nonzero()[j], kk:, :] + delta[step.clone().nonzero()[j]]).clamp(0, 100)
            a_fut = a_aug
        opt.zero_grad()
        mu, lv = model(x_hist, a_fut)
        w = torch.linspace(1.0, 0.6, H, device=DEVICE)
        loss = (w * crit(mu, lv, t_true).mean(dim=0)).sum() / H
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); total += loss.item()
    return total / E.STEPS

# ============ 因果一致性验证 (两种扰动模式) ============
def check_causal(model, name):
    model.eval()
    np.random.seed(7)
    idxs = np.random.choice(range(len(E.test_raw) - W - H), 50, replace=False)
    dT_first = {s: [] for s in [1, 3, 8, 12]}
    dT_step = {s: [] for s in [1, 3, 8, 12]}
    for i in idxs:
        x_hist = torch.FloatTensor(E.test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(E.test_raw[i+W:i+W+H, E.VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu_b, _ = model(x_hist, a_fut)
        bp = mu_b[0].cpu().numpy()
        a1 = a_fut.clone(); a1[0, 0, 1] = torch.clamp(a1[0, 0, 1] + 10, 0, 100)
        with torch.no_grad():
            mu1, _ = model(x_hist, a1)
        pp1 = mu1[0].cpu().numpy()
        a2 = a_fut.clone(); a2[0, 10:, 1] = torch.clamp(a2[0, 10:, 1] + 10, 0, 100)
        with torch.no_grad():
            mu2, _ = model(x_hist, a2)
        pp2 = mu2[0].cpu().numpy()
        for s in [1, 3, 8, 12]:
            dT_first[s].append(pp1[s] - bp[s]); dT_step[s].append(pp2[s] - bp[s])
    return {s: float(np.mean(dT_first[s])) for s in [1, 3, 8, 12]}, \
           {s: float(np.mean(dT_step[s])) for s in [1, 3, 8, 12]}

if __name__ == '__main__':
    model = E.build_model('M7').to(DEVICE)
    crit = E.BetaNLLLoss(beta=-0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    best_m, pc, be = float('inf'), 0, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        crit.beta = -0.3
        nll = train_epoch_aug(model, E.train_raw, opt, crit)
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
    print(f"  M7aug 训练完成 best@{be} ({time.time()-t0:.0f}s)")
    mae = E.eval_rollout(model, E.test_raw, True)
    print(f"  Rollout(test, °C): {mae[0]:.4f}→{mae[-1]:.4f} avg={mae.mean():.4f}")

    # 验证 + 画图
    f_m7, s_m7 = check_causal(E.load_wm() if hasattr(E, 'load_wm') else None, 'M7') if False else (None, None)
    # 加载原 M7
    m7 = E.build_model('M7').to(DEVICE).eval()
    ck7 = torch.load('results/exp_025_M7/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
    m7.load_state_dict(ck7['model_state_dict'])
    f7, s7 = check_causal(m7, 'M7')
    fa, sa = check_causal(model, 'M7aug')
    print(f"\n  首步扰动 t1/t3/t8/t12:  M7   {[f'{f7[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"                            M7aug {[f'{fa[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"  持续阶跃 t1/t3/t8/t12:  M7   {[f'{s7[s]:+.3f}' for s in [1,3,8,12]]}")
    print(f"                            M7aug {[f'{sa[s]:+.3f}' for s in [1,3,8,12]]}")
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ts = [1, 3, 8, 12]
    axes[0].plot(ts, [f7[s] for s in ts], 'o-', color='#c0504d', label='M7 (orig)')
    axes[0].plot(ts, [fa[s] for s in ts], 's--', color='#4f81bd', label='M7aug (perturb-train)')
    axes[0].axhline(0, color='gray', lw=0.7); axes[0].set_title('(a) First-step pulse (+10% V2)')
    axes[0].set_xlabel('Prediction step t (×10s)'); axes[0].set_ylabel('ΔT (°C)'); axes[0].legend()
    axes[1].plot(ts, [s7[s] for s in ts], 'o-', color='#c0504d', label='M7 (orig)')
    axes[1].plot(ts, [sa[s] for s in ts], 's--', color='#4f81bd', label='M7aug (perturb-train)')
    axes[1].axhline(0, color='gray', lw=0.7)
    axes[1].set_title('(b) Sustained step from t=10 (+10% V2) — physical sign should be negative')
    axes[1].set_xlabel('Prediction step t (×10s)'); axes[1].legend()
    fig.suptitle('Action→temperature response: original M7 vs perturb-train M7aug', fontsize=10.5)
    fig.tight_layout()
    fig.savefig('figures/fig_aug_effect.png', dpi=180, bbox_inches='tight')
    print('Saved: figures/fig_aug_effect.png')
