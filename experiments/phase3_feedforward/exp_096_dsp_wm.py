#!/usr/bin/env python3
"""
exp_096_dsp_wm.py — 训练 ΔSP 动作世界模型 (2026-08-04)
=========================================================
动作通道从 2 维阀位 Δ 改为 1 维 ΔSP (SP_t − SP_{t-1}), 使 WM 学"ΔSP→温度"因果过渡。
M5 结构复用: DirectWM(18步, prob=False, action_dim=1→act_in=H_OUT)
状态含绝对 SP (累积回填), 动作 = 未来 ΔSP 序列 (现场部署对称: 路线型监督模式)
训练 ~30min (800K 样本, MSE loss), 同 exp_025 pipeline
用法: python exp_096_dsp_wm.py [--smoke]
"""
import os, sys, time
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import experiments.phase1_dynamics.exp_025_unified_benchmark as E

SMOKE = '--smoke' in sys.argv
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
H_OUT = E.H_OUT                              # 18 (模块级)
W = E.cfg.WINDOW_SIZE                        # 96

# ===== 数据: 构造 ΔSP 列 =====
raw = E.data_all
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])  # ΔSP
raw_dsp = np.concatenate([E.data_all, dsp[:, None]], 1)
I_DSP = raw_dsp.shape[1] - 1
n_train, n_val_end = 495407, 601566  # 同 exp_025: tr 495407, val 106159, test 106159
train_raw = raw_dsp[:n_train]; test_raw = raw_dsp[n_val_end:]
del raw_dsp, dsp
print(f"[data] train {train_raw.shape} test {test_raw.shape}")

# ===== M5-DSP 模型 (子类: action_enc 1维) =====
class M5DSP(E.DirectWM):
    """M5 确定性 WM, 动作 = ΔSP (1维, act_in=H_OUT)"""
    def __init__(self):
        super().__init__(use_action=True, use_patch=True, per_variable=True, use_varattn=True, probabilistic=False)
        d = E.cfg.D_MODEL
        self.action_enc = nn.Sequential(
            nn.Linear(H_OUT, d * 2), nn.GELU(), nn.Dropout(E.cfg.DROPOUT))

model = M5DSP().to(DEVICE).to(DTYPE)
print(f"[model] M5-DSP: action_enc input {H_OUT}")

# ===== 训练参数 =====
LR = 1e-3; WD = 1e-5; BS = 256
NEPOCH = 4 if SMOKE else 100
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
N_TRAIN = len(train_raw)

t0 = time.time()
for ep in range(NEPOCH):
    model.train()
    idxs = np.random.permutation(N_TRAIN - W - H_OUT)
    losses, n = 0.0, 0
    for bi in range(0, len(idxs), BS):
        bid = idxs[bi:bi+BS]
        X = [train_raw[i:i+W, :40] for i in bid]       # 40列状态 (不含ΔSP)
        A = [train_raw[i+W:i+W+H_OUT, I_DSP] for i in bid]  # ΔSP [H], 动作
        Y = [train_raw[i+W:i+W+H_OUT, E.TARGET_IDX] for i in bid]
        x = torch.FloatTensor(np.stack(X)).to(DEVICE)       # [BS, W, 40]
        a = torch.FloatTensor(np.stack(A)).unsqueeze(-1).to(DEVICE)  # [BS, H, 1]
        y = torch.FloatTensor(np.stack(Y)).to(DEVICE)       # [BS, H]
        mu, _ = model(x, a)                                 # [BS, H]
        loss = nn.functional.mse_loss(mu, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses += loss.item() * len(bid); n += len(bid)
    # test
    model.eval()
    te_errs = []
    with torch.no_grad():
        te_idxs = np.arange(len(test_raw) - W - H_OUT)
        np.random.shuffle(te_idxs)
        for i in te_idxs[:200]:
            xh = torch.FloatTensor(test_raw[i:i+W, :40]).unsqueeze(0).to(DEVICE)
            af = torch.FloatTensor(test_raw[i+W:i+W+H_OUT, I_DSP]).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            mu, _ = model(xh, af)
            tt = test_raw[i+W:i+W+H_OUT, E.TARGET_IDX]
            te_errs.append(np.abs(mu[0].cpu().numpy() - tt).mean())
    print(f"  ep {ep+1:3d} | train loss {losses/n:.4f} | test MAE {np.mean(te_errs):.4f}°C | {time.time()-t0:.0f}s")

os.makedirs('results/exp_096_dsp_wm/checkpoints', exist_ok=True)
torch.save({'model_state_dict': model.state_dict(), 'epoch': NEPOCH},
           'results/exp_096_dsp_wm/checkpoints/best_model.pth')
print(f"\n[done] saved results/exp_096_dsp_wm/checkpoints/best_model.pth")
