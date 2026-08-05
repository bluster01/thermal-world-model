#!/usr/bin/env python3
"""
exp_100_m7dsp_h60.py — M7 概率架构 + ΔSP 动作 + H=60 (600s) 世界模型 (2026-08-05)
==============================================================================
动机: SP 跟随时标 600s (97% 响应, exp_099_follow_timescale), 预测窗口 18 步 (180s)
只覆盖 17% 响应 → 加长到 H=60 (600s), 用 phase1 最优概率模型 M7 架构
(BetaNLL β=-0.3 fixed, VarAttn+PerVarTCN+RevIN), 动作通道换 1维 ΔSP (M7-DSP)。
协议: 同 exp_025 M7 (BS 256, STEPS 500/ep, AdamW LR 1e-3 WD 1e-5, w=linspace(1,0.6,H),
      ReduceLROnPlateau), E.H_OUT 模块级 patch 到 60。
评测 (训练后自动):
  1. 134 SP 阶跃事件 60 步 MAE/方向/分层 (0-18 步段与 H=18 可比)
  2. 动作增益测试: pred(real ΔSP) vs pred(ΔSP=0) — 600s 末点响应 + 时间剖面
     物理基准: 600s 97% → 2.0°C@ΔSP2.07; 判定 响应≥0.6°C(30%) 且方向≥80%
用法: python exp_100_m7dsp_h60.py [--smoke]
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

import causal_eval as CE

SMOKE = '--smoke' in sys.argv
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E.cfg.WINDOW_SIZE
H = 60                                  # 预测长度 600s
E.H_OUT = H                             # 模块级 patch: DirectWM 构造时用 H=60
n_train, n_val_end = 495407, 601566

raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_V1 = E.NUMERIC_COLS.index('一级减温调节门阀位')
I_V2 = E.NUMERIC_COLS.index('二级减温调节门阀位')
I_LD = E.NUMERIC_COLS.index('机组负荷')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40
train_raw = raw41[:n_train]; test_raw = raw41[n_val_end:]
print(f"[data] train {train_raw.shape} test {test_raw.shape} | H={H}")

# ===== 数值安全的 BetaNLL (H=60 修复: σ 下限保护防 β<0 膨胀 nan) =====
class SafeBetaNLL(E.BetaNLLLoss):
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -6., 20.)          # σ ≥ e^-3 ≈ 0.05 (归一化空间), 防极小σ梯度爆炸
        v = torch.exp(lv) + 1e-4
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0:
            nll = v.detach()**self.beta * nll
        return nll.mean()

# ===== M7-DSP: M7 概率架构 + ΔSP 动作 =====
class M7DSP(E.DirectWM):
    def __init__(self):
        super().__init__(use_action=True, use_patch=True, per_variable=True,
                         use_varattn=True, beta_mode='fixed')   # 概率默认 True, β 见训练段
        d = E.cfg.D_MODEL
        self.action_enc = nn.Sequential(
            nn.Linear(H, d * 2), nn.GELU(), nn.Dropout(E.cfg.DROPOUT))

    def forward(self, x_hist, a_future=None):
        mu, lv = super().forward(x_hist, a_future)
        if lv is not None:
            # denorm 内 exp(lv/2) 数值保护: 未 clamp 的 lv 上溢(>88) → inf → 梯度 nan
            # (H=60 下 β<0 膨胀正反馈早期即剧烈, debug_nan_h60 定位 ep2 batch453)
            lv = torch.clamp(lv, -6., 20.)
        return mu, lv

model = M7DSP().to(DEVICE)
n_param = sum(p.numel() for p in model.parameters())
# β=-0.3 (H=18 定案) 在 H=60 下膨胀正反馈数值不稳定 (exp lv 上溢 nan, debug_nan_h60 定位)
# → β=0 (标准高斯 NLL): 保留概率架构, 去掉 σ 加权正则; σ 头由 NLL 天然平衡
BETA = 0.0
print(f"[model] M7-DSP H={H}: {n_param/1e6:.2f}M params (probabilistic, beta={BETA})")

# ===== 训练 (同 exp_025 M7 协议, β 适配) =====
BS, STEPS = 256, 500
crit = SafeBetaNLL(beta=BETA)
opt = torch.optim.AdamW(model.parameters(), lr=E.cfg.LEARNING_RATE, weight_decay=E.cfg.WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
NEPOCH = 4 if SMOKE else E.cfg.EPOCHS
t0 = time.time()
best_mae, best_ep, curve = None, None, []
for ep in range(1, NEPOCH + 1):
    model.train()
    losses, n = 0.0, 0
    for _ in range(STEPS):
        idxs = np.random.randint(0, len(train_raw) - W - H, size=BS)
        X = [train_raw[i:i+W, :40] for i in idxs]
        A = [train_raw[i+W:i+W+H, I_DSP] for i in idxs]
        Y = [train_raw[i+W:i+W+H, E.TARGET_IDX] for i in idxs]
        x = torch.FloatTensor(np.stack(X)).to(DEVICE)
        a = torch.FloatTensor(np.stack(A)).unsqueeze(-1).to(DEVICE)
        y = torch.FloatTensor(np.stack(Y)).to(DEVICE)
        mu, lv = model(x, a)
        w = torch.linspace(1.0, 0.6, H, device=DEVICE)
        loss = (w * crit(mu, lv, y).mean(dim=0)).sum() / H
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        losses += loss.item(); n += 1
    # test (200 随机窗口, 60 步 MAE)
    model.eval()
    te_errs = []
    with torch.no_grad():
        te_idxs = np.arange(len(test_raw) - W - H)
        np.random.shuffle(te_idxs)
        for i in te_idxs[:200]:
            xh = torch.FloatTensor(test_raw[i:i+W, :40]).unsqueeze(0).to(DEVICE)
            af = torch.FloatTensor(test_raw[i+W:i+W+H, I_DSP]).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            mu, _ = model(xh, af)
            tt = test_raw[i+W:i+W+H, E.TARGET_IDX]
            te_errs.append(np.abs(mu[0].cpu().numpy() - tt).mean())
    te_mae = float(np.mean(te_errs))
    print(f"  ep {ep:3d} | loss {losses/n:.4f} | test MAE(60步) {te_mae:.4f}°C | {time.time()-t0:.0f}s")
    if ep % 10 == 0:
        model.eval()
        with torch.no_grad():
            lvs = []
            for i in te_idxs[:20]:
                xh = torch.FloatTensor(test_raw[i:i+W, :40]).unsqueeze(0).to(DEVICE)
                af = torch.FloatTensor(test_raw[i+W:i+W+H, I_DSP]).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                _, lv = model(xh, af)
                lvs.append(lv[0].cpu().numpy())
            lvs = np.concatenate(lvs)
            print(f"         [lv] mean {lvs.mean():+.2f} min {lvs.min():+.2f} max {lvs.max():+.2f} (σ {np.exp(np.clip(lvs, -6, 20)/2).mean():.3f})")
    sched.step(te_mae)
    if best_mae is None or te_mae < best_mae:
        best_mae, best_ep = te_mae, ep
        os.makedirs('results/exp_100_m7dsp_h60/checkpoints', exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                   'results/exp_100_m7dsp_h60/checkpoints/best_model.pth')
    curve.append((ep, losses / n, te_mae))

with open('results/exp_100_m7dsp_h60/metrics.json', 'w') as f:
    json.dump({'best_epoch': best_ep, 'best_test_mae': best_mae, 'curve': curve, 'H': H}, f, indent=2)
print(f"\n[done] best ep {best_ep} MAE {best_mae:.4f}°C")

# ===== 评测 1: 134 事件 60 步 MAE/方向/分层 =====
ck = torch.load('results/exp_100_m7dsp_h60/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
model.load_state_dict(ck['model_state_dict']); model.eval()

dsp_abs = np.abs(np.diff(raw[:, I_SP]))
onsets = []
for i in np.where(dsp_abs > 1.0)[0] + 1:
    if not onsets or i - onsets[-1] >= 60:
        onsets.append(i)
stable = [o for o in onsets if o + 60 < N and
          np.abs(np.diff(raw[max(0, o-20):min(N, o+20), I_LD])).max() <= 3.0]
kept = [o for o in stable if np.abs(raw[o:o+61, I_SP] - raw[o, I_SP]).max() <= 0.3]
print(f"[events] {len(kept)}")

def pred(s, a_override=None):
    if s < 0 or s + W + H >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    if a_override is None:
        a = CE.build_action(raw41, s, W, H, I_DSP)
    else:
        a = a_override
    a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        mu, _ = model(win, a_f)
    return mu[0].cpu().numpy()

errs_all, dirs_all = [], []
errs_18 = []
for o in kept:
    s = o - W
    p = pred(s)
    if p is None: continue
    actual = raw[o:o+H, I_T]
    prev_T = raw[o-1, I_T]
    errs_all.append(np.abs(p - actual).mean())
    errs_18.append(np.abs(p[:18] - actual[:18]).mean())          # 0-180s 段 (与 H=18 可比)
    dirs_all.append(1 if np.sign(p[-1]-prev_T) == np.sign(actual[-1]-prev_T) else 0)
errs_all, dirs_all, errs_18 = map(np.array, (errs_all, dirs_all, errs_18))
print(f"\n===== SP 事件 60 步预测 (n={len(errs_all)}) =====")
print(f"  全程60步 MAE {errs_all.mean():.3f} | 前18步(180s) MAE {errs_18.mean():.3f} (H=18参照: 0.301) | 方向(600s) {dirs_all.mean()*100:.0f}%")

# ===== 评测 2: 动作增益测试 (real vs zero) =====
resp = []
for o in kept:
    s = o - W
    p_real = pred(s)
    p_zero = pred(s, np.zeros(H))
    if p_real is None or p_zero is None: continue
    resp.append((p_real - p_zero, raw[o, I_SP] - raw[o-1, I_SP]))
resp_dT = np.array([r[0] for r in resp])          # [n, 60] 响应轨迹
dsp_v = np.array([r[1] for r in resp])
print(f"\n===== 动作增益测试 (real ΔSP vs ΔSP=0, n={len(resp)}) =====")
print(f"  物理基准: 600s 响应比例 97% → ΔSP2.07 期望响应 ~2.0°C")
for k, lab in ((5, '60s'), (11, '120s'), (17, '180s'), (29, '300s'), (41, '420s'), (59, '600s')):
    r = resp_dT[:, k]
    ok = (np.sign(r) == np.sign(dsp_v)).mean() * 100
    print(f"  {lab:5s} | 响应 {np.abs(r).mean():+.4f}°C | 方向 {ok:.0f}%")
r_end = resp_dT[:, 59]
dir_end = (np.sign(r_end) == np.sign(dsp_v)).mean() * 100
verdict = 'PASS' if np.abs(r_end).mean() >= 0.6 and dir_end >= 80 else 'FAIL'
print(f"  [判定] 600s 响应 {np.abs(r_end).mean():.3f}°C (需≥0.6) 方向 {dir_end:.0f}% (需≥80) → {verdict}")

with open('results/exp_100_m7dsp_h60/gain_test.json', 'w') as f:
    json.dump({'resp_600s': float(np.abs(r_end).mean()), 'dir_600s': float(dir_end),
               'resp_profile': {str(k): float(np.abs(resp_dT[:, k]).mean()) for k in (5, 11, 17, 29, 41, 59)},
               'mae_60': float(errs_all.mean()), 'mae_18': float(errs_18.mean()),
               'dir_60': float(dirs_all.mean()), 'phys_ref_600s': 2.0, 'verdict': verdict}, f, indent=2)
print('Saved: results/exp_100_m7dsp_h60/gain_test.json')
