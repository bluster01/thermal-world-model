#!/usr/bin/env python3
"""
exp_097_action_probe.py — ΔSP 动作通道响应性诊断 (2026-08-05)
================================================================
用户反馈: case 图趋势对不上 → 怀疑 M5-DSP 对 ΔSP 动作无响应 (通道增益≈0)
诊断: 同一事件, 真实 ΔSP vs ΔSP=0 输入 → 预测差异
  - 响应幅度: |ΔT_pred(real) − ΔT_pred(zero)| (180s 末点 + 全程)
  - 响应方向正确率: sign(ΔT_pred(real) − ΔT_pred(zero)) == sign(ΔSP)
  - 对比 M5: 真实阀位 vs 阀位保持 (通道增益参照)
  - 对比沙盒 M5 (exp_095 已知有响应) 作为基准
用法: python exp_097_action_probe.py
"""
import os, sys
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E.cfg.WINDOW_SIZE
H_OUT = E.H_OUT
raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_V1 = E.NUMERIC_COLS.index('一级减温调节门阀位')
I_V2 = E.NUMERIC_COLS.index('二级减温调节门阀位')
I_LD = E.NUMERIC_COLS.index('机组负荷')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)

class M5DSP(E.DirectWM):
    def __init__(self):
        super().__init__(use_action=True, use_patch=True, per_variable=True,
                         use_varattn=True, probabilistic=False)
        d = E.cfg.D_MODEL
        self.action_enc = nn.Sequential(
            nn.Linear(H_OUT, d * 2), nn.GELU(), nn.Dropout(E.cfg.DROPOUT))

ck_m5 = torch.load('results/exp_025_M5/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
m5 = E.build_model('M5').to(DEVICE).eval()
m5.load_state_dict(ck_m5['model_state_dict'])
ck_dsp = torch.load('results/exp_096_dsp_wm/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
m5dsp = M5DSP().to(DEVICE).eval()
m5dsp.load_state_dict(ck_dsp['model_state_dict'])
print('[load] M5 + M5-DSP OK')

# 事件筛选 (同 exp_097)
dsp_abs = np.abs(np.diff(raw[:, I_SP]))
onsets = []
for i in np.where(dsp_abs > 1.0)[0] + 1:
    if not onsets or i - onsets[-1] >= 60:
        onsets.append(i)
stable = [o for o in onsets if o + 60 < N and
          np.abs(np.diff(raw[max(0, o-20):min(N, o+20), I_LD])).max() <= 3.0]
kept = [o for o in stable if np.abs(raw[o:o+61, I_SP] - raw[o, I_SP]).max() <= 0.3]
print(f"[events] {len(kept)}")

def pred_dsp(s, a_override=None):
    """M5-DSP: a_override=None 用真实 ΔSP, 否则用给定序列 [H]"""
    if s < 0 or s + W + H_OUT >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    if a_override is None:
        a = np.diff(raw41[s+W-1:s+W+H_OUT, 40])
    else:
        a = a_override
    a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        mu, _ = m5dsp(win, a_f)
    return mu[0].cpu().numpy()

def pred_m5(s, a_override=None):
    """M5: a_override=None 用真实阀位, 否则 [H,2] 常数"""
    if s < 0 or s + W + H_OUT >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    if a_override is None:
        a = raw[s+W:s+W+H_OUT, I_V1:I_V2+1]
    else:
        a = a_override
    a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        mu, _ = m5(win, a_f)
    return mu[0].cpu().numpy()

# ===== 诊断 1: ΔSP 扰动响应 =====
print("\n===== 诊断1: M5-DSP 对 ΔSP 动作的响应 (134 事件) =====")
resp_amp, resp_dir, resp_err = [], [], []
for o in kept:
    s = o - W
    p_real = pred_dsp(s)
    p_zero = pred_dsp(s, np.zeros(H_OUT))
    p_neg  = pred_dsp(s, -np.abs(np.diff(raw41[s+W-1:s+W+H_OUT, 40])))  # 反方向
    if p_real is None or p_zero is None: continue
    dT = p_real[-1] - p_zero[-1]                       # 动作引起的末点温差
    ds = raw[o, I_SP] - raw[o-1, I_SP]                 # 实际 ΔSP
    resp_amp.append(abs(dT))
    resp_dir.append(1 if np.sign(dT) == np.sign(ds) else 0)
    resp_err.append(np.abs(p_real - p_zero).mean())
resp_amp = np.array(resp_amp); resp_dir = np.array(resp_dir); resp_err = np.array(resp_err)
print(f"  动作响应幅度 |ΔT(real)−ΔT(0)| @180s: mean {resp_amp.mean():.4f}°C | med {np.median(resp_amp):.4f} | p90 {np.percentile(resp_amp,90):.4f}")
print(f"  全程响应 (18步平均差): mean {resp_err.mean():.4f}°C")
print(f"  响应方向正确率 (sign(ΔT响应)==sign(ΔSP)): {resp_dir.mean()*100:.1f}%")
print(f"  对比: 实际温度 180s 变化幅度 mean {np.abs(raw[np.array(kept)+H_OUT-1, I_T] - raw[np.array(kept)-1, I_T]).mean():.3f}°C")

# 分层
print("\n  分层响应幅度:")
for lo, hi, lab in ((3, 99, '大 |ΔSP|>3'), (2, 3, '中 2-3'), (1, 2, '小 1-2')):
    idx = [i for i, o in enumerate(kept) if lo < abs(raw[o, I_SP]-raw[o-1, I_SP]) <= hi]
    if idx:
        print(f"  {lab:12s} n={len(idx):3d} | 幅度 {resp_amp[idx].mean():.4f}°C | 方向正确 {resp_dir[idx].mean()*100:.0f}%")

# ===== 诊断 2: M5 阀位扰动响应 (参照) =====
print("\n===== 诊断2: M5 对阀位动作的响应 (对照) =====")
v_amp, v_dir = [], []
for o in kept:
    s = o - W
    p_real = pred_m5(s)
    v0 = raw[s+W, I_V1:I_V2+1]
    p_hold = pred_m5(s, np.tile(v0, (H_OUT, 1)))
    if p_real is None or p_hold is None: continue
    dT = p_real[-1] - p_hold[-1]
    dV = raw[s+W+H_OUT-1, I_V2] - raw[s+W, I_V2]
    v_amp.append(abs(dT))
    v_dir.append(1 if np.sign(dT) == np.sign(dV) else 0)
v_amp = np.array(v_amp); v_dir = np.array(v_dir)
print(f"  M5 阀位响应幅度: mean {v_amp.mean():.4f}°C | 方向正确 {v_dir.mean()*100:.1f}%")

# ===== 诊断 3: 我选的 3 个 case 明细 =====
print("\n===== 诊断3: 已选 case 明细 (onset 659852 / 57860 / 326938) =====")
for o in (659852, 57860, 326938):
    s = o - W
    p_real = pred_dsp(s); p_zero = pred_dsp(s, np.zeros(H_OUT))
    p_m5 = pred_m5(s)
    actual = raw[o:o+H_OUT, I_T]
    prev_T = raw[o-1, I_T]
    ds = raw[o, I_SP] - raw[o-1, I_SP]
    print(f"  onset={o} ΔSP={ds:+.2f}")
    print(f"    实际: ΔT180s={actual[-1]-prev_T:+.3f} | 全程σ={actual.std():.3f}")
    print(f"    M5-DSP 真实ΔSP: ΔT180s={p_real[-1]-prev_T:+.3f} | MAE={np.abs(p_real-actual).mean():.3f}")
    print(f"    M5-DSP ΔSP=0:    ΔT180s={p_zero[-1]-prev_T:+.3f} | 响应ΔT={p_real[-1]-p_zero[-1]:+.4f}")
    print(f"    M5 真实阀位:     ΔT180s={p_m5[-1]-prev_T:+.3f} | MAE={np.abs(p_m5-actual).mean():.3f}")
