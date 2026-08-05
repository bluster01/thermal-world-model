#!/usr/bin/env python3
"""
exp_099_phys_calib.py — B方案: 物理响应校准的可行性诊断 (2026-08-05)
=====================================================================
前提矛盾: M5-DSP MAE 0.301 但对 ΔSP 扰动响应仅 0.06°C — 若 ΔSP 因果效应 ~1.3°C
且模型完全未学, SP事件MAE应~1°C。故怀疑模型经状态路径(SP水平/阀位/趋势)已预测
大部分事件轨迹, 直接叠加物理响应会双计。
诊断: 模型预测残差 vs ΔSP 的相关性
  - 残差(实际−预测)@180s vs ΔSP: 斜率>0 → 欠响应, 叠加有效; ≈0 → 已捕捉, 叠加有害
  - 残差时间剖面: 分层(大/中/小)残差均值曲线, 看系统性偏差形态
  - 残差与 SP 阶跃方向/阀位变化的相关性 (共因代理)
用法: python exp_099_phys_calib.py
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

ck = torch.load('results/exp_096_dsp_wm/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
model = M5DSP().to(DEVICE).eval()
model.load_state_dict(ck['model_state_dict'])
print(f"[load] M5-DSP (exp_096, ep{ck['epoch']})")

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
    if s < 0 or s + W + H_OUT >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    if a_override is None:
        a = np.diff(raw41[s+W-1:s+W+H_OUT, 40])
    else:
        a = a_override
    a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        mu, _ = model(win, a_f)
    return mu[0].cpu().numpy()

# ===== 残差收集 =====
rows = []
for o in kept:
    s = o - W
    p = pred(s)
    if p is None: continue
    actual = raw[o:o+H_OUT, I_T]
    prev_T = raw[o-1, I_T]
    ds = raw[o, I_SP] - raw[o-1, I_SP]
    dV = raw[o+H_OUT-1, I_V2] - raw[o-1, I_V2]
    dL = raw[o+H_OUT-1, I_LD] - raw[o-1, I_LD]
    rows.append(dict(onset=o, dsp=ds, dv=dV, dl=dL, prev_T=prev_T,
                     resid=p - actual,          # 18 步残差 (预测−实际)
                     dT_act=actual[-1] - prev_T,
                     dT_pred=p[-1] - prev_T))
print(f"[rows] {len(rows)}")

resid_end = np.array([r['resid'][-1] for r in rows])
dsp_v = np.array([r['dsp'] for r in rows])
dv_v = np.array([r['dv'] for r in rows])
dl_v = np.array([r['dl'] for r in rows])
dT_act = np.array([r['dT_act'] for r in rows])
dT_pred = np.array([r['dT_pred'] for r in rows])

print("\n===== 残差(预测−实际)@180s 与各变量的相关 =====")
for name, v in (('ΔSP', dsp_v), ('Δ阀位', dv_v), ('Δ负荷', dl_v)):
    slope = np.polyfit(v, resid_end, 1)[0] if v.max()-v.min() > 1e-9 else float('nan')
    corr = np.corrcoef(v, resid_end)[0, 1] if v.std() > 1e-9 else float('nan')
    print(f"  {name:6s} | 相关 {corr:+.3f} | 回归斜率 {slope:+.4f}°C/单位")
print(f"  残差均值 {resid_end.mean():+.3f}°C | 残差σ {resid_end.std():.3f}")
print(f"  实际ΔT 均值 {dT_act.mean():+.3f} | 预测ΔT 均值 {dT_pred.mean():+.3f}")

print("\n===== 分层残差 (预测−实际, 180s 末点) =====")
for lo, hi, lab in ((3, 99, '大 |ΔSP|>3'), (2, 3, '中 2-3'), (1, 2, '小 1-2'), (0, 1, '平稳')):
    idx = [i for i, r in enumerate(rows) if lo < abs(r['dsp']) <= hi]
    if not idx: continue
    re_ = np.array([rows[i]['resid'] for i in idx])
    print(f"  {lab:12s} n={len(idx):3d} | 末点残差 {re_[:,-1].mean():+.3f} | 全程MAE {np.abs(re_).mean():.3f} | "
          f"残差剖面[0,6,12,17] {np.round(re_[:,[0,6,12,17]].mean(0),3)}")

print("\n===== 残差剖面 vs ΔSP 符号 (共因方向检查) =====")
for sgn, lab in ((1, 'ΔSP>0'), (-1, 'ΔSP<0')):
    idx = [i for i, r in enumerate(rows) if np.sign(r['dsp']) == sgn]
    if not idx: continue
    re_ = np.array([rows[i]['resid'] for i in idx])
    print(f"  {lab:8s} n={len(idx):3d} | 末点残差 {re_[:,-1].mean():+.3f} | ΔT实际 {np.mean([rows[i]['dT_act'] for i in idx]):+.3f}")

# ===== 物理响应叠加模拟 (K=1, τ=180s): 残差能否被修正 =====
print("\n===== 物理响应叠加模拟 (K=1.0, τ=180s) =====")
def phys_resp(dsp_seq, K=1.0, tau=180.0):
    """一阶惯性响应: T_resp[k] = K * Σ_{j<=k} ΔSP[j]*(1-exp(-(k-j)*10/tau))"""
    H = len(dsp_seq); out = np.zeros(H)
    for k in range(H):
        for j in range(k+1):
            out[k] += dsp_seq[j] * (1 - np.exp(-(k-j)*10/tau))
    return K * out

for K, tau in ((1.0, 180.0), (1.0, 120.0), (1.0, 240.0), (0.7, 180.0), (1.3, 180.0)):
    maes, maes_cal, corr_new = [], [], []
    for r in rows:
        s = r['onset'] - W
        a_seq = np.diff(raw41[s+W-1:s+W+H_OUT, 40])
        pr = phys_resp(a_seq, K, tau)
        p = r['resid'] + r['prev_T']  # 重建预测? 不需要 — 直接用残差
        # 修正后残差 = 原残差 − 物理响应 (响应加到预测上)
        rc = r['resid'] - pr
        maes.append(np.abs(r['resid']).mean())
        maes_cal.append(np.abs(rc).mean())
    m0, m1 = np.mean(maes), np.mean(maes_cal)
    print(f"  K={K:.1f} τ={tau:.0f}s | MAE {m0:.3f} → {m1:.3f} | Δ {m1-m0:+.3f} {'✓改善' if m1 < m0 else '✗变差'}")
