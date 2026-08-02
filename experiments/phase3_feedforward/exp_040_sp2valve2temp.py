#!/usr/bin/env python3
"""
exp_040_sp2valve2temp.py — 验证: SP→阀位映射→M7→温度 管线
============================================================
架构: SP调整 → 前馈映射(Δa=−0.82·ΔSP 或 g(t)曲线) → 阀位序列 → M7 → 温度
验证: SP 跳变事件上, 管线预测的温度响应 vs 真实温度响应
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import build_model
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
from exp_027_dwm_mpc import W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX
sys.argv = _argv

ALPHA_FF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.82  # SP→二级阀前馈增益
MODE = sys.argv[2] if len(sys.argv) > 2 else 'const'  # const: Δa=−α·ΔSP; curve: g(t) 分步

wm = build_model('M7').to(DEVICE).eval()
ck = torch.load("results/exp_025_M7/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
wm.load_state_dict(ck['model_state_dict'])

# g(t) 曲线 (exp_037 拟合: 快速到位 -1.4/°C)
g = np.array([-0.59, -1.31, -1.48, -1.49, -1.34, -1.25, -1.25, -1.32, -1.39,
              -1.43, -1.45, -1.47, -1.50, -1.52, -1.54, -1.53, -1.53, -1.50])

# ===== 事件: SP 变化起点 =====
sp, a2, pv = test_raw[:, SP_IDX], test_raw[:, VALVE_IDX[1]], test_raw[:, TARGET_IDX]
dsp = np.diff(sp)
starts = np.where((np.abs(dsp) > 1.5) & (np.abs(np.concatenate([[0], dsp[:-1]])) < 0.3))[0]
starts = [s for s in starts if s > W + 50 and s < len(test_raw) - W - 60]
print(f"SP 跳变事件: {len(starts)}")

# ===== 管线预测 vs 真实 =====
# 真实: SP 跳变后, 温度响应 = pv[e+1:e+61] - pv[e]  (60步)
# 管线: SP跳变 ΔSP → 阀位序列 a_ff → M7(窗口, a_ff) → 温度
preds, actuals = [], []
dir_ok = 0
for e in starts:
    ds = sp[e+1] - sp[e]
    # 映射: 阀位序列 (相对跳变前)
    if MODE == 'const':
        a_ff = a2[e] - ALPHA_FF * ds * np.ones(60)   # 阶跃响应: 立即到位
    else:
        a_ff = a2[e] + np.concatenate([g, [g[-1]]*42]) * ds  # g(t) 曲线 (60步)
    a_ff = np.clip(a_ff, 0, 100)
    # M7 预测: 窗口 [e-W+1, e+1] (跳变前窗口), 未来 60 步分批 (M7 每次 18 步)
    xh = torch.FloatTensor(test_raw[e+1-W:e+1]).unsqueeze(0).to(DEVICE)
    a1_real = test_raw[e+1:e+61, VALVE_IDX[0]]  # 一级阀保持真实
    temp_pred = []
    for t0 in range(0, 60, H_OUT):
        n_take = min(H_OUT, 60 - t0)
        af2 = a_ff[t0:t0+n_take]
        af1 = a1_real[t0:t0+n_take]
        # 填充到 18 步 (M7 固定输入)
        if n_take < H_OUT:
            af2 = np.concatenate([af2, [af2[-1]]*(H_OUT-n_take)])
            af1 = np.concatenate([af1, [af1[-1]]*(H_OUT-n_take)])
        af = torch.FloatTensor(np.stack([af1, af2], 1)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = wm(xh, af)
            temp_pred.extend(mu[0].cpu().numpy()[:n_take])
    pred = np.array(temp_pred[:60])
    actual = pv[e+1:e+61]
    # 相对响应 (去起点)
    pred_rel = pred - pred[0]
    actual_rel = actual - actual[0]
    preds.append(pred_rel); actuals.append(actual_rel)
    if np.sign(pred_rel[-1]) == np.sign(actual_rel[-1]):
        dir_ok += 1

preds, actuals = np.array(preds), np.array(actuals)
print(f"\n方向一致率 (60步末): {dir_ok}/{len(starts)} = {dir_ok/len(starts)*100:.0f}%")
print(f"响应幅度 (30步): 预测 {np.mean(np.abs(preds[:,29])):.3f} vs 实际 {np.mean(np.abs(actuals[:,29])):.3f}")
print(f"响应幅度 (60步): 预测 {np.mean(np.abs(preds[:,59])):.3f} vs 实际 {np.mean(np.abs(actuals[:,59])):.3f}")
# 相关系数 (逐事件预测 vs 实际轨迹)
corrs = [np.corrcoef(p, a)[0,1] for p, a in zip(preds, actuals) if np.std(a) > 0.1]
print(f"轨迹相关 (预测vs实际, 有效事件): 均值 {np.mean(corrs):.3f}")
# MAE
mae = np.abs(preds - actuals).mean(0)
print(f"MAE 轨迹: 10步 {mae[9]:.3f} | 30步 {mae[29]:.3f} | 60步 {mae[59]:.3f}")

json.dump({'mode': MODE, 'alpha_ff': ALPHA_FF, 'n': len(starts),
           'dir_acc': dir_ok/len(starts), 'mae30': float(mae[29]), 'mae60': float(mae[59]),
           'corr_mean': float(np.mean(corrs))},
          open(f"results/exp_040_pipeline_{MODE}_{ALPHA_FF}.json", 'w'), indent=2)
print(f"\nSaved: results/exp_040_pipeline_{MODE}_{ALPHA_FF}.json")
