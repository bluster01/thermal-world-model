#!/usr/bin/env python3
"""
exp_037_sp_ff_fit.py — SP突变→阀位响应 前馈拟合 (联合优化前置)
================================================================
从数据学: 给定 ΔSP 突变 → 预期阀位响应轨迹 Δa(t), t=1..18
方法: 事件研究 (train+test 所有 |ΔSP|>1.5 事件) + 基线校正 + 回归
输出: 响应曲线 g(t), 比例系数 α, 留出验证 R²
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import train_raw, val_raw, test_raw, SP_IDX, VALVE_IDX, TARGET_IDX

raw = np.concatenate([train_raw, val_raw, test_raw])
N = len(raw)
a2 = raw[:, VALVE_IDX[1]]
sp = raw[:, SP_IDX]
pv = raw[:, TARGET_IDX]

# ===== 1. 事件提取 =====
dsp = np.diff(sp)
events = np.where(np.abs(dsp) > 1.5)[0]
events = [e for e in events if e > 120 and e < N - 120]
print(f"SP 跳变事件 (|ΔSP|>1.5°C): {len(events)}")

# ===== 2. 基线校正: 非事件时段的阀位漂移 =====
np.random.seed(0)
non = np.random.choice([i for i in range(500, N-120) if np.abs(dsp[i]) < 0.1], 2000, replace=False)
base_move = np.zeros(18)
for i in non:
    base_move += np.abs(a2[i+1:i+19] - a2[i])
base_move /= len(non)
print(f"基线阀位漂移 (非事件, 18步): {base_move.mean():.3f}")

# ===== 3. 事件响应 =====
resps = []  # (ΔSP, Δa_sum_18, Δa_sum_60, 响应曲线)
for e in events:
    ds = sp[e+1] - sp[e]
    da18 = a2[e+18] - a2[e]      # 事件后18步总变化 (带符号)
    da60 = a2[e+60] - a2[e]
    curve = a2[e+1:e+19] - a2[e]  # 分步响应
    resps.append((ds, da18, da60, curve))
resps = np.array([(r[0], r[1], r[2]) for r in resps], dtype=float)

# ===== 4. 回归: Δa_sum = α·ΔSP + β =====
X = np.stack([resps[:, 0], np.ones(len(resps))], 1)
y18 = resps[:, 1]
beta, *_ = np.linalg.lstsq(X, y18, rcond=None)
alpha18, b18 = beta
yhat = X @ beta
r2_18 = 1 - np.sum((y18 - yhat)**2) / np.sum((y18 - np.mean(y18))**2)
print(f"\nΔa(18步) = {alpha18:.3f}·ΔSP + {b18:.3f}  R²={r2_18:.4f}")
print(f"  含义: SP 突变 +1°C → 二级阀预期净变化 {alpha18:+.3f} 阀位单位")

y60 = resps[:, 2]
beta60, *_ = np.linalg.lstsq(np.stack([resps[:, 0], np.ones(len(resps))], 1), y60, rcond=None)
r2_60 = 1 - np.sum((y60 - (np.stack([resps[:,0], np.ones(len(resps))],1) @ beta60))**2) / np.sum((y60 - np.mean(y60))**2)
print(f"Δa(60步) = {beta60[0]:.3f}·ΔSP + {beta60[1]:.3f}  R²={r2_60:.4f}")

# ===== 5. 分步响应曲线 (符号对齐: 上调SP → 阀位↓(关小减温) ) =====
curves = []
for e in events:
    ds = sp[e+1] - sp[e]
    if abs(ds) < 0.1: continue
    curves.append((a2[e+1:e+19] - a2[e]) / ds)  # 每步响应/ΔSP
curves = np.array(curves)
g = np.mean(curves, 0)
print(f"\n分步响应 g(t) = Δa(t)/ΔSP:")
print("  " + " ".join([f"{x:+.3f}" for x in g[:9]]))
print("  " + " ".join([f"{x:+.3f}" for x in g[9:]]))
print(f"  g(t) 符号: 上调SP应关小阀(负), g 均值 {g.mean():+.3f} (方向{'✓' if np.sign(g.mean())==-1 else '✗'})")

# ===== 6. 留出验证 (test 事件) =====
te = np.where(np.abs(np.diff(test_raw[:, SP_IDX])) > 1.5)[0]
te = [e for e in te if e > 120 and e < len(test_raw) - 120]
errs = []
for e in te:
    ds = test_raw[e+1, SP_IDX] - test_raw[e, SP_IDX]
    pred = alpha18 * ds + b18
    actual = test_raw[e+18, VALVE_IDX[1]] - test_raw[e, VALVE_IDX[1]]
    errs.append(abs(pred - actual))
print(f"\n留出验证 (test {len(te)} 事件): 预测误差 MAE = {np.mean(errs):.3f} 阀位单位")
print(f"  (基线常数预测误差: {np.mean([abs(b18 - (test_raw[e+18,VALVE_IDX[1]]-test_raw[e,VALVE_IDX[1]])) for e in te]):.3f})")

json.dump({'alpha18': float(alpha18), 'b18': float(b18), 'r2_18': float(r2_18),
           'alpha60': float(beta60[0]), 'r2_60': float(r2_60),
           'g_curve': g.tolist(), 'holdout_mae': float(np.mean(errs))},
          open("results/exp_037_sp_ff.json", 'w'), indent=2)
print("\nSaved: results/exp_037_sp_ff.json")
