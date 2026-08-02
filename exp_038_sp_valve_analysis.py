#!/usr/bin/env python3
"""exp_038_sp_valve_analysis.py — 修正版 SP→阀位 事件研究
1. 事件 = SP 变化起点 (合并连续变化)
2. 方向正确率 (修复 bug: 比较实际响应 vs 预期方向)
3. 30步响应幅度回归 Δa = α·ΔSP
4. 分箱线性检验 + 留出验证
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import train_raw, val_raw, test_raw, SP_IDX, VALVE_IDX, TARGET_IDX

raw = np.concatenate([train_raw, val_raw, test_raw])
sp, a2 = raw[:, SP_IDX], raw[:, VALVE_IDX[1]]
dsp = np.diff(sp)

# ===== 1. 事件: SP 变化起点 =====
starts = np.where((np.abs(dsp) > 0.3) & (np.abs(np.concatenate([[0], dsp[:-1]])) < 0.3))[0]
starts = [s for s in starts if s > 120 and s < len(raw) - 120]
print(f"SP 变化起点事件: {len(starts)}")

# ===== 2. 响应: 30 步净变化 =====
resps = np.array([(sp[s+1]-sp[s], a2[s+30]-a2[s]) for s in starts])
ds, da = resps[:, 0], resps[:, 1]

# ===== 3. 方向正确率 (修复版) =====
pred_dir = -np.sign(ds)          # SP下调→开大(+), 上调→关小(-)
actual_dir = np.sign(da)
strict = pred_dir == actual_dir
strong = np.abs(da) > 1.0
print(f"\n方向正确率 (全部): {strict.mean()*100:.0f}%")
print(f"  强响应事件 (|Δa|>1): {strong.sum()}, 其中方向正确: {(strong & strict).sum()} ({(strong & strict).sum()/max(strong.sum(),1)*100:.0f}%)")
print(f"  弱响应/无响应 (|Δa|≤1): {(~strong).sum()} ({(~strong).sum()/len(da)*100:.0f}%)")

# ===== 4. 幅度回归 =====
X = np.stack([ds, np.ones(len(ds))], 1)
beta, *_ = np.linalg.lstsq(X, da, rcond=None)
yhat = X @ beta
r2 = 1 - np.sum((da - yhat)**2) / np.sum((da - np.mean(da))**2)
print(f"\nΔa(30步) = {beta[0]:.3f}·ΔSP + {beta[1]:.3f}  R² = {r2:.4f}")

# ===== 5. 分箱线性 =====
print("\n分箱 (ΔSP → Δa 均值):")
for lo, hi in [(-8,-3), (-3,-1.5), (-1.5,-0.3), (0.3,1.5), (1.5,3), (3,8)]:
    m = (ds >= lo) & (ds < hi)
    if m.sum() > 0:
        print(f"  ΔSP∈[{lo:+.1f},{hi:+.1f}): n={m.sum():3d}  Δa均值={da[m].mean():+.2f}  中位={np.median(da[m]):+.2f}")

# ===== 6. 响应速度: 5步/10步/30步 =====
print("\n响应速度 (上调/下调分开):")
for sign, name in [(1, '上调'), (-1, '下调')]:
    idx = np.where(ds * sign > 0)[0]
    if len(idx) == 0: continue
    s5 = [a2[starts[i]+5] - a2[starts[i]] for i in idx]
    s10 = [a2[starts[i]+10] - a2[starts[i]] for i in idx]
    s30 = [da[i] for i in idx]
    print(f"  {name}SP (n={len(idx)}): 5步={np.mean(s5):+.2f}  10步={np.mean(s10):+.2f}  30步={np.mean(s30):+.2f}")

# ===== 7. 留出验证 (test 集) =====
n_tr = len(train_raw) + len(val_raw)
te_raw = raw[n_tr:]
te_starts = [s for s in starts if s >= n_tr]
te_ds = np.array([sp[s+1]-sp[s] for s in te_starts])
te_da = np.array([a2[s+30]-a2[s] for s in te_starts])
pred = beta[0] * te_ds + beta[1]
err_pred = np.abs(te_da - pred).mean()
err_const = np.abs(te_da - np.mean(da)).mean()
print(f"\n留出验证 (test {len(te_starts)} 事件): 预测 MAE={err_pred:.3f} vs 常数基线 {err_const:.3f} ({ (1-err_pred/err_const)*100:+.1f}%)")
print(f"  方向正确率 (test): {(np.sign(te_da)==-np.sign(te_ds)).mean()*100:.0f}%")

json.dump({'n_events': len(starts), 'alpha': float(beta[0]), 'b': float(beta[1]),
           'r2': float(r2), 'dir_acc': float(strict.mean()), 'dir_acc_strong': float((strong & strict).sum()/max(strong.sum(),1)),
           'holdout_mae_pred': float(err_pred), 'holdout_mae_const': float(err_const)},
          open("results/exp_038_sp_valve.json", 'w'), indent=2)
print("\nSaved: results/exp_038_sp_valve.json")
