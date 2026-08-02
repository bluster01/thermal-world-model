#!/usr/bin/env python3
"""
exp_033_pi_identify.py — 路线B: PI 控制器行为辨识
===================================================
真实控制回路: SP → PI(SP−PV) → 二级阀指令/阀位 → 温度
从数据辨识简化 PI 模拟器: Δa₂(t) = Kp·e(t) + Ki·Σe + a₂(t−1) 残差结构
  或 增量形式: Δa₂ = Kp·Δe + Ki·e(t−1) (增量式PI)
输出: PI 参数 + 拟合精度 (R²) + 与真实阀位的轨迹对比
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

# ===== 数据准备 =====
# 二级减温阀 (主汽温 PI 回路执行机构)
a2 = test_raw[:, VALVE_IDX[1]]
sp = test_raw[:, SP_IDX]
pv = test_raw[:, TARGET_IDX]
e = sp - pv  # 偏差

# 增量式 PI: Δa₂(t+1) = Kp·(e(t+1)−e(t)) + Ki·e(t+1)
de = np.diff(e)
da2 = np.diff(a2)
# 对齐: da2[t+1] ↔ (de[t+1], e[t+1]), t=0..N-3
X = np.stack([de[1:], e[1:-1], np.ones(len(de)-1)], 1)
y = da2[1:]
# 剔除异常点 (|da|>30 的阀位跳变)
mask = np.abs(y) < 30
X, y = X[mask], y[mask]
beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
Kp, Ki, c = beta
yhat = X @ beta
r2 = 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)
print(f"=== 增量式 PI 辨识 (二级阀) ===")
print(f"  Kp (比例, Δe): {Kp:.4f}")
print(f"  Ki (积分, e):  {Ki:.4f}")
print(f"  常数: {c:.4f}")
print(f"  R² = {r2:.4f} (n={len(y)})")
print(f"  含义: SP+1°C 持续 → 每步阀位变化 {Ki:.3f} + 突变瞬间 {Kp:.3f}")

# 位置式 (直接拟合 a 而非 Δa): a2(t) = a2(t-1) + Kp·de + Ki·e
# 验证: 用辨识的 PI 在 test 上滚动仿真
def pi_sim(sp_seq, pv_seq, a0, Kp, Ki, c):
    a = np.zeros(len(sp_seq)); a[0] = a0
    e_prev = sp_seq[0] - pv_seq[0]
    for t in range(1, len(sp_seq)):
        e_t = sp_seq[t] - pv_seq[t]
        a[t] = a[t-1] + Kp*(e_t - e_prev) + Ki*e_t + c
        a[t] = np.clip(a[t], 0, 100)
        e_prev = e_t
    return a

# 测试: 随机 10 条 500 步段
np.random.seed(0)
errs = []
for _ in range(10):
    i = np.random.randint(1000, len(test_raw) - 600)
    a_sim = pi_sim(sp[i:i+500], pv[i:i+500], a2[i], Kp, Ki, c)
    errs.append(np.abs(a_sim - a2[i:i+500]).mean())
print(f"\n滚动仿真 MAE (10段×500步): {np.mean(errs):.3f} 阀位单位 (二级阀范围 0-100)")
print(f"真实阀位 std ≈ {a2.std():.2f}, 相对误差 {np.mean(errs)/a2.std()*100:.1f}%")

json.dump({'Kp': float(Kp), 'Ki': float(Ki), 'c': float(c), 'r2': float(r2),
           'sim_mae': float(np.mean(errs))},
          open("results/exp_033_pi_params.json", 'w'), indent=2)
print("\nSaved: results/exp_033_pi_params.json")
