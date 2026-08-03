#!/usr/bin/env python3
"""
exp_067_arx_identification.py — P1: ARX 系统辨识 + 1-18步 direct 验证
======================================================================
ARX(T 自回归 12 阶 + 阀位1/2 各 12 阶 + SP 6 阶) 最小二乘
对比 M7 的 direct 1-18 步 MAE (同 exp_057 的 500 固定窗口协议)
用法: python exp_067_arx_identification.py [--order 12]
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
from src import config as cfg
sys.argv = _argv

ORDER = int(os.environ.get('ORDER', 12))   # 温度/阀位自回归阶次
SP_ORDER = int(os.environ.get('SP_ORDER', 6))
W = cfg.WINDOW_SIZE
TIDX, VIDX, SIDX = E.TARGET_IDX, E.VALVE_IDX, E.SP_IDX
train_raw = E.train_raw

# ============ 1. 构造 ARX 回归矩阵 (训练段) ============
t0 = time.time()
P = max(ORDER, SP_ORDER)
T = len(train_raw)
X_rows, y_rows = [], []
V1, V2 = train_raw[:, VIDX[0]], train_raw[:, VIDX[1]]
Tv, Sp = train_raw[:, TIDX], train_raw[:, SIDX]
for i in range(P, T - 1):
    r = np.concatenate([
        Tv[i-ORDER:i][::-1],          # 温度自回归 12 阶 (最近的在前)
        V1[i-ORDER:i][::-1], V2[i-ORDER:i][::-1],  # 阀位 12 阶
        Sp[i-SP_ORDER:i][::-1],       # SP 6 阶
        [1.0],                        # 常数
    ])
    X_rows.append(r); y_rows.append(Tv[i + 1])
X = np.array(X_rows, dtype=np.float64)   # [N, 42]
y = np.array(y_rows, dtype=np.float64)
print(f"ARX 回归矩阵: {X.shape} ({time.time()-t0:.1f}s)")
coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
print(f"ARX 拟合: R²={1 - np.sum((y-pred)**2)/np.sum((y-y.mean())**2):.4f} | 一步MAE={np.abs(pred-y).mean():.4f}°C")

# ============ 2. 1-18 步 direct 验证 (同 exp_057: 500 固定窗口) ============
def arx_forecast(i0, h, test_raw):
    """从窗口起点 i0 递归预测 h 步"""
    Tv, V1, V2, Sp = (test_raw[:, TIDX], test_raw[:, VIDX[0]],
                      test_raw[:, VIDX[1]], test_raw[:, SIDX])
    t_hist = list(Tv[i0:i0+ORDER])
    out = []
    for k in range(h):
        idx = i0 + k
        r = np.concatenate([
            t_hist[::-1][:ORDER],
            V1[idx:idx+ORDER][::-1], V2[idx:idx+ORDER][::-1],
            Sp[idx+1-SP_ORDER:idx+1][::-1], [1.0]])
        pred = float(r @ coef)
        out.append(pred)
        t_hist.append(pred); t_hist.pop(0)
    return np.array(out)

N = len(E.test_raw)
np.random.seed(42)
idxs = np.random.randint(0, N - W - 18, 500)
errs = np.zeros((500, 18))
for i, i0 in enumerate(idxs):
    mu = arx_forecast(i0 + W, 18, E.test_raw)
    tgt = E.test_raw[i0 + W:i0 + W + 18, TIDX]
    errs[i] = np.abs(mu - tgt)
step_mae = errs.mean(0)
print("\nARX direct 1-18 步 MAE (°C):")
print("  " + "  ".join(f"{i+1}:{v:.3f}" for i, v in enumerate(step_mae)))
print(f"  均值(1-18步): {step_mae.mean():.3f}")
print("\n对照 M7 (exp_057): 1步0.074 5步0.179 10步0.311 18步0.500")

# ============ 3. 保存 ============
os.makedirs('results/exp_067_linear_mpc', exist_ok=True)
np.save('results/exp_067_linear_mpc/arx_coef.npy', coef)
np.save('results/exp_067_linear_mpc/arx_order.npy', np.array([ORDER, SP_ORDER]))
np.save('results/exp_067_linear_mpc/arx_step_mae.npy', step_mae)
print(f"\nSaved: results/exp_067_linear_mpc/arx_coef.npy (+order/step_mae)")
