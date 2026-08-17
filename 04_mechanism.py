#!/usr/bin/env python3
"""机制分析：rollout 中沿程顺序违例与误差的关系"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
OUTPUTS = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"]
PAIRS = {  # (低, 高)
    "sh1_out < sh2_in": (1, 2),
    "sh2_out < main": (3, 4),
    "sh2_in < sh2_out": (2, 3),
    "sh1_out < sh1_in": (1, 0),
    "sh1_in < sh2_in": (0, 2),
}
variants = ["v0", "v1", "v0b", "v2o", "v2b", "v2"]
print(f"{'variant':8} {'rmse_main':>10} {'viol_any':>9} {'main<sh2out':>11} {'err_if_viol':>11} {'err_if_ok':>10}")
for v in variants:
    d = np.load(os.path.join(OUT, f"rollout_{v}.npz"))
    p, t = d["preds"], d["truths"]
    err = np.abs(p[:, 4] - t[:, 4])
    viol = np.zeros(len(p), dtype=bool)
    for name, (lo, hi) in PAIRS.items():
        viol |= (p[:, lo] >= p[:, hi])
    print(f"{v:8} {np.sqrt(np.mean(err**2)):10.2f} {viol.mean():9.3f} "
          f"{(p[:,3]>=p[:,4]).mean():11.3f} {err[viol].mean() if viol.any() else 0:11.2f} "
          f"{err[~viol].mean():10.2f}")

# v0 平台段检查
d = np.load(os.path.join(OUT, "rollout_v0.npz"))
p, t = d["preds"], d["truths"]
main_p = p[:, 4]
plat = (np.abs(main_p - 562.5) < 0.35) & (np.arange(len(main_p)) > 100)
if plat.any():
    i0 = np.where(plat)[0][0]
    print(f"\nv0 平台段 (预测主汽温≈562.5): step {i0}~{i0+20} (即 {i0*10/60:.0f}~{(i0+20)*10/60:.0f} min)")
    print(f"{'step':>6} {'sh1_in':>8} {'sh1_out':>8} {'sh2_in':>8} {'sh2_out':>8} {'main_pred':>9} {'main_true':>9}")
    for i in range(i0, i0 + 8):
        print(f"{i:6d} {p[i,0]:8.1f} {p[i,1]:8.1f} {p[i,2]:8.1f} {p[i,3]:8.1f} {p[i,4]:9.2f} {t[i,4]:9.2f}")
    print(f"... {OUTPUTS}")
else:
    print("\nv0 无平台段(阈值未命中)")

# 各变体 5 温度平均预测值 (漂移方向)
print("\nrollout 内 5 温度预测均值 vs 真实均值:")
for v in variants:
    d = np.load(os.path.join(OUT, f"rollout_{v}.npz"))
    p, t = d["preds"], d["truths"]
    print(f"{v:8} pred_mean={p.mean(0).round(1)}")
print(f"{'truth':8} pred_mean={t.mean(0).round(1)}")
