#!/usr/bin/env python3
"""
事件研究 (Event Study) — 减温阀开/关阀事件 → 主汽温滞后响应
=============================================================
目的: 确立物理滞后时标真值, 回答:
  开阀后多久主汽温开始降? (10s? 30s? 60-90s?)
  这是评判世界模型响应曲线是否符合物理的唯一基准。

方法:
1. 找二级减温阀阀位突增/突减事件 (Δv >= 阈值)
2. 以事件时刻为 0, 取事件前 30s 基线, 对齐主汽温 0-150s 轨迹
3. 平均所有事件 → 平均响应曲线 ΔT(t)
4. 同时给互相关全景: valve(t) vs 主汽温(TARGET) 各滞后 0-150s 的 r
"""
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import config as cfg
from data_loader import load_raw_data

state_data, delta_actions, valve_abs = load_raw_data()
T = state_data[:, cfg.TARGET_IDX]   # 末级过热器出口汽温 (主汽温)
V1 = valve_abs[:, 0]                # 一级减温调节门阀位
V2 = valve_abs[:, 1]                # 二级减温调节门阀位
dt = 10.0                           # 采样间隔 10s

print(f"数据: {len(T)} 点 @ {dt}s = {len(T)*dt/3600:.1f}h")
print(f"主汽温: mean={T.mean():.1f}°C std={T.std():.2f}")
print(f"二级阀位: mean={V2.mean():.2f}% std={V2.std():.2f} min={V2.min():.1f} max={V2.max():.1f}")
print(f"一级阀位: mean={V1.mean():.2f}% std={V1.std():.2f}")

# ===== 1. 事件定义 =====
def find_events(v, thr_up, thr_dn, min_gap=6):
    """阀位突增(开阀) / 突减(关阀)事件"""
    dv = np.diff(v, prepend=v[0])
    ev_up, ev_dn = [], []
    for i in range(1, len(v)-1):
        if dv[i] >= thr_up and (not ev_up or i - ev_up[-1] >= min_gap):
            ev_up.append(i)
        if dv[i] <= -thr_dn and (not ev_dn or i - ev_dn[-1] >= min_gap):
            ev_dn.append(i)
    return np.array(ev_up), np.array(ev_dn)

def event_response(T, events, pre=3, post=15):
    """事件响应: 以事件前 pre 步均值为基线, 对齐事件后 post 步"""
    resp = []
    for i in events:
        if i - pre < 0 or i + post >= len(T):
            continue
        base = T[i-pre:i].mean()
        resp.append(T[i:i+post+1] - base)
    return np.array(resp) if resp else np.zeros((0, post+1))

for thr in [2.0, 3.0, 5.0]:
    up, dn = find_events(V2, thr, thr)
    r_up = event_response(T, up)
    r_dn = event_response(T, dn)
    print(f"\n===== 二级阀 阈值±{thr}%: 开阀{len(up)}次 / 关阀{len(dn)}次 =====")
    if len(r_up) > 10:
        m = r_up.mean(0)
        t = np.arange(len(m)) * dt
        # 找最小温度点 (最大降温) 及时间
        imin = np.argmin(m)
        print(f"  开阀→主汽温: t0={m[0]:+.3f} t30s={m[3]:+.3f} t60s={m[6]:+.3f} "
              f"t90s={m[9]:+.3f} t120s={m[12]:+.3f} t150s={m[15]:+.3f}")
        print(f"  最大降温 {m[imin]:+.3f}°C @ {t[imin]:.0f}s")
        print(f"  响应轨迹: " + " ".join(f"{x:+.2f}" for x in m))
    if len(r_dn) > 10:
        m = r_dn.mean(0)
        t = np.arange(len(m)) * dt
        imax = np.argmax(m)
        print(f"  关阀→主汽温: t0={m[0]:+.3f} t30s={m[3]:+.3f} t60s={m[6]:+.3f} "
              f"t90s={m[9]:+.3f} t120s={m[12]:+.3f} t150s={m[15]:+.3f}")
        print(f"  最大升温 {m[imax]:+.3f}°C @ {t[imax]:.0f}s")

# ===== 2. 互相关全景 =====
print("\n===== 互相关: 二级阀位(超前) vs 主汽温 =====")
print("r[valve(t-lag) vs T(t)]: lag>0 表示阀位超前")
dT = np.diff(T, prepend=T[0])
V2c = V2 - V2.mean()
dTc = dT - dT.mean()
lags = list(range(1, 16))  # 10-150s
for lag in [1, 3, 5, 6, 7, 8, 9, 10, 12, 15]:
    if lag >= len(V2c): continue
    r = np.corrcoef(V2c[:-lag], dTc[lag:])[0, 1]
    print(f"  valve(t-{lag*10:3d}s) vs dT/dt: r={r:+.4f}")
# 负相关最强者 (物理因果 = 阀位超前导致温度变化)
r_all = []
for lag in lags:
    if lag >= len(V2c): break
    r_all.append(np.corrcoef(V2c[:-lag], dTc[lag:])[0, 1])
r_all = np.array(r_all)
imin = np.argmin(r_all)
print(f"  负相关最强: r={r_all[imin]:+.4f} @ lag={lags[imin]*10}s ← 物理因果时标")
imax = np.argmax(r_all)
print(f"  正相关最强: r={r_all[imax]:+.4f} @ lag={lags[imax]*10}s ← PID反馈时标")

# ===== 3. 温度对阀位的"稳态"响应 (长窗) =====
print("\n===== 大时间窗: 开阀后 1-10min 温度趋势 =====")
for thr in [3.0]:
    up, _ = find_events(V2, thr, thr)
    r_long = event_response(T, up, pre=6, post=60)  # 10min
    if len(r_long) > 10:
        m = r_long.mean(0)
        t = np.arange(len(m)) * dt
        imin = np.argmin(m)
        print(f"  开阀{len(up)}次: 最大降温 {m[imin]:+.3f}°C @ {t[imin]:.0f}s")
        print(f"  轨迹(min): " + " ".join(f"{x:+.2f}" for x in m[::6]))
