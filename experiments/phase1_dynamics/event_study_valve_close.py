#!/usr/bin/env python3
"""
关阀专项事件研究 — 关阀 → 主汽温响应 (与开阀对称性对比)
=========================================================
开阀分析发现: 开阀后前 90s 微升 +0.3°C, 120s+ 才转降 (大滞后物理)。
关阀初步结果反直觉: 关阀后温度持续下降 (t150s -0.38°C) — 需解析:

1. 共因: PID 在温度下降时才关阀 → 关阀时刻温度本就在降 (事件前趋势)
2. 物理: 关阀=减温水减少 → 蒸汽升温, 但滞后 60-90s+
3. 问题: 150s 窗口内物理升温是否被共因下降掩盖? 更长窗口 (10min) 是否回升?

方法:
- 长窗口: 关阀后 0-10min 响应 (看最终是否回升 = 物理升温)
- 基线校正: 用事件前 30s 趋势外推, 分离共因趋势 vs 事件后增量
- 对称性: 开阀(降温物理) vs 关阀(升温物理) 响应曲线对比
- 分类: 按事件时温度趋势 (升/降) 分组, 检验共因强度
"""
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config as cfg
from data_loader import load_raw_data

state_data, delta_actions, valve_abs = load_raw_data()
T = state_data[:, cfg.TARGET_IDX]   # 末级过热器出口汽温 (主汽温)
V1 = valve_abs[:, 0]
V2 = valve_abs[:, 1]
dt = 10.0

print(f"数据: {len(T)} 点 @ {dt}s")


def find_events(v, thr_up, thr_dn, min_gap=6):
    dv = np.diff(v, prepend=v[0])
    ev_up, ev_dn = [], []
    for i in range(1, len(v)-1):
        if dv[i] >= thr_up and (not ev_up or i - ev_up[-1] >= min_gap):
            ev_up.append(i)
        if dv[i] <= -thr_dn and (not ev_dn or i - ev_dn[-1] >= min_gap):
            ev_dn.append(i)
    return np.array(ev_up), np.array(ev_dn)


def event_response(T, events, pre=6, post=60):
    resp = []
    for i in events:
        if i - pre < 0 or i + post >= len(T):
            continue
        base = T[i-pre:i].mean()
        resp.append(T[i:i+post+1] - base)
    return np.array(resp) if resp else np.zeros((0, post+1))


def trend_adjusted(T, events, pre=3):
    """基线趋势校正: 事件前 pre 步线性趋势外推, 返回事件后残差"""
    out = []
    for i in events:
        if i - pre < 0 or i + 60 >= len(T):
            continue
        seg = T[i-pre:i]
        slope = (seg[-1] - seg[0]) / (len(seg) - 1) if len(seg) > 1 else 0.
        extrap = T[i] + slope * np.arange(0, 61)
        out.append(T[i:i+61] - extrap)
    return np.array(out) if out else np.zeros((0, 61))


print("\n" + "="*100)
print("1. 关阀长窗口 (10min) — 物理升温是否在长程显现?")
print("="*100)
for thr in [2.0, 3.0, 5.0]:
    up, dn = find_events(V2, thr, thr)
    r = event_response(T, dn, pre=6, post=60)
    if len(r) > 10:
        m = r.mean(0); t = np.arange(len(m)) * dt
        imax = np.argmax(m)
        print(f"\n  [关阀±{thr}%] n={len(dn)}")
        print(f"  t0={m[0]:+.3f} t30s={m[3]:+.3f} t60s={m[6]:+.3f} t90s={m[9]:+.3f} "
              f"t150s={m[15]:+.3f} t300s={m[30]:+.3f} t600s={m[60]:+.3f}")
        print(f"  峰值 {m[imax]:+.3f}°C @ {t[imax]:.0f}s")
    r_up = event_response(T, up, pre=6, post=60)
    if len(r_up) > 10:
        m = r_up.mean(0); t = np.arange(len(m)) * dt
        imin = np.argmin(m)
        print(f"  [开阀±{thr}%] n={len(up)}: 谷值 {m[imin]:+.3f}°C @ {t[imin]:.0f}s "
              f"(t600s={m[60]:+.3f})")

print("\n" + "="*100)
print("2. 基线趋势校正 — 分离共因趋势 vs 事件后物理增量")
print("="*100)
for thr in [3.0, 5.0]:
    up, dn = find_events(V2, thr, thr)
    ra = trend_adjusted(T, dn)
    if len(ra) > 10:
        m = ra.mean(0); t = np.arange(len(m)) * dt
        imax = np.argmax(m)
        print(f"\n  [关阀±{thr}% 趋势校正] n={len(dn)}")
        print(f"  t0={m[0]:+.3f} t30s={m[3]:+.3f} t60s={m[6]:+.3f} t90s={m[9]:+.3f} "
              f"t120s={m[12]:+.3f} t150s={m[15]:+.3f} t300s={m[30]:+.3f} t600s={m[60]:+.3f}")
        print(f"  峰值 {m[imax]:+.3f}°C @ {t[imax]:.0f}s ← 物理升温(去共因后)")
    ra_up = trend_adjusted(T, up)
    if len(ra_up) > 10:
        m = ra_up.mean(0); t = np.arange(len(m)) * dt
        imin = np.argmin(m)
        print(f"  [开阀±{thr}% 趋势校正] n={len(up)}: 谷值 {m[imin]:+.3f}°C @ {t[imin]:.0f}s")

print("\n" + "="*100)
print("3. 对称性 — 开阀 vs 关阀 响应曲线 (原始+趋势校正)")
print("="*100)
thr = 3.0
up, dn = find_events(V2, thr, thr)
r_up = event_response(T, up, pre=6, post=30).mean(0)
r_dn = event_response(T, dn, pre=6, post=30).mean(0)
ra_up = trend_adjusted(T, up).mean(0)
ra_dn = trend_adjusted(T, dn).mean(0)
t = np.arange(len(r_up)) * dt
print(f"\n  {'time':>6} | {'开阀原始':>10} {'关阀原始':>10} | {'开阀校正':>10} {'关阀校正':>10}")
for i in [0, 3, 6, 9, 12, 15, 20, 25, 30]:
    print(f"  {t[i]:5.0f}s | {r_up[i]:+10.3f} {r_dn[i]:+10.3f} | {ra_up[i]:+10.3f} {ra_dn[i]:+10.3f}")

print("\n" + "="*100)
print("4. 事件工况分类 — 关阀发生在温度上升段还是下降段? (共因方向)")
print("="*100)
for thr in [3.0, 5.0]:
    up, dn = find_events(V2, thr, thr)
    for name, evs in [('开阀', up), ('关阀', dn)]:
        if len(evs) < 10: continue
        # 事件前 30s 温度趋势
        slopes = []
        for i in evs:
            if i - 3 < 0: continue
            seg = T[i-3:i]
            slopes.append((seg[-1] - seg[0]))
        slopes = np.array(slopes)
        rising = (slopes > 0).mean()
        print(f"  [{name}±{thr}%] n={len(evs)}: 事件前30s温度上升中 {rising*100:.0f}% / 下降中 {(1-rising)*100:.0f}%"
              f" (平均斜率 {slopes.mean():+.4f}°C/步)")

print("\n" + "="*100)
print("5. 一级阀开/关 对照 (间接路径)")
print("="*100)
for thr in [3.0, 5.0]:
    up, dn = find_events(V1, thr, thr)
    r_up = event_response(T, up, pre=6, post=30)
    r_dn = event_response(T, dn, pre=6, post=30)
    if len(r_up) > 10:
        m = r_up.mean(0)
        print(f"  [一级开阀±{thr}%] n={len(up)}: t0={m[0]:+.3f} t60s={m[6]:+.3f} "
              f"t120s={m[12]:+.3f} t300s={m[30]:+.3f}")
    if len(r_dn) > 10:
        m = r_dn.mean(0)
        print(f"  [一级关阀±{thr}%] n={len(dn)}: t0={m[0]:+.3f} t60s={m[6]:+.3f} "
              f"t120s={m[12]:+.3f} t300s={m[30]:+.3f}")
