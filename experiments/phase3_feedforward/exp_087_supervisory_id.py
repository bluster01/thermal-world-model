#!/usr/bin/env python3
"""
exp_087_supervisory_id.py — 现场串级结构辨识 (监督模式支撑, 2026-08-04)
========================================================================
目标: 确立监督模式虚拟世界所需的现场结构事实 — MPC输出SP(idx36)后的真实执行链:
  链1: SP(idx36 二级减温调节阀设定) → 阀位(idx39 二级减温调节门阀位)
  链2: 阀位(idx39) → 导前温度(idx29 二级减温器出口温度)
  链3: 导前温度(idx29) → 出口汽温(idx30 末级过热器出口汽温)
  链4: 副调设定值(idx31 二级减温中间设定值) 与 SP(idx36) 的关系 (现场副回路角色)

方法: 事件研究 (SP/阀位阶跃事件 → 对齐 → 平均响应曲线) + 时滞峰值互相关
  - 事件定义: |Δx| > 阈值 且 事件间隔 ≥ 30 步 (避免重叠污染)
  - 对齐窗口: onset−20 .. onset+60 步 (10s/步)
  - 输出: 各链 时标(响应起始)/方向/幅度 表 + 响应曲线图 (全英文标签)
教训应用: 采样相对onset; 不急着下结论, 先看曲线形状; 列索引用列名查找防错位

用法: python exp_087_supervisory_id.py [--smoke]
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
raw = E.data_all  # 全段 [N, 40]
cols = E.NUMERIC_COLS

def cidx(name):
    return cols.index(name)

# ===== 列定位 (列名查找, 防索引错位) =====
I_SP   = cidx('二级减温调节阀设定')      # 监督层输出对象
I_V2   = cidx('二级减温调节门阀位')      # 二级减温阀 (主执行器)
I_LEAD = cidx('二级减温器出口温度')      # 导前温度 (副变量)
I_T    = cidx('末级过热器出口汽温')      # 出口汽温 (主变量)
I_MID  = cidx('二级减温中间设定值')      # 现场副回路 SP
I_V1   = cidx('一级减温调节门阀位')
print(f"[cols] SP={I_SP} V2={I_V2} LEAD={I_LEAD} T={I_T} MID={I_MID}")

N = len(raw)
THR_SP = 0.5    # SP 阶跃阈值 (°C)
THR_V  = 1.5    # 阀位阶跃阈值 (%)
GAP    = 30     # 事件最小间隔 (步)
PRE, POST = 20, 60

def find_events(x, thr, gap=GAP):
    """|Δx|>thr 的事件 onset 列表 (间隔≥gap)"""
    d = np.abs(np.diff(x))
    idxs = np.where(d > thr)[0] + 1
    onsets = []
    for i in idxs:
        if not onsets or i - onsets[-1] >= gap:
            onsets.append(i)
    return onsets

def event_response(x_src, x_resp, onsets, pre=PRE, post=POST):
    """事件对齐平均响应: 返回 (t_axis, mean, std, n) 相对 onset"""
    segs = []
    for o in onsets:
        if o - pre < 0 or o + post >= N:
            continue
        base = x_resp[o - 1]              # onset 前一刻值
        segs.append(x_resp[o:o + post] - base)
    if not segs:
        return None
    m = np.stack(segs)
    return m.mean(0), m.std(0) / np.sqrt(len(segs)), len(segs)

def first_rise(mean, resp_idx, thr_frac=0.3, min_idx=2):
    """响应起始步: 超过峰值 30% 的第一个索引 (跳过 onset 后 1 步, 防伪影)"""
    pk = np.abs(mean).max()
    if pk < 1e-9:
        return None
    for j in range(min_idx, len(mean)):
        if np.abs(mean[j]) >= thr_frac * pk:
            return j
    return None

# ===== 事件检测 =====
sp_ev = find_events(raw[:, I_SP], THR_SP)
v2_ev = find_events(raw[:, I_V2], THR_V)
print(f"[events] SP 阶跃 {len(sp_ev)} 个 | V2 阶跃 {len(v2_ev)} 个")
if SMOKE:
    sp_ev, v2_ev = sp_ev[:8], v2_ev[:8]

results = {}
t_ax = np.arange(POST) * 10  # s

# ===== 链1: SP → 阀位 =====
r = event_response(raw[:, I_SP], raw[:, I_V2], sp_ev)
if r:
    m, se, n = r
    j = first_rise(m, None)
    results['SP->V2'] = dict(n=n, onset_step=j, onset_s=j * 10 if j else None,
                             dir=np.sign(m[j] if j else 0),
                             amp=float(m[j]) if j else 0.0, peak=float(m.max()), peak_s=int(m.argmax()) * 10)

# ===== 链2: 阀位 → 导前温度 (纯阀位事件, 同方向) =====
r = event_response(raw[:, I_V2], raw[:, I_LEAD], v2_ev)
if r:
    m, se, n = r
    j = first_rise(m, None)
    results['V2->LEAD'] = dict(n=n, onset_step=j, onset_s=j * 10 if j else None,
                               dir=np.sign(m[j] if j else 0),
                               amp=float(m[j]) if j else 0.0, peak=float(m.max()), peak_s=int(m.argmax()) * 10)

# ===== 链3: 导前温度 → 出口汽温 (用 SP 事件, 导前温度先动) =====
r = event_response(raw[:, I_SP], raw[:, I_T], sp_ev)
if r:
    m, se, n = r
    j = first_rise(m, None)
    results['SP->T'] = dict(n=n, onset_step=j, onset_s=j * 10 if j else None,
                            dir=np.sign(m[j] if j else 0),
                            amp=float(m[j]) if j else 0.0, peak=float(m.max()), peak_s=int(m.argmax()) * 10)

# ===== 链4: SP vs 副调设定值 (MID) 关系 =====
r = event_response(raw[:, I_SP], raw[:, I_MID], sp_ev)
if r:
    m, se, n = r
    j = first_rise(m, None)
    results['SP->MID'] = dict(n=n, onset_step=j, onset_s=j * 10 if j else None,
                              dir=np.sign(m[j] if j else 0),
                              amp=float(m[j]) if j else 0.0, peak=float(m.max()), peak_s=int(m.argmax()) * 10)

# ===== 导前温度领先出口的互相关 (SP 事件窗内) =====
lead_ev = find_events(raw[:, I_LEAD], 1.0)
r_lead = event_response(raw[:, I_LEAD], raw[:, I_T], lead_ev[:200])
r_lead2 = event_response(raw[:, I_LEAD], raw[:, I_LEAD], lead_ev[:200])
if r_lead and r_lead2:
    mT, seT, nT = r_lead
    mL, seL, nL = r_lead2
    jT = first_rise(mT, None); jL = first_rise(mL, None)
    results['LEAD->T'] = dict(n=nT, lead_onset_step=jL, t_onset_step=jT,
                              lead_onset_s=jL * 10 if jL else None, t_onset_s=jT * 10 if jT else None,
                              lead_advance=(jT - jL) * 10 if (jT and jL) else None)

print("\n===== 串级结构辨识结果 =====")
for k, v in results.items():
    print(f"  {k}: {v}")

# ===== 短程细节 (前 15 步 = 150s) =====
print("\n===== 短程响应 (前 15 步, 10s/步) =====")
for name, resp in [('SP->V2', event_response(raw[:, I_SP], raw[:, I_V2], sp_ev)),
                   ('V2->LEAD', event_response(raw[:, I_V2], raw[:, I_LEAD], v2_ev)),
                   ('SP->T', event_response(raw[:, I_SP], raw[:, I_T], sp_ev)),
                   ('SP->MID', event_response(raw[:, I_SP], raw[:, I_MID], sp_ev))]:
    if resp:
        m, se, n = resp
        row = ' '.join(f'{v:+.3f}' for v in m[:15])
        print(f"  {name} (n={n}): {row}")

# ===== 图: 三链响应曲线 =====
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
def plot_ax(ax, resp, title, ylab, ref=None, ref_lab=None):
    m, se, n = resp
    ax.plot(t_ax, m, 'o-', ms=3, lw=1.5, label=f'{n} events (mean±SE)')
    ax.fill_between(t_ax, m - 1.96 * se, m + 1.96 * se, alpha=0.15)
    if ref is not None:
        ax.plot(t_ax, ref, '--', lw=1.2, label=ref_lab)
    ax.axhline(0, color='gray', lw=0.7)
    ax.set_title(title); ax.set_xlabel('Time since onset (s)'); ax.set_ylabel(ylab)
    ax.legend(fontsize=8)

r1 = event_response(raw[:, I_SP], raw[:, I_V2], sp_ev)
r2 = event_response(raw[:, I_V2], raw[:, I_LEAD], v2_ev)
r3 = event_response(raw[:, I_SP], raw[:, I_T], sp_ev)
r4 = event_response(raw[:, I_SP], raw[:, I_MID], sp_ev)
if r1: plot_ax(axes[0, 0], r1, '(a) SP step -> valve V2', 'Valve V2 Δ (%)')
if r2: plot_ax(axes[0, 1], r2, '(b) Valve V2 step -> lead temp (desuperheater out)', 'Lead temp Δ (°C)')
if r3: plot_ax(axes[1, 0], r3, '(c) SP step -> outlet steam temp', 'Outlet temp Δ (°C)')
if r4: plot_ax(axes[1, 1], r4, '(d) SP step -> inner-loop SP (MID)', 'MID Δ (°C)')
fig.tight_layout()
fig.savefig('figures/fig_supervisory_cascade_id.png', dpi=170, bbox_inches='tight')
print('\nSaved: figures/fig_supervisory_cascade_id.png')
