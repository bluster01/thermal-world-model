#!/usr/bin/env python3
"""
exp_097_sandbox_eval.py — 沙盒 vs 现场预测精度 + ΔSP 通道消融 (2026-08-05)
=========================================================================
设计稿 phase3_sandbox_design.md 实验 2+3:
  实验2: 世界模型作为 SP 规划沙盒 — 输入实际 SP/阀位轨迹, 预测温度 vs 现场实际
  实验3: M5 (动作=阀位) vs M5-DSP (动作=ΔSP) 消融 — ΔSP 通道信息无损验证
协议 (同 exp_093/095):
  - 557 个 SP 阶跃事件 (|ΔSP|>1°C, 间隔≥60步, 工况稳定, 阶跃后保持)
  - 起点: onset−96 (窗口末=阶跃前) + 提前 90s 起点 (现场前馈视角)
  - 沙盒输入 = 现场实际执行序列 (条件预测, 非真闭环)
指标: 180s MAE/RMSE/p90/方向正确率 × 分层 (大|ΔSP|>3 / 中2-3 / 小1-2 / 平稳)
判定: 实验2 PASS RMSE≤0.4 且方向≥80%; 实验3 Δ(MAE) ≤ +0.05 信息无损
用法: python exp_097_sandbox_eval.py [--smoke]
"""
import os, sys, json
import numpy as np
import torch, torch.nn as nn
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E.cfg.WINDOW_SIZE          # 96
H_OUT = E.H_OUT                # 18
raw = E.data_all               # [N, 40] 物理值
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_V1 = E.NUMERIC_COLS.index('一级减温调节门阀位')
I_V2 = E.NUMERIC_COLS.index('二级减温调节门阀位')
I_LD = E.NUMERIC_COLS.index('机组负荷')

# ΔSP 列 (第 41 列): 动作通道
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40

# ===== 模型 =====
class M5DSP(E.DirectWM):
    """M5-DSP: 动作=1维ΔSP (同 exp_096 定义, 独立声明避免 import 副作用)"""
    def __init__(self):
        super().__init__(use_action=True, use_patch=True, per_variable=True,
                         use_varattn=True, probabilistic=False)
        d = E.cfg.D_MODEL
        self.action_enc = nn.Sequential(
            nn.Linear(H_OUT, d * 2), nn.GELU(), nn.Dropout(E.cfg.DROPOUT))

ck_m5 = torch.load('results/exp_025_M5/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
m5 = E.build_model('M5').to(DEVICE).eval()
m5.load_state_dict(ck_m5['model_state_dict'])
print('[load] M5 OK (exp_025_M5)')

ck_dsp = torch.load('results/exp_096_dsp_wm/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
m5dsp = M5DSP().to(DEVICE).eval()
m5dsp.load_state_dict(ck_dsp['model_state_dict'])
print(f"[load] M5-DSP OK (exp_096, ep{ck_dsp['epoch']})")

# ===== 事件筛选 (同 exp_095) =====
dsp_abs = np.abs(np.diff(raw[:, I_SP]))
onsets = []
for i in np.where(dsp_abs > 1.0)[0] + 1:
    if not onsets or i - onsets[-1] >= 60:
        onsets.append(i)
stable = [o for o in onsets if o + 60 < N and
          np.abs(np.diff(raw[max(0, o-20):min(N, o+20), I_LD])).max() <= 3.0]
kept = [o for o in stable if np.abs(raw[o:o+61, I_SP] - raw[o, I_SP]).max() <= 0.3]
events = kept[:200] if SMOKE else kept
print(f"[events] SP阶跃+稳定+SP保持: {len(events)}")

# 平稳基线 (随机非阶跃窗口)
rng = np.random.default_rng(42)
n_ev = len(events)
calm = []
for _ in range(n_ev):
    while True:
        c = int(rng.integers(W + 60, N - 60))
        if np.abs(np.diff(raw[c-20:c+20, I_SP])).max() <= 0.15 and c not in events:
            calm.append(c); break
print(f"[events] 平稳基线: {len(calm)}")

# ===== 评测 =====
def predict(model, s, kind):
    """窗口末=s−1, 未来 H 步动作用现场实际轨迹. kind='m5'|'dsp'"""
    if s < 0 or s + W + H_OUT >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)      # [1, W, 40]
    with torch.no_grad():
        if kind == 'm5':
            a = raw[s+W:s+W+H_OUT, I_V1:I_V2+1]                       # [H, 2] 实际阀位
            if len(a) < H_OUT:
                a = np.pad(a, ((0, H_OUT-len(a)), (0, 0)), mode='edge')
            a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)      # [1, 36]
            mu, _ = model(win, a_f)
        else:
            a = np.diff(raw41[s+W-1:s+W+H_OUT, I_DSP])               # [H] 实际 ΔSP
            a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)      # [1, 18]
            mu, _ = model(win, a_f)
    return mu[0].cpu().numpy()  # [H]

def eval_event(o, lead):
    """返回 (m5_pred, dsp_pred) 或 None"""
    s = o - W - lead
    if s < 0 or s + W + H_OUT >= N:
        return None
    p_m5 = predict(m5, s, 'm5')
    p_dsp = predict(m5dsp, s, 'dsp')
    if p_m5 is None or p_dsp is None:
        return None
    return p_m5, p_dsp

def metrics(preds, actuals, prev_T):
    """preds/actuals: [n, H], prev_T: [n] 起点前温度. 返回 dict"""
    maes = np.abs(preds - actuals).mean(1)
    rmses = np.sqrt(((preds - actuals) ** 2).mean(1))
    d_act = actuals[:, -1] - prev_T
    d_pred = preds[:, -1] - prev_T
    dir_ok = (np.sign(d_pred) == np.sign(d_act)) * 1.0
    return dict(mae=maes, rmse=rmses, dir=dir_ok)

def report(name, m):
    print(f"  {name:28s} n={len(m['mae']):4d} | MAE {m['mae'].mean():.3f} (med {np.median(m['mae']):.3f}, p90 {np.percentile(m['mae'],90):.3f}) | RMSE {m['rmse'].mean():.3f} | 方向 {m['dir'].mean()*100:.0f}%")

results = {'events': [], 'calm': []}
for o in events:
    r = eval_event(o, 0)
    if r is None: continue
    p_m5, p_dsp = r
    actual = raw[o:o+H_OUT, I_T]
    prev_T = raw[o-1, I_T]
    ds_step = raw[o, I_SP] - raw[o-1, I_SP]
    results['events'].append(dict(
        onset=o, mag=ds_step,
        m5=metrics(p_m5[None], actual[None], np.array([prev_T])),
        dsp=metrics(p_dsp[None], actual[None], np.array([prev_T]))))
for c in calm:
    p_m5 = predict(m5, c - W, 'm5')
    p_dsp = predict(m5dsp, c - W, 'dsp')
    if p_m5 is None or p_dsp is None: continue
    actual = raw[c:c+H_OUT, I_T]
    prev_T = raw[c-1, I_T]
    results['calm'].append(dict(
        onset=c, mag=0.0,
        m5=metrics(p_m5[None], actual[None], np.array([prev_T])),
        dsp=metrics(p_dsp[None], actual[None], np.array([prev_T]))))

# ===== 汇总 =====
def collect(rows, key, split=None):
    if split is None:
        return {k: np.concatenate([r[key][k] for r in rows]) for k in ('mae', 'rmse', 'dir')}
    return {k: np.concatenate([r[key][k] for r in rows if split(r)]) for k in ('mae', 'rmse', 'dir')}

ev, ca = results['events'], results['calm']
print(f"\n===== 实验2: 沙盒 vs 现场预测精度 (180s 预测, onset 起点) =====")
for key, nm in (('m5', 'M5 (act=valve)'), ('dsp', 'M5-DSP (act=ΔSP)')):
    report(f'{nm} [全事件]', collect(ev, key))
    for lo, hi, lab in ((3, 99, '大动作 |ΔSP|>3'), (2, 3, '中动作 2-3'), (1, 2, '小动作 1-2')):
        m = collect(ev, key, lambda r, lo=lo, hi=hi: lo < abs(r['mag']) <= hi)
        if len(m['mae']) > 0:
            report(f'{nm} [{lab}]', m)
    report(f'{nm} [平稳基线]', collect(ca, key))

print(f"\n===== 实验3: ΔSP 通道消融 (M5-DSP vs M5) =====")
for src, lab, spl in (
        (ev, '全事件', None),
        (ev, '大动作', lambda r: 3 < abs(r['mag'])),
        (ev, '中动作', lambda r: 2 < abs(r['mag']) <= 3),
        (ev, '小动作', lambda r: 1 < abs(r['mag']) <= 2),
        (ca, '平稳基线', None)):
    m5m = collect(src, 'm5', spl)
    dspm = collect(src, 'dsp', spl)
    dm = dspm['mae'].mean() - m5m['mae'].mean()
    print(f"  {lab:10s} n={len(m5m['mae']):4d} | M5 MAE {m5m['mae'].mean():.3f} | M5-DSP {dspm['mae'].mean():.3f} | Δ {dm:+.3f} {'✓≤+0.05' if dm <= 0.05 else '✗>+0.05'} | 方向 M5 {m5m['dir'].mean()*100:.0f}% vs DSP {dspm['dir'].mean()*100:.0f}%")

# 提前 90s 起点 (现场前馈视角)
ev_lead = []
for o in events:
    r = eval_event(o, 9)
    if r is None: continue
    p_m5, p_dsp = r
    actual = raw[o:o+H_OUT, I_T]
    prev_T = raw[o-1, I_T]
    ev_lead.append((metrics(p_m5[None], actual[None], np.array([prev_T])),
                    metrics(p_dsp[None], actual[None], np.array([prev_T]))))
if ev_lead:
    m5l = {k: np.concatenate([a[0][k] for a in ev_lead]) for k in ('mae', 'rmse', 'dir')}
    dspl = {k: np.concatenate([a[1][k] for a in ev_lead]) for k in ('mae', 'rmse', 'dir')}
    print(f"\n[提前90s起点] M5 MAE {m5l['mae'].mean():.3f} 方向 {m5l['dir'].mean()*100:.0f}% | M5-DSP MAE {dspl['mae'].mean():.3f} 方向 {dspl['dir'].mean()*100:.0f}%")

# ===== 判定 (设计稿) =====
all_m5 = collect(ev, 'm5'); all_dsp = collect(ev, 'dsp')
ok2 = all_m5['rmse'].mean() <= 0.4 and all_m5['dir'].mean() >= 0.8
ok2d = all_dsp['rmse'].mean() <= 0.4 and all_dsp['dir'].mean() >= 0.8
ok3 = (all_dsp['mae'].mean() - all_m5['mae'].mean()) <= 0.05
print(f"\n[判定] 实验2 (M5 沙盒): RMSE {all_m5['rmse'].mean():.3f} 方向 {all_m5['dir'].mean()*100:.0f}% → {'PASS' if ok2 else 'FAIL'}")
print(f"[判定] 实验2 (M5-DSP 沙盒): RMSE {all_dsp['rmse'].mean():.3f} 方向 {all_dsp['dir'].mean()*100:.0f}% → {'PASS' if ok2d else 'FAIL'}")
print(f"[判定] 实验3 (ΔSP 信息无损): ΔMAE {all_dsp['mae'].mean()-all_m5['mae'].mean():+.3f} → {'PASS' if ok3 else 'FAIL'}")

# ===== 留痕 json =====
os.makedirs('results/exp_097_sandbox_eval', exist_ok=True)
ser = {'n_events': len(ev), 'n_calm': len(ca)}
for key, nm in (('m5', 'M5'), ('dsp', 'M5-DSP')):
    ser[nm] = {'all_mae': float(collect(ev, key)['mae'].mean()),
               'all_rmse': float(collect(ev, key)['rmse'].mean()),
               'all_dir': float(collect(ev, key)['dir'].mean()),
               'calm_mae': float(collect(ca, key)['mae'].mean())}
    for lo, hi, lab in ((3, 99, 'large'), (2, 3, 'mid'), (1, 2, 'small')):
        m = collect(ev, key, lambda r, lo=lo, hi=hi: lo < abs(r['mag']) <= hi)
        ser[nm][f'{lab}_mae'] = float(m['mae'].mean())
        ser[nm][f'{lab}_dir'] = float(m['dir'].mean())
with open('results/exp_097_sandbox_eval/metrics.json', 'w') as f:
    json.dump(ser, f, indent=2)
print('\nSaved: results/exp_097_sandbox_eval/metrics.json')

# ===== 图 =====
os.makedirs('figures', exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
# (a) 误差分布
for key, c, lab in (('m5', '#4f81bd', 'M5'), ('dsp', '#c0504d', 'M5-DSP')):
    m = collect(ev, key)
    axes[0].hist(m['mae'], bins=30, alpha=0.6, color=c, label=f'{lab} (n={len(m["mae"])})')
axes[0].axvline(collect(ev, 'm5')['mae'].mean(), color='#4f81bd', ls='--', lw=1.2)
axes[0].axvline(collect(ev, 'dsp')['mae'].mean(), color='#c0504d', ls='--', lw=1.2)
axes[0].set_xlabel('180s prediction MAE (°C)'); axes[0].set_ylabel('Count')
axes[0].set_title('(a) Error distribution, SP-step events'); axes[0].legend(fontsize=8)
# (b) 分层对比
labs = ['All', 'Large\n|ΔSP|>3', 'Mid\n2-3', 'Small\n1-2', 'Calm']
m5v, dspv = [], []
for lo, hi in ((0, 99), (3, 99), (2, 3), (1, 2), (0, 1)):
    src = ev if lo > 0 else ca
    spl = (lambda r, lo=lo, hi=hi: lo < abs(r['mag']) <= hi) if lo > 0 else None
    m5v.append(collect(src, 'm5', spl)['mae'].mean())
    dspv.append(collect(src, 'dsp', spl)['mae'].mean())
x = np.arange(len(labs)); wdt = 0.36
axes[1].bar(x - wdt/2, m5v, wdt, color='#4f81bd', label='M5 (act=valve)')
axes[1].bar(x + wdt/2, dspv, wdt, color='#c0504d', label='M5-DSP (act=ΔSP)')
for xi, (a, b) in enumerate(zip(m5v, dspv)):
    axes[1].text(xi - wdt/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=8)
    axes[1].text(xi + wdt/2, b + 0.01, f'{b:.3f}', ha='center', fontsize=8)
axes[1].set_xticks(x); axes[1].set_xticklabels(labs, fontsize=8)
axes[1].set_ylabel('Mean prediction MAE (°C)')
axes[1].set_title('(b) MAE by action magnitude'); axes[1].legend(fontsize=8)
# (c) 方向正确率分层
m5dir, dspdir = [], []
for lo, hi in ((0, 99), (3, 99), (2, 3), (1, 2)):
    spl = lambda r, lo=lo, hi=hi: lo < abs(r['mag']) <= hi
    m5dir.append(collect(ev, 'm5', spl)['dir'].mean() * 100)
    dspdir.append(collect(ev, 'dsp', spl)['dir'].mean() * 100)
x = np.arange(4)
axes[2].bar(x - wdt/2, m5dir, wdt, color='#4f81bd', label='M5')
axes[2].bar(x + wdt/2, dspdir, wdt, color='#c0504d', label='M5-DSP')
axes[2].axhline(80, color='gray', ls=':', lw=1)
for xi, (a, b) in enumerate(zip(m5dir, dspdir)):
    axes[2].text(xi - wdt/2, a + 1, f'{a:.0f}', ha='center', fontsize=8)
    axes[2].text(xi + wdt/2, b + 1, f'{b:.0f}', ha='center', fontsize=8)
axes[2].set_xticks(x); axes[2].set_xticklabels(['All', 'Large', 'Mid', 'Small'], fontsize=8)
axes[2].set_ylabel('Direction accuracy (%)'); axes[2].set_ylim(0, 110)
axes[2].set_title('(c) Direction accuracy by magnitude'); axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig('figures/fig_sandbox_m5_vs_dsp.png', dpi=170, bbox_inches='tight')
print('Saved: figures/fig_sandbox_m5_vs_dsp.png')

# ===== case 图: 大/中/平稳 代表事件 =====
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 4.6))
pick = [(3, 99, 'Large SP step'), (2, 3, 'Mid SP step'), (0, 1, 'Calm baseline')]
for ax, (lo, hi, title) in zip(axes2, pick):
    pool = [r for r in ev if lo < abs(r['mag']) <= hi] if lo > 0 else ca
    if not pool: continue
    r0 = pool[len(pool) // 2]
    o = r0['onset']
    p_m5, p_dsp = r0['m5']['mae'], r0['dsp']['mae']
    t_ax = np.arange(-W, H_OUT) * 10  # 窗口+预测
    ax.plot(t_ax[:W], raw[o-W:o, I_T], color='gray', lw=0.9, alpha=0.6, label='History')
    ax.plot(np.arange(H_OUT) * 10, raw[o:o+H_OUT, I_T], color='black', lw=1.8, label='Field (actual)')
    s = o - W
    ax.plot(np.arange(H_OUT) * 10, predict(m5, s, 'm5'), color='#4f81bd', lw=1.4, ls='--', label='M5 sandbox')
    ax.plot(np.arange(H_OUT) * 10, predict(m5dsp, s, 'dsp'), color='#c0504d', lw=1.4, ls='--', label='M5-DSP sandbox')
    sp_s = raw[o-1, I_SP]
    ax.plot([-W*10, H_OUT*10], [sp_s, sp_s], color='green', lw=0.7, ls=':', label=f'SP={sp_s:.1f}°C')
    ax.axvline(0, color='orange', lw=0.8)
    ax.set_xlabel('Time since SP step (s)'); ax.set_ylabel('Outlet temp (°C)')
    ax.set_title(f'({title}) |ΔSP|={abs(r0["mag"]):.1f}°C')
    ax.legend(fontsize=7)
fig2.tight_layout()
fig2.savefig('figures/fig_sandbox_cases.png', dpi=170, bbox_inches='tight')
print('Saved: figures/fig_sandbox_cases.png')
