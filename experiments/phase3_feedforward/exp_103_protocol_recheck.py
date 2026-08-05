#!/usr/bin/env python3
"""
exp_103_protocol_recheck.py — 修正动作编码后重测已有 checkpoint (2026-08-05)
============================================================================
设计稿: docs/causal_eval_framework.md  L0 + 重测

背景 (决定性 bug):
  第40列 raw41[:, I_DSP] = np.diff(SP) 本身已是一阶差分。
    训练: A = train_raw[i+W : i+W+H, I_DSP]              → 一阶差分 ✅
    评测: a = np.diff(raw41[s+W-1 : s+W+H, I_DSP])        → 二阶差分 ❌
  一次 +2 阶跃: 训练看到 [2,0,0,...] (净变+2), 评测喂入 [2,-2,0,...] (净变0, 偶极子)。
  → exp_097/098/100/101/102 全部动作增益数字在自相消动作下测得, 作废。

本脚本 (无需重训, 仅推理):
  Part A  P0.2 往返一致性自检 (断言 build_action 与训练取法逐元素相同)
  Part B  bug 复现: 展示旧/新动作序列差异 + 净变量对比
  Part C  用修正编码重测 4 个 ckpt, 新旧口径并列, 判断结论是否翻转
用法: python exp_103_protocol_recheck.py [--models M5DSP,M7DSP,M9DSP60,M9DSP18]
"""
import os, sys, json
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

import causal_eval as CE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E.cfg.WINDOW_SIZE
n_train, n_val_end = 495407, 601566
OUT = 'results/exp_103_protocol_recheck'
os.makedirs(OUT, exist_ok=True)

raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40
train_raw = raw41[:n_train]
print(f"[data] N={N} | I_SP={I_SP} I_T={I_T} I_DSP={I_DSP}")


# ===================================================== 模型注册表
def _mk_directwm(H, probabilistic, beta_mode):
    """DirectWM 子类: 动作通道 1 维 ΔSP (action_enc 输入 H 而非 H*2)"""
    E.H_OUT = H

    class _M(E.DirectWM):
        def __init__(self):
            super().__init__(use_action=True, use_patch=True, per_variable=True,
                             use_varattn=True, probabilistic=probabilistic,
                             beta_mode=beta_mode)
            self.action_enc = nn.Sequential(
                nn.Linear(H, E.cfg.D_MODEL * 2), nn.GELU(), nn.Dropout(E.cfg.DROPOUT))

        def forward(self, x, a=None):
            mu, lv = super().forward(x, a)
            if lv is not None:
                lv = torch.clamp(lv, -6., 20.)
            return mu, lv
    return _M()


def _mk_timexer(H):
    E.H_OUT = H

    class _M(E.TimeXerWM):
        def __init__(self):
            super().__init__(probabilistic=True, beta_mode='fixed')
            self.act_lin = nn.Linear(H, E.cfg.D_MODEL)

        def forward(self, x, a=None):
            mu, lv = super().forward(x, a)
            if lv is not None:
                lv = torch.clamp(lv, -6., 20.)
            return mu, lv
    return _M()


# name -> (H, ckpt_path, builder, 旧口径参考)
REGISTRY = {
    'M5DSP':   (18, 'results/exp_096_dsp_wm/checkpoints/best_model.pth',
                lambda: _mk_directwm(18, False, 'warmup'),
                {'src': 'exp_097', 'resp_180s': 0.050, 'dir_180s': 75.0, 'mae_18': 0.301}),
    'M7DSP':   (60, 'results/exp_100_m7dsp_h60/checkpoints/best_model.pth',
                lambda: _mk_directwm(60, True, 'fixed'),
                {'src': 'exp_100', 'resp_600s': 0.212, 'dir_600s': 44.8, 'mae_18': 0.398}),
    'M9DSP60': (60, 'results/exp_101_m9dsp_h60/checkpoints/best_model.pth',
                lambda: _mk_timexer(60),
                {'src': 'exp_101', 'resp_600s': 0.170, 'dir_600s': 41.0,
                 'resp_180s': 0.106, 'dir_180s': 88.8, 'mae_18': 0.361}),
    'M9DSP18': (18, 'results/exp_102_m9dsp_h18/checkpoints/best_model.pth',
                lambda: _mk_timexer(18),
                {'src': 'exp_102', 'resp_180s': 0.080, 'dir_180s': 64.9, 'mae_18': 0.349}),
    'M5DSP_DO': (18, 'results/exp_098_dsp_dropout/checkpoints/best_model.pth',
                 lambda: _mk_directwm(18, False, 'warmup'),
                 {'src': 'exp_098', 'resp_180s': 0.060, 'dir_180s': 67.0}),
}

PROFILE = {18: [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s')],
           60: [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s'),
                (29, '300s'), (41, '420s'), (59, '600s')]}


# ===================================================== Part A: P0.2 往返一致性
print("\n" + "=" * 70)
print("Part A — P0.2 往返一致性自检 (build_action vs 训练取法)")
print("=" * 70)
for H in (18, 60):
    CE.assert_train_eval_identity(train_raw, raw41, W, H, I_DSP, n_probe=300)
    print(f"  H={H:2d}  PASS  (300 个随机样本逐元素相同)")


# ===================================================== Part B: bug 复现
print("\n" + "=" * 70)
print("Part B — bug 复现: 旧(二阶差分) vs 新(一阶差分) 动作序列")
print("=" * 70)


def action_old(s, H):
    """已废弃的错误写法, 仅用于对比"""
    return np.diff(raw41[s + W - 1: s + W + H, I_DSP]).astype(np.float32)


events, dsp_vals = CE.select_events(raw, I_SP, I_LD, H=60)
print(f"  事件数 n={len(events)} (与 exp_100/101/102 同协议)")

o_demo = events[len(events) // 2]
s_demo = o_demo - W
a_new = CE.build_action(raw41, s_demo, W, 18, I_DSP)
a_old = action_old(s_demo, 18)
print(f"\n  样例 onset={o_demo}  ΔSP={raw[o_demo, I_SP] - raw[o_demo - 1, I_SP]:+.2f}")
print(f"    新(一阶) 前6步 {np.round(a_new[:6], 3)}  | 累积净变 {a_new.sum():+.3f}")
print(f"    旧(二阶) 前6步 {np.round(a_old[:6], 3)}  | 累积净变 {a_old.sum():+.3f}")

net_new = np.array([CE.build_action(raw41, o - W, W, 18, I_DSP).sum() for o in events])
net_old = np.array([action_old(o - W, 18) .sum() for o in events])
print(f"\n  全部 {len(events)} 事件的 18 步累积 ΔSP:")
print(f"    新: |净变| 均值 {np.abs(net_new).mean():.3f}  与真实ΔSP相关 {np.corrcoef(net_new, dsp_vals)[0,1]:+.3f}")
print(f"    旧: |净变| 均值 {np.abs(net_old).mean():.3f}  与真实ΔSP相关 {np.corrcoef(net_old, dsp_vals)[0,1]:+.3f}")
print(f"  → 旧口径下动作输入与真实干预幅度的相关性被破坏, 动作增益必然被低估")


# ===================================================== Part C: 重测
print("\n" + "=" * 70)
print("Part C — 修正编码重测 (新旧口径并列, 无需重训)")
print("=" * 70)

want = None
for i, a in enumerate(sys.argv):
    if a == '--models' and i + 1 < len(sys.argv):
        want = [x.strip() for x in sys.argv[i + 1].split(',')]
names = want or list(REGISTRY.keys())

results = {}
for name in names:
    if name not in REGISTRY:
        print(f"\n[skip] {name}: 未注册")
        continue
    H, ckpt, builder, ref = REGISTRY[name]
    if not os.path.exists(ckpt):
        print(f"\n[skip] {name}: ckpt 不存在 {ckpt}")
        continue
    print(f"\n--- {name} (H={H}, 旧口径来源 {ref['src']}) ---")
    model = builder().to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(sd['model_state_dict'], strict=False)
    if missing or unexpected:
        print(f"  [warn] state_dict 不完全匹配 missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"         missing 示例: {list(missing)[:3]}")
    wrap = CE.ModelWrapper(model, raw, raw41, W, H, I_DSP, DEVICE)

    ev, dv = CE.select_events(raw, I_SP, I_LD, H=H)
    m_new, p_real, keep = CE.model_response(wrap, ev, dv)
    ev_k = np.array(ev)[keep]
    dv_k = dv[keep]

    # 旧口径复现 (同一 ckpt, 仅动作编码不同) —— 用于确认差异确实来自 bug
    m_old = []
    for o in ev_k:
        s = o - W
        p1 = wrap.predict(s, action_old(s, H))
        p0 = wrap.predict(s, 0.0)
        if p1 is not None and p0 is not None:
            m_old.append((p1 - p0) / (raw[o, I_SP] - raw[o - 1, I_SP]))
    m_old = np.array(m_old, dtype=np.float32)

    # 预测精度 (不受动作编码影响的部分也一并回归)
    mae_all, mae_18, dir_end = [], [], []
    for o, p in zip(ev_k, p_real):
        act = raw[o:o + H, I_T]
        mae_all.append(np.abs(p - act).mean())
        mae_18.append(np.abs(p[:18] - act[:18]).mean())
        dir_end.append(np.sign(p[-1] - raw[o - 1, I_T]) == np.sign(act[-1] - raw[o - 1, I_T]))
    mae_all = float(np.mean(mae_all)); mae_18 = float(np.mean(mae_18))
    dir_end = float(np.mean(dir_end))
    print(f"  n={len(ev_k)} | MAE(全{H}步) {mae_all:.3f} | MAE(前18步) {mae_18:.3f} "
          f"| 末点方向 {dir_end*100:.0f}%  [旧参考 mae_18={ref.get('mae_18', float('nan')):.3f}]")

    # 动作增益剖面: 新 vs 旧
    # 注意此处 dir 仍是 sign(ΔSP) 口径 (与旧数字可比); DiD 真值口径见 exp_104
    print(f"  {'时刻':>6} | {'新响应':>9} {'新方向':>7} | {'旧响应':>9} {'旧方向':>7} | {'响应倍数':>8}")
    prof = {}
    for k, lab in PROFILE[H]:
        rn, ro = m_new[:, k] * dv_k, m_old[:, k] * dv_k     # 还原为 °C (未除 ΔSP)
        dn = float((np.sign(rn) == np.sign(dv_k)).mean())
        do = float((np.sign(ro) == np.sign(dv_k)).mean())
        an, ao = float(np.abs(rn).mean()), float(np.abs(ro).mean())
        ratio = an / ao if ao > 1e-9 else float('inf')
        print(f"  {lab:>6} | {an:8.4f}°C {dn*100:6.0f}% | {ao:8.4f}°C {do*100:6.0f}% | {ratio:7.2f}x")
        prof[lab] = dict(k=int(k), resp_new=an, dir_new=dn, resp_old=ao, dir_old=do, ratio=ratio)

    # 归一化增益 (°C per °C of ΔSP) — 与物理基准可比的口径
    print(f"  归一化增益 (°C/°C ΔSP, 新口径):")
    for k, lab in PROFILE[H]:
        print(f"    {lab:>6} {m_new[:, k].mean():+.4f}   (物理参考: 180s≈0.17, 600s≈0.97)")

    results[name] = dict(H=H, n=int(len(ev_k)), mae_all=mae_all, mae_18=mae_18,
                         dir_end=dir_end, profile=prof, ref_old=ref,
                         gain_norm={lab: float(m_new[:, k].mean()) for k, lab in PROFILE[H]})
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

with open(f'{OUT}/recheck.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT}/recheck.json")

# ===================================================== 判决
print("\n" + "=" * 70)
print("判决 (是否推翻 exp_100-102 的 FAIL 结论)")
print("=" * 70)
for name, r in results.items():
    H = r['H']
    key = '600s' if H == 60 else '180s'
    p = r['profile'][key]
    flip = p['ratio'] > 1.5 or (p['dir_new'] - p['dir_old']) > 0.10
    tag = '结论可能翻转' if flip else '结论不变'
    print(f"  {name:9s} {key} 响应 {p['resp_old']:.3f}→{p['resp_new']:.3f}°C ({p['ratio']:.2f}x) | "
          f"方向 {p['dir_old']*100:.0f}%→{p['dir_new']*100:.0f}% | {tag}")
print("\n下一步: 无论翻转与否, 都需 exp_104 建立 DiD 真值 —")
print("        当前方向口径仍是 sign(ΔSP), 而 exp_099 显示 180s 物理响应比例中位仅 0.17。")
