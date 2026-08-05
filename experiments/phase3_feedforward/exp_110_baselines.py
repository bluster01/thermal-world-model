#!/usr/bin/env python3
"""
exp_110_baselines.py — P4: 补基线行 (M5-DSP / M7DSP / M9DSP) + A1phys 对照
=====================================================================
用 P2 DiD 真值 (n_ev=79) + CFI_agg 评测所有模型。
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
import causal_arch as CA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E.cfg.WINDOW_SIZE; N_FEAT = E.N_FEAT; TARGET_IDX = E.TARGET_IDX
n_train, n_val_end = 495407, 601566
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')

raw = E.data_all
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40

gt = np.load('results/cfe_groundtruth_p2/did_response.npz')
PROFILE60 = [(2,'30s'),(5,'60s'),(11,'120s'),(17,'180s'),(29,'300s'),(41,'420s'),(59,'600s')]
PROFILE18 = [(2,'30s'),(5,'60s'),(11,'120s'),(17,'180s')]

# ======================== 旧架构 model wrapper ========================
def build_old_model(name, H):
    """DirectWM / TimeXerWM (exp_025 体系)"""
    E.H_OUT = H
    if 'M5' in name:
        m = E.DirectWM(use_action=True, use_patch=True, per_variable=True,
                       use_varattn=True, probabilistic=False)
        m.action_enc = nn.Sequential(nn.Linear(H, E.cfg.D_MODEL*2), nn.GELU(),
                                     nn.Dropout(E.cfg.DROPOUT))
    elif 'M7' in name:
        m = E.DirectWM(use_action=True, use_patch=True, per_variable=True,
                       use_varattn=True, probabilistic=True, beta_mode='fixed')
        m.action_enc = nn.Sequential(nn.Linear(H, E.cfg.D_MODEL*2), nn.GELU(),
                                     nn.Dropout(E.cfg.DROPOUT))
    elif 'M9' in name:
        m = E.TimeXerWM(probabilistic=True, beta_mode='fixed')
        m.act_lin = nn.Linear(H, E.cfg.D_MODEL)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m.to(DEVICE).eval()

def predict_old(model, s, H, a_override=None):
    """旧架构预测，a_override=None用真实ΔSP"""
    if s < 0 or s + W + H >= len(raw): return None
    x = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    a = a_override if a_override is not None else CE.build_action(raw41, s, W, H, I_DSP)
    if isinstance(model, E.TimeXerWM):
        at = torch.FloatTensor(a).reshape(1, H, 1).to(DEVICE)
    else:
        at = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        mu, _ = model(x, at)
    return mu[0].cpu().numpy()

# ======================== 评测函数 ========================
def eval_model(name, ckpt_path, H, is_new_arch=False):
    if not os.path.exists(ckpt_path):
        return None
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

    if is_new_arch:
        if 'physcs' in name:
            model = CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H, intervention='phys',
                                        cumsum_out=True, probabilistic=True).to(DEVICE).eval()
        elif 'phys' in name:
            model = CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H, intervention='phys',
                                        cumsum_out=False, probabilistic=True).to(DEVICE).eval()
        elif 'mlp' in name:
            model = CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H, intervention='mlp',
                                        cumsum_out=False, probabilistic=True).to(DEVICE).eval()
        else:
            model = CA.TimeXerCausalWM(N_FEAT, TARGET_IDX, H, head_mode='glb',
                                       probabilistic=True).to(DEVICE).eval()
    else:
        model = build_old_model(name, H)
    model.load_state_dict(sd['model_state_dict'])
    wrap = CE.ModelWrapper(model, raw, raw41, W, H, I_DSP, DEVICE)

    # P2 events
    ev, dv = CE.select_events(raw, I_SP, I_LD, H=H, lo=n_train, W=W,
                              thr=0.6, sp_hold=0.5, load_stable=5.0)
    gt_key = H
    gt_onsets = gt[f'onsets{gt_key}']; gt_r = gt[f'r{gt_key}']
    gt_R = gt[f'R_true{gt_key}']; gt_ceil = gt[f'sgn_ceiling{gt_key}']
    gt_gc = gt[f'gain_ceiling{gt_key}']

    onset_to_i = {int(o): i for i, o in enumerate(gt_onsets)}
    m_a, r_a = [], []
    for o in ev:
        s = o - W
        p1 = wrap.predict(s)
        p0 = wrap.predict(s, 0.0)
        if p1 is None or p0 is None: continue
        if int(o) not in onset_to_i: continue
        ds = raw[o, I_SP] - raw[o-1, I_SP]
        if abs(ds) < 1e-8: continue
        m_a.append((p1 - p0) / ds)
        r_a.append(gt_r[onset_to_i[int(o)]])

    if len(m_a) == 0: return None
    m_a = np.array(m_a, dtype=np.float32); r_a = np.array(r_a, dtype=np.float32)
    pk = PROFILE60 if H == 60 else PROFILE18
    cfe = CE.causal_metrics(m_a, r_a, gt_R, gt_ceil, pk, gain_ceiling=gt_gc)
    agg = CE.cfi_agg(cfe, pk)

    # 预测精度 (test set random 200)
    test_raw41 = raw41[n_val_end:]
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(test_raw41) - W - H, size=200)
    errs = []
    for i in idxs:
        p = wrap.predict(i + n_val_end)
        if p is not None:
            errs.append(np.abs(p - test_raw41[i+W:i+W+H, TARGET_IDX]).mean())
    mae = float(np.mean(errs)) if errs else float('nan')

    return dict(name=name, H=H, n_ev=len(m_a), mae=mae,
                cfi_agg=agg['cfi'], gain_mean=agg['gain_mean'],
                gain_span=agg['gain_span'], early_sign_ok=agg['early_sign_ok'],
                shape=cfe['shape_corr'])

# ======================== Run ========================
BASE_NEW = 'results/exp_106_causal_arch'
configs = [
    # 基线 (旧架构)
    ('M5-DSP',  'results/exp_096_dsp_wm/checkpoints/best_model.pth', 18, False),
    ('M7DSP',   'results/exp_100_m7dsp_h60/checkpoints/best_model.pth', 60, False),
    ('M9DSP60', 'results/exp_101_m9dsp_h60/checkpoints/best_model.pth', 60, False),
    # 新架构对照
    ('A1phys (best_causal)', f'{BASE_NEW}/A1phys_s0/checkpoints/best_causal.pth', 60, True),
    ('A1phys (best_mae)',    f'{BASE_NEW}/A1phys_s0/checkpoints/best_mae.pth', 60, True),
    # P3A: freeze variants
    ('A1phys ff10 (best_mae)',   f'{BASE_NEW}/A1phys_s0_ff10/checkpoints/best_mae.pth', 60, True),
    ('A1phys ff10 (best_causal)',f'{BASE_NEW}/A1phys_s0_ff10/checkpoints/best_causal.pth', 60, True),
    ('A1phys ff20 (best_mae)',   f'{BASE_NEW}/A1phys_s0_ff20/checkpoints/best_mae.pth', 60, True),
    ('A1phys ff20 (best_causal)',f'{BASE_NEW}/A1phys_s0_ff20/checkpoints/best_causal.pth', 60, True),
    # P3B: gain calibration
    ('A1phys ff10+lg0.5 (best_mae)',   f'{BASE_NEW}/A1phys_s0_ff10_lg0.5/checkpoints/best_mae.pth', 60, True),
    ('A1phys ff10+lg0.5 (best_causal)',f'{BASE_NEW}/A1phys_s0_ff10_lg0.5/checkpoints/best_causal.pth', 60, True),
    ('A1physcs (best_causal)', f'{BASE_NEW}/A1physcs_s0/checkpoints/best_causal.pth', 60, True),
    ('A1physcs (best_mae)',    f'{BASE_NEW}/A1physcs_s0/checkpoints/best_mae.pth', 60, True),
    ('B1glb (best_causal)',  f'{BASE_NEW}/B1glb_s0/checkpoints/best_causal.pth', 60, True),
]

print(f"{'Model':25s} | {'H':>2s} | {'MAE':>7s} | {'CFI_agg':>7s} | {'gain_mean':>9s} | {'span':>6s} | {'early':>5s} | {'SHAPE':>6s}")
print("-" * 100)
for name, ckpt, H, is_new in configs:
    r = eval_model(name, ckpt, H, is_new)
    if r:
        print(f"{name:25s} | {r['H']:2d} | {r['mae']:7.4f} | {r['cfi_agg']:7.3f} | "
              f"{r['gain_mean']:9.3f} | {r['gain_span']:6.3f} | "
              f"{'OK' if r['early_sign_ok'] else 'FAIL':>5s} | {r['shape']:+6.3f}")
    else:
        print(f"{name:25s} | SKIP")
