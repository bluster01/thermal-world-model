#!/usr/bin/env python3
"""DiD 真值评测 L5 + 全量对比 (H=18/60 × flat/linspace)"""
import os, sys, json
import numpy as np
import torch

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
n_train = 495407
n_val_end = 601566
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')

raw = E.data_all
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40

gt = np.load('results/cfe_groundtruth_p2/did_response.npz')
print(f"[gt] P2 expanded: R_true60[600s]={gt['R_true60'][-1]:+.4f}, n_ev={len(gt['r60'])}")
PROFILE18 = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s')]
PROFILE60 = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s'),
             (29, '300s'), (41, '420s'), (59, '600s')]

def build_model(label, H):
    """根据 label 选择架构"""
    if label.startswith('B1'):
        return CA.TimeXerCausalWM(N_FEAT, TARGET_IDX, H,
                                  head_mode='glb' if 'glb' in label or 'B1glb' in label else 'flatten',
                                  probabilistic=True).to(DEVICE).eval()
    elif label.startswith('A1physcs'):
        return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H,
                                   intervention='phys', cumsum_out=True, probabilistic=True).to(DEVICE).eval()
    elif label.startswith('A1phys'):
        return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H,
                                   intervention='phys', cumsum_out=False, probabilistic=True).to(DEVICE).eval()
    elif label.startswith('A1mlp_cs'):
        return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H,
                                   intervention='mlp', cumsum_out=True, probabilistic=True).to(DEVICE).eval()
    elif label.startswith('A1mlp'):
        return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H,
                                   intervention='mlp', cumsum_out=False, probabilistic=True).to(DEVICE).eval()
    elif label.startswith('A1both'):
        return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H,
                                   intervention='both', cumsum_out=False, probabilistic=True).to(DEVICE).eval()
    else:
        return CA.TimeXerCausalWM(N_FEAT, TARGET_IDX, H,
                                  head_mode='glb', probabilistic=True).to(DEVICE).eval()

def eval_ckpt(ckpt_path, H, label):
    if not os.path.exists(ckpt_path):
        return None
    model = build_model(label, H)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd['model_state_dict'])
    wrap = CE.ModelWrapper(model, raw, raw41, W, H, I_DSP, DEVICE)
    
    # P2: val+test range
    ev, dv = CE.select_events(raw, I_SP, I_LD, H=H, lo=n_train, W=W,
                              thr=0.6, sp_hold=0.5, load_stable=5.0)
    m, _, keep = CE.model_response(wrap, ev, dv)
    ev_k = np.array(ev)[keep]; dv_k = dv[keep]
    
    gt_key = H
    gt_onsets = gt[f'onsets{gt_key}']; gt_r = gt[f'r{gt_key}']
    gt_R = gt[f'R_true{gt_key}']; gt_ceil = gt[f'sgn_ceiling{gt_key}']
    
    onset_to_i = {int(o): i for i, o in enumerate(gt_onsets)}
    m_a, r_a = [], []
    for i, o in enumerate(ev_k):
        if int(o) in onset_to_i:
            m_a.append(m[i]); r_a.append(gt_r[onset_to_i[int(o)]])
    if len(m_a) == 0: return None
    
    m_a = np.array(m_a, dtype=np.float32); r_a = np.array(r_a, dtype=np.float32)
    pk = PROFILE60 if H == 60 else PROFILE18
    gc = gt[f'gain_ceiling{gt_key}']
    cfe = CE.causal_metrics(m_a, r_a, gt_R, gt_ceil, pk, gain_ceiling=gc)
    agg = CE.cfi_agg(cfe, pk)
    p = cfe['profile']['600s' if H == 60 else '180s']
    return dict(label=label, H=H, n_ev=len(m_a),
                cfi_agg=agg['cfi'], gain_mean=agg['gain_mean'], gain_span=agg['gain_span'],
                early_sign_ok=agg['early_sign_ok'],
                gain_old=p['gain_raw'], sgn=p['sgn_pair'], sgn_ceil=p['sgn_ceiling'],
                shape=cfe['shape_corr'], ttp_err=cfe['ttp_err'],
                ep=int(sd.get('ep', -1)), mae=float(sd.get('mae', np.nan)))

BASE = 'results/exp_106_causal_arch'
configs = [
    ('B1glb H=60 linspace', f'{BASE}/B1glb_s0/checkpoints/best_causal.pth', 60),
    ('B1glb H=60 flat',     f'{BASE}/B1glb_s0_flatw/checkpoints/best_causal.pth', 60),
    ('B1glb H=18 linspace', f'{BASE}/B1glb_H18_s0/checkpoints/best_causal.pth', 18),
    ('B1glb H=18 flat',     f'{BASE}/B1glb_H18_s0_flatw/checkpoints/best_causal.pth', 18),
    # A1 系列对照
    ('A1phys H=60',         f'{BASE}/A1phys_s0/checkpoints/best_causal.pth', 60),
    ('A1mlp H=60',          f'{BASE}/A1mlp_s0/checkpoints/best_causal.pth', 60),
    ('A1both H=60',         f'{BASE}/A1both_s0/checkpoints/best_causal.pth', 60),
]

print(f"{'Model':22s} | {'H':>2s} | {'CFI_agg':>7s} | {'gain_mean':>9s} | {'span':>6s} | {'early':>5s} | {'old CFI':>7s} | {'SHAPE':>6s}")
print("-" * 90)
for label, ckpt, H in configs:
    r = eval_ckpt(ckpt, H, label)
    if r:
        print(f"{label:22s} | {r['H']:2d} | {r['cfi_agg']:7.3f} | {r['gain_mean']:9.3f} | "
              f"{r['gain_span']:6.3f} | {'OK' if r['early_sign_ok'] else 'FAIL':>5s} | "
              f"{r['gain_old']:7.3f} | {r['shape']:+6.3f}")
    else:
        print(f"{label:22s} | SKIP")
