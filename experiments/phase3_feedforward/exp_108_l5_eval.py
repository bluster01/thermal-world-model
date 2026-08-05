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
n_val_end = 601566
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')

raw = E.data_all
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40

gt = np.load('results/cfe_groundtruth/did_response.npz')

PROFILE18 = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s')]
PROFILE60 = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s'),
             (29, '300s'), (41, '420s'), (59, '600s')]

def build_model(H, head_mode='glb'):
    return CA.TimeXerCausalWM(N_FEAT, TARGET_IDX, H,
                              head_mode=head_mode, probabilistic=True).to(DEVICE).eval()

def eval_ckpt(ckpt_path, H, label):
    if not os.path.exists(ckpt_path):
        return None
    model = build_model(H)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd['model_state_dict'])
    wrap = CE.ModelWrapper(model, raw, raw41, W, H, I_DSP, DEVICE)
    
    ev, dv = CE.select_events(raw, I_SP, I_LD, H=H, lo=n_val_end, W=W)
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
    cfe = CE.causal_metrics(m_a, r_a, gt_R, gt_ceil, pk)
    key_step = '600s' if H == 60 else '180s'
    cfi = CE.cfi(cfe, key_step)
    p = cfe['profile'][key_step]
    return dict(label=label, H=H, n_ev=len(m_a), cfi=cfi,
                gain=p['gain'], sgn=p['sgn_pair'], sgn_ceil=p['sgn_ceiling'],
                shape=cfe['shape_corr'], ttp_err=cfe['ttp_err'],
                ep=int(sd.get('ep', -1)), mae=float(sd.get('mae', np.nan)))

BASE = 'results/exp_106_causal_arch'
configs = [
    ('B1glb H=60 linspace', f'{BASE}/B1glb_s0/checkpoints/best_causal.pth', 60),
    ('B1glb H=60 flat',     f'{BASE}/B1glb_s0_flatw/checkpoints/best_causal.pth', 60),
    ('B1glb H=18 linspace', f'{BASE}/B1glb_H18_s0/checkpoints/best_causal.pth', 18),
    ('B1glb H=18 flat',     f'{BASE}/B1glb_H18_s0_flatw/checkpoints/best_causal.pth', 18),
]

print(f"{'Model':22s} | {'H':>2s} | {'CFI':>6s} | {'GAIN':>7s} | {'SGN':>6s} (ceil) | {'SHAPE':>6s} | {'TTP':>5s}")
print("-" * 85)
for label, ckpt, H in configs:
    r = eval_ckpt(ckpt, H, label)
    if r:
        print(f"{label:22s} | {r['H']:2d} | {r['cfi']:6.3f} | {r['gain']:7.3f} | "
              f"{r['sgn']:.3f} ({r['sgn_ceil']:.3f}) | {r['shape']:+6.3f} | {r['ttp_err']:+5d}")
    else:
        print(f"{label:22s} | SKIP")
