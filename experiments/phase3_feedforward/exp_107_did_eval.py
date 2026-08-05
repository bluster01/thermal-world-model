#!/usr/bin/env python3
"""
exp_107_did_eval.py — DiD 真值口径正式评测 L4 消融模型 (2026-08-05)
===================================================================
对 exp_106 训练的 7 变体, 用 exp_104 的 DiD 真值做 L2 评测:
  SGN_pair / SGN_agg / GAIN / SHAPE / TTP / MONO / HET
对比口径: best_mae vs best_causal ckpt (P0.5)
"""
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
W = E.cfg.WINDOW_SIZE
N_FEAT = E.N_FEAT
TARGET_IDX = E.TARGET_IDX
n_val_end = 601566
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')

OUT = 'results/exp_107_did_eval'
os.makedirs(OUT, exist_ok=True)

raw = E.data_all
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40

# 载入 DiD 真值 (NPZ: per-event data)
gt_path = 'results/cfe_groundtruth/did_response.npz'
gt = np.load(gt_path)
print(f"[gt] {gt_path}: R_true60[600s]={gt['R_true60'][-1]:+.4f}, n_ev={len(gt['r60'])}")

# 模型注册
VARIANTS = {
    'A1mlp':    dict(H=60, kind='res', intervention='mlp',  cumsum_out=False),
    'A1phys':   dict(H=60, kind='res', intervention='phys', cumsum_out=False),
    'A1both':   dict(H=60, kind='res', intervention='both', cumsum_out=False),
    'A1mlp_cs': dict(H=60, kind='res', intervention='mlp',  cumsum_out=True),
    'A1physcs': dict(H=60, kind='res', intervention='phys', cumsum_out=True),
    'B1glb':    dict(H=60, kind='timexer', head_mode='glb'),
    'B1flat':   dict(H=60, kind='timexer', head_mode='flatten'),
}

PROFILE_K = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s'),
             (29, '300s'), (41, '420s'), (59, '600s')]

def build_model(variant):
    v = VARIANTS[variant]
    H = v['H']
    if v['kind'] == 'res':
        return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H,
                                   intervention=v['intervention'],
                                   cumsum_out=v['cumsum_out'], probabilistic=True)
    return CA.TimeXerCausalWM(N_FEAT, TARGET_IDX, H,
                              head_mode=v['head_mode'], probabilistic=True)

def eval_one(name, ckpt_type, H):
    """ckpt_type: 'best_mae' or 'best_causal'"""
    ckpt_path = f'results/exp_106_causal_arch/{name}_s0/checkpoints/{ckpt_type}.pth'
    if not os.path.exists(ckpt_path):
        return None
    
    model = build_model(name).to(DEVICE).eval()
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd['model_state_dict'])
    
    wrap = CE.ModelWrapper(model, raw, raw41, W, H, I_DSP, DEVICE)
    
    # 用 test-only 事件 + DiD 真值评测
    ev, dv = CE.select_events(raw, I_SP, I_LD, H=H, lo=n_val_end, W=W)
    if len(ev) == 0:
        return None
    
    m, _, keep = CE.model_response(wrap, ev, dv)
    ev_k = np.array(ev)[keep]
    dv_k = dv[keep]
    
    # 对齐 DiD 真值的事件 (按 onset 匹配)
    gt_onsets = gt[f'onsets{H // 10}0']
    gt_r = gt[f'r{H // 10}0']
    gt_R = gt[f'R_true{H // 10}0']
    gt_ceil = gt[f'sgn_ceiling{H // 10}0']
    
    # 建 onset→index 映射
    onset_to_i = {int(o): i for i, o in enumerate(gt_onsets)}
    m_aligned, r_aligned = [], []
    for i, o in enumerate(ev_k):
        if int(o) in onset_to_i:
            m_aligned.append(m[i])
            r_aligned.append(gt_r[onset_to_i[int(o)]])
    
    if len(m_aligned) == 0:
        return None
    
    m_a = np.array(m_aligned, dtype=np.float32)
    r_a = np.array(r_aligned, dtype=np.float32)
    
    cfe = CE.causal_metrics(m_a, r_a, gt_R, gt_ceil, PROFILE_K)
    cfi = CE.cfi(cfe, '600s')
    
    # 也跑旧口径 (sign(ΔSP)) 便于对照
    prof_old = {}
    for k, lab in PROFILE_K:
        resp_c = m_a[:, k] * dv_k[:len(m_a)]
        prof_old[lab] = dict(
            gain_norm=float(m_a[:, k].mean()),
            dir_dsp=float((np.sign(resp_c) == np.sign(dv_k[:len(m_a)])).mean()))
    
    return dict(name=name, ckpt=ckpt_type, H=H, n_ev=len(m_a),
                cfe=cfe, cfi=cfi, prof_old=prof_old,
                ep=int(sd.get('ep', -1)), mae=float(sd.get('mae', np.nan)))

# ===================================================== 跑全部
print("\n" + "=" * 80)
print("DiD 真值口径评测 (SGN/GAIN/SHAPE/TTP/MONO)")
print("=" * 80)

all_results = []
for name in VARIANTS:
    for ckpt_type in ['best_mae', 'best_causal']:
        res = eval_one(name, ckpt_type, 60)
        if res is None:
            print(f"  [{name:10s}/{ckpt_type:12s}] SKIP (no ckpt)")
            continue
        
        cfe = res['cfe']
        p600 = cfe['profile']['600s']
        p180 = cfe['profile']['180s']
        print(f"\n  [{name:10s}/{ckpt_type:12s}] n={res['n_ev']} ep={res['ep']}")
        print(f"    CFI={res['cfi']:.3f} | SHAPE={cfe['shape_corr']:+.3f} | TTP_err={cfe['ttp_err']:+d}")
        print(f"    600s: SGN_pair={p600['sgn_pair']:.3f} (ceil={p600['sgn_ceiling']:.3f}) "
              f"| GAIN={p600['gain']:.3f} | resp_model={p600['resp_model']:+.3f} vs resp_true={p600['resp_true']:+.3f}")
        print(f"    180s: SGN_pair={p180['sgn_pair']:.3f} (ceil={p180['sgn_ceiling']:.3f}) "
              f"| GAIN={p180['gain']:.3f} | resp_model={p180['resp_model']:+.3f} vs resp_true={p180['resp_true']:+.3f}")
        all_results.append(res)

# ===================================================== 汇总表
print("\n" + "=" * 80)
print("汇总: DiD 真值口径 (best_causal ckpt)")
print("=" * 80)

best_causal = [r for r in all_results if r['ckpt'] == 'best_causal']
best_causal.sort(key=lambda r: r['cfi'] if r['cfi'] else -999, reverse=True)

print(f"{'变体':10s} | {'MAE':>7s} | {'CFI':>6s} | {'600s GAIN':>9s} {'SGN':>6s} | {'180s GAIN':>9s} {'SGN':>6s} | {'SHAPE':>6s} | {'TTP_err':>7s}")
print("-" * 90)
for r in best_causal:
    cfe = r['cfe']
    p600 = cfe['profile']['600s']
    p180 = cfe['profile']['180s']
    print(f"{r['name']:10s} | {r['mae']:7.4f} | {r['cfi']:6.3f} | "
          f"{p600['gain']:8.3f} {p600['sgn_pair']:.2f} | "
          f"{p180['gain']:8.3f} {p180['sgn_pair']:.2f} | "
          f"{cfe['shape_corr']:+5.2f} | {cfe['ttp_err']:+6d}")

# 关键对比: best_mae vs best_causal
print("\n" + "=" * 80)
print("P0.5 双 ckpt 对比 (best_mae vs best_causal)")
print("=" * 80)
for name in VARIANTS:
    mae = next((r for r in all_results if r['name']==name and r['ckpt']=='best_mae'), None)
    cau = next((r for r in all_results if r['name']==name and r['ckpt']=='best_causal'), None)
    if mae and cau:
        p600m = mae['cfe']['profile']['600s']
        p600c = cau['cfe']['profile']['600s']
        print(f"  {name:10s}  MAE: {mae['cfi']:.3f} → CFI ckpt: {cau['cfi']:.3f}  "
              f"| gain: {p600m['gain']:.3f} → {p600c['gain']:.3f}")

# Save
with open(f'{OUT}/did_eval.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved: {OUT}/did_eval.json ({len(all_results)} results)")
