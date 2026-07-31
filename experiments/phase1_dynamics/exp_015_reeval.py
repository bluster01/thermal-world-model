"""
exp_015_reeval.py — 阀位实验重新评估 (纠正命名: 一级/二级减温阀)
=================================================================
复用已训练模型权重, 对 一级减温阀(adim=0) / 二级减温阀(adim=1)
分别做多步敏感性分析 (扰动 step0 → 观察 t=1..12)

用法: python exp_015_reeval.py <exp_id> [delta|abs]
  exp_id: exp_006|exp_011_B|exp_011_C|exp_012|exp_013|exp_014_A|exp_014_B
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import WorldModel

EXP_ID = sys.argv[1] if len(sys.argv) > 1 else 'exp_012'
MODE = sys.argv[2] if len(sys.argv) > 2 else 'abs'  # delta|abs

# 模型 → checkpoint 路径
CKPT_MAP = {
    'exp_006': 'results/exp_006/checkpoints/best_model.pth',
    'exp_011_B': 'results/exp_011_B_bypass/checkpoints/best_model.pth',
    'exp_011_C': 'results/exp_011_C_film/checkpoints/best_model.pth',
    'exp_012': 'results/exp_012_absvalve/checkpoints/best_model.pth',
    'exp_013': 'results/exp_013_signreg_l0.1_d5.0/checkpoints/best_model.pth',
    'exp_014_A': 'results/exp_014_A/checkpoints/best_model.pth',
    'exp_014_B': 'results/exp_014_B/checkpoints/best_model.pth',
}

# 特殊模型需要特殊构造
SPECIAL = {'exp_011_B', 'exp_011_C', 'exp_014_A', 'exp_014_B'}

VALVE_NAMES = ['一级减温阀', '二级减温阀']


def load_model(exp_id, device):
    ckpt_path = CKPT_MAP[exp_id]
    if exp_id == 'exp_011_B':
        # Bypass 模型: 状态 RevIN + 动作 MLP (从 exp_011_action_fix.py 复制)
        from exp_011_action_fix import ModelB_Bypass
        model = ModelB_Bypass()
    elif exp_id == 'exp_011_C':
        from exp_011_action_fix import ModelC_FiLM
        model = ModelC_FiLM()
    elif exp_id in ('exp_014_A', 'exp_014_B'):
        import sys as _sys
        if len(_sys.argv) < 2 or _sys.argv[1] not in ('A', 'B'):
            _sys.argv = ['exp_014_delay_fix.py', 'A']
        import importlib
        mod = importlib.import_module('exp_014_delay_fix')
        model = mod.WorldModel_ActionHistory()
    else:
        model = WorldModel(
            n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
            window_size=cfg.WINDOW_SIZE, d_model=cfg.D_MODEL,
            n_heads=cfg.N_HEADS, n_var_layers=cfg.N_VAR_LAYERS,
            n_tcn_layers=cfg.N_TCN_LAYERS, dropout=cfg.DROPOUT,
            rollout_mode='sliding', probabilistic=True,
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


@torch.no_grad()
def eval_sensitivity(model, raw, device, mode, n=150, H=13):
    """对每个阀做多步扰动敏感性"""
    W = cfg.WINDOW_SIZE
    N = len(raw)
    np.random.seed(42)
    idxs = np.random.choice(range(N - W - H), n, replace=False)

    for adim in range(2):
        vname = VALVE_NAMES[adim]
        print(f"\n  [{vname}] 扰动 step0 → Δ主汽温 (t=1..12)")
        header = "  " + "Δ".rjust(7) + "  " + "  ".join([f"t={s}".rjust(9) for s in [1,2,3,5,8,12]])
        print(header)
        for d in ([-10,-5,-2,-1,1,2,5,10] if mode == 'abs' else [-0.2,-0.1,-0.05,0.05,0.1,0.2]):
            dT = {s: [] for s in [1,2,3,5,8,12]}
            for i in idxs:
                sh = raw[i:i+W, :cfg.N_STATE].copy()
                ah = raw[i:i+W, cfg.N_STATE:].copy()
                fa = raw[i+W:i+W+H, cfg.N_STATE:].copy()
                # baseline rollout
                xt = torch.FloatTensor(np.concatenate([sh, ah], 1)).unsqueeze(0).to(device)
                at = torch.FloatTensor(fa).unsqueeze(0).to(device)
                trb = model.rollout(xt, at, mode='sliding')
                bp = trb[0, :, cfg.TARGET_IDX].cpu().numpy()
                # perturb step0 action (abs: +d; delta: *(1+d))
                ap = ah.copy()
                if mode == 'abs':
                    ap[-1, adim] = np.clip(ap[-1, adim] + d, 0, 100)
                    fap = fa.copy(); fap[0, adim] = np.clip(fap[0, adim] + d, 0, 100)
                else:
                    ap[-1, adim] *= (1 + d)
                    fap = fa.copy(); fap[0, adim] *= (1 + d)
                xt = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(device)
                at = torch.FloatTensor(fap).unsqueeze(0).to(device)
                trp = model.rollout(xt, at, mode='sliding')
                pp = trp[0, :, cfg.TARGET_IDX].cpu().numpy()
                for s in [1,2,3,5,8,12]:
                    dT[s].append(pp[s] - bp[s])
            row = f"  {d:>+7.1f}  " + "  ".join([f"{np.mean(dT[s]):>+9.4f}" for s in [1,2,3,5,8,12]])
            print(row)


@torch.no_grad()
def eval_rollout(model, raw, device, H=18, n=300):
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(device)
        fa = torch.FloatTensor(raw[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(device)
        tt = raw[i+W:i+W+H, cfg.TARGET_IDX]
        tr = model.rollout(xt, fa, mode='sliding')
        err[j] = np.abs(tr[0,:,cfg.TARGET_IDX].cpu().numpy()-tt)
    return err.mean(0)


def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model: {EXP_ID} | Mode: {MODE}")

    state_data, delta_actions, valve_abs = load_raw_data()
    if MODE == 'abs':
        raw = np.concatenate([state_data, valve_abs], axis=1)
    else:
        raw = np.concatenate([state_data, delta_actions], axis=1)
    n_val = int(len(raw) * 0.85)
    test = raw[n_val:]
    print(f"Test: {len(test)} | {'绝对阀位' if MODE=='abs' else '差分阀位'}")

    model = load_model(EXP_ID, device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    mae = eval_rollout(model, test, device)
    print(f"\nRollout(test): {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f})")

    print(f"\n多步敏感性:")
    eval_sensitivity(model, test, device, MODE)


if __name__ == '__main__':
    main()
