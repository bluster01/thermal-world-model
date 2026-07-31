"""
exp_014_delay_fix.py — 延迟因果修复: 动作历史增强 + 多步监督
==============================================================
基于 exp_006 管线 + exp_012 绝对阀位

A. Action History: 显式加入滞后阀位特征 [a_t, a_{t-3}, a_{t-6}, a_{t-9}]
   → 输入: 11状态窗口 + 4×2 滞后阀位 (显式延迟模式匹配)

B. Multi-step Supervision: 在 A 基础上, loss 强化第 3/5 步预测监督
   → 权重 [1.0, 0.8, 0.6, 0.4, 0.2] → [1.0, 0.6, 1.0, 0.6, 1.2]
   (远步权重不衰减, 强制学习远期因果)

用法: python exp_014_delay_fix.py A|B
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import WorldModel, RevIN, PatchEmbedding, PerVariableTCN, VariableAttention

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'A'
assert VARIANT in ('A', 'B'), "Usage: python exp_014_delay_fix.py A|B"

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
ROLLOUT_K, BETA, BETA_WARMUP = 5, -0.3, 20
BS, STEPS = 256, 500

# 滞后点: 0, 3, 6, 9 步前 (0/30/60/90s 前)
LAG_STEPS = [0, 3, 6, 9]
N_LAGS = len(LAG_STEPS)
# 多步监督权重 (B): 强化 3/5 步
if VARIANT == 'B':
    ROLLOUT_W = [1.0, 0.6, 1.0, 0.6, 1.2]
else:
    ROLLOUT_W = [1.0, 0.8, 0.6, 0.4, 0.2]

print(f"Variant: {VARIANT} | Lags: {LAG_STEPS} | Weights: {ROLLOUT_W}")


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


# ======================================================================
#   WorldModel with Action History (lagged valve features)
# ======================================================================
class WorldModel_ActionHistory(nn.Module):
    """状态走原 RevIN+TCN 路径, 滞后阀位作为显式额外特征注入 decoder

    输入 x: [B, W, N_state + N_action]  (绝对阀位)
    构造: 滞后阀位 [B, N_LAGS*N_action] → MLP → [B, d_model]
          与状态特征拼接 → VarAttn → decoder
    """
    def __init__(self):
        super().__init__()
        d = cfg.D_MODEL
        # State encoder (复用原结构, 但 RevIN 只归一化状态)
        self.revin = RevIN(cfg.N_STATE)
        self.patch = PatchEmbedding(cfg.WINDOW_SIZE, 16, 8, d)
        self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)

        # 滞后动作嵌入
        self.lag_embed = nn.Sequential(
            nn.Linear(N_LAGS * cfg.N_ACTION, d * 2),
            nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 2, d),
        )

        # VarAttn: N_state + 1 action token
        n_tokens = cfg.N_STATE + 1
        self.var_attn = nn.ModuleList([VariableAttention(d, cfg.N_HEADS, cfg.DROPOUT)
                                       for _ in range(cfg.N_VAR_LAYERS)])
        self.decoder = nn.Sequential(
            nn.Linear(n_tokens * d, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 2, cfg.N_STATE * 2),
        )
        self.probabilistic = True

    def forward(self, x):
        """x: [B, W, N_state + N_action] 绝对阀位"""
        B, W, _ = x.shape
        d = cfg.D_MODEL
        s = x[:, :, :cfg.N_STATE]
        a = x[:, :, cfg.N_STATE:]  # [B, W, 2]

        # 状态: RevIN → Patch → TCN
        s_norm = self.revin(s, mode='norm')
        var_tokens = [self.patch(s_norm[:, :, i]) for i in range(cfg.N_STATE)]
        var_tokens = torch.stack(var_tokens, 1)  # [B, Ns, np, d]
        var_tokens = var_tokens.reshape(B * cfg.N_STATE, self.np, d)
        s_repr = self.tcn(var_tokens).reshape(B, cfg.N_STATE, d)

        # 滞后动作特征: 取 a_{t}, a_{t-3}, a_{t-6}, a_{t-9}
        lag_vals = []
        for lag in LAG_STEPS:
            lag_vals.append(a[:, -1 - lag, :])  # [B, 2]
        lag_feat = torch.cat(lag_vals, dim=1)   # [B, N_LAGS*2]
        a_token = self.lag_embed(lag_feat).unsqueeze(1)  # [B, 1, d]

        # 拼接 + VarAttn
        tokens = torch.cat([s_repr, a_token], 1)  # [B, Ns+1, d]
        for attn in self.var_attn:
            tokens, _ = attn(tokens)

        # Decode (含动作 token)
        raw = self.decoder(tokens.reshape(B, -1))

        # Denorm
        ms = self.revin._mean[:, :, :cfg.N_STATE]
        ss = self.revin._std[:, :, :cfg.N_STATE]
        w = self.revin.weight[:cfg.N_STATE]
        b = self.revin.bias[:cfg.N_STATE]
        mu_n = raw[:, :cfg.N_STATE]; lv_n = raw[:, cfg.N_STATE:]
        mu_n2 = mu_n.unsqueeze(1)
        if self.revin.affine: mu_n2 = (mu_n2 - b) / (w + self.revin.eps)
        mu = (mu_n2 * ss + ms).squeeze(1)
        sig = torch.exp(lv_n * 0.5) * ss.squeeze(1)
        lv = 2.0 * torch.log(sig + 1e-8)
        return mu, lv

    def rollout(self, x_hist, a_seq, mode='sliding', return_stats=False):
        B, W, _ = x_hist.shape
        H = a_seq.shape[1]
        sw = x_hist[:, :, :cfg.N_STATE]
        aw = x_hist[:, :, cfg.N_STATE:]
        preds = []
        for t in range(H):
            mu, _ = self(torch.cat([sw, aw], 2))
            preds.append(mu)
            sw = torch.cat([sw[:, 1:, :], mu.unsqueeze(1)], 1)
            aw = torch.cat([aw[:, 1:, :], a_seq[:, t:t+1, :]], 1)
        return torch.stack(preds, 1)


# ======================================================================
#  Standard model with abs valves (control: exp_012 re-run)
# ======================================================================
def make_std_model():
    return WorldModel(
        n_state=cfg.N_STATE, n_action=cfg.N_ACTION,
        window_size=cfg.WINDOW_SIZE, d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS, n_var_layers=cfg.N_VAR_LAYERS,
        n_tcn_layers=cfg.N_TCN_LAYERS, dropout=cfg.DROPOUT,
        rollout_mode='sliding', probabilistic=True,
    )


# ======================================================================
#  Training
# ======================================================================
def train_epoch(model, raw, opt, crit, use_abs):
    model.train(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw)
    total = 0.
    for _ in range(STEPS):
        idxs = np.random.randint(0, N-W-K, size=BS)
        xb, ab, sb = [], [], []
        for i in idxs:
            sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
            xb.append(np.concatenate([sw, aw], 1))
            ab.append(raw[i+W:i+W+K, cfg.N_STATE:])
            sb.append(raw[i+W:i+W+K, :cfg.N_STATE])
        xt = torch.FloatTensor(np.stack(xb)).to(DEVICE)
        at = torch.FloatTensor(np.stack(ab)).to(DEVICE)
        st = torch.FloatTensor(np.stack(sb)).to(DEVICE)
        opt.zero_grad()
        ss = xt[:,:,:cfg.N_STATE]; aa = xt[:,:,cfg.N_STATE:]
        tsl = 0.
        for k in range(K):
            mu, lv = model(torch.cat([ss, aa], 2))
            tsl += ROLLOUT_W[k] * crit(mu, lv, st[:,k])
            ss = torch.cat([ss[:,1:], mu.unsqueeze(1).detach()], 1)
            aa = torch.cat([aa[:,1:], at[:,k:k+1]], 1)
        tsl.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); total += tsl.item()
    return total/STEPS


@torch.no_grad()
def validate(model, raw, n=200):
    model.eval(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw)
    m0, m4 = 0., 0.
    for _ in range(n):
        i = np.random.randint(0, N-W-K)
        sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
        fa = torch.FloatTensor(raw[i+W:i+W+K, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        st = torch.FloatTensor(raw[i+W:i+W+K, :cfg.N_STATE]).unsqueeze(0).to(DEVICE)
        tr = model.rollout(xt, fa, mode='sliding')
        m0 += (tr[0,0,cfg.TARGET_IDX]-st[0,0,cfg.TARGET_IDX]).abs().item()
        m4 += (tr[0,min(4,K-1),cfg.TARGET_IDX]-st[0,min(4,K-1),cfg.TARGET_IDX]).abs().item()
    return m0/n, m4/n


@torch.no_grad()
def eval_rollout(model, raw, H=18, n=500):
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1)).unsqueeze(0).to(DEVICE)
        fa = torch.FloatTensor(raw[i+W:i+W+H, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
        tt = raw[i+W:i+W+H, cfg.TARGET_IDX]
        tr = model.rollout(xt, fa, mode='sliding')
        err[j] = np.abs(tr[0,:,cfg.TARGET_IDX].cpu().numpy()-tt)
    return err.mean(0)


@torch.no_grad()
def eval_sensitivity(model, raw, n=500):
    """多步敏感性: 扰动 step0 动作, 观察 step1-10 的 ΔT"""
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-12), n, replace=False)
    abs_deltas = [-10., -5., -2., -1., 1., 2., 5., 10.]
    results = {}
    for adim in range(cfg.N_ACTION):
        name = ['喷水阀', '其他阀'][adim]
        print(f"\n  [{name} 多步扰动] (绝对值单位)")
        header = "  " + "".join([f"{'Δ':>7}   "]) + "".join([f"{f't={s}':>9}  " for s in [1,2,3,5,7,10]])
        print(header)
        r = {}
        for d in abs_deltas:
            dT_steps = {s: [] for s in [1,2,3,5,7,10]}
            for i in idxs:
                sh = raw[i:i+W, :cfg.N_STATE].copy()
                ah = raw[i:i+W, cfg.N_STATE:].copy()
                fa = raw[i+W:i+W+12, cfg.N_STATE:].copy()
                xt = torch.FloatTensor(np.concatenate([sh, ah], 1)).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fa).unsqueeze(0).to(DEVICE)
                trb = model.rollout(xt, at, mode='sliding')
                bp = trb[0,:,cfg.TARGET_IDX].cpu().numpy()
                ap = ah.copy(); ap[-1, adim] = np.clip(ap[-1, adim] + d, 0, 100)
                fap = fa.copy(); fap[0, adim] = np.clip(fap[0, adim] + d, 0, 100)
                xt = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fap).unsqueeze(0).to(DEVICE)
                trp = model.rollout(xt, at, mode='sliding')
                pp = trp[0,:,cfg.TARGET_IDX].cpu().numpy()
                for s in [1,2,3,5,7,10]:
                    dT_steps[s].append(pp[s]-bp[s])
            row = f"  {d:>+7.1f}   "
            for s in [1,2,3,5,7,10]:
                arr = np.array(dT_steps[s])
                row += f"{arr.mean():>+9.4f}  "
                r[f'{d}_{s}'] = float(arr.mean())
            print(row)
        results[f'action_{adim}'] = r
    return results


# ======================================================================
def main():
    state_data, delta_actions, valve_abs = load_raw_data()
    # 绝对阀位 (exp_012 已验证有效)
    raw = np.concatenate([state_data, valve_abs], axis=1)
    n_total = len(raw)
    n_train = int(n_total * 0.70); n_val = int(n_total * 0.85)
    train, val, test = raw[:n_train], raw[n_train:n_val], raw[n_val:]
    print(f"Data: {len(train)}+{len(val)}+{len(test)} | 绝对阀位")

    exp_dir = f"results/exp_014_{VARIANT}"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

    if VARIANT == 'A':
        model = WorldModel_ActionHistory().to(DEVICE)
        desc = "Action History (lagged valves)"
    else:
        model = WorldModel_ActionHistory().to(DEVICE)
        desc = "Action History + Multi-step weights"

    print(f"Model: {desc} | Params: {sum(p.numel() for p in model.parameters()):,}")

    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        tl = train_epoch(model, train, opt, crit, True)
        v0, v4 = validate(model, val); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{tl:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}")
        if v4 < best_m - 0.001: best_m, be, pc = v4, ep, 0; torch.save(
            {'epoch':ep, 'model_state_dict':model.state_dict()},
            f"{exp_dir}/checkpoints/best_model.pth")
        else: pc += 1
        if pc >= cfg.EARLY_STOPPING_PATIENCE: print(f"  Stop@{ep} best@{be}"); break

    ck = torch.load(f"{exp_dir}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    print(f"\nTrain: {(time.time()-t0)/60:.1f}min Best@{be}")

    mae = eval_rollout(model, test)
    print(f"\nRollout(test): {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f})")

    print(f"\nMulti-step Sensitivity:")
    sens = eval_sensitivity(model, test)

    result = {'variant': VARIANT, 'desc': desc, 'lags': LAG_STEPS, 'weights': ROLLOUT_W,
              'best_epoch': be, 'rollout_mae': mae.tolist(), 'sensitivity': sens}
    with open(f"{exp_dir}/results.json", 'w') as f: json.dump(result, f, indent=2)
    print(f"\nSaved: {exp_dir}/results.json")


if __name__ == '__main__':
    main()
