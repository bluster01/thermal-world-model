"""
exp_016_ablation_sweep.py — 符号正则/权重/延迟 消融矩阵
=========================================================
基线 = exp_015: 绝对阀位 + 双阀符号正则(λ=0.1) + 多步权重W1 + 无滞后

控制变量法, 每次只变一个维度:

λ 扫描 (固定 W1, L0):
  L0_W1_l0.00  无符号正则 (对照=exp_012 双阀版)
  L0_W1_l0.01  弱正则
  L0_W1_l0.10  = exp_015 基线
  L0_W1_l1.00  强正则

权重扫描 (固定 λ=0.1, L0):
  L0_W0_l0.10  标准权重 [1.0,0.8,0.6,0.4,0.2]
  L0_W1_l0.10  = exp_015 基线 (多步权重)
  L0_W2_l0.10  均匀权重 [1.0,1.0,1.0,1.0,1.0]

延迟扫描 (固定 λ=0.1, W1):
  L0_W1_l0.10  = exp_015 基线 (无滞后)
  L3_W1_l0.10  滞后 [0,3,6,9] (30-90s)
  L6_W1_l0.10  滞后 [0,6,12,18] (60-180s)

用法: python exp_016_ablation_sweep.py <config_name>
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import WorldModel

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else 'L0_W1_l0.10'

# ===== 配置表 =====
CONFIGS = {
    # λ 扫描
    'L0_W1_l0.00': dict(lags=[0], weights=[1.0, 0.6, 1.0, 0.6, 1.2], lam=0.00),
    'L0_W1_l0.01': dict(lags=[0], weights=[1.0, 0.6, 1.0, 0.6, 1.2], lam=0.01),
    'L0_W1_l0.10': dict(lags=[0], weights=[1.0, 0.6, 1.0, 0.6, 1.2], lam=0.10),
    'L0_W1_l1.00': dict(lags=[0], weights=[1.0, 0.6, 1.0, 0.6, 1.2], lam=1.00),
    # 权重扫描
    'L0_W0_l0.10': dict(lags=[0], weights=[1.0, 0.8, 0.6, 0.4, 0.2], lam=0.10),
    'L0_W2_l0.10': dict(lags=[0], weights=[1.0, 1.0, 1.0, 1.0, 1.0], lam=0.10),
    # 延迟扫描
    'L3_W1_l0.10': dict(lags=[0, 3, 6, 9], weights=[1.0, 0.6, 1.0, 0.6, 1.2], lam=0.10),
    'L6_W1_l0.10': dict(lags=[0, 6, 12, 18], weights=[1.0, 0.6, 1.0, 0.6, 1.2], lam=0.10),
}

CFG = CONFIGS[CONFIG_NAME]
LAGS = CFG['lags']
ROLLOUT_W = CFG['weights']
SIGN_LAMBDA = CFG['lam']
N_LAGS = len(LAGS)
ROLLOUT_K, BETA, BETA_WARMUP = 5, -0.3, 20
BS, STEPS = 256, 500
SIGN_DELTA = 5.0
SIGN_WARMUP = 20
CONSTRAINED_DIMS = [0, 1]

print(f"Config: {CONFIG_NAME} | lags={LAGS} | weights={ROLLOUT_W} | λ={SIGN_LAMBDA}")


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


# ===== 模型: 支持滞后动作特征 =====
from world_model import RevIN, PatchEmbedding, PerVariableTCN, VariableAttention

class WorldModel_Lag(nn.Module):
    """标准 WorldModel + 可选滞后动作特征注入 decoder"""
    def __init__(self):
        super().__init__()
        d = cfg.D_MODEL
        self.revin = RevIN(cfg.N_STATE)
        self.patch = PatchEmbedding(cfg.WINDOW_SIZE, 16, 8, d)
        self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)

        if N_LAGS > 1:
            self.lag_embed = nn.Sequential(
                nn.Linear(N_LAGS * cfg.N_ACTION, d * 2),
                nn.GELU(), nn.Dropout(cfg.DROPOUT),
                nn.Linear(d * 2, d),
            )
            n_tokens = cfg.N_STATE + 1
        else:
            # lags=[0] → 标准模型 (动作在输入拼接, 无额外token)
            self.lag_embed = None
            n_tokens = cfg.N_STATE

        self.var_attn = nn.ModuleList([VariableAttention(d, cfg.N_HEADS, cfg.DROPOUT)
                                       for _ in range(cfg.N_VAR_LAYERS)])
        self.decoder = nn.Sequential(
            nn.Linear(n_tokens * d, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 2, cfg.N_STATE * 2),
        )
        self.probabilistic = True

    def forward(self, x):
        B, W, _ = x.shape
        d = cfg.D_MODEL
        s = x[:, :, :cfg.N_STATE]
        a = x[:, :, cfg.N_STATE:]

        s_norm = self.revin(s, mode='norm')
        var_tokens = [self.patch(s_norm[:, :, i]) for i in range(cfg.N_STATE)]
        var_tokens = torch.stack(var_tokens, 1)
        var_tokens = var_tokens.reshape(B * cfg.N_STATE, self.np, d)
        s_repr = self.tcn(var_tokens).reshape(B, cfg.N_STATE, d)

        if self.lag_embed is not None:
            lag_vals = [a[:, -1 - lag, :] for lag in LAGS]
            lag_feat = torch.cat(lag_vals, dim=1)
            a_token = self.lag_embed(lag_feat).unsqueeze(1)
            tokens = torch.cat([s_repr, a_token], 1)
        else:
            tokens = s_repr
        for attn in self.var_attn:
            tokens, _ = attn(tokens)

        raw = self.decoder(tokens.reshape(B, -1))

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


# ===== Data =====
state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
train_data, val_data = raw_data[:n_train], raw_data[n_train:n_val_end]
test_data = raw_data[n_val_end:]
print(f"Data: {len(train_data)}+{len(val_data)}+{len(test_data)} | 绝对阀位")


# ===== Training =====
def train_epoch(model, raw, opt, crit, epoch):
    model.train(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw)
    total_nll, total_sign = 0., 0.
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
        nll_loss = 0.
        for k in range(K):
            mu, lv = model(torch.cat([ss, aa], 2))
            nll_loss += ROLLOUT_W[k] * crit(mu, lv, st[:,k])
            ss = torch.cat([ss[:,1:], mu.detach().unsqueeze(1)], 1)
            aa = torch.cat([aa[:,1:], at[:,k:k+1]], 1)

        sign_loss = torch.tensor(0., device=DEVICE)
        if SIGN_LAMBDA > 0 and epoch > SIGN_WARMUP:
            ss0 = xt[:,:,:cfg.N_STATE]
            aa0 = xt[:,:,cfg.N_STATE:].clone()
            mu_orig, _ = model(torch.cat([ss0, aa0], 2))
            t_orig = mu_orig[:, cfg.TARGET_IDX]
            for adim in CONSTRAINED_DIMS:
                aa_open = aa0.clone(); aa_open[:, -1, adim] += SIGN_DELTA
                mu_open, _ = model(torch.cat([ss0, aa_open], 2))
                t_open = mu_open[:, cfg.TARGET_IDX]
                aa_close = aa0.clone(); aa_close[:, -1, adim] -= SIGN_DELTA
                mu_close, _ = model(torch.cat([ss0, aa_close], 2))
                t_close = mu_close[:, cfg.TARGET_IDX]
                sign_loss += F.relu(t_open - t_orig).mean() + F.relu(t_orig - t_close).mean()

        loss = nll_loss + SIGN_LAMBDA * sign_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        total_nll += nll_loss.item(); total_sign += sign_loss.item()
    return total_nll / STEPS, total_sign / STEPS


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
def eval_sensitivity(model, raw, n=200):
    model.eval(); W = cfg.WINDOW_SIZE; N = len(raw)
    np.random.seed(42); idxs = np.random.choice(range(N-W-13), n, replace=False)
    abs_deltas = [-10., -5., -2., -1., 1., 2., 5., 10.]
    results = {}
    for adim in range(2):
        vname = ['一级减温阀', '二级减温阀'][adim]
        r = {}
        for d in abs_deltas:
            dT = {s: [] for s in [1, 3, 8, 12]}
            for i in idxs:
                sh = raw[i:i+W, :cfg.N_STATE].copy()
                ah = raw[i:i+W, cfg.N_STATE:].copy()
                fa = raw[i+W:i+W+13, cfg.N_STATE:].copy()
                xt = torch.FloatTensor(np.concatenate([sh, ah], 1)).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fa).unsqueeze(0).to(DEVICE)
                trb = model.rollout(xt, at, mode='sliding')
                bp = trb[0,:,cfg.TARGET_IDX].cpu().numpy()
                ap = ah.copy(); ap[-1, adim] = np.clip(ap[-1, adim]+d, 0, 100)
                fap = fa.copy(); fap[0, adim] = np.clip(fap[0, adim]+d, 0, 100)
                xt = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fap).unsqueeze(0).to(DEVICE)
                trp = model.rollout(xt, at, mode='sliding')
                pp = trp[0,:,cfg.TARGET_IDX].cpu().numpy()
                for s in [1, 3, 8, 12]:
                    dT[s].append(pp[s]-bp[s])
            for s in [1, 3, 8, 12]:
                r[f'{d}_{s}'] = float(np.mean(dT[s]))
        results[f'action_{adim}'] = r
    return results


def main():
    model = WorldModel_Lag().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    exp_dir = f"results/exp_016_{CONFIG_NAME}"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    best_since_warmup = False
    for ep in range(1, cfg.EPOCHS + 1):
        crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        nll, sign = train_epoch(model, train_data, opt, crit, ep)
        v0, v4 = validate(model, val_data); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{nll:7.0f} | Sign:{sign:.4f} | V0:{v0:.4f} | V4:{v4:.4f}")
        # 只在符号正则生效后 (epoch > SIGN_WARMUP) 选择 best; λ=0 无正则则正常早停
        reg_active = (SIGN_LAMBDA > 0)
        if (not reg_active or ep > SIGN_WARMUP) and not best_since_warmup:
            best_since_warmup = True
            if reg_active: print(f"  [符号正则生效, 开始选 best]")
        if not reg_active or ep > SIGN_WARMUP:
            if v4 < best_m - 0.001: best_m, be, pc = v4, ep, 0; torch.save(
                {'epoch':ep, 'model_state_dict':model.state_dict()},
                f"{exp_dir}/checkpoints/best_model.pth")
            else: pc += 1
            if pc >= cfg.EARLY_STOPPING_PATIENCE: print(f"  Stop@{ep} best@{be}"); break

    ck = torch.load(f"{exp_dir}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    print(f"\nTrain: {(time.time()-t0)/60:.1f}min Best@{be}")

    mae = eval_rollout(model, test_data)
    print(f"\nRollout(test): {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f})")
    sens = eval_sensitivity(model, test_data)
    print(f"  Sens (二级 ±10, t1/t12): {sens['action_1']['10.0_1']:+.3f} / {sens['action_1']['10.0_12']:+.3f}")

    result = {'config': CONFIG_NAME, 'lags': LAGS, 'weights': ROLLOUT_W, 'lambda': SIGN_LAMBDA,
              'best_epoch': be, 'rollout_mae': mae.tolist(), 'sensitivity': sens}
    with open(f"{exp_dir}/results.json", 'w') as f: json.dump(result, f, indent=2)
    print(f"Saved: {exp_dir}/results.json")


if __name__ == '__main__':
    main()
