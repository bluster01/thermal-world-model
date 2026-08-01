"""
exp_017_multistep_signreg.py — 长程符号正则 (匹配真实物理滞后)
=================================================================
问题: exp_016 单步符号正则 (∂T/∂a<0 on t+1) 是伪物理 — 事件研究
(event_study_valve.py) 证明主汽温对二级减温阀的真实响应是 60-90s+
大滞后: 开阀后前 90s 主汽温微升 (+0.3°C, 减温水未传到), 120s+ 才转降,
10min 达 -3.4°C。强制单步降温把物理从 120s 滞后扭曲成 10s 伪响应,
且把 120s 的真实降温翻转成升温 (exp_016 L3_W1_l0.10: t1 -0.281 / t12 +0.278)。

新设计 (基于物理真值):
- ROLLOUT_K: 5 → 12 (监督覆盖 120s)
- 长程正则 (C): 只约束 t>=8 (80s+) 响应方向 ∂T/∂a<0, 短程 (t<8) 自由 —
  匹配 "前 90s 微升, 120s+ 转降" 的物理
- 对照 (B): K=12 无正则, 分离 "K12 长监督本身" vs "长程正则"

用法: python exp_017_multistep_signreg.py <config>
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import RevIN, PatchEmbedding, PerVariableTCN, VariableAttention

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else 'A'

CONFIGS = {
    # A: 多步正则全步约束 (原设计, 已被事件研究证伪 — 保留作反例)
    'A': dict(lags=[0, 3, 6, 9], weights=None, lam=0.10, sign_start=0),
    # B: 对照 — K=12 无正则 (分离 K12 监督 vs 正则)
    'B': dict(lags=[0, 3, 6, 9], weights=None, lam=0.00, sign_start=0),
    # C: 长程正则 — 只约束 t>=8 (80s+) 响应为负, 短程自由 (匹配事件研究真值:
    #    开阀后前 90s 微升+0.3°C, 120s+ 才转降 — 短程约束=伪物理)
    'C': dict(lags=[0, 3, 6, 9], weights=None, lam=0.10, sign_start=8),
}

CFG_C = CONFIGS[CONFIG_NAME]
LAGS = CFG_C['lags']
SIGN_LAMBDA = CFG_C['lam']
SIGN_START = CFG_C['sign_start']
N_LAGS = len(LAGS)

# K=12 覆盖 120s; W1 模式扩展到 12 步 (奇偶交替, 末步 1.2)
ROLLOUT_K, BETA, BETA_WARMUP = 12, -0.3, 20
ROLLOUT_W = [1.0, 0.6, 1.0, 0.6, 1.0, 0.6, 1.0, 0.6, 1.0, 0.6, 1.0, 1.2]
BS, STEPS = 256, 500
SIGN_DELTA = 5.0
SIGN_WARMUP = 20
CONSTRAINED_DIMS = [0, 1]

print(f"Config: {CONFIG_NAME} | lags={LAGS} | K={ROLLOUT_K} | λ={SIGN_LAMBDA}")


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


class WorldModel_Lag(nn.Module):
    """与 exp_016 完全一致: 滞后动作特征注入 decoder"""
    def __init__(self):
        super().__init__()
        d = cfg.D_MODEL
        self.revin = RevIN(cfg.N_STATE)
        self.patch = PatchEmbedding(cfg.WINDOW_SIZE, 16, 8, d)
        self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)

        self.lag_embed = nn.Sequential(
            nn.Linear(N_LAGS * cfg.N_ACTION, d * 2),
            nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 2, d),
        )
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
        B, W, _ = x.shape
        d = cfg.D_MODEL
        s = x[:, :, :cfg.N_STATE]
        a = x[:, :, cfg.N_STATE:]

        s_norm = self.revin(s, mode='norm')
        var_tokens = [self.patch(s_norm[:, :, i]) for i in range(cfg.N_STATE)]
        var_tokens = torch.stack(var_tokens, 1)
        var_tokens = var_tokens.reshape(B * cfg.N_STATE, self.np, d)
        s_repr = self.tcn(var_tokens).reshape(B, cfg.N_STATE, d)

        lag_vals = [a[:, -1 - lag, :] for lag in LAGS]
        lag_feat = torch.cat(lag_vals, dim=1)
        a_token = self.lag_embed(lag_feat).unsqueeze(1)
        tokens = torch.cat([s_repr, a_token], 1)
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


def rollout_from(model, xt, at, K):
    """基准/扰动共用: 从 xt 滚动 K 步, 返回 [B, K, N_STATE] mu 序列"""
    ss = xt[:, :, :cfg.N_STATE].clone()
    aa = xt[:, :, cfg.N_STATE:].clone()
    mus = []
    for k in range(K):
        mu, _ = model(torch.cat([ss, aa], 2))
        mus.append(mu)
        ss = torch.cat([ss[:, 1:], mu.detach().unsqueeze(1)], 1)
        aa = torch.cat([aa[:, 1:], at[:, k:k+1]], 1)
    return torch.stack(mus, 1)


def rollout_from_lv(model, xt, at, K):
    """返回 (mu, lv) 序列, 供 NLL 训练使用"""
    ss = xt[:, :, :cfg.N_STATE].clone()
    aa = xt[:, :, cfg.N_STATE:].clone()
    mus, lvs = [], []
    for k in range(K):
        mu, lv = model(torch.cat([ss, aa], 2))
        mus.append(mu); lvs.append(lv)
        ss = torch.cat([ss[:, 1:], mu.detach().unsqueeze(1)], 1)
        aa = torch.cat([aa[:, 1:], at[:, k:k+1]], 1)
    return torch.stack(mus, 1), torch.stack(lvs, 1)


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

        mus_base, lvs_base = rollout_from_lv(model, xt, at, K)
        nll_loss = 0.
        for k in range(K):
            nll_loss += ROLLOUT_W[k] * crit(mus_base[:, k], lvs_base[:, k], st[:, k])

        sign_loss = torch.tensor(0., device=DEVICE)
        if SIGN_LAMBDA > 0 and epoch > SIGN_WARMUP:
            # 只约束 t >= SIGN_START 的步 (C: 80s+ 长程物理响应区)
            t_base = mus_base[:, SIGN_START:, cfg.TARGET_IDX]  # [B, K-SIGN_START]
            for adim in CONSTRAINED_DIMS:
                for d in (+SIGN_DELTA, -SIGN_DELTA):
                    aa0 = xt[:, :, cfg.N_STATE:].clone()
                    aa0[:, -1, adim] = torch.clamp(aa0[:, -1, adim] + d, 0, 100)
                    at_p = at.clone()
                    at_p[:, 0, adim] = torch.clamp(at_p[:, 0, adim] + d, 0, 100)
                    xt_p = torch.cat([xt[:, :, :cfg.N_STATE], aa0], 2)
                    mus_p = rollout_from(model, xt_p, at_p, K)
                    t_p = mus_p[:, SIGN_START:, cfg.TARGET_IDX]
                    if d > 0:  # 开阀 → 长程每步更低
                        sign_loss += F.relu(t_p - t_base).mean()
                    else:      # 关阀 → 长程每步更高
                        sign_loss += F.relu(t_base - t_p).mean()

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
    """与 exp_016 完全一致: H=18 步评测, 便于直接对比"""
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
    """与 exp_016 完全一致: 扰动历史末位+未来首位动作"""
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

    exp_dir = f"results/exp_017_{CONFIG_NAME}"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        nll, sign = train_epoch(model, train_data, opt, crit, ep)
        v0, v4 = validate(model, val_data); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{nll:7.0f} | Sign:{sign:.4f} | V0:{v0:.4f} | V4:{v4:.4f}")
        reg_active = (SIGN_LAMBDA > 0)
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

    result = {'config': CONFIG_NAME, 'lags': LAGS, 'weights': ROLLOUT_W,
              'lambda': SIGN_LAMBDA, 'rollout_k': ROLLOUT_K,
              'best_epoch': be, 'rollout_mae': mae.tolist(), 'sensitivity': sens}
    with open(f"{exp_dir}/results.json", 'w') as f: json.dump(result, f, indent=2)
    print(f"Saved: {exp_dir}/results.json")


if __name__ == '__main__':
    main()
