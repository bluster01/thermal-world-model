"""
exp_018_lstm_baseline.py — 普通时序预测模型对照 (审稿人 R2-M1: 零 baseline)
=============================================================================
与最终世界模型 (L3_W1_l0.00, K=5 无正则) 完全同协议, 只换 backbone:

- 同数据切分 (70/15/15 时序) / 同绝对阀位 / 同 K=5 rollout loss / 同 β-NLL
- 同 seed 42 评测 (H=18 rollout + 敏感性 ±10)
- 变体:
  A: LSTM-dynamics  含动作 (与世界模型同输入输出协议, 直接对比架构)
  B: LSTM-pure      无动作 (纯预测, 类比 Exp-0 直接多步预测)

用法: python exp_018_lstm_baseline.py <A|B>
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data
from world_model import RevIN

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'A'

ROLLOUT_K, BETA, BETA_WARMUP = 5, -0.3, 20
ROLLOUT_W = [1.0, 0.6, 1.0, 0.6, 1.2]
BS, STEPS = 256, 500
LSTM_HIDDEN = 64
LSTM_LAYERS = 2

INCLUDE_ACTION = (VARIANT == 'A')
print(f"Config: {VARIANT} | include_action={INCLUDE_ACTION} | LSTM h={LSTM_HIDDEN} l={LSTM_LAYERS} | K={ROLLOUT_K}")


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


class LSTMDynamics(nn.Module):
    """LSTM 动力学模型: [B, W, dim] → RevIN → LSTM → MLP → (μ, logσ²)
    加入 RevIN 与世界模型同条件 (审稿人公平性要求)"""
    def __init__(self):
        super().__init__()
        d = cfg.N_STATE + (cfg.N_ACTION if INCLUDE_ACTION else 0)
        self.revin = RevIN(d)  # 状态+动作一起归一化 (动作绝对值 0-100 量纲也大)
        self.lstm = nn.LSTM(d, LSTM_HIDDEN, LSTM_LAYERS, dropout=cfg.DROPOUT, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 64), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(64, cfg.N_STATE * 2),
        )
        self.probabilistic = True

    def forward(self, x):
        # x: [B, W, d]
        x_n = self.revin(x, mode='norm')
        h, _ = self.lstm(x_n)
        raw = self.head(h[:, -1])  # 最后时刻隐状态 (归一化空间)
        mu_n = raw[:, :cfg.N_STATE]
        lv_n = raw[:, cfg.N_STATE:]
        # 反归一化 μ (全仿射逆), σ 只乘 std (与世界模型一致)
        ms = self.revin._mean[:, :, :cfg.N_STATE]
        ss = self.revin._std[:, :, :cfg.N_STATE]
        w = self.revin.weight[:cfg.N_STATE]
        b = self.revin.bias[:cfg.N_STATE]
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
        aw = x_hist[:, :, cfg.N_STATE:] if INCLUDE_ACTION else torch.zeros(B, W, 0, device=x_hist.device)
        preds = []
        for t in range(H):
            x = torch.cat([sw, aw], 2) if INCLUDE_ACTION else sw
            mu, _ = self(x)
            preds.append(mu)
            sw = torch.cat([sw[:, 1:, :], mu.unsqueeze(1)], 1)
            if INCLUDE_ACTION:
                aw = torch.cat([aw[:, 1:, :], a_seq[:, t:t+1, :]], 1)
        return torch.stack(preds, 1)


# ===== Data (与 exp_016 完全一致) =====
state_data, delta_actions, valve_abs = load_raw_data()
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
train_data, val_data = raw_data[:n_train], raw_data[n_train:n_val_end]
test_data = raw_data[n_val_end:]
print(f"Data: {len(train_data)}+{len(val_data)}+{len(test_data)} | 绝对阀位 | include_action={INCLUDE_ACTION}")


def train_epoch(model, raw, opt, crit, epoch):
    model.train(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw)
    total_nll = 0.
    for _ in range(STEPS):
        idxs = np.random.randint(0, N-W-K, size=BS)
        xb, ab, sb = [], [], []
        for i in idxs:
            sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
            xb.append(np.concatenate([sw, aw], 1) if INCLUDE_ACTION else sw)
            ab.append(raw[i+W:i+W+K, cfg.N_STATE:])
            sb.append(raw[i+W:i+W+K, :cfg.N_STATE])
        xt = torch.FloatTensor(np.stack(xb)).to(DEVICE)
        at = torch.FloatTensor(np.stack(ab)).to(DEVICE)
        st = torch.FloatTensor(np.stack(sb)).to(DEVICE)
        opt.zero_grad()

        ss = xt[:, :, :cfg.N_STATE]
        aa = xt[:, :, cfg.N_STATE:] if INCLUDE_ACTION else torch.zeros(BS, W, 0, device=DEVICE)
        nll_loss = 0.
        for k in range(K):
            x = torch.cat([ss, aa], 2) if INCLUDE_ACTION else ss
            mu, lv = model(x)
            nll_loss += ROLLOUT_W[k] * crit(mu, lv, st[:, k])
            ss = torch.cat([ss[:, 1:], mu.detach().unsqueeze(1)], 1)
            if INCLUDE_ACTION:
                aa = torch.cat([aa[:, 1:], at[:, k:k+1]], 1)

        nll_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        total_nll += nll_loss.item()
    return total_nll / STEPS


@torch.no_grad()
def validate(model, raw, n=200):
    model.eval(); W, K = cfg.WINDOW_SIZE, ROLLOUT_K; N = len(raw)
    m0, m4 = 0., 0.
    for _ in range(n):
        i = np.random.randint(0, N-W-K)
        sw = raw[i:i+W, :cfg.N_STATE]; aw = raw[i:i+W, cfg.N_STATE:]
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1) if INCLUDE_ACTION else sw).unsqueeze(0).to(DEVICE)
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
        xt = torch.FloatTensor(np.concatenate([sw, aw], 1) if INCLUDE_ACTION else sw).unsqueeze(0).to(DEVICE)
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
        r = {}
        for d in abs_deltas:
            dT = {s: [] for s in [1, 3, 8, 12]}
            for i in idxs:
                sh = raw[i:i+W, :cfg.N_STATE].copy()
                ah = raw[i:i+W, cfg.N_STATE:].copy()
                fa = raw[i+W:i+W+13, cfg.N_STATE:].copy()
                xt = torch.FloatTensor(np.concatenate([sh, ah], 1) if INCLUDE_ACTION else sh).unsqueeze(0).to(DEVICE)
                at = torch.FloatTensor(fa).unsqueeze(0).to(DEVICE)
                trb = model.rollout(xt, at, mode='sliding')
                bp = trb[0,:,cfg.TARGET_IDX].cpu().numpy()
                if INCLUDE_ACTION:
                    ap = ah.copy(); ap[-1, adim] = np.clip(ap[-1, adim]+d, 0, 100)
                    fap = fa.copy(); fap[0, adim] = np.clip(fap[0, adim]+d, 0, 100)
                    xt = torch.FloatTensor(np.concatenate([sh, ap], 1)).unsqueeze(0).to(DEVICE)
                    at = torch.FloatTensor(fap).unsqueeze(0).to(DEVICE)
                    trp = model.rollout(xt, at, mode='sliding')
                    pp = trp[0,:,cfg.TARGET_IDX].cpu().numpy()
                else:
                    pp = bp  # 无动作 → 敏感性=0 (协议完整性)
                for s in [1, 3, 8, 12]:
                    dT[s].append(pp[s]-bp[s])
            for s in [1, 3, 8, 12]:
                r[f'{d}_{s}'] = float(np.mean(dT[s]))
        results[f'action_{adim}'] = r
    return results


def main():
    model = LSTMDynamics().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    exp_dir = f"results/exp_018_{VARIANT}"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        crit.beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        nll = train_epoch(model, train_data, opt, crit, ep)
        v0, v4 = validate(model, val_data); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{nll:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}")
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
    if INCLUDE_ACTION:
        print(f"  Sens (二级 ±10, t1/t12): {sens['action_1']['10.0_1']:+.3f} / {sens['action_1']['10.0_12']:+.3f}")

    result = {'config': VARIANT, 'include_action': INCLUDE_ACTION, 'model': 'LSTM',
              'hidden': LSTM_HIDDEN, 'layers': LSTM_LAYERS, 'best_epoch': be,
              'rollout_mae': mae.tolist(), 'sensitivity': sens}
    with open(f"{exp_dir}/results.json", 'w') as f: json.dump(result, f, indent=2)
    print(f"Saved: {exp_dir}/results.json")


if __name__ == '__main__':
    main()
