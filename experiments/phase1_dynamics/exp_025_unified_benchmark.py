"""
exp_025_unified_benchmark.py — Phase 1 收口 v2: RevIN 公平协议横屏对比 + Direct WM 消融
=========================================================================================
v2 修正 (2026-08-01):
- 归一化: 所有模型默认 RevIN (实例归一化, 与 exp_023 一致, 公平同条件)
- 全局 min-max 仅作为 M6 (RevIN 消融对照)
- VarAttn 确认有用 (v1 结论), 加入 M0 全量
- 顺序: 先消融 M0-M6, 再 baseline B1-B6 (全部 + RevIN)
- loss 在物理空间 (σ 用窗口统计, 与 exp_023 相同)

模型:
  M0  Direct WM 全量 = RevIN + Patch + PerVarTCN + VarAttn + 动作注入 + β-NLL
  M1  -动作   M2 -Patch   M3 -PerVarTCN(合并)   M4 -VarAttn   M5 确定性head
  M6  全局min-max版 (RevIN 消融对照)
  B1  TCN     B2 LSTM    B3 GRU    B4 iTransformer   B5 DLinear   B6 Exp-0重跑
用法: python exp_025_unified_benchmark.py <M0..M6|B1..B6>
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config as cfg
from world_model import RevIN, PatchEmbedding, PerVariableTCN, VariableAttention

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else 'M0'

BETA, BETA_WARMUP = -0.3, 20
BS, STEPS = 256, 500
H_OUT = 18
EPOCHS = cfg.EPOCHS
PATIENCE = cfg.EARLY_STOPPING_PATIENCE

# ===== 数据 =====
CSV_PATH = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE)
df_full = pd.read_csv(CSV_PATH)
NUMERIC_COLS = [c for c in df_full.columns if c != 'date']
N_FEAT = len(NUMERIC_COLS)
TARGET_IDX = NUMERIC_COLS.index('末级过热器出口汽温')
VALVE_IDX = [NUMERIC_COLS.index('一级减温调节门阀位'),
             NUMERIC_COLS.index('二级减温调节门阀位')]

data_all = df_full[NUMERIC_COLS].values.astype(np.float32)
data_all = np.nan_to_num(data_all, nan=0.0)
n_total = len(data_all)
n_train = int(n_total * 0.70); n_val_end = int(n_total * 0.85)
train_raw, val_raw = data_all[:n_train], data_all[n_train:n_val_end]
test_raw = data_all[n_val_end:]

# 全局 min-max (仅 M6 用) — 进出口归一化: 数据管线归一化, 模型吃归一化数据
train_min = train_raw.min(0); train_max = train_raw.max(0)
span_g = np.maximum(train_max - train_min, 1e-6)
def norm_g(x): return (x - train_min) / span_g
train_data_norm = norm_g(train_raw); val_data_norm = norm_g(val_raw); test_data_norm = norm_g(test_raw)
print(f"数据: {len(train_raw)}+{len(val_raw)}+{len(test_raw)} | 40列 | 目标idx={TARGET_IDX}")


# ===== 损失 =====
class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()

class MSELoss_(nn.Module):
    def forward(self, mu, lv, tgt): return nn.functional.mse_loss(mu, tgt)


# ===== RevIN 基类: 所有模型共享的归一化+反归一化 (公平同条件) =====
class RevINModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.revin = RevIN(N_FEAT)
        self.use_revin = True

    def denorm_out(self, mu_n, lv_n):
        """归一化输出 → 物理空间 (用当前窗口 RevIN 统计, 同 exp_023)"""
        ms = self.revin._mean[:, :, TARGET_IDX]    # [B,1]
        ss = self.revin._std[:, :, TARGET_IDX]     # [B,1]
        w = self.revin.weight[TARGET_IDX]; b = self.revin.bias[TARGET_IDX]
        mu_n2 = mu_n
        if self.revin.affine: mu_n2 = (mu_n2 - b) / (w + self.revin.eps)
        mu = mu_n2 * ss + ms
        if lv_n is not None:
            sig = torch.exp(lv_n * 0.5) * ss
            lv = 2.0 * torch.log(sig + 1e-8)
            return mu, lv
        return mu, None


# ===== 模型 =====
class DirectWM(RevINModel):
    """Direct WM 模块化 (flags 控制组件) — 默认含 VarAttn"""
    def __init__(self, use_action=True, use_patch=True, per_variable=True,
                 use_varattn=True, probabilistic=True, beta_mode='warmup'):
        super().__init__()
        self.beta_mode = beta_mode  # 'warmup' | 'fixed' | 'warmup_pos'
        d = cfg.D_MODEL; W = cfg.WINDOW_SIZE
        self.use_action, self.use_patch = use_action, use_patch
        self.per_variable, self.probabilistic = per_variable, probabilistic
        self.use_varattn = use_varattn
        if use_patch:
            self.patch = PatchEmbedding(W, 16, 8, d); self.np = self.patch.n_patches
        else:
            self.proj = nn.Linear(1, d); self.np = W
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)
        self.varattn = VariableAttention(d, 4, cfg.DROPOUT) if use_varattn else None
        a_dim = d * 2 if use_action else 0
        n_tokens = N_FEAT if per_variable else 1
        self.action_enc = nn.Sequential(
            nn.Linear(H_OUT * 2, d * 2), nn.GELU(), nn.Dropout(cfg.DROPOUT)) if use_action else None
        in_dim = n_tokens * d + a_dim
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, H_OUT * 2 if probabilistic else H_OUT),
        )

    def forward(self, x_hist, a_future=None):
        B = x_hist.shape[0]; d = cfg.D_MODEL
        x_n = self.revin(x_hist, mode='norm')
        if self.use_patch:
            var_tokens = [self.patch(x_n[:, :, i]) for i in range(N_FEAT)]
            var_tokens = torch.stack(var_tokens, 1)  # [B, N, np, d]
        else:
            var_tokens = [self.proj(x_n[:, :, i:i+1]) for i in range(N_FEAT)]
            var_tokens = torch.stack(var_tokens, 1)  # [B, N, W, d]
        if self.per_variable:
            vt = var_tokens.reshape(B * N_FEAT, self.np, d)
            s_repr = self.tcn(vt).reshape(B, N_FEAT, d)  # [B, N, d]
        else:
            vt = var_tokens.permute(0, 2, 1, 3).reshape(B, self.np * N_FEAT, d)
            s_repr = self.tcn(vt)  # [B, d]
        if self.varattn:
            assert self.per_variable, "VarAttn 需要 per-variable 表示"
            s_repr, _ = self.varattn(s_repr)
        if self.use_action:
            a_feat = self.action_enc(a_future.reshape(B, -1))
            z = torch.cat([s_repr.reshape(B, -1), a_feat], 1)
        else:
            z = s_repr.reshape(B, -1)
        raw = self.decoder(z)
        if self.probabilistic:
            raw = raw.reshape(B, H_OUT, 2)
            mu_n, lv_n = raw[..., 0], raw[..., 1]
        else:
            mu_n, lv_n = raw.reshape(B, H_OUT), None
        return self.denorm_out(mu_n, lv_n)


class GlobalMinMaxWM(nn.Module):
    """M6: 全局 min-max 归一化版 (RevIN 消融对照) — 进出口归一化
    数据管线归一化 (get_data 返回归一化数据), 模型吃归一化输入, 输出归一化 μ/σ。
    loss 在归一化空间 (σ 初始 1 天然合理, 无膨胀问题)。eval 时反归一化到物理。"""
    def __init__(self, probabilistic=True):
        super().__init__()
        d = cfg.D_MODEL; W = cfg.WINDOW_SIZE
        self.probabilistic = probabilistic
        self.use_revin = False
        self.use_action = True
        self.use_patch = True
        self.norm_output = True  # 输出在归一化空间 (eval 需反归一化)
        self.patch = PatchEmbedding(W, 16, 8, d); self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)
        self.varattn = VariableAttention(d, 4, cfg.DROPOUT)
        self.action_enc = nn.Sequential(
            nn.Linear(H_OUT * 2, d * 2), nn.GELU(), nn.Dropout(cfg.DROPOUT))
        self.decoder = nn.Sequential(
            nn.Linear(N_FEAT * d + d * 2, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, d * 4), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(d * 4, H_OUT * 2 if probabilistic else H_OUT),
        )

    def forward(self, x_hist, a_future=None):
        B = x_hist.shape[0]; d = cfg.D_MODEL
        var_tokens = [self.patch(x_hist[:, :, i]) for i in range(N_FEAT)]
        var_tokens = torch.stack(var_tokens, 1).reshape(B * N_FEAT, self.np, d)
        s_repr = self.tcn(var_tokens).reshape(B, N_FEAT, d)
        s_repr, _ = self.varattn(s_repr)
        a_feat = self.action_enc(a_future.reshape(B, -1))
        z = torch.cat([s_repr.reshape(B, -1), a_feat], 1)
        raw = self.decoder(z)
        if self.probabilistic:
            raw = raw.reshape(B, H_OUT, 2)
            return raw[..., 0], raw[..., 1]  # 归一化空间
        return raw.reshape(B, H_OUT), None


class TCNBaseline(RevINModel):
    """B1 TCN: 全变量拼接 → 共享TCN → direct 输出"""
    def __init__(self, probabilistic=True):
        super().__init__()
        d = cfg.D_MODEL
        self.probabilistic = probabilistic
        self.proj = nn.Linear(N_FEAT, d)
        self.tcn = PerVariableTCN(cfg.WINDOW_SIZE, d, cfg.N_TCN_LAYERS, cfg.DROPOUT)
        self.head = nn.Linear(d, H_OUT * 2 if probabilistic else H_OUT)
    def forward(self, x_hist, a_future=None):
        x_n = self.revin(x_hist, mode='norm')   # [B, W, N]
        x = self.proj(x_n)                       # [B, W, d]
        x = self.tcn(x)                          # [B, d]
        out = self.head(x).reshape(-1, H_OUT, 2 if self.probabilistic else 1)
        if self.probabilistic:
            return self.denorm_out(out[..., 0], out[..., 1])
        return self.denorm_out(out[..., 0], None)


class RecurrentBaseline(RevINModel):
    """B2/B3 LSTM/GRU: 统一循环骨干 direct 输出"""
    def __init__(self, kind='lstm', probabilistic=True):
        super().__init__()
        d = cfg.D_MODEL
        self.probabilistic = probabilistic
        rnn_cls = nn.LSTM if kind == 'lstm' else nn.GRU
        self.rnn = rnn_cls(N_FEAT, d, 2, dropout=cfg.DROPOUT, batch_first=True)
        self.head = nn.Linear(d, H_OUT * 2 if probabilistic else H_OUT)
    def forward(self, x_hist, a_future=None):
        x_n = self.revin(x_hist, mode='norm')
        out, _ = self.rnn(x_n)
        out = self.head(out[:, -1]).reshape(-1, H_OUT, 2 if self.probabilistic else 1)
        if self.probabilistic:
            return self.denorm_out(out[..., 0], out[..., 1])
        return self.denorm_out(out[..., 0], None)


class iTransformerBaseline(RevINModel):
    """B4 iTransformer: 变量通道注意力 + 线性时序映射 (变量即token)"""
    def __init__(self, probabilistic=True):
        super().__init__()
        d = cfg.D_MODEL
        self.probabilistic = probabilistic
        self.seq_lin = nn.Linear(cfg.WINDOW_SIZE, d)
        self.var_attn = VariableAttention(d, 4, cfg.DROPOUT)
        self.head = nn.Linear(d, H_OUT * 2 if probabilistic else H_OUT)
    def forward(self, x_hist, a_future=None):
        x_n = self.revin(x_hist, mode='norm')
        xt = self.seq_lin(x_n.permute(0, 2, 1))  # [B, N, d]
        z, _ = self.var_attn(xt)
        zt = z[:, TARGET_IDX]
        out = self.head(zt).reshape(-1, H_OUT, 2 if self.probabilistic else 1)
        if self.probabilistic:
            return self.denorm_out(out[..., 0], out[..., 1])
        return self.denorm_out(out[..., 0], None)


class DLinearBaseline(RevINModel):
    """B5 DLinear: 每变量 W→H 线性 + 变量加权"""
    def __init__(self, probabilistic=True):
        super().__init__()
        self.probabilistic = probabilistic
        self.lin = nn.Linear(cfg.WINDOW_SIZE, H_OUT)
        self.var_w = nn.Linear(N_FEAT, 1)
        self.head = nn.Linear(1, 2 if probabilistic else 1)
    def forward(self, x_hist, a_future=None):
        x_n = self.revin(x_hist, mode='norm')
        z = self.lin(x_n.permute(0, 2, 1))          # [B, N, H]
        z = self.var_w(z.permute(0, 2, 1)).squeeze(-1)  # [B, H]
        out = self.head(z.unsqueeze(-1)).squeeze(-1)    # [B, H, 1/2]→[B,H]
        out = out.reshape(-1, H_OUT, 2 if self.probabilistic else 1)
        if self.probabilistic:
            return self.denorm_out(out[..., 0], out[..., 1])
        return self.denorm_out(out[..., 0], None)


def build_model(mid):
    if mid == 'M0': return DirectWM(use_action=True, use_patch=True, per_variable=True, use_varattn=True)
    if mid == 'M1': return DirectWM(use_action=False, use_patch=True, per_variable=True, use_varattn=True)
    if mid == 'M2': return DirectWM(use_action=True, use_patch=False, per_variable=True, use_varattn=True)
    if mid == 'M3': return DirectWM(use_action=True, use_patch=True, per_variable=False, use_varattn=False)  # 合并序列无变量维度, VarAttn 无意义
    if mid == 'M4': return DirectWM(use_action=True, use_patch=True, per_variable=True, use_varattn=False)
    if mid == 'M5': return DirectWM(use_action=True, use_patch=True, per_variable=True, use_varattn=True, probabilistic=False)
    if mid == 'M6': return GlobalMinMaxWM()
    if mid == 'M7': return DirectWM(use_action=True, use_patch=True, per_variable=True, use_varattn=True, beta_mode='fixed')   # β 固定 -0.3, 无 warmup
    if mid == 'M8': return DirectWM(use_action=True, use_patch=True, per_variable=True, use_varattn=True, beta_mode='warmup_pos')  # β warmup 到 +0.3 (对照)
    if mid == 'B1': return TCNBaseline()
    if mid == 'B2': return RecurrentBaseline('lstm')
    if mid == 'B3': return RecurrentBaseline('gru')
    if mid == 'B4': return iTransformerBaseline()
    if mid == 'B5': return DLinearBaseline()
    # B6 = Exp-0 架构重跑 (无动作 TCN-iTransformer) — 与 M1 同构, sanity check
    if mid == 'B6': return DirectWM(use_action=False, use_patch=True, per_variable=True, use_varattn=True)
    raise ValueError(mid)


# ===== 训练 =====
def get_data(mid):
    if mid == 'M6':
        # 进出口 min-max: 数据管线归一化 (train 统计), 含动作列
        return train_data_norm, val_data_norm, test_data_norm
    return train_raw, val_raw, test_raw  # 其他模型吃原始数据, RevIN 在模型内部


def train_epoch(model, raw, opt, crit, probabilistic):
    model.train(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    total = 0.
    for _ in range(STEPS):
        idxs = np.random.randint(0, N-W-H, size=BS)
        xh, af, tt = [], [], []
        for i in idxs:
            xh.append(raw[i:i+W]); af.append(raw[i+W:i+W+H, VALVE_IDX])
            tt.append(raw[i+W:i+W+H, TARGET_IDX])
        x_hist = torch.FloatTensor(np.stack(xh)).to(DEVICE)
        a_fut = torch.FloatTensor(np.stack(af)).to(DEVICE)
        t_true = torch.FloatTensor(np.stack(tt)).to(DEVICE)
        opt.zero_grad()
        mu, lv = model(x_hist, a_fut)
        w = torch.linspace(1.0, 0.6, H, device=DEVICE)
        if probabilistic:
            loss = (w * crit(mu, lv, t_true).mean(dim=0)).sum() / H
        else:
            loss = (w * crit(mu, None, t_true).mean(dim=0)).sum() / H
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); total += loss.item()
    return total / STEPS


@torch.no_grad()
def validate(model, raw, probabilistic, n=200):
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    norm_out = getattr(model, 'norm_output', False)
    m0, m4 = 0., 0.
    for _ in range(n):
        i = np.random.randint(0, N-W-H)
        xh = torch.FloatTensor(raw[i:i+W]).unsqueeze(0).to(DEVICE)
        af = torch.FloatTensor(raw[i+W:i+W+H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        mu, _ = model(xh, af)
        # 目标: raw 是归一化数据 (M6) 或物理数据 (其他) — 统一转到物理比较
        if norm_out:
            mu = mu * span_g[TARGET_IDX] + train_min[TARGET_IDX]
            t0 = raw[i+W, TARGET_IDX] * span_g[TARGET_IDX] + train_min[TARGET_IDX]
            t4 = raw[i+W+4, TARGET_IDX] * span_g[TARGET_IDX] + train_min[TARGET_IDX]
        else:
            t0, t4 = raw[i+W, TARGET_IDX], raw[i+W+4, TARGET_IDX]
        m0 += (mu[0,0]-t0).abs().item()
        m4 += (mu[0,4]-t4).abs().item()
    return m0/n, m4/n


@torch.no_grad()
def eval_rollout(model, raw, probabilistic, n=500):
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    norm_out = getattr(model, 'norm_output', False)
    np.random.seed(42); idxs = np.random.choice(range(N-W-H), n, replace=False)
    err = np.zeros((n, H))
    for j, i in enumerate(idxs):
        xh = torch.FloatTensor(raw[i:i+W]).unsqueeze(0).to(DEVICE)
        af = torch.FloatTensor(raw[i+W:i+W+H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        mu, _ = model(xh, af)
        if norm_out:
            mu = mu * span_g[TARGET_IDX] + train_min[TARGET_IDX]
            tt = raw[i+W:i+W+H, TARGET_IDX] * span_g[TARGET_IDX] + train_min[TARGET_IDX]
        else:
            tt = raw[i+W:i+W+H, TARGET_IDX]
        err[j] = np.abs(mu[0].cpu().numpy() - tt)
    return err.mean(0)


@torch.no_grad()
def eval_sensitivity(model, raw, n=200):
    """扰动未来动作首步 → 轨迹响应 (exp_023 协议: 首步干预, 打破共因)
    关键: 全步扰动测到的是 PID 共因统计(高阀位↔高温), 首步扰动才测因果"""
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    np.random.seed(7); idxs = np.random.choice(range(N-W-H), n, replace=False)
    abs_deltas = [-10., -5., -2., -1., 1., 2., 5., 10.]
    norm_out = getattr(model, 'norm_output', False)
    # 物理扰动 d (阀位单位) → 归一化空间扰动 (M6 动作已归一化)
    d_scale = np.ones(2)
    clamp_max = 100.0
    if norm_out:
        d_scale = 1.0 / (span_g[VALVE_IDX] + 1e-9)
        clamp_max = 1.0
    results = {}
    for adim in range(2):
        r = {}
        for d in abs_deltas:
            dT = {s: [] for s in [1, 3, 8, 12]}
            d_n = d * d_scale[adim]
            for i in idxs:
                x_hist = torch.FloatTensor(raw[i:i+W]).unsqueeze(0).to(DEVICE)
                a_fut = torch.FloatTensor(raw[i+W:i+W+H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
                mu_b, _ = model(x_hist, a_fut)
                bp = mu_b[0].cpu().numpy()
                a_p = a_fut.clone(); a_p[0, 0, adim] = torch.clamp(a_p[0, 0, adim] + d_n, 0, clamp_max)
                mu_p, _ = model(x_hist, a_p)
                pp = mu_p[0].cpu().numpy()
                for s in [1, 3, 8, 12]:
                    dT[s].append(pp[s] - bp[s])
            for s in [1, 3, 8, 12]:
                r[f'{d}_{s}'] = float(np.mean(dT[s]))
        results[f'action_{adim}'] = r
    if norm_out:
        # 反归一化: 输出是归一化空间, 扰动 d 也是归一化空间 → 结果 *span 到物理
        for adim in range(2):
            for k in results[f'action_{adim}']:
                results[f'action_{adim}'][k] = float(results[f'action_{adim}'][k] * span_g[TARGET_IDX])
    return results


@torch.no_grad()
def eval_sigma_calib(model, raw, n=300):
    """|error|/σ 校准 (理想=1.0) — 仅概率模型"""
    model.eval(); W = cfg.WINDOW_SIZE; H = H_OUT; N = len(raw)
    norm_out = getattr(model, 'norm_output', False)
    np.random.seed(11); idxs = np.random.choice(range(N-W-H), n, replace=False)
    ratios = []
    for i in idxs:
        xh = torch.FloatTensor(raw[i:i+W]).unsqueeze(0).to(DEVICE)
        af = torch.FloatTensor(raw[i+W:i+W+H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        mu, lv = model(xh, af)
        if lv is None: return None
        if norm_out:
            mu = mu * span_g[TARGET_IDX] + train_min[TARGET_IDX]
            tt = raw[i+W:i+W+H, TARGET_IDX] * span_g[TARGET_IDX] + train_min[TARGET_IDX]
        else:
            tt = raw[i+W:i+W+H, TARGET_IDX]
        sig = torch.exp(lv * 0.5)
        if norm_out:
            sig = sig * span_g[TARGET_IDX]
        err = (mu[0].cpu().numpy() - tt)
        ratios.append(np.abs(err) / (sig[0].cpu().numpy() + 1e-8))
    return float(np.mean(ratios))


def main():
    np.random.seed(42); torch.manual_seed(42)  # 训练固定 seed (公平性)
    model = build_model(MODEL_ID).to(DEVICE)
    prob = model.probabilistic
    tr, va, te = get_data(MODEL_ID)
    print(f"Config: {MODEL_ID} | Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"probabilistic={prob} | RevIN={getattr(model,'use_revin',False)}")

    exp_dir = f"results/exp_025_{MODEL_ID}"
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

    crit = BetaNLLLoss(beta=BETA) if prob else MSELoss_()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    best_m, pc, be = float('inf'), 0, 0; t0 = time.time()
    beta_mode = getattr(model, 'beta_mode', 'warmup')
    for ep in range(1, EPOCHS + 1):
        if not prob:
            crit_beta = 0.
        elif beta_mode == 'fixed':
            crit_beta = BETA  # 全程固定 -0.3, 无 warmup
        elif beta_mode == 'warmup_pos':
            crit_beta = 0. if ep <= BETA_WARMUP else 0.3 * min((ep - BETA_WARMUP) / 10, 1.)  # warmup 到 +0.3
        else:
            crit_beta = 0. if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.)
        crit.beta = crit_beta
        nll = train_epoch(model, tr, opt, crit, prob)
        v0, v4 = validate(model, va, prob); sched.step(v4)
        if ep % 10 == 0 or ep == 1:
            print(f"  E{ep:3d} | NLL:{nll:7.0f} | V0:{v0:.4f} | V4:{v4:.4f}")
        if v4 < best_m - 0.001: best_m, be, pc = v4, ep, 0; torch.save(
            {'epoch': ep, 'model_state_dict': model.state_dict()},
            f"{exp_dir}/checkpoints/best_model.pth")
        else: pc += 1
        if pc >= PATIENCE: print(f"  Stop@{ep} best@{be}"); break

    ck = torch.load(f"{exp_dir}/checkpoints/best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    train_min_ = time.time() - t0

    mae = eval_rollout(model, te, prob)
    print(f"\nRollout(test, °C): {mae[0]:.4f}→{mae[-1]:.4f} (×{mae[-1]/mae[0]:.1f}) avg={mae.mean():.4f}")

    result = {'model': MODEL_ID, 'params': sum(p.numel() for p in model.parameters()),
              'best_epoch': be, 'train_min': train_min_ / 60,
              'rollout_mae_degC': mae.tolist(), 'avg_mae_degC': float(mae.mean())}

    # 敏感性 (仅带动作模型): 首步扰动协议 (exp_023 同款)
    if getattr(model, 'use_action', False):
        sens = eval_sensitivity(model, te)
        result['sensitivity_degC'] = sens
        s1 = sens.get('action_1', {})
        print(f"  Sens 二级阀首步扰动 ±10: t1={s1.get('10.0_1', float('nan')):+.3f} "
              f"t8={s1.get('10.0_8', float('nan')):+.3f} t12={s1.get('10.0_12', float('nan')):+.3f}")

    # σ 校准 (仅概率模型)
    if prob:
        cal = eval_sigma_calib(model, te)
        if cal is not None:
            result['sigma_calib'] = cal
            print(f"  σ 校准 |err|/σ: {cal:.2f} (理想 1.0)")

    json.dump(result, open(f"{exp_dir}/results.json", 'w'), indent=2)
    print(f"Saved: {exp_dir}/results.json")


if __name__ == '__main__':
    main()
