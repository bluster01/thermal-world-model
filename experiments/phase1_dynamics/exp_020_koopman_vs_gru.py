#!/usr/bin/env python3
"""
exp_020_koopman_vs_gru.py — Koopman 线性化 Decoder vs GRU Decoder 对比实验
=============================================================================
目标: 验证 Koopman 线性动力学是否能在不损失精度的情况下替代 GRU

三条路线:
  A: GRU baseline (当前主力)
  B: Koopman (线性隐空间, z_{t+1}=K·z_t+B·a_t)
  C: Neural ODE (ODE求解器, 连续时间)

设计:
  - 同一 Encoder (TCN+VarAttn), 不同 Decoder
  - 训练 50 epoch, 对比 rollout MAE@H=18 和每步耗时
  - 目标变量: 末级过热器出口汽温 (cfg.TARGET_IDX=10 in 11-dim state)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import config as cfg
from data_loader import load_raw_data

DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ===== 配置 =====
ROLLOUT_K = 5
BS, STEPS_PER_EPOCH = 256, 300
N_EPOCHS = 20
LR = 0.001

# ===== BetaNLL Loss =====
class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0., eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps
    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20., 20.)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu)**2 / v)
        if self.beta != 0: nll = v.detach()**self.beta * nll
        return nll.mean()


# ===== 共享 Encoder =====
from world_model import RevIN, PatchEmbedding, PerVariableTCN, VariableAttention

class SharedEncoder(nn.Module):
    """TCN+VarAttn Encoder, 输出 z_t ∈ R^{d_model}"""
    def __init__(self, n_state, d_model=64):
        super().__init__()
        self.n_state = n_state
        self.d_model = d_model
        self.revin = RevIN(n_state)
        self.patch_embed = PatchEmbedding(cfg.WINDOW_SIZE, 16, 8, d_model)
        self.n_patches = self.patch_embed.n_patches
        self.tcn = PerVariableTCN(self.n_patches, d_model, cfg.N_TCN_LAYERS, cfg.DROPOUT)
        self.var_attn = nn.ModuleList([
            VariableAttention(d_model, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_VAR_LAYERS)
        ])
        self.z_proj = nn.Linear(n_state * d_model, d_model)

    def forward(self, s):
        """s: [B, W, n_state] → z_t: [B, d_model]"""
        B, W, _ = s.shape
        d = self.d_model
        s_norm = self.revin(s, mode='norm')
        var_tokens = [self.patch_embed(s_norm[:, :, i]) for i in range(self.n_state)]
        var_tokens = torch.stack(var_tokens, 1)
        var_tokens = var_tokens.reshape(B * self.n_state, self.n_patches, d)
        s_repr = self.tcn(var_tokens).reshape(B, self.n_state, d)
        for attn in self.var_attn:
            s_repr, _ = attn(s_repr)
        return self.z_proj(s_repr.reshape(B, -1))


# ===== Decoder A: GRU (baseline) =====
class GRUDecoder(nn.Module):
    def __init__(self, d_hidden=64, n_action=2, n_state=11):
        super().__init__()
        self.action_embed = nn.Sequential(
            nn.Linear(n_action, d_hidden // 4), nn.GELU(),
            nn.Linear(d_hidden // 4, d_hidden // 2),
        )
        self.gru_cell = nn.GRUCell(d_hidden + d_hidden // 2, d_hidden)
        self.state_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2), nn.GELU(),
            nn.Linear(d_hidden * 2, n_state * 2),
        )
        self.n_state = n_state

    def rollout(self, z_t, a_seq):
        B, H, _ = a_seq.shape
        h = z_t
        traj_mu, traj_lv = [], []
        for t in range(H):
            a_emb = self.action_embed(a_seq[:, t, :])
            h = self.gru_cell(torch.cat([h, a_emb], -1), h)
            raw = self.state_head(h)
            traj_mu.append(raw[:, :self.n_state])
            traj_lv.append(raw[:, self.n_state:])
        return torch.stack(traj_mu, 1), torch.stack(traj_lv, 1)


# ===== Decoder B: Koopman =====
class KoopmanDecoder(nn.Module):
    """
    Koopman 线性化 Decoder:
      z_{t+1} = K @ z_t + B(a_t)
      其中 K = diag(λ) 用特征值参数化, 保证稳定性 |λ| ≤ 1
    """
    def __init__(self, d_hidden=64, n_action=2, n_state=11):
        super().__init__()
        self.d_hidden = d_hidden

        # 控制矩阵 B: 动作 → 隐空间偏移
        self.B = nn.Sequential(
            nn.Linear(n_action, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, d_hidden),
        )

        # Koopman 特征值 (对角化 K)
        # 参数化: λ_k = tanh(α_k) · exp(i·2π·σ(β_k)), 保证 |λ| < 1
        self.alpha = nn.Parameter(torch.zeros(d_hidden))  # 衰减率参数
        self.beta = nn.Parameter(torch.randn(d_hidden) * 0.1)  # 频率参数

        # 解码器: 隐空间 → 状态 (μ, log_σ²)
        self.decoder = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2), nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(d_hidden * 2, n_state * 2),
        )
        self.n_state = n_state

    def get_K_matrix(self):
        """构造对角化 Koopman 矩阵 K = diag(λ_k)"""
        mag = torch.tanh(self.alpha)  # |λ| ∈ [0, 1)
        # 频率约束在 [0, 0.5] 内 (避免 Nyquist)
        freq = torch.sigmoid(self.beta) * 0.4
        re = mag * torch.cos(2 * np.pi * freq)
        im = mag * torch.sin(2 * np.pi * freq)
        # 实特征值 (对角)
        return torch.diag(re)

    def stability_loss(self):
        """约束: 所有特征值模长 ≤ 0.995"""
        mag = torch.tanh(self.alpha)
        return F.relu(mag - 0.995).sum()

    def rollout(self, z_t, a_seq):
        B, H, _ = a_seq.shape
        K = self.get_K_matrix()  # [d_hidden, d_hidden]
        z = z_t
        traj_mu, traj_lv = [], []
        for t in range(H):
            z = z @ K.T + self.B(a_seq[:, t, :])  # 线性演化!
            raw = self.decoder(z)
            traj_mu.append(raw[:, :self.n_state])
            traj_lv.append(raw[:, self.n_state:])
        return torch.stack(traj_mu, 1), torch.stack(traj_lv, 1)


# ===== Decoder C: Neural ODE (简化版) =====
class NeuralODEDecoder(nn.Module):
    """
    用 Euler 步模拟 ODE 求解器 (避免 torchdiffeq 依赖)
    dh/dt = f_NN(h, a_emb), 步长 dt=0.1, 积分 10 步 = 1 时间单位
    """
    def __init__(self, d_hidden=64, n_action=2, n_state=11, n_steps=10, dt=0.1):
        super().__init__()
        self.n_steps = n_steps
        self.dt = dt
        self.action_embed = nn.Sequential(
            nn.Linear(n_action, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, d_hidden),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 4), nn.Tanh(),
            nn.Linear(d_hidden * 4, d_hidden),
        )
        self.state_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2), nn.GELU(),
            nn.Linear(d_hidden * 2, n_state * 2),
        )
        self.n_state = n_state

    def step_ode(self, h, a_emb):
        """Euler step: h += dt * f(h, a_emb)"""
        dh = self.dynamics(torch.cat([h, a_emb], -1))
        return h + self.dt * dh

    def rollout(self, z_t, a_seq):
        B, H, _ = a_seq.shape
        h = z_t
        traj_mu, traj_lv = [], []
        for t in range(H):
            a_emb = self.action_embed(a_seq[:, t, :])
            # ODE 积分 (n_steps 小步)
            for _ in range(self.n_steps):
                h = self.step_ode(h, a_emb)
            raw = self.state_head(h)
            traj_mu.append(raw[:, :self.n_state])
            traj_lv.append(raw[:, self.n_state:])
        return torch.stack(traj_mu, 1), torch.stack(traj_lv, 1)


# ===== 完整 WorldModel =====
class PhysicsWorldModel(nn.Module):
    def __init__(self, decoder_type='gru'):
        super().__init__()
        self.encoder = SharedEncoder(cfg.N_STATE, cfg.D_MODEL)
        if decoder_type == 'gru':
            self.decoder = GRUDecoder(cfg.D_MODEL, cfg.N_ACTION, cfg.N_STATE)
        elif decoder_type == 'koopman':
            self.decoder = KoopmanDecoder(cfg.D_MODEL, cfg.N_ACTION, cfg.N_STATE)
        elif decoder_type == 'neural_ode':
            self.decoder = NeuralODEDecoder(cfg.D_MODEL, cfg.N_ACTION, cfg.N_STATE)
        self.decoder_type = decoder_type

    def forward(self, s_hist, a_seq):
        """s_hist: [B, W, n_state], a_seq: [B, H, n_action]"""
        z_t = self.encoder(s_hist)
        return self.decoder.rollout(z_t, a_seq)


# ===== 数据加载 =====
state_data, delta_actions, valve_abs = load_raw_data()
# 用绝对阀位作为动作 (valve_abs 是 2 维: 一级/二级减温阀位)
raw_data = np.concatenate([state_data, valve_abs], axis=1)
n_total = len(raw_data)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.85)
print(f"Data: {n_total} total, {n_train} train, {n_val-n_train} val, {n_total-n_val} test")


# ===== 训练 =====
def train_model(decoder_type, n_epochs=N_EPOCHS):
    print(f"\n{'='*60}\nTraining: {decoder_type}\n{'='*60}")
    model = PhysicsWorldModel(decoder_type).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = BetaNLLLoss(beta=-0.3)
    W = cfg.WINDOW_SIZE
    K = ROLLOUT_K
    train_raw = raw_data[:n_train]
    val_raw = raw_data[n_train:n_val]

    best_val_mae = float('inf')
    history = []

    for epoch in range(1, n_epochs + 1):
        # --- Train ---
        model.train()
        total_nll, total_stab = 0., 0.
        for step in range(STEPS_PER_EPOCH):
            idxs = np.random.randint(0, len(train_raw) - W - K, size=BS)
            s_hist_b, a_seq_b, s_true_b = [], [], []
            for i in idxs:
                s_hist_b.append(train_raw[i:i+W, :cfg.N_STATE])
                a_seq_b.append(train_raw[i+W:i+W+K, cfg.N_STATE:])
                s_true_b.append(train_raw[i+W:i+W+K, :cfg.N_STATE])
            sh = torch.FloatTensor(np.stack(s_hist_b)).to(DEVICE)
            aq = torch.FloatTensor(np.stack(a_seq_b)).to(DEVICE)
            st = torch.FloatTensor(np.stack(s_true_b)).to(DEVICE)

            opt.zero_grad()
            mu_traj, lv_traj = model(sh, aq)  # [B, K, n_state]
            nll = sum(crit(mu_traj[:, k], lv_traj[:, k], st[:, k]) for k in range(K))
            stab = model.decoder.stability_loss() if decoder_type == 'koopman' else torch.tensor(0.)
            loss = nll + 0.01 * stab
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total_nll += nll.item()
            total_stab += stab.item()

        # --- Val ---
        model.eval()
        val_mae = 0.
        with torch.no_grad():
            for _ in range(100):
                i = np.random.randint(0, len(val_raw) - W - K)
                sh = torch.FloatTensor(val_raw[i:i+W, :cfg.N_STATE]).unsqueeze(0).to(DEVICE)
                aq = torch.FloatTensor(val_raw[i+W:i+W+K, cfg.N_STATE:]).unsqueeze(0).to(DEVICE)
                st = torch.FloatTensor(val_raw[i+W:i+W+K, :cfg.N_STATE]).unsqueeze(0).to(DEVICE)
                mu_traj, _ = model(sh, aq)
                val_mae += (mu_traj[0, -1, cfg.TARGET_IDX] - st[0, -1, cfg.TARGET_IDX]).abs().item()
        val_mae /= 100

        # --- Koopman 谱信息 ---
        if decoder_type == 'koopman':
            alpha = model.decoder.alpha.detach().cpu()
            mag = torch.tanh(alpha)
            n_unstable = (mag > 0.99).sum().item()
            print(f"Epoch {epoch:3d} | NLL={total_nll/STEPS_PER_EPOCH:.4f} "
                  f"stab={total_stab/STEPS_PER_EPOCH:.4f} "
                  f"val_MAE={val_mae:.4f} "
                  f"|λ|_max={mag.max():.4f} unstable={n_unstable}")
        else:
            print(f"Epoch {epoch:3d} | NLL={total_nll/STEPS_PER_EPOCH:.4f} "
                  f"val_MAE={val_mae:.4f}")

        history.append({'epoch': epoch, 'nll': total_nll/STEPS_PER_EPOCH, 'val_mae': val_mae})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_val_mae


# ===== 主实验 =====
if __name__ == '__main__':
    results = {}
    for dtype in ['gru', 'koopman', 'neural_ode']:
        t0 = time.time()
        model, hist, best_mae = train_model(dtype)
        elapsed = time.time() - t0

        # 速度 benchmark: 100次 rollout H=18
        W = cfg.WINDOW_SIZE
        test_raw = raw_data[n_val:]
        n_samples = 100
        s_batch, a_batch = [], []
        for i in range(n_samples):
            s_batch.append(test_raw[i:i+W, :cfg.N_STATE])
            a_batch.append(test_raw[i+W:i+W+18, cfg.N_STATE:])
        s_test = torch.FloatTensor(np.stack(s_batch)).to(DEVICE)
        a_test = torch.FloatTensor(np.stack(a_batch)).to(DEVICE)

        model.eval()
        with torch.no_grad():
            t1 = time.time()
            for _ in range(100):
                model(s_test, a_test)
            infer_time = (time.time() - t1) / (100 * 100) * 1000  # ms per rollout

        results[dtype] = {
            'best_val_mae': best_mae,
            'train_time': elapsed,
            'infer_ms': infer_time,
            'history': hist,
        }
        print(f"\n{dtype}: val_MAE={best_mae:.4f}, train={elapsed:.0f}s, infer={infer_time:.2f}ms/rollout")

    # --- 对比总结 ---
    print("\n" + "="*60)
    print("对比总结")
    print("="*60)
    baseline_mae = results['gru']['best_val_mae']
    baseline_speed = results['gru']['infer_ms']
    for dtype in ['gru', 'koopman', 'neural_ode']:
        r = results[dtype]
        mae_delta = (r['best_val_mae'] - baseline_mae) / baseline_mae * 100
        speed_ratio = baseline_speed / r['infer_ms']
        print(f"  {dtype:12s}: MAE={r['best_val_mae']:.4f} ({mae_delta:+.1f}%), "
              f"infer={r['infer_ms']:.2f}ms ({speed_ratio:.1f}x vs GRU), "
              f"train={r['train_time']:.0f}s")

    # 保存
    out_dir = f"results/exp_020_koopman_vs_gru"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", 'w') as f:
        json.dump({k: {'best_val_mae': v['best_val_mae'],
                       'train_time': v['train_time'],
                       'infer_ms': v['infer_ms']}
                   for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")
