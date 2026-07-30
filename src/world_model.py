"""
world_model.py — 动作条件化世界模型 v2
基于 Exp-0 TCN-iTransformer 改造:
- 输入: [s_{t-W:t} ‖ a_{t-W:t}] → 16维 (14状态 + 2动作)
- Encoder: TCN+VarAttn → z_t (历史压缩, 只跑一次)
- Decoder: GRU Cell → 逐步接收动作 a_t → 预测 ŝ_{t+1}
- 两种 rollout 模式可切换:
  * 'gru':    Encoder 1次 + GRU Cell H次 (推荐)
  * 'sliding': Encoder H次 + 滑动窗口 (对照)

v2 改动: +GRU Cell rollout, +encode()接口, +可切换模式
"""
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg


class RevIN(nn.Module):
    """可逆实例归一化 (同Exp-0)"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            x = x * self._std + self._mean
        return x


class PatchEmbedding(nn.Module):
    """Patching (同Exp-0)"""
    def __init__(self, seq_len, patch_len, stride, d_model):
        super().__init__()
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
    
    def forward(self, x):
        patches = x.unfold(dimension=1, size=cfg.PATCH_LEN, step=cfg.STRIDE)
        out = self.proj(patches)
        out = self.norm(out)
        out = out + self.pos_embed
        return out


class VariableAttention(nn.Module):
    """变量间注意力 (同Exp-0)"""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
    
    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, D)
        x = self.norm1(x + self.W_o(context))
        x = self.norm2(x + self.ffn(x))
        return x, attn


class LightTCNBlock(nn.Module):
    """轻量TCN块 (同Exp-0)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.chomp = padding
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    
    def forward(self, x):
        out = self.conv(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = self.act(out)
        out = self.drop(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class PerVariableTCN(nn.Module):
    """Per-variable TCN编码器 (共享参数, 同Exp-0)"""
    def __init__(self, n_patches, d_model, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        layers = []
        for i in range(n_layers):
            layers.append(LightTCNBlock(d_model, d_model, kernel_size=3, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        return x


class GRUStateDecoder(nn.Module):
    """
    GRU 状态解码器 (v2 新增)
    - 将 Encoder 输出 z_t 作为初始隐状态
    - 逐步接收动作 a_t, 预测全状态 ŝ_{t+1}
    - 隐状态 h 自动累积历史信息, 无需反复编码
    """
    def __init__(self, d_hidden, n_action, n_state, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_action = n_action
        
        # 动作嵌入
        self.action_embed = nn.Sequential(
            nn.Linear(n_action, d_hidden // 4),
            nn.GELU(),
            nn.Linear(d_hidden // 4, d_hidden // 2),
        )
        
        # GRU: 输入 = z_t拼接后的隐状态 + action嵌入
        # 实际输入维度 = d_hidden + d_hidden//2
        self.gru_cell = nn.GRUCell(d_hidden + d_hidden // 2, d_hidden)
        
        # 状态预测头
        self.state_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, n_state),
        )
    
    def forward(self, z_t, a_t, h_prev=None):
        """
        单步解码
        
        Args:
            z_t: [B, d_hidden] Encoder输出的历史压缩 (第一次传入, 后续为None时只用h_prev)
            a_t: [B, n_action] 当前步动作
            h_prev: [B, d_hidden] 上一步隐状态 (None时用z_t初始化)
        
        Returns:
            s_next: [B, n_state] 预测的下一状态
            h_new: [B, d_hidden] 更新后的隐状态
        """
        if h_prev is None:
            h_prev = z_t
        
        # 拼接条件: 历史编码 + 当前动作
        a_emb = self.action_embed(a_t)                    # [B, d_hidden//2]
        gru_input = torch.cat([h_prev, a_emb], dim=-1)   # [B, d_hidden + d_hidden//2]
        
        h_new = self.gru_cell(gru_input, h_prev)         # [B, d_hidden]
        s_next = self.state_head(h_new)                   # [B, n_state]
        
        return s_next, h_new
    
    def rollout(self, z_t, a_seq):
        """
        从 z_t 出发, 沿 a_seq 展开 H 步
        
        Args:
            z_t: [B, d_hidden]
            a_seq: [B, H, n_action]
        
        Returns:
            s_traj: [B, H, n_state] 预测状态轨迹
        """
        B, H, _ = a_seq.shape
        s_traj = []
        h = z_t
        
        for t in range(H):
            s_next, h = self.forward(z_t=None, a_t=a_seq[:, t, :], h_prev=h)
            s_traj.append(s_next)
        
        return torch.stack(s_traj, dim=1)  # [B, H, n_state]


class WorldModel(nn.Module):
    """
    动作条件化世界模型 v2
    
    架构:
      Encoder: [s_win ‖ a_win] → TCN+VarAttn → z_t  (历史编码, 只跑一次)
      Decoder: z_t + a_t → GRU Cell → ŝ_{t+1}        (逐步展开)
    
    两种 rollout 模式:
      - 'gru':     Encoder 1次 + GRU Cell H次 (默认, 推荐)
      - 'sliding': Encoder H次 (对照, 冗余)
    """
    def __init__(self, n_state=14, n_action=2, window_size=96,
                 d_model=64, n_heads=4, n_var_layers=2, n_tcn_layers=2,
                 patch_len=16, stride=8, dropout=0.1,
                 rollout_mode='gru'):
        super().__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_total = n_state + n_action
        self.window_size = window_size
        self.d_model = d_model
        self.rollout_mode = rollout_mode
        
        # --- Encoder (同v1) ---
        self.revin = RevIN(self.n_total)
        self.patch_embed = PatchEmbedding(window_size, patch_len, stride, d_model)
        self.n_patches = self.patch_embed.n_patches
        self.var_encoder = PerVariableTCN(self.n_patches, d_model, n_tcn_layers, dropout)
        self.var_attention_layers = nn.ModuleList([
            VariableAttention(d_model, n_heads, dropout)
            for _ in range(n_var_layers)
        ])
        
        # --- z_t 投影: 变量表示 → 隐状态 ---
        self.z_proj = nn.Linear(self.n_total * d_model, d_model)
        
        # --- Decoder: GRU Cell (v2 新增) ---
        self.state_decoder_gru = GRUStateDecoder(
            d_hidden=d_model, n_action=n_action, n_state=n_state, dropout=dropout
        )
        
        # --- 保留 v1 的直接解码器 (用于一步预测训练) ---
        self.state_decoder_direct = nn.Sequential(
            nn.Linear(self.n_total * d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_state),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """
        Encoder: 将历史窗口压缩为隐状态 z_t
        
        Input:  [B, W, N_total]
        Output: [B, d_model]
        """
        B, W, N = x.shape
        
        # 1. RevIN
        x_norm = self.revin(x, mode='norm')
        
        # 2. Per-variable Patching + TCN
        var_tokens = []
        for i in range(N):
            patches_i = self.patch_embed(x_norm[:, :, i])
            var_tokens.append(patches_i)
        var_tokens = torch.stack(var_tokens, dim=1)
        
        var_tokens = var_tokens.reshape(B * N, self.n_patches, self.d_model)
        var_repr = self.var_encoder(var_tokens)
        var_repr = var_repr.reshape(B, N, self.d_model)
        
        # 3. 变量间 Attention
        for attn_layer in self.var_attention_layers:
            var_repr, _ = attn_layer(var_repr)
        
        # 4. → z_t
        var_repr_flat = var_repr.reshape(B, N * self.d_model)
        z_t = self.z_proj(var_repr_flat)
        
        return z_t
    
    def forward(self, x, return_attention=False):
        """
        一步预测 (同v1, 保留兼容)
        
        x: [B, W, N_total] → s_next: [B, N_state]
        """
        B, W, N = x.shape
        
        x_norm = self.revin(x, mode='norm')
        
        var_tokens = []
        for i in range(N):
            patches_i = self.patch_embed(x_norm[:, :, i])
            var_tokens.append(patches_i)
        var_tokens = torch.stack(var_tokens, dim=1)
        
        var_tokens = var_tokens.reshape(B * N, self.n_patches, self.d_model)
        var_repr = self.var_encoder(var_tokens)
        var_repr = var_repr.reshape(B, N, self.d_model)
        
        attn_weights = None
        for attn_layer in self.var_attention_layers:
            var_repr, attn_weights = attn_layer(var_repr)
        
        var_repr_flat = var_repr.reshape(B, N * self.d_model)
        s_next_norm = self.state_decoder_direct(var_repr_flat)
        
        # RevIN 逆变换
        mean_s = self.revin._mean[:, :, :self.n_state]
        std_s = self.revin._std[:, :, :self.n_state]
        w_s = self.revin.weight[:self.n_state]
        b_s = self.revin.bias[:self.n_state]
        s_next_norm_expand = s_next_norm.unsqueeze(1)
        if self.revin.affine:
            s_next_norm_expand = (s_next_norm_expand - b_s) / (w_s + self.revin.eps)
        s_next = s_next_norm_expand * std_s + mean_s
        s_next = s_next.squeeze(1)
        
        if return_attention:
            return s_next, attn_weights
        return s_next
    
    def rollout(self, x_hist, a_seq, mode=None):
        """
        自回归展开 H 步
        
        Args:
            x_hist: [B, W, N_total] 历史窗口
            a_seq:  [B, H, N_action] H步动作序列
            mode:   'gru' | 'sliding' (默认使用 self.rollout_mode)
        
        Returns:
            s_traj: [B, H, N_state]
        """
        if mode is None:
            mode = self.rollout_mode
        
        if mode == 'gru':
            return self._rollout_gru(x_hist, a_seq)
        elif mode == 'sliding':
            return self._rollout_sliding(x_hist, a_seq)
        else:
            raise ValueError(f"Unknown rollout mode: {mode}")
    
    def _rollout_gru(self, x_hist, a_seq):
        """
        GRU模式: Encoder 1次 + GRU Cell H次
        
        x_hist: [B, W, N_total]
        a_seq:  [B, H, N_action]
        """
        z_t = self.encode(x_hist)  # 只跑一次 Encoder (已存储 RevIN stats)
        s_traj_norm = self.state_decoder_gru.rollout(z_t, a_seq)  # [B, H, n_state] (归一化空间)
        
        # RevIN denorm: 用 encode() 存储的 mean/std
        mean_s = self.revin._mean[:, :, :self.n_state]  # [B, 1, n_state]
        std_s = self.revin._std[:, :, :self.n_state]
        if self.revin.affine:
            w_s = self.revin.weight[:self.n_state]
            b_s = self.revin.bias[:self.n_state]
            s_traj_norm = (s_traj_norm - b_s) / (w_s + self.revin.eps)
        s_traj = s_traj_norm * std_s + mean_s
        
        return s_traj
    
    def _rollout_sliding(self, x_hist, a_seq):
        """
        滑动窗口模式 (v1, 保留对照): Encoder H次
        """
        B, H = a_seq.shape[0], a_seq.shape[1]
        states = x_hist[:, :, :self.n_state]
        actions = x_hist[:, :, self.n_state:]
        
        s_pred = []
        for t in range(H):
            x_t = torch.cat([states, actions], dim=2)
            s_next = self.forward(x_t)
            s_pred.append(s_next)
            states = torch.cat([states[:, 1:, :], s_next.unsqueeze(1)], dim=1)
            actions = torch.cat([actions[:, 1:, :], a_seq[:, t:t+1, :]], dim=1)
        
        return torch.stack(s_pred, dim=1)


if __name__ == '__main__':
    model = WorldModel(rollout_mode='gru')
    print(f"WorldModel v2 | Params: {sum(p.numel() for p in model.parameters()):,}")
    
    x_hist = torch.randn(4, 96, 16)
    a_seq = torch.randn(4, 18, 2)
    
    # 一步预测
    s_next = model(x_hist)
    print(f"Forward:  {x_hist.shape} → {s_next.shape}")
    
    # Encoder
    z = model.encode(x_hist)
    print(f"Encode:   {x_hist.shape} → z_t {z.shape}")
    
    # GRU rollout
    s_traj_gru = model.rollout(x_hist, a_seq, mode='gru')
    print(f"Rollout(GRU):   {x_hist.shape} + a_seq {a_seq.shape} → {s_traj_gru.shape}")
    
    # Sliding rollout
    s_traj_sliding = model.rollout(x_hist, a_seq, mode='sliding')
    print(f"Rollout(sliding): {x_hist.shape} + a_seq {a_seq.shape} → {s_traj_sliding.shape}")
