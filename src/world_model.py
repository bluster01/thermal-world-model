"""
world_model.py — 动作条件化世界模型
基于 Exp-0 TCN-iTransformer 改造:
- 输入: [s_{t-W:t} ‖ a_{t-W:t}] → 16维 (14状态 + 2动作)
- 输出: s_{t+1} (14维全状态)
- 改动: +动作通道, +全状态输出, +自回归展开
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
        """x: [B, T] → [B, n_patches, d_model]"""
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
        """x: [B*N, n_patches, d_model] → [B*N, d_model]"""
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        return x


class WorldModel(nn.Module):
    """
    动作条件化世界模型 (基于 Exp-0 TCN-iTransformer 改造)
    
    输入: [B, W, N_state+N_action]  (W=96, N_state=14, N_action=2)
    输出: [B, N_state]              (全状态预测)
    """
    def __init__(self, n_state=14, n_action=2, window_size=96,
                 d_model=64, n_heads=4, n_var_layers=2, n_tcn_layers=2,
                 patch_len=16, stride=8, dropout=0.1):
        super().__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_total = n_state + n_action
        self.window_size = window_size
        self.d_model = d_model
        
        # 1. RevIN (对所有输入通道归一化)
        self.revin = RevIN(self.n_total)
        
        # 2. Patching (共享)
        self.patch_embed = PatchEmbedding(window_size, patch_len, stride, d_model)
        self.n_patches = self.patch_embed.n_patches
        
        # 3. Per-variable TCN
        self.var_encoder = PerVariableTCN(self.n_patches, d_model, n_tcn_layers, dropout)
        
        # 4. 变量间 Attention
        self.var_attention_layers = nn.ModuleList([
            VariableAttention(d_model, n_heads, dropout)
            for _ in range(n_var_layers)
        ])
        
        # 5. 全状态解码器 (16变量 → 14维状态)
        self.state_decoder = nn.Sequential(
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
    
    def forward(self, x, return_attention=False):
        """
        x: [B, W, N_total]  (W=窗口, N_total=16)
        → s_next: [B, N_state]
        """
        B, W, N = x.shape
        
        # 1. RevIN
        x_norm = self.revin(x, mode='norm')
        
        # 2. Per-variable Patching + TCN
        var_tokens = []
        for i in range(N):
            xi = x_norm[:, :, i]
            patches_i = self.patch_embed(xi)
            var_tokens.append(patches_i)
        var_tokens = torch.stack(var_tokens, dim=1)  # [B, N, n_patches, d_model]
        
        # 3. TCN编码
        var_tokens = var_tokens.reshape(B * N, self.n_patches, self.d_model)
        var_repr = self.var_encoder(var_tokens)  # [B*N, d_model]
        var_repr = var_repr.reshape(B, N, self.d_model)  # [B, N, d_model]
        
        # 4. 变量间 Attention
        attn_weights = None
        for attn_layer in self.var_attention_layers:
            var_repr, attn_weights = attn_layer(var_repr)
        
        # 5. 全状态解码
        var_repr_flat = var_repr.reshape(B, N * self.d_model)  # [B, N*d_model]
        s_next_norm = self.state_decoder(var_repr_flat)  # [B, N_state]
        
        # 6. RevIN 逆变换 (只对状态变量部分)
        # NOTE: RevIN存储的是全通道的mean/std, 这里只取状态通道
        mean_s = self.revin._mean[:, :, :self.n_state]
        std_s = self.revin._std[:, :, :self.n_state]
        w_s = self.revin.weight[:self.n_state]
        b_s = self.revin.bias[:self.n_state]
        s_next_norm_expand = s_next_norm.unsqueeze(1)  # [B, 1, N_state]
        if self.revin.affine:
            s_next_norm_expand = (s_next_norm_expand - b_s) / (w_s + self.revin.eps)
        s_next = s_next_norm_expand * std_s + mean_s
        s_next = s_next.squeeze(1)  # [B, N_state]
        
        if return_attention:
            return s_next, attn_weights
        return s_next
    
    def rollout(self, s_0, a_seq, window_states=None, window_actions=None):
        """
        自回归展开 H 步
        
        Args:
            s_0: [B, N_state] 初始状态
            a_seq: [B, H, N_action] H步动作序列
            window_states: [B, W, N_state] 初始历史窗口状态 (可选)
            window_actions: [B, W, N_action] 初始历史窗口动作 (可选)
        
        Returns:
            s_pred: [B, H, N_state] 预测状态轨迹
        """
        B = s_0.shape[0]
        H = a_seq.shape[1]
        
        # 初始化窗口
        if window_states is None:
            window_states = s_0.unsqueeze(1).repeat(1, self.window_size, 1)
        if window_actions is None:
            window_actions = a_seq[:, 0:1, :].repeat(1, self.window_size, 1)
        
        s_pred = []
        s_cur = s_0
        
        for t in range(H):
            # 构造输入: [s_win ‖ a_win]
            x_t = torch.cat([window_states, window_actions], dim=2)  # [B, W, N_total]
            
            # 预测下一步
            s_next = self.forward(x_t)  # [B, N_state]
            s_pred.append(s_next)
            
            # 滑动窗口
            window_states = torch.cat([window_states[:, 1:, :], s_next.unsqueeze(1)], dim=1)
            window_actions = torch.cat([window_actions[:, 1:, :], a_seq[:, t:t+1, :]], dim=1)
            s_cur = s_next
        
        return torch.stack(s_pred, dim=1)  # [B, H, N_state]


if __name__ == '__main__':
    # 快速测试
    model = WorldModel()
    x = torch.randn(4, 96, 16)
    s_next = model(x)
    print(f"Input: {x.shape} → Output: {s_next.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试 rollout
    s_0 = torch.randn(4, 14)
    a_seq = torch.randn(4, 18, 2)
    s_traj = model.rollout(s_0, a_seq)
    print(f"Rollout: s_0 {s_0.shape} + a_seq {a_seq.shape} → trajectory {s_traj.shape}")
