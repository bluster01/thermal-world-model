#!/usr/bin/env python3
"""36_direct_wm.py — Direct WM v2（协议修正版）

来源 [PHASE-REF]: 主仓 experiments/phase1_dynamics/exp_023_direct_aligned.py
+ src/world_model.py (RevIN/PatchEmbedding/LightTCNBlock/PerVariableTCN)。
架构: 40列全特征历史(W=96) + 未来双阀位序列(H=18) → 18步主汽温 (μ, logσ²) via β-NLL。

移植差异 (协议修复, 相对 exp_023):
  1. 数据: 本仓 CSV 开发段 [0,40000) 冻结边界, 不碰 reserved [40000,50000)
  2. 折匹配 Q32，训练/验证/评测严格不重叠。
  3. 固定验证/评测索引；H18 是唯一 checkpoint selector。
  4. 评测口径: 物理空间 rollout MAE + 持续 ±2% valve-only 条件响应。
  5. 归一化: RevIN 40列 (内部), 输出 denorm 到物理尺度 — 与框架 v2 协议一致

用法:
  python 36_direct_wm.py --dry-run
  python 36_direct_wm.py --execute --seed 42 --seed 0 --seed 7
"""
import argparse, hashlib, json, os, subprocess, tempfile, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CSV = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据_cleaned_10s.csv"
WIN_START, WIN_DEV = 70686, 40000          # 开发段边界 (reserved [40000,50000) 禁触)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 协议常量 (phase1 原生) =====
W, H = 96, 18                              # 历史窗口 16min, 预测 18步 (180s)
D_MODEL, PATCH_LEN, STRIDE = 64, 16, 8
N_TCN_LAYERS, DROPOUT = 2, 0.1
BETA, BETA_WARMUP = -0.3, 20
BS, STEPS = 256, 500
EPOCHS, PATIENCE = 100, 20
LR, WD = 1e-3, 1e-5

# ===== 折 (与 Q32 h_now 同协议) =====
FOLDS = {
    "F0": {"train": (0, 20000), "val": (20000, 25000), "eval": (25000, 30000)},
    "F1": {"train": (0, 30000), "val": (30000, 35000), "eval": (35000, 40000)},
}
OUT_DIR = "out/direct_wm_v2"
EVAL_N = 200
SENSITIVITY_DELTAS = (-2.0, 2.0)
ANCHORS = (1, 3, 8, 12, 17)


# ---------------- 数据 ----------------
def load_dev(csv_path=CSV):
    df = pd.read_csv(csv_path, nrows=WIN_START + WIN_DEV).iloc[WIN_START: WIN_START + WIN_DEV]
    df = df.ffill().bfill().reset_index(drop=True)
    num_cols = [c for c in df.columns if c != "date"]
    return df[num_cols].values.astype(np.float32), num_cols


# ---------------- 模型 (自包含移植) ----------------
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps; self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            x = x * self._std + self._mean
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model):
        super().__init__()
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, x):
        patches = x.unfold(dimension=1, size=PATCH_LEN, step=STRIDE)
        out = self.proj(patches)
        out = self.norm(out)
        return out + self.pos_embed


class LightTCNBlock(nn.Module):
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
        out = out.transpose(1, 2); out = self.norm(out); out = out.transpose(1, 2)
        out = self.act(out); out = self.drop(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class PerVariableTCN(nn.Module):
    def __init__(self, n_patches, d_model, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        layers = [LightTCNBlock(d_model, d_model, kernel_size=3, dilation=2 ** i, dropout=dropout)
                  for i in range(n_layers)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        return x[:, :, -1]


class DirectAligned(nn.Module):
    """40列历史 + 未来双阀位 → 18步主汽温 (μ, logσ²)"""
    def __init__(self, n_feat, target_idx):
        super().__init__()
        self.target_idx = target_idx
        self.revin = RevIN(n_feat)
        self.patch = PatchEmbedding(W, PATCH_LEN, STRIDE, D_MODEL)
        self.np = self.patch.n_patches
        self.tcn = PerVariableTCN(self.np, D_MODEL, N_TCN_LAYERS, DROPOUT)
        self.action_enc = nn.Sequential(
            nn.Linear(H * 2, D_MODEL * 2), nn.GELU(), nn.Dropout(DROPOUT))
        self.decoder = nn.Sequential(
            nn.Linear(n_feat * D_MODEL + D_MODEL * 2, D_MODEL * 4), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 4, D_MODEL * 4), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL * 4, H * 2))

    def forward(self, x_hist, a_future):
        B = x_hist.shape[0]
        x_n = self.revin(x_hist, mode="norm")
        var_tokens = torch.stack([self.patch(x_n[:, :, i]) for i in range(x_n.shape[2])], 1)
        var_tokens = var_tokens.reshape(B * x_n.shape[2], self.np, D_MODEL)
        s_repr = self.tcn(var_tokens).reshape(B, x_n.shape[2], D_MODEL)
        a_feat = self.action_enc(a_future.reshape(B, -1))
        z = torch.cat([s_repr.reshape(B, -1), a_feat], 1)
        raw = self.decoder(z).reshape(B, H, 2)
        mu_n, lv_n = raw[..., 0], raw[..., 1]
        ms = self.revin._mean[:, :, self.target_idx]
        ss = self.revin._std[:, :, self.target_idx]
        w = self.revin.weight[self.target_idx]
        b = self.revin.bias[self.target_idx]
        if self.revin.affine:
            mu_n = (mu_n - b) / (w + self.revin.eps)
        mu = mu_n * ss + ms
        sig = torch.exp(lv_n * 0.5) * ss
        lv = 2.0 * torch.log(sig + 1e-8)
        return mu, lv


class BetaNLLLoss(nn.Module):
    def __init__(self, beta=0.0, eps=1e-6):
        super().__init__(); self.beta = beta; self.eps = eps

    def forward(self, mu, lv, tgt):
        lv = torch.clamp(lv, -20.0, 20.0)
        v = torch.exp(lv) + self.eps
        nll = 0.5 * (lv + (tgt - mu) ** 2 / v)
        if self.beta != 0:
            nll = v.detach() ** self.beta * nll
        # Keep [batch, horizon].  The caller owns horizon weighting.
        return nll


def fixed_indices(seg, n=EVAL_N):
    """Frozen, seed-independent coverage of one contiguous block."""
    a, b = seg
    hi = b - W - H
    if hi <= a:
        raise ValueError(f"segment too short: {seg}")
    return np.linspace(a, hi - 1, min(n, hi - a), dtype=np.int64)


def indices_sha256(indices):
    payload = ",".join(str(int(i)) for i in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


# ---------------- 训练 ----------------
def train_epoch(model, raw, seg, opt, crit, rng, steps=STEPS, batch_size=BS):
    model.train(); N = len(raw)
    a, b = seg
    total = 0.0
    for _ in range(steps):
        idxs = rng.integers(a, b - W - H, size=batch_size)
        xh = np.stack([raw[i:i + W] for i in idxs])
        af = np.stack([raw[i + W:i + W + H, VALVE_IDX] for i in idxs])
        tt = np.stack([raw[i + W:i + W + H, TARGET_IDX] for i in idxs])
        x_hist = torch.FloatTensor(xh).to(DEVICE)
        a_fut = torch.FloatTensor(af).to(DEVICE)
        t_true = torch.FloatTensor(tt).to(DEVICE)
        opt.zero_grad()
        mu, lv = model(x_hist, a_fut)
        wgt = torch.linspace(1.0, 0.6, H, device=DEVICE)
        loss = (wgt * crit(mu, lv, t_true).mean(dim=0)).sum() / wgt.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / steps


@torch.no_grad()
def validate(model, raw, indices):
    model.eval()
    h18 = 0.0
    for i in indices:
        x_hist = torch.FloatTensor(raw[i:i + W]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(raw[i + W:i + W + H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        mu, _ = model(x_hist, a_fut)
        h18 += (mu[0, -1] - raw[i + W + H - 1, TARGET_IDX]).abs().item()
    return h18 / len(indices)


@torch.no_grad()
def eval_rollout(model, raw, indices):
    """物理空间逐步 |err|: [H]"""
    model.eval()
    err = np.zeros((len(indices), H))
    for j, i in enumerate(indices):
        x_hist = torch.FloatTensor(raw[i:i + W]).unsqueeze(0).to(DEVICE)
        a_fut = torch.FloatTensor(raw[i + W:i + W + H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        tt = raw[i + W:i + W + H, TARGET_IDX]
        mu, _ = model(x_hist, a_fut)
        err[j] = np.abs(mu[0].cpu().numpy() - tt)
    return err.mean(0)


@torch.no_grad()
def apply_persistent_shift(a_future, action_dim, delta):
    shifted = a_future.clone()
    shifted[:, :, action_dim] = torch.clamp(
        shifted[:, :, action_dim] + delta, 0.0, 100.0)
    actual = shifted[:, :, action_dim] - a_future[:, :, action_dim]
    return shifted, float(actual.mean().item())


def eval_sensitivity(model, raw, indices):
    """Persistent valve-only ±2% conditional response; no W coupling is implied."""
    model.eval()
    results = {}
    for adim in range(2):
        r = {}
        for d in SENSITIVITY_DELTAS:
            dT = {s: [] for s in ANCHORS}
            actual_dose = []
            for i in indices:
                x_hist = torch.FloatTensor(raw[i:i + W]).unsqueeze(0).to(DEVICE)
                a_fut = torch.FloatTensor(raw[i + W:i + W + H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
                mu_b, _ = model(x_hist, a_fut)
                bp = mu_b[0].cpu().numpy()
                a_p, applied = apply_persistent_shift(a_fut, adim, d)
                actual_dose.append(applied)
                mu_p, _ = model(x_hist, a_p)
                pp = mu_p[0].cpu().numpy()
                for s in ANCHORS:
                    dT[s].append(pp[s] - bp[s])
            key = f"{d:+.1f}"
            r[key] = {"requested_valve_pct": d,
                      "actual_mean_valve_pct": float(np.mean(actual_dose)),
                      "delta_t_by_step": {str(s): float(np.mean(dT[s])) for s in ANCHORS}}
        results[f"action_{adim}"] = r
    return results


def train_one(raw, fold, seed, checkpoint_dir, smoke=False):
    """单折单种子: 训练 + 评测"""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = DirectAligned(len(NUM_COLS), TARGET_IDX).to(DEVICE)
    crit = BetaNLLLoss(beta=BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    n_ep = 1 if smoke else EPOCHS
    steps = 1 if smoke else STEPS
    batch_size = 4 if smoke else BS
    val_indices = fixed_indices(FOLDS[fold]["val"])
    eval_indices = fixed_indices(FOLDS[fold]["eval"])
    checkpoint = os.path.join(checkpoint_dir, f"direct_wm_{fold}_s{seed}.pt")
    best_v, best_ep, pc = float("inf"), 0, 0
    t0 = time.time()
    for ep in range(1, n_ep + 1):
        crit.beta = 0.0 if ep <= BETA_WARMUP else BETA * min((ep - BETA_WARMUP) / 10, 1.0)
        nll = train_epoch(model, raw, FOLDS[fold]["train"], opt, crit, rng, steps, batch_size)
        vh18 = validate(model, raw, val_indices)
        sched.step(vh18)
        if smoke or (vh18 < best_v - 0.001):
            best_v, best_ep, pc = vh18, ep, 0
            torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, checkpoint)
        else:
            pc += 1
        if not smoke and pc >= PATIENCE:
            print(f"  [F{fold[-1]} s{seed}] stop@{ep} best@{best_ep}")
            break

    ck = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    with open(checkpoint, "rb") as f:
        checkpoint_sha = hashlib.sha256(f.read()).hexdigest()
    model.eval()

    mae = eval_rollout(model, raw, eval_indices)
    sens = eval_sensitivity(model, raw, eval_indices)
    print(f"  [F{fold[-1]} s{seed}] train {time.time()-t0:.0f}s | rollout step17={mae[-1]:.4f} (step0={mae[0]:.4f})")
    return {"fold": fold, "seed": seed, "best_epoch": best_ep,
            "best_validation_h18_mae": best_v, "rollout_mae": mae.tolist(),
            "sensitivity": sens, "evaluation_indices": eval_indices.tolist(),
            "evaluation_indices_sha256": indices_sha256(eval_indices),
            "checkpoint_sha256": checkpoint_sha, "checkpoint_retained": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, action="append", default=None)
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--output", default=OUT_DIR)
    ap.add_argument("--smoke", action="store_true")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    seeds = args.seed or [42, 0, 7]
    if args.execute and not args.smoke and seeds != [42, 0, 7]:
        raise ValueError("frozen execute requires seeds in exact order: 42, 0, 7")

    protocol = {
        "experiment": "direct_wm_endpoint_v2",
        "estimand": "logged-future-action conditional oracle; persistent valve-only perturbation",
        "folds": FOLDS,
        "window": W,
        "horizon": H,
        "seeds": seeds,
        "checkpoint_selector": "fixed_validation_H18_MAE",
        "evaluation": {"n": EVAL_N, "index_rule": "linspace", "deltas_pct": SENSITIVITY_DELTAS,
                       "anchors": ANCHORS},
        "reserved_rows": [40000, 50000],
    }
    if args.dry_run:
        print(json.dumps({**protocol, "authorized_to_train": False}, indent=2))
        return

    results_path = os.path.join(args.output, "results.json")
    if os.path.exists(results_path):
        raise FileExistsError(f"refusing to overwrite frozen result: {results_path}")

    global NUM_COLS, TARGET_IDX, VALVE_IDX
    raw, NUM_COLS = load_dev(args.csv)
    TARGET_IDX = NUM_COLS.index("末级过热器出口汽温")
    VALVE_IDX = [NUM_COLS.index("一级减温调节门阀位"), NUM_COLS.index("二级减温调节门阀位")]
    print(f"[36] dev rows={len(raw)} | n_feat={len(NUM_COLS)} | target_idx={TARGET_IDX} | "
          f"valve_idx={VALVE_IDX} | device={DEVICE} | smoke={args.smoke}")

    os.makedirs(args.output, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="direct_wm_v2_") as checkpoint_dir:
        for fold in ["F0", "F1"]:
            for seed in seeds:
                results.append(train_one(raw, fold, seed, checkpoint_dir, smoke=args.smoke))
                if args.smoke:
                    print("SMOKE_OK")
                    return
    summary = {**protocol, "results": results}
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    with open(__file__, "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {"experiment": "direct_wm_endpoint_v2",
                "source": "[PHASE-REF] exp_023_direct_aligned.py + world_model.py (blueprint)",
                "data_path": args.csv, "data_array_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                "feature_columns": NUM_COLS, "dev_rows": 40000, "reserved_rows_loaded": False,
                "training_performed": True, "device": DEVICE, "torch_version": torch.__version__,
                "git_commit": git_commit, "script_sha256": script_sha,
                "checkpoints_retained": False,
                "linux_return_allowlist": ["results.json", "manifest.json", "stdout.log"]}
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
