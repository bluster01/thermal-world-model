"""v0.5 external black-box baseline pack on the canonical protocol (2026-08-23).

Question: does the final world model beat pure black-box forecasters on main
steam temperature accuracy under an IDENTICAL protocol?  Fairness rules:
- same canonical record (side A), same train/val split (SPLIT_TRAIN/VAL)
- same eval window set as production eval (sample_windows, seed 50_000, 256)
- same information set as the t1 arms' oracle-mode eval: history (obs 5 +
  boundary 7 + actions 2, 96 steps) + future actions (18x2) + future boundary
  (18x7).  Target: future main-steam channel (obs ch4), H=18.
- 3 seeds per baseline; instance-norm on the target channel (phase1 lesson:
  per-window normalization is essential for temperature).

Baselines: N4SID-style ridge (numpy), DLinear, LSTM, GRU, iTransformer-lite.
Evidence-level comparison (not a frozen gate): writes
results/final_wm/v05_blackbox_comparison_20260823.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.final_wm import matrix_spec as ms
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL, CanonicalRecord, sample_windows

RECORD = "artifacts/final_wm/canonical_sideA.npz"
OUT_JSON = Path("results/final_wm/v05_blackbox_comparison_20260823.json")
SEEDS = (0, 1, 2)
H = ms.HORIZON
W = ms.HISTORY_STEPS
TARGET = 4           # main steam (final outlet)
N_HIST = 14          # 5 obs + 7 boundary + 2 actions
N_FUT = 9            # 2 actions + 7 boundary (oracle-mode future info)
STEPS = 3000
EVAL_EVERY = 500
PATIENCE_EVALS = 3
BATCH = 128
LR = 1e-3
TRAIN_POOL = 20000  # materialized once; per-step sampling was CPU-bound otherwise


def batch_arrays(batch, device):
    hist = torch.cat(
        [batch.history.obs, batch.history.boundary, batch.history.actions], dim=-1
    ).to(device)                                    # (B, W, 14)
    fut = torch.cat([batch.future_actions, batch.future_boundary], dim=-1).to(device)
    tgt = batch.future_obs[:, :, TARGET].to(device)  # (B, H)
    return hist, fut, tgt


class InstanceNorm:
    """Per-window target-channel centering: predict deviation from history mean."""

    @staticmethod
    def center(hist, tgt):
        mu = hist[:, :, TARGET].mean(dim=1, keepdim=True)
        return mu, tgt - mu

    @staticmethod
    def restore(mu, pred):
        return pred + mu


class LSTMForecaster(nn.Module):
    def __init__(self, cell="lstm", hidden=128, layers=2):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.inp = nn.Linear(N_HIST, hidden)
        self.rnn = rnn(hidden, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + H * N_FUT, 256), nn.GELU(), nn.Linear(256, H))

    def forward(self, hist, fut):
        h, _ = self.rnn(self.inp(hist))
        z = torch.cat([h[:, -1], fut.reshape(fut.shape[0], -1)], dim=-1)
        return self.head(z)


class DLinearForecaster(nn.Module):
    """Linear map on (flattened history + future covariates) of the target deviation."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(W * N_HIST + H * N_FUT, H)

    def forward(self, hist, fut):
        x = torch.cat([hist.reshape(hist.shape[0], -1), fut.reshape(fut.shape[0], -1)], dim=-1)
        return self.lin(x)


class ITransformerForecaster(nn.Module):
    """Variables-as-tokens over history + future-covariate tokens, flatten head."""

    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.hist_tok = nn.Linear(W, d)
        self.fut_tok = nn.Linear(H, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True,
                                           dropout=0.1, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear((N_HIST + N_FUT) * d, H)

    def forward(self, hist, fut):
        ht = self.hist_tok(hist.permute(0, 2, 1))   # (B, 14, d)
        ft = self.fut_tok(fut.permute(0, 2, 1))     # (B, 9, d)
        z = self.enc(torch.cat([ht, ft], dim=1))
        return self.head(z.reshape(z.shape[0], -1))


def channel_stats(record):
    mask = record.split == SPLIT_TRAIN
    hist_all = torch.cat([record.obs, record.boundary, record.actions], dim=-1)
    tr = hist_all[mask]
    return tr.mean(0), tr.std(0).clamp_min(1e-3)


def build_train_bank(record, device):
    """One sample_windows call over the train split, materialized on GPU.
    (Calling sample_windows per training step scans the 530k-row split mask
    on CPU and dominated wall-clock in the first attempt.)"""
    gen = torch.Generator().manual_seed(15_000)
    batch = sample_windows(record, SPLIT_TRAIN, TRAIN_POOL, W, H, gen)
    hist, fut, tgt = batch_arrays(batch, device)
    return hist, fut, tgt


def train_one(name, model, bank, record, mean, std, device, seed):
    torch.manual_seed(10_000 + seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(20_000 + seed)
    bh, bf, bt = bank
    n_pool = bh.shape[0]
    best, best_state, stale = float("inf"), None, 0
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, n_pool, (BATCH,), generator=gen)
        hist, fut, tgt = bh[idx], bf[idx], bt[idx]
        hist_n = (hist - mean.to(device)) / std.to(device)
        mu_c, dev = InstanceNorm.center(hist, tgt)
        pred = model(hist_n, fut)
        loss = nn.functional.mse_loss(pred, dev)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            vm = evaluate(model, record, mean, std, device, n=128, seed=30_000)
            if vm < best:
                best, best_state, stale = vm, {k: v.detach().clone() for k, v in
                                               model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= PATIENCE_EVALS:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  [{name} seed{seed}] stopped@{step} val_mae128={best:.4f}")
    return model


@torch.no_grad()
def evaluate(model, record, mean, std, device, n, seed):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    errs, done = [], 0
    while done < n:
        batch = sample_windows(record, SPLIT_VAL, min(64, n - done), W, H, gen)
        hist, fut, tgt = batch_arrays(batch, device)
        hist_n = (hist - mean.to(device)) / std.to(device)
        mu_c, _ = InstanceNorm.center(hist, tgt)
        pred = InstanceNorm.restore(mu_c, model(hist_n, fut))
        errs.append((pred - tgt).abs().cpu())
        done += errs[-1].shape[0]
    model.train()  # execution-side fix: cudnn RNN backward requires train mode
    return float(torch.cat(errs).mean())


def n4sid_ridge(record, device, seed):
    """Direct multi-horizon ridge on standardized history + future covariates."""
    gen = torch.Generator().manual_seed(40_000 + seed)
    xs, ys = [], []
    for _ in range(200):
        batch = sample_windows(record, SPLIT_TRAIN, 256, W, H, gen)
        hist, fut, tgt = batch_arrays(batch, "cpu")
        mu_c, dev = InstanceNorm.center(hist, tgt)
        x = torch.cat([hist.reshape(len(hist), -1), fut.reshape(len(fut), -1)], -1)
        xs.append(x); ys.append(dev)
    X = torch.cat(xs).numpy().astype(np.float64)
    Y = torch.cat(ys).numpy().astype(np.float64)
    Xm, Xs = X.mean(0), X.std(0) + 1e-6
    Xn = (X - Xm) / Xs
    W_ = np.linalg.solve(Xn.T @ Xn + 10.0 * np.eye(Xn.shape[1]), Xn.T @ Y)

    def predict(hist, fut):
        mu_c = hist[:, :, TARGET].mean(dim=1, keepdim=True)
        x = torch.cat([hist.reshape(len(hist), -1), fut.reshape(len(fut), -1)], -1)
        x = x.detach().cpu().numpy().astype(np.float64)
        dev = ((x - Xm) / Xs) @ W_
        return torch.from_numpy(dev.astype(np.float32)).to(hist.device) + mu_c
    return predict


@torch.no_grad()
def full_eval(predict_fn, record, mean, std, device, n=256):
    gen = torch.Generator().manual_seed(50_000)  # production eval window set
    errs, tgts, done = [], [], 0
    while done < n:
        batch = sample_windows(record, SPLIT_VAL, min(32, n - done), W, H, gen)
        hist, fut, tgt = batch_arrays(batch, device)
        pred = predict_fn(hist, fut)
        errs.append((pred - tgt).abs().cpu()); tgts.append(tgt.cpu())
        done += errs[-1].shape[0]
    e = torch.cat(errs); t = torch.cat(tgts)
    return {
        "step_curve_mae": [float(e[:, i].mean()) for i in range(H)],
        "step17_mae": float(e[:, -1].mean()),
        "H1_mae": float(e[:, :1].mean()), "H6_mae": float(e[:, :6].mean()),
        "H18_mae": float(e.mean()),
        "H18_mape_pct": float(((e / t.clamp_min(1e-6)).mean()) * 100.0),
    }


@torch.no_grad()
def bb_step_response(predict_fn, record, device, valve_index, seed, n=32):
    """R1-style direction probe adapted to the black-box output horizon:
    hold window-end boundary/actions, step one valve +0.05, compare the
    mean of the last 6 of the 18 predicted steps (3-min horizon -- the
    EASIER short-horizon version of the 60-step/10-min R1 probe; sign
    failures here are conservative evidence against the black-box)."""
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, SPLIT_VAL, n, W, 1, gen)
    hist = torch.cat([batch.history.obs, batch.history.boundary,
                      batch.history.actions], dim=-1).to(device)
    bnd0 = batch.future_boundary[:, 0].to(device)
    act0 = batch.future_actions[:, 0].to(device)
    base = torch.cat([act0.unsqueeze(1).repeat(1, H, 1),
                      bnd0.unsqueeze(1).repeat(1, H, 1)], dim=-1)
    step = base.clone()
    step[:, :, valve_index] = (step[:, :, valve_index] + 0.05).clamp(max=1.0)
    pb = predict_fn(hist, base)
    ps = predict_fn(hist, step)
    delta = (ps[:, -6:] - pb[:, -6:]).mean(dim=1)
    return {"mean_delta_c": float(delta.mean()),
            "frac_negative": float((delta < 0).float().mean()), "n_windows": n}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    record = CanonicalRecord(RECORD)
    mean, std = channel_stats(record)
    bank = build_train_bank(record, device)
    out = {"record": RECORD, "protocol": "canonical sideA val, 256 windows (seed 50_000), "
           "inputs=hist(obs+bnd+act)+future act+bnd (oracle parity with t1 eval), "
           "target=main-steam ch4 H18", "baselines": {}}

    for name, ctor in (("lstm", lambda: LSTMForecaster("lstm")),
                       ("gru", lambda: LSTMForecaster("gru")),
                       ("dlinear", DLinearForecaster),
                       ("itransformer", ITransformerForecaster)):
        for seed in SEEDS:
            model = train_one(name, ctor(), bank, record, mean, std, device, seed)
            hist_scale = lambda hist: (hist - mean.to(device)) / std.to(device)
            def pf(hist, fut, m=model, hs=hist_scale):
                return InstanceNorm.restore(
                    hist[:, :, TARGET].mean(dim=1, keepdim=True), m(hs(hist), fut))
            entry = {"accuracy": full_eval(pf, record, mean, std, device)}
            for valve, tag in ((0, "v1"), (1, "v2")):
                entry[f"{tag}_18step"] = bb_step_response(
                    pf, record, device, valve, seed=80_000 + seed)
            out["baselines"].setdefault(name, {})[f"seed{seed}"] = entry
            acc = entry["accuracy"]
            print(f"  [{name} seed{seed}] step17={acc['step17_mae']:.3f}C H18={acc['H18_mae']:.3f}C "
                  f"v1_fracneg={entry['v1_18step']['frac_negative']:.3f} "
                  f"v2_fracneg={entry['v2_18step']['frac_negative']:.3f}")
            del model
            torch.cuda.empty_cache()

    for seed in (0,):
        pf = n4sid_ridge(record, device, seed)
        out["baselines"].setdefault("n4sid_ridge", {})[f"seed{seed}"] = {
            "accuracy": full_eval(pf, record, mean, std, device)}
        print(f"  [n4sid seed{seed}] step17={out['baselines']['n4sid_ridge'][f'seed{seed}']['accuracy']['step17_mae']:.3f}C")

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[report] written {OUT_JSON}")


if __name__ == "__main__":
    main()
