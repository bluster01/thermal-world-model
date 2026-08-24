"""Black-box baselines: per-load-segment accuracy (execution-side extension).

Trains the v0.5 black-box pack (lstm/gru/dlinear/itransformer, 3 seeds)
with the SAME functions as v05_blackbox_baselines, then evaluates each
model per-load-quintile (main-steam H18 MAE) on the production window set
(256 windows, sample_windows seed 50_000), load = window-start steam flow
(same binning rule as the physical-arm per-segment script).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from experiments.final_wm.v05_blackbox_baselines import (
    ITransformerForecaster,
    DLinearForecaster,
    InstanceNorm,
    LSTMForecaster,
    TARGET,
    W,
    H,
    batch_arrays,
    build_train_bank,
    channel_stats,
    train_one,
    RECORD,
)
from src.final_wm.data import CanonicalRecord, SPLIT_VAL, sample_windows

DEVICE = "cuda"
OUT = Path("/tmp/grid_out")
SEEDS = (0, 1, 2)
N_WIN = 256


@torch.no_grad()
def seg_eval(pf, record, device, n=N_WIN):
    gen = torch.Generator().manual_seed(50_000)
    errs, loads, done = [], [], 0
    while done < n:
        bsz = min(32, n - done)
        batch = sample_windows(record, SPLIT_VAL, bsz, W, H, gen)
        hist, fut, tgt = batch_arrays(batch, device)
        pred = pf(hist, fut)
        errs.append((pred - tgt).abs().cpu())
        loads.append(batch.future_boundary[:, 0, 0])
        done += bsz
    e = torch.cat(errs)
    l = torch.cat(loads).numpy()
    per_win = e.mean(dim=1)
    edges = np.quantile(l, np.linspace(0, 1, 6)[1:-1])
    bid = np.clip(np.digitize(l, edges), 0, 4)
    return {
        "H18_mae": float(e.mean()),
        "per_quintile": [float(per_win[bid == b].mean()) for b in range(5)],
        "counts": [int((bid == b).sum()) for b in range(5)],
        "load_edges": edges.tolist(),
    }


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    record = CanonicalRecord(RECORD)
    mean, std = channel_stats(record)
    bank = build_train_bank(record, DEVICE)
    out = {"protocol": "sideA val 256 win seed50k oracle, target main-steam H18",
           "baselines": {}}
    for name, ctor in (("lstm", lambda: LSTMForecaster("lstm")),
                       ("gru", lambda: LSTMForecaster("gru")),
                       ("dlinear", DLinearForecaster),
                       ("itransformer", ITransformerForecaster)):
        for seed in SEEDS:
            model = train_one(name, ctor(), bank, record, mean, std, DEVICE, seed)
            hist_scale = lambda hist: (hist - mean.to(DEVICE)) / std.to(DEVICE)

            def pf(hist, fut, m=model, hs=hist_scale):
                return InstanceNorm.restore(
                    hist[:, :, TARGET].mean(dim=1, keepdim=True), m(hs(hist), fut))

            entry = seg_eval(pf, record, DEVICE)
            out["baselines"].setdefault(name, {})[f"seed{seed}"] = entry
            print(f"{name} seed{seed}: H18 {entry['H18_mae']:.3f} | "
                  f"Q {['%.2f' % v for v in entry['per_quintile']]}", flush=True)
    (OUT / "bb_per_segment.json").write_text(json.dumps(out, indent=2))
    print("saved bb_per_segment.json", flush=True)


if __name__ == "__main__":
    main()
