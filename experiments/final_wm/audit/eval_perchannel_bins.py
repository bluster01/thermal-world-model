"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""Fig3 数据补齐: (1) per-channel MAE (v0.2 closure_steam seed0) (2) v0.2 分箱均值"""
import sys
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[3]))
import numpy as np
import torch

from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
import experiments.final_wm.matrix_spec as ms

GRID = "/home/bluster/.hermes/workspace/adhoc2_lumped_enthalpy/out/iapws_surrogate.npz"  # env-specific
CKPT = "artifacts/final_wm/checkpoints/t1_closure_steam_seed0.pt"
DEV = "cuda"
HIST, HOR = ms.HISTORY_STEPS, ms.HORIZON
CHAN = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "final"]

record = CanonicalRecord("artifacts/final_wm/canonical_sideA.npz")
spec = next(s for s in ms.t1_specs((0,)) if s.arm == "closure_steam")
model = build_world_model(spec, load_grid_properties(GRID)).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=False)["state_dict"])
model.eval()

gen = torch.Generator().manual_seed(7)
resids, feats = [], []
with torch.no_grad():
    for _ in range(4):
        b = sample_windows(record, SPLIT_VAL, 64, HIST, HOR, gen)
        h = b.history.__class__(
            obs=b.history.obs.to(DEV), actions=b.history.actions.to(DEV),
            boundary=b.history.boundary.to(DEV),
        )
        out = model.forecast(h, b.future_actions.to(DEV), boundary_mode="oracle",
                             true_future_boundary=b.future_boundary.to(DEV))
        r = (out.temps_mu - b.future_obs.to(DEV)).cpu().numpy()   # (B,H,5)
        resids.append(r)
        feats.append(b.history.boundary[:, -1, 0].cpu().numpy())  # 主汽流量 D
R = np.concatenate(resids, 0); F = np.concatenate(feats, 0)

print("per-channel MAE (closure_steam seed0, v0.2):")
for hk, hi in [("H1", 0), ("H6", 5), ("H18", 17)]:
    mae = np.abs(R[:, hi]).mean(axis=0)
    print(f"  {hk}: " + " ".join(f"{c}={v:.3f}" for c, v in zip(CHAN, mae)))

print("\npersistence per-channel MAE (val):")
o = record.obs.numpy()[record.split.numpy() == 1].astype(np.float64)
for hk, h in [("H1", 1), ("H6", 6), ("H18", 18)]:
    e = np.abs(o[h:] - o[:-h])
    print(f"  {hk}: " + " ".join(f"{c}={v:.3f}" for c, v in zip(CHAN, e.mean(axis=0))))

print("\n分箱均值残差 (final 通道 H1, 5箱 by 主汽流量, v0.2):")
q = np.quantile(F, np.linspace(0, 1, 6)[1:-1])
bb = np.digitize(F, q)
r_final = R[:, 0, 4]
means = [f"{r_final[bb == i].mean():+.2f}" for i in range(5)]
print("  " + " ".join(means))
np.savez("/tmp/fig3_data.npz", resid=R, feat=F, bin_means=np.array([r_final[bb == i].mean() for i in range(5)]))
print("saved /tmp/fig3_data.npz")
