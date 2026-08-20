"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""v0.2 全通道分箱比率 + sh1_in 箱均值 (最终口径)"""
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
    for _ in range(8):
        b = sample_windows(record, SPLIT_VAL, 64, HIST, HOR, gen)
        h = b.history.__class__(
            obs=b.history.obs.to(DEV), actions=b.history.actions.to(DEV),
            boundary=b.history.boundary.to(DEV),
        )
        out = model.forecast(h, b.future_actions.to(DEV), boundary_mode="oracle",
                             true_future_boundary=b.future_boundary.to(DEV))
        resids.append((out.temps_mu - b.future_obs.to(DEV)).cpu().numpy())
        feats.append(b.history.boundary[:, -1, 0].cpu().numpy())
R = np.concatenate(resids, 0)   # (B,H,5)
F = np.concatenate(feats, 0)


def ratio(r, f, nbins=5):
    q = np.quantile(f, np.linspace(0, 1, nbins + 1)[1:-1])
    bb = np.digitize(f, q)
    tv = float(np.var(r))
    w = [(bb == i).sum() for i in range(nbins)]
    means = np.array([r[bb == i].mean() for i in range(nbins)])
    bt = float(np.average((means - r.mean()) ** 2, weights=w))
    return bt / tv if tv > 0 else 0.0, means

print("v0.2 分箱比率 (箱间均值方差/总方差, by 主汽流量 D):")
for hk, hi in [("H1", 0), ("H18", 17)]:
    row = []
    for ci, c in enumerate(CHAN):
        r0, means = ratio(R[:, hi, ci], F)
        row.append(f"{c}={r0:.3f}")
    print(f"  {hk}: " + "  ".join(row))

print("\nsh1_in H1 五箱均值残差 (v0.2):")
_, means = ratio(R[:, 0, 0], F)
print("  " + " ".join(f"{m:+.2f}" for m in means))
print("final H1 五箱均值残差 (v0.2):")
_, means = ratio(R[:, 0, 4], F)
print("  " + " ".join(f"{m:+.2f}" for m in means))
