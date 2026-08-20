"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""消融: 再湿项 (aW=0) vs 全模型, 60步 +5% v2 终端响应"""
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
HIST = ms.HISTORY_STEPS

record = CanonicalRecord("artifacts/final_wm/canonical_sideA.npz")
spec = next(s for s in ms.t1_specs((0,)) if s.arm == "closure_steam")
model = build_world_model(spec, load_grid_properties(GRID)).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=False)["state_dict"])
model.eval()

def rollout(h, fb_all, fa_all):
    st = model._initial_state(h)
    outs = []
    for t in range(60):
        res = model.closure(st, fb_all[:, t]) if model.closure is not None else None
        step = model.transition.step(st, fb_all[:, t], fa_all[:, t], residual=res)
        st = step.state
        outs.append(model.transition.output_temperatures(st, fb_all[:, t], fa_all[:, t]))
    return torch.stack(outs, dim=1)

gen = torch.Generator().manual_seed(3)
b2 = sample_windows(record, SPLIT_VAL, 16, HIST, 60, gen)
h2 = b2.history.__class__(obs=b2.history.obs.to(DEV), actions=b2.history.actions.to(DEV),
                          boundary=b2.history.boundary.to(DEV))
fb2 = torch.cat([h2.boundary[:, -1:].expand(-1, 60, -1), b2.future_boundary.to(DEV)], dim=1)[:, :60]
fa2b = b2.future_actions.to(DEV)[:, :60].clone()
fa2p = fa2b.clone(); fa2p[:, :, 1] = fa2p[:, :, 1] * 1.05

def probe(tag):
    with torch.no_grad():
        base = rollout(h2, fb2, fa2b)
        pert = rollout(h2, fb2, fa2p)
    d = (pert - base)[:, -1, 4].cpu().numpy()
    print(f"{tag}: final 终端 ΔT mean={d.mean():+.3f}  frac_neg={(d<0).mean():.2f}")

probe("全模型")
with torch.no_grad():
    model.transition.raw["aW1"].data.fill_(-60.0)
    model.transition.raw["aW2"].data.fill_(-60.0)
probe("aW=0 (去再湿)")
