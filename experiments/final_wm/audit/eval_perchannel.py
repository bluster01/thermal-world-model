"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""Fig3 数据 (官方 evaluate_windows 路径): per-channel MAE + 分箱"""
import sys
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[3]))
import numpy as np
import torch

from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from src.final_wm.evaluation import evaluate_windows
import experiments.final_wm.matrix_spec as ms

GRID = "/home/bluster/.hermes/workspace/adhoc2_lumped_enthalpy/out/iapws_surrogate.npz"  # env-specific
CKPT = "artifacts/final_wm/checkpoints/t1_closure_steam_seed0.pt"
DEV = "cuda"
HIST, HOR = ms.HISTORY_STEPS, ms.HORIZON

record = CanonicalRecord("artifacts/final_wm/canonical_sideA.npz")
spec = next(s for s in ms.t1_specs((0,)) if s.arm == "closure_steam")
model = build_world_model(spec, load_grid_properties(GRID)).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=False)["state_dict"])
model.eval()

with torch.no_grad():
    res = evaluate_windows(model, record, 1, n_windows=128, batch_size=32,
                           history_steps=HIST, horizon=HOR, boundary_mode="oracle",
                           seed=10000, device=DEV)
print("官方 evaluate_windows (closure_steam seed0):")
for name in ("nll", "mae", "crps"):
    v = getattr(res, name, None)
    if v is not None:
        vv = v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        print(f"  {name}: shape={vv.shape} overall_mean={vv.mean():.3f}")
        if vv.ndim == 2:
            ph = vv.mean(axis=0)
            print(f"    per-horizon H1={ph[0]:.3f} H6={ph[5]:.3f} H12={ph[11]:.3f} H18={ph[-1]:.3f}")
