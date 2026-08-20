"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""Fig 3a 数据: 物理WM ±2% 阀位阶跃响应探针 (v0.2 closure_steam seed0, 侧A)
对比: Direct WM v2 实测响应值 (来自审计: F1 -0.005/-0.010/-0.015°C)"""
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

record = CanonicalRecord("artifacts/final_wm/canonical_sideA.npz")
spec = next(s for s in ms.t1_specs((0,)) if s.arm == "closure_steam")
model = build_world_model(spec, load_grid_properties(GRID)).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=False)["state_dict"])
model.eval()

# 取一组 val 窗口, 在基线轨迹上做 ±2% 阀位扰动, 测各通道温度响应
gen = torch.Generator().manual_seed(42)
b = sample_windows(record, SPLIT_VAL, 8, HIST, HOR, gen)
hist = b.history.__class__(
    obs=b.history.obs.to(DEV), actions=b.history.actions.to(DEV),
    boundary=b.history.boundary.to(DEV),
)
fa = b.future_actions.to(DEV).clone()
fb = b.future_boundary.to(DEV)

with torch.no_grad():
    base = model.forecast(hist, fa, boundary_mode="oracle", true_future_boundary=fb)
    resp = {}
    for name, dv in [("+2% v1", 0), ("+2% v2", 1), ("-2% v1", 0), ("-2% v2", 1)]:
        fa2 = fa.clone()
        fa2[:, :, dv] = fa2[:, :, dv] * (1.02 if name.startswith("+") else 0.98)
        out = model.forecast(hist, fa2, boundary_mode="oracle", true_future_boundary=fb)
        d = (out.temps_mu - base.temps_mu).cpu().numpy()  # (B,H,5)
        resp[name] = d.mean(axis=0)  # (H,5)

print("物理WM v0.2 阀位阶跃响应 (均值, °C): 通道=[sh1_in, sh1_out, sh2_in, sh2_out, final]")
for name in resp:
    print(f"{name}: H1={resp[name][0].round(3)} H6={resp[name][5].round(3)} H18={resp[name][-1].round(3)}")
print("\nDirect WM v2 同探针: ±2% -> -0.005/-0.010/-0.015°C (F1, 审计值)")
print("能量平衡参考: 约 -0.45 ~ -0.87°C / 2%")
np.savez("/tmp/step_response.npz",
         v1p=resp["+2% v1"], v2p=resp["+2% v2"], v1m=resp["-2% v1"], v2m=resp["-2% v2"])
print("\nsaved /tmp/step_response.npz")
