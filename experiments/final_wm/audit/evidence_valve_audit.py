"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""物理WM 阀门证据链审计:
1) 学习参数实值 (th/gamma/M/UA/tau_evap)
2) 模型喷水量 vs 实测 W 对比
3) H1 响应量级拆解 (输出方程瞬时混合 vs 状态动力学)
4) 长时反转定位: 关 closure 重测 60 步响应
"""
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

tr = model.transition
print("=== 学习参数实值 (prior -> learned) ===")
for name in ["th1", "th2", "th1d", "th2d", "gamma1", "gamma2", "tau_evap",
             "M0", "M1", "M2", "UA0", "UA1", "UA2", "Cm0", "Cm1", "Cm2", "tauB"]:
    prior = tr.priors.get(name)
    val = tr.val(name).item()
    print(f"  {name}: prior={prior}  learned={val:.4f}")

# 实测 W vs 模型喷水
o = record.obs.numpy(); sp = record.split.numpy()
bnd = record.boundary.numpy()
act = record.actions.numpy()
W_tph = bnd[:, 6]  # t/h
KAPPA = 0.2777778  # t/h -> kg/s
W_real_kgps = W_tph * KAPPA
with torch.no_grad():
    v1 = torch.tensor(act[:, 0][sp == 1][:5000], dtype=torch.float32, device=DEV)
    v2 = torch.tensor(act[:, 1][sp == 1][:5000], dtype=torch.float32, device=DEV)
    pm = torch.tensor(bnd[sp == 1, 2][:5000], dtype=torch.float32, device=DEV)
    th1, th2 = tr.th_of(pm)
    dsw1 = (th1 * tr.varphi(v1, 1)).cpu().numpy()
    dsw2 = (th2 * tr.varphi(v2, 2)).cpu().numpy()
W_m = W_real_kgps[sp == 1][:5000]
print(f"\n=== 喷水量对比 (val) ===")
print(f"  实测 W: mean={W_m.mean():.2f} kg/s  p90={np.quantile(W_m,0.9):.2f}")
print(f"  模型 dsw1: mean={dsw1.mean():.2f}  p90={np.quantile(dsw1,0.9):.2f}")
print(f"  模型 dsw2: mean={dsw2.mean():.2f}  p90={np.quantile(dsw2,0.9):.2f}")
print(f"  模型 dsw1+dsw2 vs 实测 W: {(dsw1+dsw2).mean():.2f} vs {W_m.mean():.2f}")
print(f"  (KAPPA 因子已含: 模型 1.0 开度 = th kg/s)")

# H1 响应拆解: 输出方程瞬时混合 vs 状态动力学
gen = torch.Generator().manual_seed(3)
b = sample_windows(record, SPLIT_VAL, 16, HIST, HOR, gen)
h = b.history.__class__(obs=b.history.obs.to(DEV), actions=b.history.actions.to(DEV),
                        boundary=b.history.boundary.to(DEV))
fa = b.future_actions.to(DEV)
fb = b.future_boundary.to(DEV)
with torch.no_grad():
    st0 = model._initial_state(h)
    # 基准: 保持原动作
    base = model.forecast(h, fa, boundary_mode="oracle", true_future_boundary=fb)
    # 输出方程瞬时项: 用 step0 状态 + 新动作直接算 output_temperatures
    t_now = tr.output_temperatures(st0, fb[:, 0], fa[:, 0])
    fa2 = fa.clone(); fa2[:, :, 0] = fa2[:, :, 0] * 1.02
    t_now2 = tr.output_temperatures(st0, fb[:, 0], fa2[:, 0])
    inst = (t_now2 - t_now).mean(axis=0).cpu().numpy()
    print(f"\n=== H1 响应拆解 (输出方程瞬时混合项, 通道=[sh1_in,sh1_out,sh2_in,sh2_out,final]) ===")
    print(f"  瞬时项 (状态未动, 仅输出方程): {inst.round(3)}")
    out2 = model.forecast(h, fa2, boundary_mode="oracle", true_future_boundary=fb)
    h1_full = (out2.temps_mu[:, 0] - base.temps_mu[:, 0]).mean(axis=0).cpu().numpy()
    print(f"  H1 全响应 (动力学1步+输出方程): {h1_full.round(3)}")
    print(f"  瞬时项占比: {(inst/h1_full).round(2)}")

# 长时反转定位: 手动 rollout 60 步, 分别测 (a) 全模型 (b) 无 closure (c) 无 metal 耦合近似不可行, 用 (b) 即可
def rollout(model, h, fb_all, fa_all, use_closure):
    st = model._initial_state(h)
    outs = []
    for t in range(60):
        if use_closure:
            res = model.closure.residual(st, fb_all[:, t]) if hasattr(model.closure, "residual") else None
        else:
            res = None
        step = model.transition.step(st, fb_all[:, t], fa_all[:, t], residual=res)
        st = step.state
        outs.append(model.transition.output_temperatures(st, fb_all[:, t], fa_all[:, t]))
    return torch.stack(outs, dim=1)  # (B,60,5)

print(f"\n=== 长时(60步)反转定位: +5% v2 ===")
b2 = sample_windows(record, SPLIT_VAL, 16, HIST, 60, gen)
h2 = b2.history.__class__(obs=b2.history.obs.to(DEV), actions=b2.history.actions.to(DEV),
                          boundary=b2.history.boundary.to(DEV))
fb2 = torch.cat([h2.boundary[:, -1:].expand(-1, 60, -1), b2.future_boundary.to(DEV)], dim=1)[:, :60]
fa2b = b2.future_actions.to(DEV)[:, :60].clone()
fa2p = fa2b.clone(); fa2p[:, :, 1] = fa2p[:, :, 1] * 1.05
with torch.no_grad():
    base_c = rollout(model, h2, fb2, fa2b, True)
    pert_c = rollout(model, h2, fb2, fa2p, True)
    base_nc = rollout(model, h2, fb2, fa2b, False)
    pert_nc = rollout(model, h2, fb2, fa2p, False)
d_full = (pert_c - base_c)[:, -1, 4].cpu().numpy()
d_nocl = (pert_nc - base_nc)[:, -1, 4].cpu().numpy()
print(f"  全模型 final 终端 ΔT: mean={d_full.mean():+.3f}  frac_neg={(d_full<0).mean():.2f}")
print(f"  关 closure final 终端 ΔT: mean={d_nocl.mean():+.3f}  frac_neg={(d_nocl<0).mean():.2f}")
