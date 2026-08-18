import os, sys, importlib.util
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("t02", os.path.join(os.getcwd(), "02_train.py"))
t02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t02)
import numpy as np
import pandas as pd
import torch

torch.manual_seed(0)
df = pd.read_csv(t02.CSV,
                 usecols=t02.EXO + t02.EXO_EXTRA + t02.OUTPUTS + [
                     "一级减温调节门阀位", "二级减温调节门阀位", "分离器出口压力", "末级过热器出口压力"],
                 dtype=np.float32).iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)
Xtr, Ytr, Itr, Itr_T = t02.e0_build_windows(df, 0, 30000, 5)
Xtr_t = torch.from_numpy(Xtr[:64]).to(t02.DEVICE)
Itr_t = torch.from_numpy(Itr[:64]).to(t02.DEVICE)
ItrT_t = torch.from_numpy(Itr_T[:64]).to(t02.DEVICE)

model = t02.E0Model().to(t02.DEVICE)
init = Itr_t
obs = ItrT_t
pm = init[:, 2]
p_out = init[:, 7]
p0 = pm + (p_out - pm) / 3.0
p1 = pm + 2.0 * (p_out - pm) / 3.0
h0 = t02.h_of_pT(p0, obs[:, 0])
h1 = t02.h_of_pT(p1, obs[:, 2])
h2 = t02.h_of_pT(p_out, obs[:, 4])
h = torch.stack([h0, h1, h2])
ts = t02.T_of_ph(torch.stack([p0, p1, p_out]), h)
Tm = ts + model.tri("dTm")[:, None]
rB = init[:, 1].clone()

# 打印 h 范围
print("h init range:", h.min().item(), h.max().item())
print("Tm init range:", Tm.min().item(), Tm.max().item())

out = model.integrate(Xtr_t, h, Tm, rB, Xtr_t.shape[1])
print("out.requires_grad:", out.requires_grad)
print("out.grad_fn:", type(out.grad_fn).__name__ if out.grad_fn else None)
print("out main range:", out[:, :, 4].min().item(), out[:, :, 4].max().item())

# 直接对 out.sum 求梯度
gs = torch.autograd.grad(out.sum(), [model.raw["k0"], model.raw["UA0"], model.raw["M0"], model.raw["dTm0"]], allow_unused=True)
print("grad k0:", gs[0], "UA0:", gs[1], "M0:", gs[2], "dTm0:", gs[3])

# 有限差分：k0 改 50%
model2 = t02.E0Model().to(t02.DEVICE)
with torch.no_grad():
    model2.raw["k0"].copy_(model.raw["k0"] + 0.5)
    out2 = model2.integrate(Xtr_t, h, ts + model2.tri("dTm")[:, None], rB, Xtr_t.shape[1])
print("FD: k0 +0.5(50%): main mean", out[:, :, 4].mean().item(), "->", out2[:, :, 4].mean().item())

# 单步图检查：h 更新一个子步
t = 0
pst = torch.stack([p0[:, t], p1[:, t], p_out[:, t]])
tss = t02.T_of_ph(pst, h)
Q = model.tri("UA")[:, None] * (Tm - tss)
hh = h + 2.0 * (init[:, 0][None, :] * (torch.stack([t02.h_sep_of(init[:, 2], init[:, 3]), (init[:, 0]*h[0])/(init[:, 0]+1.0), (init[:, 0]*h[1])/(init[:, 0]+1.0)]) - h) + Q) / model.tri("M")[:, None]
print("single-step h grad_fn:", hh.grad_fn)
g2 = torch.autograd.grad(hh.sum(), [model.raw["k0"], model.raw["M0"], model.raw["UA0"], model.raw["dTm0"]], allow_unused=True)
print("single-step grad k0:", g2[0], "M0:", g2[1], "UA0:", g2[2], "dTm0:", g2[3])
