import os, sys, importlib.util
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("t02", os.path.join(os.getcwd(), "02_train.py"))
t02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t02)
import numpy as np
import pandas as pd
import torch

torch.manual_seed(0)
np.random.seed(0)
df = pd.read_csv(t02.CSV,
                 usecols=t02.EXO + t02.EXO_EXTRA + t02.OUTPUTS + [
                     "一级减温调节门阀位", "二级减温调节门阀位", "分离器出口压力", "末级过热器出口压力"],
                 dtype=np.float32).iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)

Xtr, Ytr, Itr, Itr_T = t02.e0_build_windows(df, 0, 30000, 5)
Xtr_t = torch.from_numpy(Xtr[:256]).to(t02.DEVICE)
Ytr_t = torch.from_numpy(Ytr[:256]).to(t02.DEVICE)
Itr_t = torch.from_numpy(Itr[:256]).to(t02.DEVICE)
ItrT_t = torch.from_numpy(Itr_T[:256]).to(t02.DEVICE)

model = t02.E0Model().to(t02.DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
w5 = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=t02.DEVICE)


def fwd():
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
    return model.integrate(Xtr_t, h, Tm, rB, Xtr_t.shape[1])


print("target main mean:", Ytr_t[:, :, 4].mean().item())
with torch.no_grad():
    pred0 = fwd()
    print("pred main mean (prior):", pred0[:, :, 4].mean().item())
    print("pred sh1_in mean (prior):", pred0[:, :, 0].mean().item(), "target:", Ytr_t[:, :, 0].mean().item())
    print("pred sh2_out mean (prior):", pred0[:, :, 3].mean().item(), "target:", Ytr_t[:, :, 3].mean().item())

print("\n--- 训练 30 步观察 ---")
for it in range(30):
    opt.zero_grad()
    pred = fwd()
    mse = (((pred - Ytr_t) ** 2) * w5).mean()
    loss = mse
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    if it % 5 == 0:
        g = {k: model.raw[k].grad.abs().item() for k in ["M0", "UA0", "Cm0", "k0", "tauB", "th1", "th2", "dTm0"]}
        print(f"iter {it}: loss={loss.item():.1f} grads={ {k: f'{v:.3g}' for k, v in g.items()} }")

with torch.no_grad():
    pred = fwd()
    print("\npred main mean (30 steps):", pred[:, :, 4].mean().item(), "target:", Ytr_t[:, :, 4].mean().item())
