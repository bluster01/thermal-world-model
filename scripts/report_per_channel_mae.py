"""READ-ONLY per-channel MAE recompute from frozen checkpoints (execution-side reporting aid).

No verdict blocks, no file writes in artifacts/. Replicates the frozen metrics
sampling exactly (split=SPLIT_VAL, n_windows=128, batch=32, seed=50_000+seed,
boundary_mode=oracle) so channel-mean H18 matches the committed metrics files.
H60 is an OUT-OF-PROTOCOL extrapolation (frozen HORIZON=18) - labeled as such.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.final_wm.data import SPLIT_VAL, CanonicalRecord
from src.final_wm.evaluation import sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

CHANNELS = ("sh1_inlet_temp", "sh1_outlet_temp", "sh2_inlet_temp", "sh2_outlet_temp", "final_outlet_temp")
DEVICE = "cuda"
N_WIN, BSZ = 256, 32
torch.set_float32_matmul_precision("high")  # mirror the frozen ledger flag (tf32)


def per_channel_mae(model, record, horizon, seed):
    gen = torch.Generator().manual_seed(seed)
    acc = torch.zeros(horizon, 5)
    done = 0
    while done < N_WIN:
        bsz = min(BSZ, N_WIN - done)
        batch = sample_windows(record, SPLIT_VAL, bsz, 96, horizon, gen)
        history = batch.history.__class__(
            obs=batch.history.obs.to(DEVICE),
            actions=batch.history.actions.to(DEVICE),
            boundary=batch.history.boundary.to(DEVICE),
        )
        res = model.forecast(history, batch.future_actions.to(DEVICE),
                             boundary_mode="oracle",
                             true_future_boundary=batch.future_boundary.to(DEVICE))
        err = (batch.future_obs.to(DEVICE) - res.temps_mu).abs()  # (B,H,5)
        acc += err.sum(dim=0).cpu()
        done += bsz
    return acc / N_WIN  # (H,5) mean over windows


def main():
    record = CanonicalRecord("artifacts/final_wm/canonical_sideA.npz")
    props = load_grid_properties("artifacts/final_wm/iapws_surrogate.npz")
    specs = [s for s in ms.t1_specs(ms.SEEDS) if s.arm == "closure_cons_norew"]
    print("== production arm: closure_cons_norew (side A), per-channel MAE [°C], mean over windows ==")
    for spec in sorted(specs, key=lambda s: s.seed):
        model = build_world_model(spec, props).to(DEVICE).eval()
        model.transition._substep = torch.compile(model.transition._substep, dynamic=False)
        ckpt = torch.load(f"artifacts/final_wm/checkpoints/t1_{spec.arm}_seed{spec.seed}.pt",
                          map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        with torch.no_grad():
            m18 = per_channel_mae(model, record, 18, 50_000 + spec.seed)      # frozen protocol
            m60 = per_channel_mae(model, record, 60, 50_000 + spec.seed)      # OUT-OF-PROTOCOL
        # self-check: channel-mean H18 must match the committed metrics file
        frozen = torch.load(f"artifacts/final_wm/metrics/t1_{spec.arm}_seed{spec.seed}.pt",
                            map_location="cpu", weights_only=False)["metrics"]["mae"]
        mine = float(m18[:18].mean())
        ref = float(frozen[:, :18].mean())
        print(f"seed={spec.seed}  [self-check] recomputed channel-mean H18={mine:.4f} vs committed={ref:.4f}")
        def agg(m):
            return {
                "H1(step1)": m[0].tolist(), "H6(mean1-6)": m[:6].mean(0).tolist(),
                "H18(mean1-18)": m[:18].mean(0).tolist(), "term18(step18)": m[17].tolist(),
            }
        print(f"seed={spec.seed}")
        for k, v in agg(m18).items():
            print(f"  [frozen] {k}: " + "  ".join(f"{c[:4]}={x:.3f}" for c, x in zip(CHANNELS, v)))
        hm60 = m60[:60].mean(0).tolist(); t60 = m60[59].tolist()
        print(f"  [OUT-OF-PROTOCOL] H60(mean1-60): " + "  ".join(f"{c[:4]}={x:.3f}" for c, x in zip(CHANNELS, hm60)))
        print(f"  [OUT-OF-PROTOCOL] term60(step60): " + "  ".join(f"{c[:4]}={x:.3f}" for c, x in zip(CHANNELS, t60)))


if __name__ == "__main__":
    main()
