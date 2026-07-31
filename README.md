# Thermal World Model

Learned world models and model predictive control for thermal power plant processes.

## Experiments

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | Dynamics modeling accuracy (1-step + multi-step unrolling) | 🟡 |
| Phase 2 | Offline MPC simulation (counterfactual control evaluation) | 🔲 |
| Phase 3 | Feedforward closed-loop validation (on real plant) | 🔲 |

---

## Phase 1: Dynamics Modeling

### Todo

| # | Task | Status |
|---|------|--------|
| 1 | Clean state + delta actions baseline | ✅ exp_004 |
| 2 | Probabilistic modeling (Gaussian NLL) | ✅ exp_005 |
| 3 | Uncertainty calibration (β-NLL) | ✅ exp_006 |
| 4 | Ablation: MSE-only control experiment | ✅ exp_007 |
| 5 | Multi-step rollout loss (K=5) | ✅ exp_003+ |
| 6 | GRU-cell decoder vs sliding window | ✅ (GRU broken, sliding wins) |
| 7 | Horizon tuning (H=5-10 for MPC) | 🔲 |
| 8 | Two-stage rollout loss (K=20, step-weights) | 🔲 |
| 9 | Latent diffusion decoder (like Graph WM paper) | 🔲 |

### Results Summary

| Exp | Loss | Step 0 MAE | Step 17 MAE | Error Growth | Train Time | Key Finding |
|-----|------|-----------|-------------|-------------|------------|-------------|
| exp_004 | MSE (v2 arch) | 0.194 | 1.012 | ×5.2 | 30.9min | Baseline |
| exp_005 | GaussNLL | **0.077** | 0.953 | ×12.3 | 30.6min | Short-step winner, σ overconfident |
| exp_006 | β-NLL (β=-0.3) | 0.083 | **0.808** | ×9.7 | 43.9min | Long-step winner, best overall |
| exp_007 | MSE (v3 arch) | 0.207 | 1.017 | ×4.9 | 18.9min | NLL loss is key, not architecture |

### Key Findings

1. **Probabilistic training is essential** — NLL-based losses outperform MSE by 31-55%
2. **β-NLL wins at long horizons** — 0.808°C at 180s vs 0.953°C (GaussNLL)
3. **σ head acts as regularizer** — even poorly calibrated variance improves μ training
4. **GRU-cell decoder is broken** — constant ~1.3°C MAE regardless of step
5. **Sliding-window encoder is necessary** — GRU's 64-dim hidden state cannot carry 96-step causal chain
6. **Rollout loss (K=5) is validated** — reduces error growth from ×12 to ×5-10

---

## Architecture

```
[s_t ‖ a_t] → ReVIN(s) ‖ Emb(a) → TCN → iTransformer → P(s_{t+1} | s_t, a_t)
                                                              ↓
                                          Autoregressive unrolling for MPC planning
```

## References

- Differentiable World Models for Offline RL + MPC (arXiv, 2026.03)
- Graph Spatiotemporal World-Model-Driven Rolling MPC (Electronics, 2026)
- Differentiable Predictive Control / Neuromancer (PNNL)
- iTransformer-SST (Zhang et al., Sensors, 2026) — baseline comparison

## Data

- Source: 1000MW coal-fired unit, 40 variables, 10s sampling
- Targets: main steam temperature, NOx, reheat steam temperature
- Control variables: spray-water valve position, feedwater flow, fuel rate
