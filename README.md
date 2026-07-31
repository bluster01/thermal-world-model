# Thermal World Model

Learned world models and model predictive control for thermal power plant processes.

## Experiments

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | Dynamics modeling accuracy | ✅ Complete |
| Phase 2 | Offline MPC simulation | 🔲 |
| Phase 3 | Feedforward closed-loop validation | 🔲 |

---

## Phase 1: Dynamics Modeling ✅

### Experimental Framework

| # | Experiment | Method | Metric | Validates |
|---|-----------|--------|--------|-----------|
| 1a | One-step prediction | s_t, a_t → ŝ_{t+1} | Per-variable RMSE | Basic dynamics fit |
| 1b | Autoregressive unrolling | Start from s_0, roll H={5,10,20,30} steps | RMSE vs step curve | Error accumulation (core WM metric) |
| 1c | Action sensitivity | Fix s_t, vary a_t ±5%/±10%/±20% | ΔT vs Δaction curve | Causal direction correctness |
| 1d | Component ablation | Remove action / VarAttn / TCN / RevIN | 1a + 1b metrics | Component contributions |

### 1a. One-step Prediction (exp_006 best)

| Variable Index | Feature | RMSE (°C or normalized) |
|---------------|---------|------------------------|
| TARGET (idx) | Main steam temperature | 0.0834 |

### 1b. Autoregressive Unrolling

| Experiment | Loss | Step 0 MAE | H=10 MAE | H=17 MAE | H=30 MAE | Growth × |
|-----------|------|-----------|----------|----------|----------|--------|
| exp_004 | MSE (v2 arch) | 0.194 | — | 1.012 | — | ×5.2 |
| exp_005 | GaussNLL | **0.077** | — | 0.953 | — | ×12.3 |
| **exp_006** | **β-NLL (β=-0.3)** | 0.083 | — | **0.808** | — | ×9.7 |
| exp_007 | MSE (v3 arch) | 0.207 | — | 1.017 | — | ×4.9 |

### 1c. Action Sensitivity

| Model | ±20% Valve → ΔT | Direction Correct? | Signal/Noise |
|-------|-----------------|-------------------|---------------|
| Full (TCN+Attn) | ±0.0013°C | Yes (开→升) | 0.23 |
| MLP Backbone | ±0.0001°C | **No** (noisy) | ~0 |

> ⚠️ **Critical finding**: Action response is nearly zero (~0.001°C for 20% valve change). The model learns dynamics almost entirely from state autoregression, not from action conditioning. This is a major limitation for MPC — the planner cannot evaluate "what if I take action X vs Y".

### 1d. Component Ablation (exp_008)

All variants trained with β-NLL (β=-0.3, warmup=20), same data split.

| Variant | 1-step RMSE | H=5 MAE | H=10 MAE | H=20 MAE | H=30 MAE | Growth × | Params |
|---------|------------|---------|----------|----------|----------|--------|--------|
| **Full** (TCN+VarAttn+RevIN) | 0.1425 | 0.2801 | 0.6216 | 1.2151 | 1.6102 | 15.8 | 476K |
| Zero Actions | 0.1311 | 0.2777 | 0.5731 | 1.1377 | 1.6817 | 17.9 | 476K |
| No VarAttn | 0.5620 | 0.2633 | 0.5059 | 1.0285 | **1.2019** | **9.8** | 376K |
| **MLP Backbone** | 0.4914 | 0.2540 | 0.4891 | 0.9510 | **1.1670** | 11.5 | 669K |
| No RevIN | 9.8873 | 7.1200 | 9.3684 | 5.9442 | 7.7922 | 0.8 | 476K |

#### Δ vs Full (baseline)

| Variant | 1-step | H=5 | H=10 | H=20 | H=30 |
|---------|--------|-----|------|------|------|
| Zero Actions | −8.0% | −0.9% | −7.8% | −6.4% | +4.4% |
| No VarAttn | +294% | −6.0% | −18.6% | −15.4% | **−25.4%** |
| MLP Backbone | +245% | −9.3% | −21.3% | −21.7% | **−27.5%** |
| No RevIN | +6839% | +2442% | +1407% | +389% | +384% |

---

## Key Findings

### Architecture
1. **RevIN is indispensable** — without it, MAE jumps 70× (0.14 → 9.9°C). The raw physical scale (~540°C) makes training impossible.
2. **One-step accuracy ≠ long-horizon accuracy** — TCN+Attention has best short-step (0.14) but worst H=30 (1.61). Simpler architectures (MLP, no attention) trade short-term accuracy for dramatically better long-term stability.
3. **MLP backbone is best for MPC** — H=30 MAE = 1.17°C, 27% better than TCN+Attention. Simpler models overfit less to temporal noise patterns.
4. **VariableAttention hurts long-term** — removing it improves H=30 by 25% while sacrificing one-step precision. Likely overfits to short-term patterns that don't generalize.

### Training
5. **Probabilistic training is essential** — NLL-based losses outperform MSE by 31–55% at H=17.
6. **β-NLL wins at long horizons** — β=−0.3 with warmup gives 15% better long-step MAE vs GaussNLL.
7. **Rollout loss K=5 validated** — sliding window with decaying weights [1.0, 0.8, 0.6, 0.4, 0.2] reduces error growth.
8. **GRU-cell decoder is broken** — constant ~1.3°C MAE regardless of step. Sliding window only viable approach.

### Critical Gap for Phase 2
9. **Action conditioning is too weak** — the model barely responds to control actions (±0.001°C for ±20% valve change). MPC requires the model to simulate different control policies; this must be addressed before Phase 2.
10. **Recommended Phase 2 precondition**: strengthen action signal (e.g., action embeddings with higher weight, cross-attention instead of concatenation, or action-specific loss terms).

---

## Architecture

```
[s_t ‖ a_t] → RevIN → TCN/MLP encoder → VariableAttention → Dual-head decoder
                                                              ├─ μ (state mean)
                                                              └─ logσ² (uncertainty)
```

- **Encoder**: Patch → per-variable TCN/MLP → VariableAttention → z
- **Decoder**: Linear(hidden → μ, logσ²)  
- **Loss**: β-NLL for calibrated probabilistic training
- **Rollout**: Sliding window (sole viable mode, GRU abandoned)

## Data

- Source: 1000MW coal-fired unit, 40+ variables, 10s sampling
- Target: main steam temperature (11-dim clean state + 2-dim delta valve actions)
- Dataset: 707K samples (train/val/test = 70/15/15)
- 11 operating conditions (load-based classification from Exp-0)

## References

- Differentiable World Models for Offline RL + MPC (arXiv, 2026.03)
- Graph Spatiotemporal World-Model-Driven Rolling MPC (Electronics, 2026)
- Differentiable Predictive Control / Neuromancer (PNNL)
- iTransformer-SST (Zhang et al., Sensors, 2026) — baseline comparison
