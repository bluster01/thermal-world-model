# Thermal World Model

Learned world models and model predictive control for thermal power plant processes.

## Overview

This project develops a **world-model-based Model Predictive Control (MPC)** framework for superheated steam temperature (SST) regulation in coal-fired thermal power plants. The core idea: train a differentiable dynamics model (ReVIN + TCN + iTransformer) from offline plant data, then use it for trajectory optimization in a receding-horizon MPC loop.

### Why world models for SST control?

- **Large time delays**: SST dynamics have 30s–120s delays from spray-water valve to sensor — SSM-based models capture long-range dependencies better than linear models
- **Nonlinear dynamics**: Load changes and disturbances break linear MPC assumptions
- **Safety constraints**: MPC naturally handles temperature bounds and actuator limits

### Architecture

```
[s_t ‖ a_t] → ReVIN(s) ‖ Emb(a) → TCN → iTransformer → P(s_{t+1} | s_t, a_t)
                                                              ↓
                                          Autoregressive unrolling for MPC planning
```

## Experiments

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | Dynamics modeling accuracy (1-step + multi-step unrolling) | 🔲 |
| Phase 2 | Offline MPC simulation (counterfactual control evaluation) | 🔲 |
| Phase 3 | Feedforward closed-loop validation (on real plant) | 🔲 |

## References

- Differentiable World Models for Offline RL + MPC (arXiv, 2026.03)
- Graph Spatiotemporal World-Model-Driven Rolling MPC (Electronics, 2026)
- Differentiable Predictive Control / Neuromancer (PNNL)
- iTransformer-SST (Zhang et al., Sensors, 2026) — baseline comparison

## Data

- Source: 1000MW coal-fired unit, 40 variables, 10s sampling
- Targets: main steam temperature, NOx, reheat steam temperature
- Control variables: spray-water valve position, feedwater flow, fuel rate
