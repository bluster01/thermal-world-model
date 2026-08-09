# exp_101 — 真实增量式 PID vs DWM-MPC 闭环对比（小实验 / pilot）

> **⚠️ 实验级别：PILOT（小实验）** — 1 seed × 20 轨迹 × 120 步，单扰动序列。
> 结果仅作方向参考，**不构成正式结论**；正式闭环对比须按审计 5C 冻结 estimand 后另行设计。

## 目的

路线A（阀位动作通道）的公平闭环 baseline 对比：今天验证过的**真实增量式 PID**
（POU #107，slope≈1.07，`RealPIDController`）当 baseline，对 DWM-MPC（`mpc_m7`）。

## 协议

- 复用 phase2 `eval_protocol` P0-A/B/C 公平协议：第三方世界集成（M0/M8/M9）、
  物理侧扰动（负荷/煤量通道）、PID/MPC 共用同一执行器、同一扰动序列。
- 三臂（同世界、同扰动、同 start_seed=42）：
  - `pid_real`：真实参数 + 副调标定 K=2.0%/°C（增量式公式已与 pid_repro 逐位数值对齐，4/4 PASS）
  - `pid_legacy`：旧主表虚拟 PID kp=40 ki=8（对照用）
  - `mpc_m7`：DWM-MPC，综合代价

## 结果（n=20/臂，mean±std）

| 臂 | RMSE | IAE | TV（动作平滑度） |
|---|---|---|---|
| pid_real | 2.033 ± 1.399 | — | **0.097 ± 0.064** |
| pid_legacy | 1.845 ± 1.319 | — | 0.325 ± 0.271 |
| mpc_m7 | 1.989 ± 1.362 | — | 0.209 ± 0.058 |

- **RMSE 三臂持平**（std ~1.3 远大于臂间差 ~0.2，无显著差异）；
- **TV：pid_real 最平滑**（≈legacy 的 1/3、MPC 的 1/2），与冒烟信号一致（n=2 时 0.06 << 0.78/0.15 的方向性保留）；
- 冒烟时 pid_real RMSE 最低的迹象**未延续**到正式 20 轨迹（正式反而略高）。

## 解读边界（勿越界引用）

1. 单 seed、单扰动序列——未做配对显著性检验；
2. RMSE 臂间差异不显著，**不能**得出"真实 PID 优于/劣于 MPC"；
3. 唯一稳定的信号是**动作平滑度**（TV），可支撑"真实 PID 是平滑 baseline"的工程论断，但这不足以推翻/支撑 MPC 叙事；
4. 本实验的定位是**路线A baseline 可行性验证**，不是 E3 的替代。

## 产物

- `pid_real.json` / `pid_legacy.json` / `mpc_m7.json`：20 轨迹逐条指标（rmse/iae/itae/tv/overtemp/max_temp 等）
- `case_curves.png`：示例 case 时序曲线
