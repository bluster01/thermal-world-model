# 世界模型控制方法对比实验方案 (baseline 复现计划)

> 2026-08-03 | 目标: 论文(Applied Energy)的世界模型驱动控制对比实验
> 任务: 伊敏6号机主汽温闭环控制 | 14维状态(含主汽温/减温阀位) | 10s采样 | 266K样本 | offline
> 现有: DWM-MPC (M7概率WM+梯度MPC, 闭环RMSE 2.50 vs PID 2.66, 150轨迹显著) + PID

## 1. 方法全景表 (12 候选, 来自三路文献调研)

| 方法 | 年份/出处 | 机制一句话 | offline | 复现成本 | 推荐 |
|---|---|---|---|---|---|
| **DWM-MPC (我们)** | — | 概率WM(β-NLL) + 梯度MPC | ✅ | 已有 | **主方法** |
| PID | — | 实际电厂回路 | ✅ | 已有 | 工程基线 |
| **线性MPC (ARX/N4SID)** | 经典/Wang 2022 Energies | 系统辨识→NMSS→QP滚动优化 | ✅ | 低(2-3天) | **必做** |
| **PETS 式 (B=5集成+CEM)** | Chua 2018 NeurIPS | bootstrap概率集成+CEM-MPC+TSinf粒子 | ✅(控制器部分) | 低(2-3天, CPU) | **必做** |
| **IQL** | Kostrikov 2022 ICLR | expectile回归学V, in-sample TD | ✅ | 低(2-3天) | **必做** |
| **TD3+BC** | Fujimoto 2021 NeurIPS | TD3+BC正则项, 单文件 | ✅ | 极低(1-2天) | **必做** |
| 确定性WM+gradMPC | 消融 | M7去掉σ输出 | ✅ | 已有改动量小 | 消融 |
| M7内SAC (MBPO协议) | Janner 2019 | 用M7当仿真器训练SAC | ✅(模型当环境) | 中(3-5天) | 机制对照 |
| TD-MPC2 | Hansen 2024 ICLR | 潜空间MLP WM+MPPI | 中(online为主,需改造) | 中高(1-2周) | 扩展 |
| DreamerV3 | Hafner 2023/Nature 2025 | RSSM+actor-critic想象rollout(H=16) | 中(官方无offline入口,需改造) | 中偏重(3-5人日+1-2天A100) | 扩展 |
| Diffuser | Janner 2022 ICML | 轨迹扩散+值引导, receding horizon | ✅(天然offline) | 中高(1-2周, GPU规划) | 扩展/探针 |
| CQL | Kumar 2020 | 保守Q下界 | ✅ | 中(方差大) | 备选佐证 |
| MBPO / GP-MPC | 2019/2020 | 需在线/数据规模不匹配 | ❌ | — | 落选 |

## 2. 论文主表设计 (6 行)

| 方法 | 类别 | 对照价值 |
|---|---|---|
| PID | 工程基线 | 实际回路性能 |
| 线性MPC (ARX/N4SID) | 传统最优控制 | 学习型 vs 辨识型 |
| PETS式 (概率集成+CEM) | 学习型MPC(采样规划) | **grad vs CEM 规划器 + 集成 vs 概率输出不确定性** |
| IQL | offline RL | 策略学习 vs 规划 |
| TD3+BC | offline RL | 最低成本策略基线 |
| **DWM-MPC (我们)** | 学习型MPC(梯度规划) | 主方法 |

## 3. 消融与机制故事

- **不确定性来源** (同框架换模型): DWM-MPC(M7概率) vs PETS(集成) vs 确定性WM → "概率输出 vs 集成 vs 无"三类不确定性对闭环性能的影响
- **规划器**: 梯度MPC (我们) vs CEM-MPC (PETS) → 2维动作下计算成本与性能
- **规划 vs 策略**: DWM-MPC/IQL/TD3+BC → MPC滚动优化 vs 直接策略映射
- **世界模型当仿真器**: M7内SAC → "学习的WM能否支撑RL训练" (MBPO协议)

## 4. 公平对比协议 (所有方法统一)

1. 同一数据集划分 (train/val/test, 与M7一致), 同一归一化 (RevIN/窗口)
2. 同一闭环评测: 扰动世界 (DIST_AMP=0.3, 3起点集×50轨迹=150) + 无扰动; 3起点集取平均
3. 同一指标集: RMSE(主)/IAE/ITAE/TV/超温积分/约束违反 + 配对Wilcoxon (150轨迹)
4. 评测基准: 每步真实SP (BENCH_SP_EACH=True)
5. 训练成本如实报告 (GPU时/CPU时), 推理成本 (单步规划耗时)
6. 训练随机性: 关键方法≥3 seed, 报mean±std

## 5. 分阶段执行计划

| 阶段 | 内容 | 预计工期 | 产出 |
|---|---|---|---|
| **P1** | 线性MPC (sysidentpy辨识ARX→QP, cvxpy) | 2-3天 | 主表行1 |
| **P2** | PETS式 (B=5集成+CEM-MPC+TSinf, 复用exp_027的MPC框架) | 2-3天 | 主表行2 + grad/CEM对照 |
| **P3** | IQL + TD3+BC (官方PyTorch, 自定义dataloader绕开gym) | 3-4天 | 主表行3,4 |
| **P4** | 消融: 确定性WM + M7内SAC | 3-5天 | 机制故事 |
| **P5** (可选) | TD-MPC2 / DreamerV3 / Diffuser | 2-4周 | 扩展对照 |

P1-P4 并行空间: P2 与 P3 可并行 (不同代码栈)。总工期 ~2周 (单人全职)。

## 6. 关键实现要点 (调研结论)

- **线性MPC**: 主汽温大迟延(60-120s)→需串级或多模型GPC; 参考文献: 蔡利军2018中国电力, Wang 2022 Energies 15(21):7935, Peng 2002 IEEE TCST
- **PETS**: B=5, 4×FC(200) swish, CEM popsize=500/elites=50/iters=5/α=0.1, TSinf 20粒子; 注意CEM horizon需≥60步(主汽温时标), 2维动作下CEM成本可控
- **IQL**: 官方 ikostrikov/implicit_q_learning (PyTorch), expectile τ=0.7, 需自定义reward (温度跟踪误差+动作惩罚)
- **TD3+BC**: sfujim/TD3_BC 单文件, 直接替换dataloader
- **M7内SAC**: 奖励= −RMSE − λ·TV, 用M7滚动预测当环境 (注意: 这是"模型当仿真器"的合法用法, 文献支持: MBPO/MORE/DeepThermal范式)
- **reward设计统一**: r = −|e| − λ_a·|Δa| (所有RL方法一致, 与MPC代价对应)

## 7. 论文故事线

"我们提出概率世界模型+梯度MPC用于火电主汽温闭环控制, 与6类基线对比: 传统(PID/线性MPC)、学习型MPC(PETS式CEM)、offline RL(IQL/TD3+BC)——在扰动世界+无扰动双协议下, DWM-MPC在RMSE/平滑度/超温上全面最优, 且揭示: 梯度规划优于CEM采样(2维动作), 概率输出优于集成不确定性(训练成本1/5), MPC优于直接策略(滚动修正)"

## 8. 风险与对策

- TD-MPC2/DreamerV3 offline改造失败 → 降级为PETS对照+文献引用, 不阻塞主表
- CQL方差大 → 只作佐证不主表
- Diffuser训练成本高 → 作扩展/未来工作
- 线性MPC在大迟延下可能性能差 → 如实报告(这正说明学习型MPC价值)
