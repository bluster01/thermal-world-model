# JEPA-B 系列设计与实现计划

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-30
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1
- Upstream Dependencies: `jepa_architecture_note_20260826.md`, `PREREG_protocol_fix_20260829.md`

## Experiment Overview

- **标题**：物理预测器不变条件下的 JEPA 状态增强 B1–B4
- **目标**：检验未来表示塑形、慢状态、训练期特权轨迹交叉预测和物理—残差分解，是否能改善主汽温世界模型的 H18 误差、跨负荷稳健性和长程漂移，同时保持喷水阀降温方向证书。
- **主假设**：收益若存在，应来自状态表示或状态演化，而不是替换 Fan2020-UDE 物理预测器、读取未来动作目标或增加输出级自由头。
- **类型**：training + analysis

## 原文到本项目的最小映射

| 来源 | 原文机制 | 本项目采用 | 明确不采用 |
|---|---|---|---|
| LeJEPA, arXiv:2511.08544v3 | 预测损失 + 各向同性高斯正则，避免 EMA/stop-grad 启发式 | 固定随机切片的高斯特征函数正则；只用于无物理语义的 latent | 不对 11 维物理态施加高斯先验 |
| LeWorldModel, arXiv:2603.19312v3 | action-conditioned latent rollout | B3 共享预测器显式读未来动作；B2 慢态更新不读当前动作 | 不替换现有物理 transition，不做 latent MPC |
| JEPA-x, arXiv:2608.24044v2 | 同轨迹双模态、共享预测器、四路自/交叉预测；训练后丢弃特权分支 | B3 使用 v2.2 已注册的 32 维富通道；另设固定错配对照 | 不称 XP-JEPA；不把未来动作送入任一 target encoder |
| Phys-JEPA, arXiv:2606.16076v1 | 物理/残差分解、static + dynamic consistency、残差防塌缩 | B4 物理块仍为 11 维 UDE 状态，4 维残差块独立正则；物理伪标签由同刻观测、边界和已记录动作确定性反演，residual target encoder 仍为动作盲 | 不把其单种子小幅改进当证据背书；不把输出误差冒充 latent consistency |

## 冻结实验矩阵

所有训练使用侧 A canonical v2.2、A5 质量门、oracle boundary、`hybrid + conservative_norew`、seed0、120 epochs、patience 20、batch 32、每 epoch 200 batch。control 与全部 B 臂用同一采样种子族；不读 test。

| ID | 仅有改动 | latent 规格 | 训练期额外损失 | 部署接口 |
|---|---|---|---|---|
| C0 | 无 | 0 | 无 | 7 boundary + 2 action → 5 温度 |
| B1 | 当前物理态预测未来观测窗表示 | 8 | pred + Gaussian-CF | 与 C0 相同，target/projector 可丢弃 |
| B2 | 低频慢态，每 6 步更新 | 4 | Gaussian-CF | 7+2 输入不变；内部状态 11+4 |
| B3 | JEPA-x 同轨迹特权分支 | 16 | 四路预测 + 双分支 Gaussian-CF | 特权分支丢弃，输入输出与 C0 相同 |
| B3-SHUFFLE | B3 结构不变，特权轨迹固定错配 | 16 | 同 B3 | 负控制，不参与候选晋级 |
| B4 | 11 维物理态 + 4 维残差态 | 4 | JEPA + residual Gaussian-CF + static + dynamic | 7+2 输入不变；内部状态 11+4 |

特权向量固定为 canonical v2.2 中 `boundary_ext(9) + aux(15) + mill_on(8)`，共 32 维；标准化统计只从 train split 和有效样本估计。原始 381 列未全部进入 canonical，不允许 runner 临时读取原 CSV。

## 判定与停止

1. 嵌套身份门：每个机制权重为 0 时，温度 rollout 与其自身对照逐位相同；B3 正确配对与 SHUFFLE 的模型结构、预算、随机种子完全相同。
2. 主门：末端测点在第 18 步的 `MAE <= 0.95 × C0`；五测点/全时程 MAE 仍完整报告，但不混入主门。
3. 稳健门：四个预注册负荷箱的 `max(MAE)/min(MAE)` 相对 C0 恶化不超过 10%。
4. 方向门：两阀、H18/H60、逐窗口支持域内，v0.3 规则（均值<0、UTC-day bootstrap CI 上界<0、正确方向率≥0.60）全部通过。保留旧备忘的 `frac_negative=1.000` 作为描述量，不再错误地作为正式硬门。
5. B3/B4 额外报告 H36/H60 末端 MAE、相对 persistence 的 rollout drift，以及 UTC-day 段级偏置；不新增通过阈值，避免看结果后改门。
6. seed0 若主门、稳健门、方向门全部通过，状态仅为 `PROMOTE_TO_FIXED_SEEDS_1_2`；否则为 `INCONCLUSIVE_EXPLORATORY_SEED0` 或 `REJECT_EXPLORATORY_SEED0`。本批不得升级正式论文 verdict。
7. 不自动重试、不补跑搜索；OOM/NaN/产物缺失直接记失败并停止该臂。

## Setup

- **语言/框架**：Python 3.11，PyTorch 2.x
- **入口命令**：`python experiments/final_wm/run_jepa_b.py --queue`
- **工作目录**：仓库根目录
- **依赖**：仓库现有依赖；不新增第三方 JEPA 包
- **环境**：Linux + CUDA；单 GPU；顺序执行，禁止多进程争抢 GPU

## Inputs

| 输入 | 路径 | 说明 |
|---|---|---|
| canonical v2.2 | `artifacts/final_wm/canonical_sideA_v2.npz` | Linux 本地重建，test locked |
| IAPWS surrogate | `artifacts/final_wm/iapws_surrogate.npz` | 已审计物性网格 |
| matrix | `configs/final_wm/jepa_b_series_v1.json` | 冻结 arms、预算、权重、判据 |
| registry | `configs/phase3_5/experiment_registry.json` | 只授权 `jepa_b_series_v1` |

## Expected Outputs

| 输出 | 路径 | 格式 | 成功标准 |
|---|---|---|---|
| per-arm ledger | `results/final_wm/jepa_b_series_v1/<arm>/ledger.jsonl` | JSONL | 每 epoch + final，含 commit/config hash |
| checkpoint | `results/final_wm/jepa_b_series_v1/<arm>/checkpoints/*.pt` | torch | 最佳 validation checkpoint；默认不入 Git |
| per-arm report | `results/final_wm/jepa_b_series_v1/<arm>/report.json` | JSON | 身份、训练、H18/H36/H60、分箱、方向、偏置齐全 |
| root report | `results/final_wm/jepa_b_series_v1/report.json` | JSON | 6/6 臂完整或逐臂失败原因明确 |

## Monitoring Configuration

- **超时**：不由 runner 猜测；外层 Linux 任务按单臂 6 h 硬超时监控
- **监控文件**：每臂 `ledger.jsonl`
- **异常**：20 个 epoch 无改进正常早停；非有限 loss、无 checkpoint、指纹不符均 fail-closed
- **恢复**：只有 matrix hash、git SHA、checkpoint 和 final ledger 同时匹配才允许跳过；不从半臂续训

## 实现步骤（TDD）

1. 新建 `tests/final_wm/test_jepa.py`：先覆盖 train-only 归一化、未来动作隔离、固定错配、Gaussian-CF、B2 六步保持、B4 物理/残差损失与机制关闭身份。
2. 新建 `src/final_wm/jepa.py`：实现数据视图、窗口采样、B1–B4 模块和统一辅助损失接口。
3. 新建 `experiments/final_wm/jepa_b_spec.py` 与 `run_jepa_b.py`：实现矩阵校验、registry 授权、顺序训练、配对评估、方向与长程报告、resume 指纹。
4. 新建 `configs/final_wm/jepa_b_series_v1.json` 与预注册稿；先 `--sanity`，再 `--quick` micro-smoke。
5. 更新 registry、TODO 与 Linux runbook；跑目标测试、全 `tests/final_wm`、registry check，审计 tracked diff 后提交并 push。
