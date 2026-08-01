# Phase 1 收口实验设计：统一横屏对比 + Direct WM 模块化消融（定稿）

> 日期: 2026-08-01 | 作者: cwoc + Hermes
> 状态: 定稿，待执行 | 依据: exp_023 架构 + 主汽温预测领域基线调研

---

## 0. 目标

1. **同协议横屏对比**：统一数据/归一化/训练/评测，Direct WM vs 通用基线 + 主汽温领域基线，证明当前模型预测最优
2. **模块化消融**：Direct WM 每个组件逐一去掉/替换，量化贡献
3. **归一化修正**：RevIN 实例归一化 → 全局 min-max（训练集统计，参数保留，反归一化同参数）

## 1. 统一数据协议（所有模型完全相同）

| 项 | 设定 |
|---|---|
| 输入 | 40 列全特征（同 exp_023，含阀位列） |
| 目标 | 单变量主汽温，H=18 直接多步 |
| 归一化 | **全局 min-max，训练集统计**（替换 RevIN 实例归一化；每列 min/max 从 train 算，存 scaler.json，val/test 用同参数） |
| 切分 | 70/15/15（同 exp_023） |
| 评测 | test 500 样本 seed 42，逐步 MAE |

## 2. 统一训练协议

- β-NLL（概率头）或 MSE（确定性头），β warmup 20
- 逐步权重 `w = linspace(1.0, 0.6, 18)`
- AdamW（lr/weight_decay 统一）+ ReduceLROnPlateau + early stopping（patience 同）
- 同 STEPS/BS（500/256）、同 seed
- 结果: `results/exp_025_<model>/results.json`（统一 schema）

## 3. 模型矩阵

### A. Direct WM 模块化消融（数据/训练不变）

| ID | 变体 | 改动 | 回答 |
|---|---|---|---|
| M0 | **Direct WM 全量（全局minmax版）** | 基线 | 当前模型公平版 |
| M1 | −未来动作注入 | 去 action_enc | 动作条件化贡献 |
| M2 | −Patch | 原始 token 直接进 TCN | patch 贡献 |
| M3 | −PerVariableTCN | 共享 TCN | per-variable 结构贡献 |
| M4 | +VarAttn | 加变量注意力（world_model 现有） | VarAttn 是否有帮助 |
| M5 | 确定性 head | MSE 替代 β-NLL | 概率头代价 |

### B. 通用基线（同数据同协议）

| ID | 模型 | 说明 |
|---|---|---|
| B1 | TCN | 时序卷积骨干 |
| B2 | LSTM | 循环骨干（exp_018 改造，direct 输出） |
| B3 | GRU | 循环变体（exp_021 改造，direct 输出） |
| B4 | iTransformer | 变量通道注意力 + 线性（时序 SOTA） |
| B5 | DLinear | 线性分解最简基线 |
| B6 | Exp-0 / TCN-iTransformer-Prob 重跑 | 原版架构统一协议重跑（旧 0.586 协议不同不可直接比） |

### C. 主汽温领域基线（调研新增）

| ID | 模型 | 出处 | 说明 |
|---|---|---|---|
| B7 | **iTransformer-SST** | MDPI Sensors 2026, 26(10):3078 | 主汽温 SOTA：iTransformer + LTC 局部时序卷积 + physics-guided 正则（延迟单调+平滑）；MSE 0.0887/MAE 0.2312，对比 LSTM/Informer/基线iTransformer |
| B8 | LSTM-MPC | IFAC 2020 (Baylor) | LSTM 预测 + PSO 加权 MPC；预测部分即 LSTM，B2 已覆盖，引用即可 |
| B9 | **DenseResLSTM-Attention** | PMC 2022 | 锅炉汽温专用：密集残差 LSTM + 注意力 + 区间不确定性；预测 3min 内 ±5% |

实现取舍: B2/B3 已实现（exp_018/021 改造为 direct 输出 + 全局 min-max）；
B7 实现 LTC + 物理正则简化版（延迟单调约束只用于评测说明，训练用 NLL+MSE 主目标）；
B9 实现 DenseResidual 骨架 + attention（不确定性头可选）。

**公平性**：所有模型同一输入（40列历史，无未来动作）+ 同 min-max + 同训练循环。
M1（去动作的 Direct WM）与 B1-B9 纯架构对比；M0（带动作）额外展示动作条件化增益。
iTransformer-SST 的 physics-guided 正则与我们 exp_013-017 的符号正则是同类物——其评测对比要注明"我们已证明单步符号正则伪物理，SST 的延迟单调约束是长程约束，不同"。

## 4. 评测指标

1. **Rollout 逐步 MAE**（H=18 + avg）——主指标
2. **σ 校准**（概率模型）：\|err\|/σ
3. **11 工况**：只跑 M0（最终模型），分析动作/精度在不同工况表现（exp_019 分类脚本复用）
4. **成本**：参数、训练时长

## 5. 输出

- 表2: 全部模型 × avg MAE / step17 / 参数 / 训练时间
- 表3: 消融 M0-M5
- 图: 逐步 MAE 曲线（全模型同图）
- 表4: M0 分工况精度 + 动作敏感性

## 6. 实现

```
exp_025_unified_benchmark.py（单一注册式脚本）
├─ data: 全局 min-max scaler（train 统计, 存 scaler.json）
├─ registry: {M0..M5, B1..B7, B9 → build_fn}
├─ train(): 统一循环
├─ eval(): 统一 rollout MAE / σ calib
└─ report: 汇总 → results/exp_025_summary.json
```

- 消融用 flags 控制组件开关
- 训练成本: 13 模型 × ~10min ≈ 2h，后台串行
- 步骤: 先 M0/M1 + B1-B6（核心对比），再 B7/B9（领域基线），最后 M2-M5 消融

## 7. 风险

| 风险 | 应对 |
|---|---|
| 全局 min-max 可能比 RevIN 差 | 正是要测的；论文诚实报告归一化 trade-off |
| 基线未调超参"欺负"基线 | 统一超参公平基准；注明未单独调优 |
| iTransformer-SST 物理正则与符号正则混淆 | 对比表注明约束时标差异（长程 vs 单步） |
| 13 模型训练时间长 | 核心 7 个先行，领域基线/消融第二批 |
