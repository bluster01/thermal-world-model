# Phase 3.5 多步动作响应算子设计

> 公式推导、代码追溯与已核验 reference ledger 见 [`../PHASE35_MS_METHODS_AND_REFERENCES.md`](../PHASE35_MS_METHODS_AND_REFERENCES.md)。

## 研究定位与边界

本工作不是重启 Phase 4，也不改写现有 Phase 3.5 的阴性审计结论。它把当前 A1phys 的核心假设收缩为一个可单独验证的问题：给定相同历史工况与一条未来阀位轨迹，模型能否稳定、因果地给出相对于参考阀位轨迹的多步温度增量响应。统一形式为

```text
T_hat[1:H] = f_free(history, future_exogenous) + g_response(context, action[1:H], reference[1:H])
```

其中 `f_free` 的接口不接收未来动作，`g_response` 必须在结构上满足 `g(a_ref, a_ref)=0`，且第 `k` 步之后的动作不得改变第 `k` 步之前的响应。动作仍是交叉映射后的实际二级减温阀反馈开度代理；阀位到喷水作用允许单调非线性映射，但不解释为质量流量。A/B 分侧训练、分侧评估，不与 SP 监督层任务混榜。

第一阶段只做合成已知真值系统辨识。这不是替代真实物理响应验证，而是建立“方法可解性”正对照：若模型连已知二阶惯性、多阶跃和脉冲都不能恢复，就没有资格进入真实数据实验；若可以恢复，也只能证明表示与优化可行，不能自动获得现场 `do(valve)` 解释。真实数据仍受 E3 common support、稳态门禁和新时间块约束。

## 统一接口与候选路线

所有候选接收 `context [B,C]`、`action [B,H]`、`reference [B,H]` 和可选初始响应状态，返回 `effect [B,H]`、响应状态轨迹及结构诊断。算子使用相同样本、损失、seed、epoch/optimizer-update 预算和 validation selector。共同的阀位映射先产生有效开度差，再由下列算子传播：

| 路线 | 表示 | 结构保证 | 用途 |
|---|---|---|---|
| Graybox-1P/2P | 一阶或串联二阶惯性 | 正时间常数、非正长期增益、稳定离散递推 | 可解性最高的论文主基线 |
| Controlled Koopman K2/K4 | 对角稳定潜状态与显式控制矩阵 | 谱半径小于 1、因果递推 | 检验更高维线性潜动力学是否有益 |
| PI-ODE | 二阶名义 ODE 加小幅动作门控神经闭合 | 正时间常数、零干预恒等式、闭合残差可惩罚 | 检验连续时间与有限失配补偿 |
| Causal DeepONet | 因果序列 branch 与时间 trunk | 参考路径相减保证零干预；branch 只见动作前缀 | 固定 horizon 的直接算子对照 |

标准 DeepONet 若一次读取完整未来动作，会把未来动作泄漏到早期响应，因此这里明确采用因果 branch。它仍是固定 horizon 算子，不宣称可无限递推。A1phys-MS 是自由预测器与任一响应算子的薄组合层；Graybox-2P 同时是当前两级 A1phys 的多步、可传递状态特例。

## 训练、验证与失败处理

合成基准包含保持、单阶跃、双阶跃、斜坡和脉冲轨迹，生成器保存真实增益、时间常数、动作形状与 seed。训练只以 train 优化，以 validation 响应 MAE 选 checkpoint；synthetic test 也需显式开关，正式真实数据 test 规则保持不变。主指标是全时程 MAE/RMSE 与 H1/H6/H18/H60 MAE；结构门禁包括零干预最大误差、未来动作泄漏、非有限 rollout、Graybox 增益/时间常数边界和 Koopman 谱半径。

多阶段训练默认不用“长期冻结再整体解冻”。基线方式是先在合成响应真值上单独辨识 `g_response`，接真实数据时再采用：自由预测器预训练或加载 → 短暂冻结自由头训练响应算子 → 小学习率联合微调。联合阶段保留独立响应损失/结构损失，防止自由头重新吸收动作。每个阶段均保存独立 checkpoint 和 validation 曲线；任一阶段出现参数塌缩、梯度非有限或结构门禁失败即 fail closed，不进入真实数据或路线冠军表。

Linux 只运行版本化矩阵与命令，回传完整 manifest、日志、checkpoint 和 validation 指标。本地负责代码、结构测试、矩阵冻结、结果复算和论文判定。第一版暂不接完整 Phase1 checkpoint、不实现闭环 controller/actuator，也不引入 Fan 方程；这些属于后续适配层，而不是本次可解性基准的必要条件。
