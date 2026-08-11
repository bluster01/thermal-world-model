# Phase 3.5 MS3-R Gate C 本地框架验证记录

日期：2026-08-11
范围：本地代码、契约、结构测试与已知真值控制；未访问真实 validation/test，未执行真实训练。

## 结论

监督标签为：

`LOCAL_FRAMEWORK_VERIFIED / REAL_TRAINING_NOT_RELEASED / OPERATOR_ROUTES_STRUCTURAL_ADAPTERS_ONLY`

当前可以确认：dual-interface measured-boundary latent MIMO 的信息流和软件外壳可执行，且能 fail-closed 地阻止未来 Tin 泄漏、oracle 选模、结构塌缩 checkpoint、超过 10% 的长期 freeze，以及秩亏数据上的双通道点辨识声明。

当前不能确认：A1phys、LPV-Koopman、PI neural ODE、DeepONet 中哪条真实数据路线更优。四个 route ID 目前共享一个稳定响应契约和结构适配核心，只用于验证共同接口；它们不是四个已经实现完毕、可以训练赛马的科学模型。

## 已验证的边界

1. `forecast_boundary` 不接收未来真实 Tin；`oracle_boundary` 只作结构上限；`scenario_boundary` 必须显式传入。
2. 主动作接口是未来 SP 场景到预测阀位；logged future valve 不能进入 residual/free 分支。
3. 局部块使用稳定、恒动作 identity 的双阀 MIMO 响应；末温块允许 full MIMO latent mixing，不强加 Gate B 已否定的末温侧别硬约束。
4. selector 只使用 `forecast_boundary`，先执行 finite/prefix/identity/leakage/non-collapse 五个硬门，再比较归一化多任务分数。
5. robust scale 只能从 train 拟合并冻结；阀位、Tin、局部 Tin−Tout、末温、长时 rollout 和结构项的权重闭合为 1。
6. warm-up 最多占总 optimizer updates 的 10%，随后必须全量 joint unfreeze。

## 已知真值控制

合成控制包含 nonlinear opening、common/differential 双阀激励、局部稳定 MIMO、真实 Tin 边界、下游跨侧 latent mixing 和有色未测扰动。固定 smoke（seed 1701，48 episodes，60 steps）的解析辨识结果为：

| 项目 | 结果 |
|---|---:|
| 输入协方差条件数 | 2.3212 |
| differential/common 能量比 | 0.4299 |
| 真值/恢复 decay | 0.88 / 0.88 |
| 局部 gain 相对误差 | 1.86e-16 |
| 恢复残差 MSE | 4.50e-32 |

该结果只证明“在已知方程、支持充分且局部真值可见时，秩审计与恢复管线自洽”。它不是端到端神经网络恢复，也不能升级真实闭环数据的因果声明。共线输入控制会触发拒绝，声明降级为只允许 `common spray mode`。

## 本地执行边界

`ms3r_gatec_model_screen.py` 当前只提供：

- `--dry-run`：打印冻结 RM0/RM1 矩阵和计算预算；
- `--synthetic-smoke`：生成已知真值诊断产物并固定 config/source/code/Git hash；
- `--real-run`：在注册表未授权时明确失败。

summary 只能聚合本地诊断，不产生 `supervisor_decision` 或自动科学 PASS。

## 软件验证

- Gate C + 状态机专项：31 passed；
- Phase 3.5 全量回归：218 passed；
- JSON/status contract、`git diff --check` 与 Python compileall：通过。

这些是软件验证计数，不是实验样本量或统计证据。

## 下一门

在授权 Linux 前，本地必须完成：

1. 四条路线各自的真实算子方程，而不是共享核心上的名称/尺度差异；
2. 同一 outer shell 下的端到端合成训练恢复，包括 free-capacity × excitation 分层；
3. response collapse、future-Tin leakage、共线输入三个负控制在训练后仍 fail-closed；
4. 真实 validation runner、单次批次矩阵和资源预算的独立冻结与审计。

在这些条件完成前，`linux_authorized_gate` 保持 `null`。
