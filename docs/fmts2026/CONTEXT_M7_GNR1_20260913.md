# FMTS 当前实验上下文：M7R1 + GNR1

2026-09-13，作者明确要求优化 token 融合模型、使用 M7 encoder，并恢复/检验物理灰箱响应。
活动任务限于两个独立注册的新 arm × 3 seeds。执行入口：
`experiments/fmts_m7fusion_20260913/RUN_LINUX.md`。

## 为什么改变

原 v0.2：GRU 0.443139、token 0.654311、持续值 0.638755°C（主汽温累计 H18 MAE）。
原 token 的两个最佳 seed 在合成历史上四个有效初态修正饱和，不能直接解释成 attention 无效。
历史 M7 是 RevIN/Patch/逐变量 TCN/VarAttn/展平读出，而非当前 state-query observer。
新 M7R1 保留这套主干并加绝对历史均值/尺度，接独立向量头、原物理锚点及修正边界。
是组合优化，不是分离 encoder/head 因果贡献的消融；不继承旧 checkpoint 或旧表数字。

原纯灰箱 rewet 开启而融合模型关闭，且原灰箱的 H18 双阀均值为正，不能支撑
“物理响应正确”。GNR1 只关闭纯灰箱 rewet 并按原预算重新拟合，不以符号选模型。
预期降温方向需由结果检验，现场增益真值/校准缺口仍在。

## 不变项

源数据/mapping/IAPWS 指纹、23 个历史变量（含 9 个新增边界）、原 indices 字节、
W96/H18/10秒、五温度 NLL、24k 更新上限、Adam0.001/batch32、主汽温 MAE 选模。
不改物理动作路径，不重训旧四臂，不搜参数、不新增种子、不开 locked test。
完整 v0.2 结果与中英文初稿保留；没有新结果前不改图中数值/论文 verdict。

## 工作分工与状态

本地：设计/注册/代码/合成测试/回传审计；Linux：冻结命令与完整产物回传。
`ready_for_linux` 不等于已启动；`results_returned` 和 `audited` 必须分别依据回执填写。
包内机器状态以 M7R1/GNR1 各自 `experiment_state.json` 为准。

M7 回传需 3 次训练、6 预测/响应回放、9 observer 健康回放（6旧+3新）。
GNR1 回传需 3 次训练、6 预测/响应回放。异常、弱结果与失败都保留，不自动继续迭代。
人工审查所有 seeds、健康诊断、精度和双阀曲线后，才能决定是否更新最终论文图表。
