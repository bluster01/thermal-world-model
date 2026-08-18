# Fan2020-UDE 物理模型证据链

本文只重组已有证据，不把历史 `PASS` 自动升级为最终世界模型结论。`raw_summaries/` 保存原分支的小型摘要；涉及模型选择或论文数字时仍须回到源分支原始 checkpoint、轨迹和执行日志。

## 1. 已有证据

| 证据节点 | 已回答的问题 | 当前证据 | 可以支持 | 不能支持 |
|---|---|---|---|---|
| E0 Fan2020 骨架 | 是否存在可计算的集总物理先验？ | 质量/能量守恒导出的低阶 ODE/DAE，并可由运行数据辨识参数 | 物理状态转移的先验骨架 | 伊敏对象已被该方程完整识别 |
| E1 物性与数据 sanity | 方程和物性计算能否数值实现？ | IAPWS 可微代理、阀位/工况检查和第一轮 rollout | 可微物理计算图可构造 | 长期 rollout 或控制可用 |
| E2 结构失配暴露 | 原始骨架是否可以直接迁移？ | [step1_summary.json](raw_summaries/step1_summary.json)、[strat_ablation_summary.json](raw_summaries/strat_ablation_summary.json) 显示明显漂移和湿/干差异 | 单纯 Fan20 参数适配不足 | 物理路线无效 |
| E3 蒸发/干燥修复 | 两相中间点缺口能否用物理状态修复？ | [fixb_evap_summary.json](raw_summaries/fixb_evap_summary.json) 中湿态局部偏差显著缩小；[fixb_qnav_summary.json](raw_summaries/fixb_qnav_summary.json) 给出后续混合模型表现 | 显式相态状态是有用候选 | 已恢复未测液滴真值或全工况闭合 |
| E4 残差权限 | 高精度残差会不会改写动作响应？ | [qnav_first_principles](raw_summaries/qnav_first_principles__summary_development.json) 与 [residual feedback probe](raw_summaries/qnav_residual_feedback_probe__summary_development.json) 表明读动作后代/状态反馈可压缩或翻转响应 | 残差必须动作隔离并做响应审计 | 已找到唯一正确的残差结构 |
| E5 动态局部化 | 非线性模型能否局部降阶？ | [step6b_lin_summary.json](raw_summaries/step6b_lin_summary.json) 中六个工况的局部线性拟合较高且局部稳定 | 可构建 A1/LPV/Koopman 学生候选 | 母模型等于真实对象或闭环稳定已证明 |
| E6 纯数据端点 | 黑箱预测是否自然携带动作响应？ | [Direct-WM v2](raw_summaries/direct_wm_v2__results.json) 的 direct H18 为 F0 0.712/F1 0.927°C，但双阀响应弱且跨折/通道方向不稳；详见 [Supervisor audit](../../../docs/ADHOC_DIRECT_WM_V2_SUPERVISOR_AUDIT_2026-08-18.md) | 高容量预测器可作 observer/backbone 候选 | 已验证 boundary/σ、物理响应或反事实 simulator |
| E7 对象/控制/初始化归因 | 干湿闭环异常来自哪里？ | [shared disturbance](raw_summaries/qnav_shared_disturbance_loop__summary_development.json)、[boundary attribution](raw_summaries/qnav_boundary_attribution_probe__summary_development.json)、[actuator identity](raw_summaries/actuator_identity_conclusion.json) 分离了若干内部机制 | observer、boundary、controller 必须成为独立接口 | 已由现场真值确认全部归因 |

## 2. 对最终 pipeline 的覆盖

| 最终模块 | 可复用资产 | 当前成熟度 | 主要缺口 |
|---|---|---|---|
| 概率 Observer | Direct WM 历史编码器、Phase 高容量 backbone | 部分 | 未证明恢复可延续的物理/latent 初态；校准仅在输出层 |
| Boundary model | Phase 预测特征、边界归因探针 | 弱 | 未来 Tin、负荷、压力的概率联合预测和无泄漏协议未闭合 |
| Controller/actuator | 串级 PID 事实、执行机构身份审计 | 部分 | 控制器 tag 映射、饱和/死区和跨工况参数仍需统一实现 |
| Fan2020-UDE transition | Ad hoc 焓模型、蒸发/干燥、阀位代理 | 部分 | 状态/参数可辨识性、干态、多折长 rollout 和能量残差位置未闭合 |
| Action-blind closure | qnav/h_now/shared/replay 系列 | 候选 | 尚无同时兼顾自然预测与真实响应的稳定冠军 |
| Observation model | 多测点输出与中间温度监督 | 部分 | 测点噪声、缺测和状态到测点的概率观测模型未统一 |
| Fast surrogate | 六工况局部线性化、Phase Koopman 实验 | 早期 | 尚未从通过门禁的统一母模型蒸馏，也没有速度—保真—闭环等价表 |

## 3. 当前可以形成的证据主线

1. **纯物理先验有用但不充分**：它提供能量状态和动作路径，同时在真实迁移中暴露结构失配。
2. **纯预测精度也不充分**：黑箱可预测自然轨迹，却可能对动作替换几乎不响应。
3. **任意残差耦合会破坏物理语义**：闭环数据中的高精度捷径能够抵消物理热惯性或重写动作响应。
4. **因此需要统一而有权限边界的生成模型**：概率 observer 负责初态，boundary model 负责未来外生条件，Fan2020-UDE 负责动作条件状态转移，action-blind closure 只补未测扰动。
5. **快速模型必须来自验证后的母模型**：A1/LPV/Koopman 是降阶或蒸馏产物，不从秩亏数据中凭空恢复独立物理通道。

## 4. 尚未闭合的决定性问题

- 同一 inferred state 在相邻窗口是否连续，能否支持 30–60 min 自由递推；
- 自然预测改进究竟来自更好的初态/边界，还是 transition 继续吸收闭环策略；
- action-blind closure 在不同容量下是否仍保持响应、能量和状态 Jacobian；
- 湿/干、负荷、日期和 A/B common/differential 支持域内，响应方向、时延和量级是否稳定；
- controller-in-loop 的稳定性是否在独立 controller、扰动和参数不确定性下成立；
- Koopman 学生能否在显著加速时保持母模型的 rollout、局部响应和闭环行为。

这些问题构成新 pipeline 的补实验队列，而不是继续修补旧编号脚本。
