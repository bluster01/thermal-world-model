# Phase 3.5 主线实验上下文与收口目标

## Material Passport

- Material Type: paper mainline and experiment-purpose specification
- Scope: Phase 3/3.5 field evidence, Phase 3.5-MS synthetic evidence, and the current MS2-J decision
- Verification Status: VERIFIED；基于截至 MS2-J test 完成提交 `5260d3f` 的实验代码、结果与审计文档（test 27/27 单次访问完成，ledger completed）
- Evidence Boundary: 阀位是可审计 plant-action proxy，不是喷水质量流量；synthetic known-truth 不是现场因果证据
- Active Decision: MS2-J 一次性 synthetic test 已完成；联合模块跨 split 复现 PASS、staged 非劣 FAIL 复现。Phase 4、MS2-D、MS3–MS5 均不自动启动，进入论文表图与 claim ledger

## 1. 一句话主线

当前论文不是要证明“已经获得完全物理世界模型”，而是要回答：

> 在现场闭环数据暂时不能识别真实阀门因果响应的条件下，能否构造一个满足动作语义、参考恒等、时间因果、方向和稳定性约束的多步响应模型，并在已知真值系统中证明它确实可解；同时用 fail-closed 门禁阻止 synthetic 可解性被误写成现场反事实能力？

因此，项目当前不以最低预测 MAE、单一 CFI 或路线冠军为目标。主要目标是把“模型是否有能力表示物理响应”和“现场数据是否足以确认该响应”拆成两个独立问题。

## 2. 为什么必须分成两条证据臂

统一模型是

\[
\widehat T=f_{free}(\text{history, context})+
g_{response}(\text{context},a_{1:H},r_{1:H}),
\qquad g_{response}(c,r,r)=0.
\]

`f_free` 负责无干预条件下的温度演化；`g_response` 负责实际阀位轨迹相对参考阀位轨迹的增量响应。未来动作不能泄漏进 `f_free`。现场喷水流量传感器不可靠，所以动作使用实际二级减温阀反馈开度；阀位到有效喷水作用只允许解释为未知的单调非线性代理。

| 证据臂 | 回答的问题 | 实验 | 当前结论 |
|---|---|---|---|
| A. 现场可识别性 | 真实 historian 是否足以建立 action→temperature reference？ | E1–E5 | E1 正对照通过；E2/E3/E5 不确定；E4 被 E3 阻断 |
| B. 架构可解性 | 如果真值已知，结构化多步响应模型能否恢复动作响应？ | MS0–MS2-J | MS1、MS2-V/C、MS2-J 双层通过（test 复现 validation：联合模块 CI 下界 0.73–0.89 >> 20%）；staged 非劣双层 FAIL（test ratio 1.14–1.20，CI 上界 >1.10） |

两臂不能互相替代：现场 `INCONCLUSIVE` 不等于架构无效；synthetic `PASS` 也不等于现场 `do(valve)` 已识别。

## 3. 现场 E1–E5 在论文中的作用

| 实验 | 真正目的 | 不是为了证明 |
|---|---|---|
| E1 动作表征 | 说明仅用 `Δ阀位` 会丢失绝对工作点，验证动作输入语义 | 绝对阀位已等于喷水流量 |
| E2 阀门非线性 | 检查现场数据是否支持 identity 之外的复杂映射 | R50 或 learned monotone 已恢复真实阀门曲线 |
| E3 经验响应 | 尝试建立稳态 held-step / 动态 trajectory 的外部响应锚点 | 相关事件天然就是因果干预 |
| E4 模型响应 | 只有 E3 通过后，才比较模型 IRF 与经验 IRF | 结构方向约束本身等于物理验证 |
| E5 SP 未执行 | 区分监督层 SP 与实际 plant action，解释 SP 变而阀位不变 | 质疑现场串级 PID 或 A/B 交叉控制事实 |

这条证据臂的主要贡献是揭示识别边界：历史闭环反馈、开关阀 support 不平衡、balance/pre-trend 不足和低基率 no-execution 事件，使当前数据不能承担强因果结论。

## 4. MS0–MS2-J 在论文中的作用

| 实验 | 主要问题 | 当前证据 |
|---|---|---|
| MS0 合同 | 不同路线是否比较同一个 action/reference estimand？ | exact reference identity、future-action leakage=0 |
| MS1 最小可解性 | 二阶惯性多步系统能否递推恢复？ | 同型 known-truth 上通过；属于正对照/inverse-crime 边界 |
| MS2-V 阀门非线性 | 绝对开度相关的单调模块是否有必要？ | 独立 test 中 learned monotone 明显优于 identity；`K/phi` 仍不可拆分辨识 |
| MS2-C 工况调度 | 增益和时间常数随 context 变化时能否恢复？ | 独立 test 中 scheduled 明显优于 global |
| MS2-J 联合耦合 | 非线性开度与工况调度同时存在时能否共同工作？ | 联合模块双层 PASS（validation + test，test CI 下界 0.73–0.89 >> 20%）；staged 双层未达 joint 的 1.10 非劣界（test ratio 1.14–1.20） |

Koopman、PI-ODE 和 DeepONet 在这里是表示能力对照，不承担“谁是最终世界模型”的赛马任务。MS2-J 的主要比较是同一灰箱中的联合模块对两个单模块消融；staged-vs-joint 只决定当前 response 内部训练策略，不回答完整 `free+response` 耦合是否需要分阶段训练。

## 5. MS2-J test：已完成，按冻结解释落地

MS2-J test 于 `5260d3f` 完成（27/27 单次访问，ledger completed，paired stratified bootstrap 10k reps）。三个预注册问题的结果：

1. **joint vs `monotone_global` / `identity_scheduled`**：3 seeds × 2 对比 = 6/6，改善 CI 下界 0.73–0.89 >> 20% 门槛 → **通过**；
2. **staged/joint 误差比**：observed 1.14–1.20，95% CI 上界 1.09–1.32 > 1.10 → **非劣失败**；
3. **staged vs Stage A**：改善 0.73–0.74，CI 下界 0.68–0.77 ≥ 20% → **通过**（训练链有效，只是不如 joint）。

结果解释按冻结表落地：

| test 结果 | Supervisor 解释 | 后续动作 |
|---|---|---|
| joint 通过；staged 非劣仍失败 | 联合模块收口；当前主训练采用 joint；staged 作为阴性消融 | **停止 synthetic 扩矩阵，进入论文表图与 claim ledger（当前状态）** |

无论哪种结果，都不启动 Phase 4，也不自动进入 MS2-D、MS3、MS4 或 MS5。

## 6. 论文最终落点

当前文章的可守贡献是：

1. 明确工业闭环温度预测与动作响应识别是两个不同任务；
2. 给出 SP—控制器—实际阀位—有效喷水代理—温度的分层动作语义；
3. 提出满足 reference identity、时间因果、方向、稳定性与递推状态合同的多步响应架构；
4. 用 known-truth 实验证明该架构在阀门非线性与工况调度下的响应级可解性；
5. 用现场 E1–E5 和 fail-closed 门禁说明：结构物理一致性不能替代经验响应 reference，证据不足时必须保持 `INCONCLUSIVE/BLOCKED`。

文章不能声称：真实阀门曲线或喷水流量已恢复、A1phys 参数是唯一物理参数、现场反事实已识别、完整状态闭合 simulator 已成立、Koopman/PINN/DeepONet 已定胜负，或模型已可直接嵌入闭环。

## 7. MS2-J 后的收口清单

1. ~~拉取一次性 test artifacts，复核 ledger、归档、trajectory pairing、结构门禁和 bootstrap~~ → 完成（`5260d3f`）；
2. ~~写 MS2-J test review，保持 validation/test 与 synthetic/field 口径分离~~ → 完成（`docs/PHASE35_MS2J_TEST_REVIEW_2026-08-10.md`）；
3. 冻结主文三张核心表：现场 E1–E5、synthetic MS1–MS2-J、claim/evidence boundary；
4. 冻结两张核心图：控制层级与 action proxy、`free + response` 架构及证据流；
5. 开始论文提纲和主文，不再用新模型实验延迟收口。
