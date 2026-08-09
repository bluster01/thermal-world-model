# 主汽温世界模型证据阶梯与缺口

> 更新：2026-08-09
> 状态：Phase 3.5 当前批次收口后的 Supervisor 判决
> 目标：明确“预测、仿真、反事实、闭环”分别需要什么证据，防止用一种能力替代另一种能力。

## Material Passport

- Material Type: project evidence-gap and experiment-gate specification
- Scope: repository code, Phase 1–3.5 results, Linux return artifacts and audit documents available at commit `548035a`
- Verification Status: ANALYZED；Phase 3.5 专项测试可复现，真实训练 checkpoint 未在本地复跑
- Data Boundary: A/B 现场 historian；喷水流量不作真值；实际阀位仅是 plant action 的代理
- Claim Boundary: 本文定义后续证据要求，不把观测闭环关联升级为随机干预因果效应

## 1. 总判决

项目目前证明的是：在历史闭环数据的 development validation 上，可以训练主汽温多步预测器，并能在架构中加入零动作、方向、惯性和单调开度等约束。

项目目前尚未证明的是：模型能够作为状态闭合的仿真器稳定递推；改变动作输入得到的是可识别的反事实；模型在由控制器生成的新动作分布下仍可信；模型嵌入闭环后能安全改善控制性能。

因此，当前原型应称为：

> predictive, physics-guided gray-box prototype under observational closed-loop validation

不能称为：

- 完全物理模型；
- 已验证的 plant simulator / digital twin；
- 已识别的 counterfactual world model；
- 可直接部署的闭环控制模型。

## 2. 世界模型的五个能力合同

这五项是递进关系，后一级不能靠前一级的指标代替。

| 合同 | 要回答的问题 | 最低证据 | 当前状态 | 核心缺口 |
|---|---|---|---|---|
| C0 状态与动作语义 | 模型中的状态、动作和扰动是否对应真实控制链？ | tag、单位、测点层级、时间对齐、staleness、split 与 provenance 可审计 | **PARTIAL** | 阀位→喷水有效作用未标定；喷水流量不可作真值；隐藏热状态与外生扰动不完整 |
| C1 预测 | 在未参与开发的未来时段，能否预测真实温度？ | 独立时间块、多时域误差、校准区间、强基线、A/B 分侧与工况分层 | **DEVELOPMENT ONLY** | 42 runs 只有 validation；没有新的独立 test；尚无稳定胜出配置 |
| C2 仿真 | 给定初态、动作轨迹和扰动，模型能否递推生成物理一致的未来状态？ | recursive rollout、状态闭合、误差增长、稳定性、守恒残差、自由运行与 teacher-forcing 分离 | **NOT ESTABLISHED** | 当前主要是定长温度预测和响应支路，不生成完整下一状态；没有长时自由滚动或独立 plant 对照 |
| C3 反事实 | 若动作轨迹换成未发生的轨迹，预测差异是否可解释为动作效应？ | action consistency、common support、无混杂设计或安全激励、经验响应 reference、placebo/negative control、CI | **BLOCKED** | E3 无双向 common support、balance 未过；E4 被阻断；架构符号约束不等于因果识别 |
| C4 闭环 | 由策略产生的新动作进入模型和现场后，性能与安全是否仍成立？ | 独立 plant/HIL、策略分布覆盖、闭环稳定性、约束违例、延迟预算、shadow→advisory→受限试验 | **NOT ESTABLISHED** | 旧 MPC 为同构 plant 且协议不公平；无可信 OPE、HIL、shadow 或现场受限闭环证据 |

只有 C0–C4 按顺序通过，才有资格把系统称为“可用于仿真、反事实推演并可嵌入闭环的世界模型”。

## 3. 当前证据到底支持到哪里

| 现有证据 | 能支持 | 不能支持 |
|---|---|---|
| Phase 1/旧 M7 的多步预测结果 | 历史协议下存在较强预测 baseline | 独立 test、动作因果性、自由滚动仿真、闭环效用 |
| Phase 3.5 的 42 个 development runs | 新协议可以训练；E1 正对照工作；各动作表示的预测差异很小 | 模型胜出；非线性阀门映射已识别；可以打开 test |
| `g_phys(constant valve)=0`、开阀长期增益非正 | 代码层结构不变量和方向先验 | 现场真实增益、因果效应、完整物理方程 |
| E3 matching | 暴露了 close-event support 和 matching balance 的不足 | “阀门没有物理响应”或“A1phys 物理响应失败” |
| SP 多时域探索 | A/B 阀位都存在明显早期响应，600 s 净效应会衰减/反转 | SP 是 plant action；600 s 单点代表完整执行链；B 侧阀门不响应 |
| G3 参数诊断 | 当前物理支路参数有塌缩，必须阻断强主张 | 两级惯性或所有 physics-guided 路线都不可行 |
| 旧 MPC 结果 | 可以作为协议失败和 objective mismatch 的方法学案例 | 世界模型控制优于 PID，或已经通过闭环验证 |

Phase 3.5 论文当前最稳的贡献不是“已经得到完整物理响应”，而是：建立动作层级、结构约束、观测事件门禁和 fail-closed 审计，并展示预测精度不能替代物理响应可识别性。

## 4. 还缺的关键证据

### P0：动作到真实 plant 响应的可识别 reference

这是当前最先卡住整条链的缺口。

- 冻结 estimand：研究“保持型阀门阶跃”还是“真实闭环阀门轨迹”，不能混用；
- 稳态主分析必须同时满足处理前负荷、主汽压力、主汽温和阀位稳定；动态工况作为单独分层，不与稳态事件合并判决；
- 动作在 60 s 内成形，并在主要响应窗内满足预注册的 hold/trajectory 条件；
- 开阀与关阀分别具有 common support，匹配后各不少于预注册样本数和独立时间块；
- balance、pre-trend、placebo onset、negative control 和 sensitivity analysis 均通过；
- 若现场允许，最强证据是小幅、安全、预注册的随机或准随机激励；否则结论必须停留在 observational response consistency。

在这项证据建立前，E4 以及所有“模型反事实与真实物理响应一致”的主张都必须保持 BLOCKED。

### P0：从预测器升级为状态闭合的仿真器

当前直接预测未来温度不等于世界仿真。至少需要：

- 明确最小状态：A/B 温度链、负荷/燃烧、压力、给水、阀门/执行机构以及必要的隐藏热状态；
- 区分可控动作、外生扰动和待预测状态，禁止把未来不可知量作为输入泄漏；
- 定义一步转移 `s(t+1)=F(s(t),a(t),d(t))`，再进行递推，而不是只输出固定 horizon 的温度向量；
- 报告 10 min、30 min、60 min 自由滚动的误差增长、漂移、越界、NaN 和物理残差；
- 同时比较 teacher-forced、open-loop rollout 和 closed-loop rollout，避免把真实未来观测不断喂回模型伪装成稳定仿真；
- 对不同动作轨迹验证 trajectory consistency 和 composition consistency，而不只看单次 constant-valve IRF。

### P1：plant-level 物理闭合与代理动作校准

A1phys 当前是低阶响应先验，不是守恒模型。若最终目标包含可信仿真，至少还需：

- 阀位→阀门有效面积/喷水有效作用的单调非线性标定，并量化滞环、死区、饱和与速率限制；
- controller/actuator 与 plant 分层：SP→控制器→阀门、阀门→喷水作用→焓/温度不能混成一个 gain；
- 质量/能量守恒残差、焓值关系和金属蓄热的可审计实现；
- 隐状态可观测性、参数可辨识性和单位量纲检查；
- Fan20 只能作为未来 central candidate，Fan17 金属蓄热与 Fan21 mismatch 必须有防双计的能量分配规则；
- 数据缺测无法识别的参数应固定、给先验或报告不可辨识，不能靠神经网络输出“看似合理”的参数。

### P1：独立反事实验证

改变模型输入并得到另一条曲线只是 sensitivity，不自动成为 counterfactual。需要至少一种外部锚点：

1. 安全随机/准随机微扰；
2. 明确 policy 和充分状态下的 sequential causal identification；
3. 可辩护的自然实验/工具变量；
4. 经独立校准的物理 simulator 或 HIL；
5. 未来新时间块上的预注册干预事件。

同时必须报告 overlap 范围。超出历史 action support 的推演只能标记 `EXTRAPOLATION`，不能给出与支持区内相同等级的置信结论。

### P1：真正独立的闭环评价

- plant 不得与 controller 内部世界模型同构；
- 冻结模型、估计器、代价函数、约束和 planner 后再接触闭环 test；
- 基线至少包含现场 PID/串级控制、线性灰箱 MPC 和纯预测顾问；
- 主指标包含温度 RMSE/IAE、最大偏差、约束违例、动作 TV/磨损、求解失败率和实时延迟；
- 对 observation delay、执行器 delay、丢包、传感器偏差和 OOD 工况做压力测试；
- 上线顺序固定为 replay → independent simulator/HIL → shadow → advisory → 小范围受限闭环；任何一级失败均不得跳级。

### P2：不确定性、鲁棒性和外部有效性

- 预测区间既要校准，也要对 rollout horizon 扩张；
- 区分 aleatoric、epistemic 与 action-support uncertainty；
- 按负荷、季节、启停/变负荷、A/B 侧、传感器状态分层；
- 建立 OOD/不可决检测和安全回退，不让模型在无支持区给出确定建议；
- 至少使用未来未开发时间块，理想情况下增加另一机组或另一运行周期外部验证；
- 报告数据漂移监测、再训练触发条件和版本回滚规则。

## 5. 最小实验路径

| Gate | 目标 | 关键实验 | 放行标准 | 失败时结论 |
|---|---|---|---|---|
| W0 语义与数据 | 关闭 tag、时间和代理动作歧义 | 数据字典、source/cache hash、阀门/SP/指令层级、staleness、未来时间块冻结 | C0 全部可审计 | 只做探索性预测 |
| W1 预测 | 建立独立预测基线 | A/B、多 horizon、未来时间块、校准区间、强基线 | 预注册非劣/优越标准与 CI 通过 | 不进入 simulator 比较 |
| W2 经验响应 | 建立 action→temperature reference | 稳态 held-step 主分析 + 动态 trajectory 次分析 + common support/placebo | E3 PASS；两方向与独立时间块充足 | 反事实继续 BLOCKED |
| W3 仿真 | 建立状态闭合递推 | one-step transition、30/60 min free rollout、物理残差与稳定性 | 无发散；误差/残差在冻结阈值内 | 仍称 predictor |
| W4 反事实 | 证明模型动作差异可对齐外部锚点 | 安全激励/自然实验/HIL；logged-vs-intervened trajectory | 方向、增益、时标、剂量与 CI 通过 | 只称 action sensitivity |
| W5 策略离线验证 | 验证 policy-generated action 分布 | support-aware OPE、独立 plant、鲁棒性和故障注入 | 性能非劣且约束/安全门通过 | 不进 shadow |
| W6 闭环分级 | 证明可嵌入且安全 | replay→HIL→shadow→advisory→受限闭环 | 每级预注册验收并有回退 | 停留在上一等级 |

W2 是当前最近的科学门；W3 是从“预测器”成为“仿真器”的结构门；W4 是“反事实”的识别门；W5–W6 才是闭环门。四者不能合并成一个 MAE 或 CFI 分数。

## 6. 论文与工程的两条叙事

### Phase 3/3.5 论文可以收口的叙事

- 现场闭环预测准确性与动作响应可识别性是两个不同目标；
- 绝对阀位及其工作点值得显式建模，但当前没有证据证明复杂非线性映射优于 identity；
- 结构物理约束可以保证代码不变量，却不能替代经验响应 reference；
- fail-closed 门禁能够阻止 prediction-only 模型被错误升级为 counterfactual model。

### 最终世界模型未来必须回答的叙事

- 模型是否生成闭合、稳定、量纲一致的状态轨迹；
- 模型是否在动作支持区内复现真实干预响应，并对支持区外明确拒答；
- planner 是否在独立 plant 和真实延迟/约束下仍改善控制；
- 系统是否具备不确定性、异常检测、安全回退和版本治理。

前一条可以形成当前文章；后一条必须由 W0–W6 的新证据逐级建立，不能从当前 Phase 3.5 数字外推。

## 7. 统计谬误扫描

覆盖 11/11：

| 类型 | 当前风险 |
|---|---|
| Simpson's paradox | 负荷/稳动态工况聚合可能掩盖反向子组 |
| Ecological fallacy | 日块或工况均值不能外推到单次动作 |
| Berkson's paradox | 事件阈值与稳定筛选会产生选择偏差 |
| Collider bias | 以处理后稳定性/阀门轨迹筛选可能条件化 collider，必须在 estimand 中预声明 |
| Base-rate neglect | E5 no-execution 和 close 事件基率极低 |
| Regression to mean | SP/温差极端时触发的动作天然伴随回归均值 |
| Survivorship bias | 只报告 matched/held 事件会隐藏漏斗损失 |
| Look-elsewhere effect | 多 horizon、阈值、caliper 与工况切分会放大偶然最好结果 |
| Garden of forking paths | 事件定义、hold 时间和模型选择必须在新数据前冻结 |
| Correlation != causation | historian 闭环关联不能写成 `do(action)` 效应 |
| Reverse causality | 温度偏差会驱动阀门动作，是主要反馈混杂路径 |

## 8. 当前优先级

1. 不再扩大 A1phys 训练矩阵；先冻结 W2 的稳态 held-step / 动态 trajectory 双 estimand。
2. 为未来新时间块保留真正盲的事件与模型 lockbox。
3. W2 仍不可识别时，Phase 3.5 以方法学和阴性结果收口，不用预测 MAE 补位。
4. 文章完成后再设计 W3 的状态闭合 simulator；这不是现有预测头的小改版。
5. 只有 W3、W4 通过后，才恢复任何 MPC/闭环路线。
