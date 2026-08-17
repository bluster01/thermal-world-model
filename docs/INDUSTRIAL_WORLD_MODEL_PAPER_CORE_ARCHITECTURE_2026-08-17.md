# 工业世界模型论文核心架构构建稿

> 状态：概念设计稿，供主窗口理解、讨论和后续冻结。
>
> 权限边界：本文档不是实验授权、代码修改计划或论文定稿。主窗口当前只需吸收上下文，不得据此修改代码、TODO、配置、实验矩阵或结论；任何实施必须等待用户另行明确授权。
>
> 当前决策：**Phase 世界模型架构是论文主角；Adhoc 物理动力学是技术核心；A1 与 Koopman 是同一接口下的低阶和快速动力学实现。**

## 1. 论文要回答的问题

核心研究问题拟定为：

> 在关键状态不可测、控制动作内生、传感器不完备且只能获得闭环运行数据的工业过程中，如何构建一个同时支持多步预测、状态仿真、支持域反事实推演和闭环控制嵌入的可信世界模型？

主汽温系统是完整工业案例，但论文对象不是某一个预测网络，也不只是一个锅炉灰盒模型。论文对象是一套工业世界模型的构建、验证和降阶部署方法。

建议的英文定位：

> A Capability- and Evidence-Guided Framework for Building Physics-State World Models from Closed-Loop Industrial Data

建议使用的边界表述：

> disturbance-conditioned, support-aware, control-usable industrial world model

在没有外生干预或可信工具变量之前，不声明任意策略下的完整 `do(valve)`、完全 plant identification 或未测喷水流量真值恢复。

## 2. 主角与核心模块的明确分工

| 层级 | 论文角色 | 具体内容 |
|---|---|---|
| 系统层 | **论文主角** | Phase 世界模型架构：状态观测、边界预测、控制器/执行器、动力学接口、动作隔离残差、rollout 与控制接口 |
| 动力学层 | **技术核心** | Adhoc 物理嵌入非线性状态模型：焓、金属蓄热、燃料滞后、喷水混合、蒸发/干燥及必要闭合项 |
| 表示层 | 主要候选 | controlled/LPV Koopman：高保真母模型和现场数据的快速全局代理 |
| 低阶层 | 基线与安全模型 | A1：局部指数模态、可解释增益与时常、在线监测和失效回退 |
| 应用层 | 工业验证 | 多步预测、条件响应、支持域反事实、场景推演、闭环控制 |

Phase 不能沿用“free head + physics head 的名称即语义”的旧叙事。新 Phase 是系统接口和职责分离框架。Adhoc 也不能独立代表世界模型，因为它目前依赖外部边界、缺少完整状态观测器，且尚未闭合所有工况的控制证据。

## 3. 统一数学架构

### 3.1 历史状态观测

设历史窗口为

\[
H_k=\{y_{k-L:k},u_{k-L:k},s_{k-L:k},c_{k-L:k}\},
\]

其中 \(y\) 为温度及其他测点，\(u\) 为阀位或有效动作，\(s\) 为 SP/控制信号，\(c\) 为负荷、压力、燃烧等测量工况。不可测物理状态由观测器估计：

\[
\hat x_k=\mathcal O_\phi(H_k).
\]

状态可包括：

\[
x=[h_{1:3},T_{m,1:3},r_B,m_{\mathrm{liq},1:2},q_{\mathrm{spray},A:B},x_{\mathrm{ctrl}},x_{\mathrm{latent}}].
\]

没有可靠测点的混合、金属壁温和未测热扰动不继续拆成多个伪白箱模块，而是合并到受稳定性和接口约束的 latent state。

### 3.2 未来边界与控制策略

世界模型 rollout 不得偷偷使用真实未来 Tin、阀位或总喷水量。未来边界由预测器或显式场景给出：

\[
\hat b_{k+1:k+H}=\mathcal B_\psi(H_k,\xi),
\]

其中 \(\xi\) 是声明的负荷、燃烧或压力场景。未来动作来自控制链：

\[
\hat u_{k+1:k+H}=\mathcal C_\omega(\hat x, s_{k+1:k+H}).
\]

SP、控制器输出、阀位和潜在有效喷水必须分层命名，避免把监督层 SP 响应与 plant-level 喷水响应混为一个 estimand。

### 3.3 动力学与残差隔离

统一状态转移写为：

\[
x_{k+1}=f_{\mathrm{phys}}(x_k,\hat b_k,\hat u_k;\eta)
+S r_\theta(x_k,\hat b_k),
\qquad
\hat y_k=g_\omega(x_k).
\]

其中：

- \(f_{\mathrm{phys}}\) 是可替换动力学接口，Adhoc 是优先候选；
- \(r_\theta\) 只承担未测扰动和结构失配，不得读取未来动作及其后代；
- \(S\) 固定残差允许进入的状态/能量位置，禁止同一闭合量无依据地重复注入金属和蒸汽能量方程；
- 所有候选共享 observer、boundary、controller、数据切分、输入权限和输出接口。

## 4. 三种动力学实现的数学关系

### 4.1 Adhoc：非线性物理母模型

Adhoc 更准确的分类是 physics-embedded neural state-space model、grey-box neural ODE 或 universal differential equation，而不是只在 loss 中加入方程残差的经典 PINN。它负责显式表达焓平衡、热惯性、喷水混合和蒸发/干燥等结构。

它首先是一个非线性状态空间：

\[
x_{k+1}=f_\eta(x_k,b_k,u_k),\qquad y_k=g_\eta(x_k).
\]

当前 Adhoc/qnav 只能作为母模型候选，不能提前当作物理真值。进入降阶前必须闭合残差注入位置、未来 \(W\) 权限、湿/干工况、多折 rollout、动作响应和闭环稳定性。

### 4.2 局部状态空间与 A1

在工况点 \((x^\ast,b^\ast,u^\ast)\) 对通过门禁的非线性模型求 Jacobian：

\[
\delta x_{k+1}=A(c)\delta x_k+B(c)\delta u_k+E(c)\delta b_k,
\]

\[
\delta y_k=C(c)\delta x_k+D(c)\delta u_k.
\]

对可控、可观主模态做降阶后，可得到 A1 形式：

\[
s_{j,k+1}=e^{-\Delta t/\tau_j(c)}s_{j,k}+b_j(c)\Delta u_k,
\qquad
\Delta \hat y_k=\sum_j w_j(c)s_{j,k}.
\]

A1 不是与 Adhoc 无关的模型，而是其局部低阶模态近似。它适合作为透明基线、局部控制模型和安全回退。

### 4.3 Controlled/LPV Koopman

Koopman 路线写为：

\[
z_k=\Psi(x_k),
\]

\[
z_{k+1}=K_0(c)z_k+\sum_i u_{i,k}K_i(c)z_k+B(c)u_k+E(c)b_k,
\qquad
y_k=C(c)z_k.
\]

若算子依赖工况，应称 controlled bilinear/LPV Koopman，而不是标准全局线性 Koopman。它应作为非线性母模型和真实现场轨迹的快速全局代理，不承担从秩亏闭环数据中恢复不存在的独立 A/B 通道信息。

### 4.4 三者的层级关系

\[
\text{Adhoc nonlinear physics}
\xrightarrow{\text{local linearization}}
\text{LPV state space}
\xrightarrow{\text{dominant modes}}
\text{A1},
\]

\[
\text{Adhoc trajectories + real data}
\xrightarrow{\text{lifting/distillation}}
\text{controlled Koopman}.
\]

因此不采用“三个动力学头直接互换、同时端到端竞争”的初始方案。先独立验证物理母模型，再生成并验证低阶与快速代理；三者最终共用相同系统接口。

## 5. 推荐训练与集成策略

不直接把成熟 Phase 预测器中的动力学模块整块替换后全量联合训练。推荐接口保持不变的逐级集成：

1. **物理闭合阶段**：单独验证 Adhoc 的能量结构、残差位置、动作权限和湿/干工况；
2. **状态观测阶段**：训练 observer 从历史恢复物理/latent 初态，检查两窗口状态延续；
3. **系统接入阶段**：冻结已验证的物理骨架，接入 Phase boundary 与 controller，只训练适配层；
4. **有限联合校准**：小学习率解冻工况调度参数与 observer，不长期冻结，也不一次全量重训；
5. **降阶阶段**：对冻结母模型做多工况 Jacobian、可控/可观分析、LPV/A1 降阶和 Koopman 蒸馏；
6. **端到端复核**：在完全相同输入权限下比较高保真母模型、Koopman 在线模型和 A1 回退模型。

局部状态空间不能只在模型完成后作为控制附件。训练过程中应同步监测谱半径、动作增益符号、主模态时常、可控/可观性和 residual 对 Jacobian 的贡献，用于发现稳定但错误的参数补偿解。

## 6. 论文能力与证据阶梯

| 层级 | 要回答的问题 | 最低验证 |
|---|---|---|
| L1 预测 | 能否预测现场轨迹？ | H18/H60、长程 rollout、blocked folds、跨工况误差 |
| L2 状态闭合 | 中间点位和状态是否自洽？ | Tin、Tin−Tout、末温联合误差，两窗口 state closure |
| L3 动作响应 | 动作是否沿正确链路传播？ | correct/wrong/lead/placebo、constant-action identity、符号/时延 |
| L4 支持域反事实 | 小动作变化能否可信推演？ | common/differential 模态、局部增益、跨日期/负荷 invariance |
| L5 控制嵌入 | 是否可用于闭环？ | 跟踪、超调、约束、湿/干稳定性、鲁棒性与推理时延 |

预测 MAE 不能单独选出最终模型。候选至少同时报告自然轨迹预测、动作响应、物理闭合、闭环控制和工程代价。

## 7. 论文候选矩阵

| ID | 候选 | 论文作用 |
|---|---|---|
| D0 | 高容量纯数据 Phase backbone | 预测上限和非物理基线 |
| D1 | Phase + scheduled MIMO A1 | 低阶、透明、稳定基线 |
| D2 | Phase + controlled/LPV Koopman | 快速非线性代理 |
| D3 | Phase + 修正后的 Adhoc 母模型 | 高保真物理候选 |
| D4 | Phase backbone + Adhoc + 动作隔离残差 | 预期完整高保真世界模型 |
| D5 | Adhoc teacher + Koopman student | 在线 rollout/MPC 候选 |
| D6 | Koopman 主模型 + A1 安全回退 | 最终工程部署候选 |

D1--D3 必须先在统一接口下独立比较。只有明确每个模块增加的能力后，才允许进入 D4--D6，避免把多个未闭合模块同时耦合后无法归因。

## 8. 文章的三条中心主张

### Claim 1：系统架构

分离未来边界、控制策略、物理动作通道和未测扰动，可以减少未来信息泄漏、闭环动作污染和 free/physics 分解不唯一。

### Claim 2：动力学机制

物理嵌入状态模型提供中间状态、动作方向和闭环响应的结构锚点；动作隔离残差只修正未测扰动与结构失配。Adhoc 是否优于 A1/Koopman必须由统一协议决定，而不是由模型名称决定。

### Claim 3：多保真工程部署

通过对通过门禁的非线性物理母模型做局部线性化、模态降阶和 Koopman 蒸馏，可形成高保真离线仿真器、快速在线世界模型和低阶安全回退模型。

## 9. 当前证据边界

以下仅是研究线索，不得直接升级为论文定论：

- 原始三段焓灰盒曾出现 rollout RMSE 12.72°C，说明解析物理结构存在系统失配；
- 蒸发状态显著修复湿态中间点，但纯物理版本 rollout 和响应时常仍不满足要求；
- qnav 的 2.463°C 条件 rollout 与湿态响应改善是重要信号，但残差重复能量注入、读取 \(W\) 和干态闭环不稳尚未闭合；
- 既有 14 维局部 LTI 对母模型的高 \(R^2\) 只证明线性化忠实，不证明母模型等于真实对象；
- 旧 883s 信号已被诊断为慢指令动态而非执行机构硬件，不能进入最终控制链叙事；
- Phase Gate C 提供了接口、稳定性和中间监督证据，但尚未证明预测优于历史 M7/M9DSP。

## 10. 论文结构建议

1. Introduction：工业世界模型与闭环数据的困难；
2. Problem formulation：estimand、输入权限、支持域和世界模型能力定义；
3. Proposed framework：Phase 系统架构和职责隔离；
4. Physics-state dynamics：Adhoc 物理母模型及残差边界；
5. Reduced representations：局部状态空间、A1 与 controlled Koopman；
6. Training and evidence protocol：分阶段校准与五级证据门禁；
7. Experiments：统一候选矩阵、预测/响应/控制结果；
8. Discussion：可辨识边界、模型失效和工业部署；
9. Conclusion：方法论贡献与适用范围。

建议篇幅重心：问题定义 15%，系统架构 20%，Adhoc 推导 25%，降阶表示 10%，实验 25%，部署与局限 5%。这保证 Phase 是叙事主角，同时让 Adhoc 保持数学和实验上的技术核心地位。

## 11. 主窗口接收规则

主窗口本轮仅需理解并保留以下共识：

1. 论文不是“Phase预测器文章”或“Adhoc灰盒文章”，而是工业世界模型构建方法论文章；
2. Phase 是系统层主角，Adhoc 是动力学技术核心；
3. A1 是局部低阶/安全模型，Koopman 是快速全局代理；
4. 不直接互换动力学头并全量端到端训练；先验证物理母模型，再做 observer、局部线性化、降阶和系统接入；
5. 当前不修改任何主线文件或实验安排，等待用户在主窗口另行授权。

