# Phase 3.5 MS3-R RM3-AV 独立审计验证批次设计

> 状态：IMPLEMENTED / LOCAL VERIFIED；位于 RM3-A 与 RM3-B 之间；Linux 尚未授权；validation-only；禁止访问 test、启动 MS4 或冻结 RM3-B 训练矩阵。

## Material Passport

- Material Type: pre-RM3-B independent-audit validation experiment specification
- Batch ID: `MS3-R / RM3-AV`（RM3 Independent Audit Validation）
- Parent evidence: RM2、RM3、RM3-A 的冻结 validation 产物与独立架构审计
- Data boundary: 现有 A/B historian rolling folds；实际阀位是有效喷水作用代理，喷水流量不作真值
- Statistical scope: `2 rolling folds × seed 0` 的机制宽筛；UTC 日/连续时间块是统计单位，seed 不是样本量
- Claim boundary: 本批验证审计命题与架构归因，不宣布模型冠军，不建立 `do(valve)`、唯一 plant gain 或闭环可用性

## 0. 为什么必须在 RM3-B 前单开本批

独立审计提出的不只是“free head 可能抢信号”或“P5 可能依赖 bypass”，而是四类相互耦合的问题：

1. **实现与审计回归**：RM2 的 `structure_penalty`、persistence、shuffled-action、分侧效应等诊断在 RM3 重构中消失；校准 JSON 仍含旧错误数字；RM3 manifest 把 P5 写成 OOF-calibrated，但 OOF/R-loss 目前没有进入 P5 训练图。
2. **架构归因不闭合**：RM3-A 匹配的是 `d_model`，没有匹配自由头所在位置；P5 独有 terminal bypass，P3/P4 独有大容量 local residual head；P3/P4 初始化又受可选分支 RNG 消耗影响。
3. **闭环反馈与动作链问题**：阀位预测接近 persistence 且过度平滑；当前 decoder 未显式读取 `SP-T`、积分误差、限幅/死区；free 与 response 的信息集近同，logged-action 响应仍受闭环内生性影响。
4. **方法与统计问题**：三极点形状可能不可辨识；早期负增益可能是对齐、闭环反馈或瞬时耦合；双侧池化掩盖侧别互换；跨候选 selector 目标不同；4000 updates 可能把收敛速度混入排名；KCI、条件数和非平稳性论文假设不能自动升级为真实闭环因果识别。

因此本批不预设独立审计正确，也不预设原架构正确。每条意见都改写为可证伪命题，并用零训练重放或成对训练给出 `SUPPORTED / REFUTED / MIXED / NOT_TESTABLE` 四态判决。RM3-B 只能消费本批结果，不能跳过本批直接堆叠新理论模块。

## 1. 审计意见完整覆盖表

| ID | 独立审计或 Supervisor 复核问题 | 待验证命题 | 直接证据 | 所属批次 |
|---|---|---|---|---|
| Q01 | §1 `structure_penalty` 删除 | 响应塌缩主要由训练锚点删除造成 | no-aux / logged-aux / integrated OOF R-loss 成对训练；predicted/logged effect | AV0 + AV1-S |
| Q02 | §2 阀位预测近 persistence | 绝对阀位损失与 decoder 信息不足造成过平滑 | persistence skill、逐步变化、跨度、频谱；Δv/roughness/PID 对照 | AV0 + AV1-V |
| Q03 | §3 RM2 诊断丢失 | 当前 RM3 报告无法发现动作分支失活 | 在冻结 checkpoint 恢复 persistence/effect/shuffle/wrong-side/lead | AV0-Z02 |
| Q04 | §4.2 容量匹配错轴 | terminal/local 优势来自自由头位置而非 `d_model` | P3+bypass、P5-no-bypass、P5-bypass-only、free 容量扫描 | AV1-H |
| Q05 | §4.3 P5 动作链疑似失活 | terminal 优势可在关闭 response 后保持 | 冻结推理 response-off + 训练 response-off；有限差分 action sensitivity | AV0-Z03/Z04 + AV1-H |
| Q06 | §4.4 P5 非 Pareto | 旧 balanced P5 是权重工作点而非架构结论 | 统一 selector 下重报；固定目标下比较 A3/A4 类权重效应 | AV0-Z10 |
| Q07 | §4.5 selector 不可比 | 不同 component weights 使 `scope_selector_score` 不可横比 | 删除跨候选 selector 排名；统一报告目标重算 | AV0-Z10 |
| Q08 | §4.6 更新上限/收敛 | 4000-update 排名混入收敛速度 | P3/P4/P5 4000 对 8000；末 500-update 斜率 | AV0-Z10 + AV1-O |
| Q09 | §5.1 符号被架构强制 | 方向正确率没有数据证据力 | 方向仅作合同；signed-response 诊断能否稳定恢复物理符号 | AV1-T |
| Q10 | §5.2 free/response 同信息集 | free 能用 history/SP 代理未来动作并压低 response | future-action probe、free small/base/large、action-shielded free | AV0-Z06 + AV1-H |
| Q11 | §5.3 初始化混淆 | P3/P4 差异部分来自可选分支改变共享模块初值 | legacy RNG 与 module-scoped identical initialization | AV0-Z11 + AV1-O |
| Q12 | §5.4 三极点不可辨识 | 1/2/3 pole、power、linear 在留出日期上无稳定区分 | blocked-date OOS shape fit；训练形状族成对比较 | AV0-Z07 + AV1-T |
| Q13 | §5.5 校准 JSON 旧错值 | 结果树与当前算法不一致 | 从冻结 NPZ 重生成、记录 source/output SHA 与算法版本 | AV0-Z01 |
| Q14 | §5.6 跨侧耦合/跨折互换 | full-MIMO cross gain 可能吸收共模或活动度差异 | common/diagonal/full 对照；gain 对工况与 valve activity 回归 | AV0-Z08 + AV1-M |
| Q15 | §5.7 早期负增益 | 负值来自采样对齐/闭环反馈/真实瞬时项中的至少一种 | lead/lag/time-shift、h=0、SP-T 条件化、dead-time 对照 | AV0-Z09 + AV1-T/V |
| Q16 | §5.8 双侧池化 | pooled MAE 掩盖侧别失败和 fold 互换 | A/B 分报、common/differential、日期/负荷/动作活跃度分层 | 所有 AV0/AV1 |
| Q17 | §6.2 E1 只有比值门 | “大家都接近零”可能假通过 | 幅值绝对下限 + effect/local 比 + 候选分歧并报 | AV0-Z04 |
| Q18 | §6.2 E2/KCI 过度解释 | 条件独立检验不能单独证明分解唯一 | KCI 仅诊断；与容量扫描、placebo、动作敏感性联合解释 | AV0-Z12 |
| Q19 | §6.2 E3 fold 差异解释不唯一 | 增益互换可能由工况或某侧被主动调节造成 | gain ~ context + per-side activity 的 UTC-day/block 回归 | AV0-Z08 |
| Q20 | §6.2 E4 条件数不敏感 | 总 Gram 条件数不能证明每侧独立激励 | per-side innovation、剂量/变化率、common/diff 模态能量与有效秩 | AV0-Z08 |
| Q21 | §6.2 E6 NNLS 非真值 | 三极点 NNLS 不能当外部物理锚点 | OOS model selection；E6 降为 trajectory diagnostic | AV0-Z07 |
| Q22 | §6.2 zero-training ablation | 冻结模型已足以定位部分旁路/响应问题 | bypass-off/only、response-off、forecast/logged/oracle/placebo 重放 | AV0-Z03 |
| Q23 | Supervisor 复核：OOF 未接训练图 | P5 当前并非 OOF-calibrated explicit response | 静态计算图追踪 + P4/P5 integrated OOF R-loss 对照 | AV0-Z06 + AV1-S |
| Q24 | Supervisor 复核：R2 flag 硬编码 | `common_only=False` 字段不能作为输入秩证据 | 重算创新协方差/有效秩；字段只作配置说明 | AV0-Z08 |
| Q25 | 闭环反馈/论文假设外推 | 非平稳、KCI、稀疏 Jacobian 不能自行解决内生性 | error/integral first stage、lead/placebo、活动度分层；结论保持 observational | AV0-Z12 + AV1-V |
| Q26 | §2 P2/M9 低于 persistence | future-SP 模型可能没有学到有效的 H60 动作/温度动力学 | P0–P5 全时程 persistence skill 与 horizon curve 重放 | AV0-Z02 |
| Q27 | §6.1 `rollout` 命名混淆 | RM2 后 1/3 teacher-forced loss 不能代表 recursive rollout | 计算图检查；teacher-forced 与真正 recursive rollout 分开报告 | AV0-Z06/Z12 |
| Q28 | CD-NOD environment 语义 | 时间索引 C 可连续，负荷/压力不能自动当外生 environment | time/domain index 与 measured context 分栏；不变性仅作诊断 | AV0-Z12 |
| Q29 | NonstatSSM/TDRL/CtrlNS 定理迁移 | 其全观测/线性或离散域/充分变化等条件未被当前 latent MIMO 自动满足 | assumption ledger，逐条件 `met/unmet/not-testable` | AV0-Z12 |
| Q30 | IDOL 瞬时边解释 | 同采样耦合或早期负增益不能天然证明瞬时因果边 | 对齐、lead/placebo、feedback conditioning 后再决定 A3 动机强度 | AV0-Z09/Z12 |
| Q31 | CaRiNG 非可逆观测解释 | 阀位代理与部分观测不自动满足 CaRiNG 的识别条件 | observation-map assumption ledger + recursive rollout，仅作架构动机 | AV0-Z12 |
| Q32 | LEAP/机制噪声独立性 | 对角 mechanism noise 在闭环中可能制造而非验证独立创新 | innovation dependence、policy residual、noise cross-covariance 诊断 | AV0-Z12 |
| Q33 | 固定 horizon 不等于状态闭合 | H60 内 latent 递推不能自动支持 30–60 min 世界模型仿真 | state-closure audit、declared-context rolling、recursive-training 对照 | AV0-Z13 + AV1-R |

## 2. 批次结构：一次设计、两类执行、一次 Supervisor 收口

| 批次 | 内容 | 训练量 | 执行方 |
|---|---|---:|---|
| `RM3-AV0` | 旧产物一致性、冻结 checkpoint 推理消融、数据/反馈/形状/收敛审计 | 0 | 本地写代码与 smoke；Linux 只作需要真实 cache/checkpoint 的批量推理 |
| `RM3-AV1` | 32 个候选的机制宽筛，固定 2 folds × seed 0 | 64 units | Linux 并行训练；不得自行改矩阵或重试 |
| `RM3-AV2` | cache-free replay、成对表、四态审计判决、RM3-B 输入清单 | 0 | 本地 Supervisor |

AV0 与 AV1 可以在同一次授权中执行，但 AV1 的矩阵 SHA、父审计 SHA、代码 SHA 和 AV0 合同测试必须先冻结。AV0 不是用来先淘汰候选；它只恢复审计能力并形成 AV1 的共同输出合同。

## 3. RM3-AV0：零训练完整审计回放

### Z01 产物、算法与 provenance 修复

- 对 RM2 的 54 个 run artifacts、RM3 的 36 个 prediction checkpoints/12 个 calibration units、RM3-A 的 30 个 checkpoints 做 ledger/manifest/shape/SHA 闭合。
- 用当前 exact active-set NNLS 从冻结 `orthogonal_residuals_validation.npz` 重生成 calibration JSON；旧文件不静默覆盖，写 `supersedes_sha256`、算法版本、输入 SHA 和生成 commit。
- 检查 RM3 manifest 的候选角色与真实训练计算图；P5 在 integrated OOF R-loss 实验通过前只能称 `joint latent + explicit response + bypass`，不能称 OOF-calibrated。

### Z02 恢复并升级 RM2 强制诊断合同

P0–P5 及 RM3-A 全部冻结模型按其输出域重新报告：

- valve/Tin/local/terminal persistence MAE 与 skill；
- predicted-action 与 logged-action 下的 response effect mean absolute、H60/H180/H600；
- logged、shuffled、wrong-side、lead、time-shift action 下 local/terminal MAE；
- A-only、B-only、common、differential action effect；
- A/B 分侧以及 UTC 日、负荷、SP 方向、valve activity 分层；
- stable poles、NaN/finite、constant-action identity、prefix causality。

这些字段升级为后续批次的跨版本强制产物合同；缺一项即 artifact incomplete，不作科学判决。

### Z03 冻结模型推理消融

对 P3/P4/P5 与 RM3-A P5 家族做同 batch、同 checkpoint 的函数干预：

1. normal；
2. bypass-off；
3. bypass-only terminal；
4. response-off；
5. predicted valve；
6. logged valve；
7. logged valve + oracle Tin；
8. logged valve + oracle Tin + oracle local drop；
9. shuffled / wrong-side / lead valve。

推理消融回答“冻结模型实际上用了什么”，不能替代 retrained ablation；两者若结论不同，必须报告为训练补偿效应。

### Z04 动作敏感性与幅值下限

在 validation 支持域内施加小幅 `±0.5/±1/±2` valve-point common/differential perturbation，报告：

- `∂local/∂valve`、`∂terminal/∂valve` 的有限差分；
- H60/H180/H600 integrated effect 与 dose-response 线性区；
- predicted/logged effect 相对 local MAE 的比例；
- correct-side 减 wrong-side、correct-lag 减 lead 的差值。

符号约束只作为合同，不计为数据通过。幅值、时延、侧别和 placebo 才是可证伪量。

### Z05 阀位策略与闭环反馈审计

- 报告实际/预测 `|Δv|`、总跨度、静止比例、功率谱/粗糙度、饱和/死区命中和 persistence skill。
- 用相同 blocked folds 比较 probe：`SP only`、`SP + current T`、`error=SP-T + integral(error)`、完整 history；输出增量解释率而非因果方向。
- free head future-action probe：只用 free 可见信息预测未来阀位；若性能高，说明 action proxy 可由历史策略泄入，但不等于 free 已实际吞掉响应。

### Z06 训练计算图与信息流审计

- 自动追踪每个 loss 到 logged future valve、future SP、future Tin、future local/terminal 的可达性。
- 验证 logged future valve 仅进入明确的训练 auxiliary，不进入被评分 forecast path。
- 验证 OOF residual/R-loss 是否对 P4/P5 参数产生非零梯度；只有 gradient reachability + one-update parameter delta 同时非零才称“已接入训练”。
- 验证 bypass 对未来 action 严格不变；验证 free/action-shielded 分支的输入权限。

### Z07 响应形状与时间常数可分辨性

在 UTC 日期块上做 fit/evaluate 分离，比较：

- linear ramp；
- power basis；
- one-pole；
- two-pole；
- three-pole；
- three-pole + dead-time。

报告 blocked OOS RMSE、参数边界命中、tau/horizon 比和跨 fold 参数漂移。三极点只有在两个 fold 都获得 OOS 改善且第三极点未贴边时才保留为支持信号；否则只是方便的稳定基底，不称真实阶次。

### Z08 双侧秩、活动度与增益互换

- 对控制历史、SP 与燃烧 context 残差化后，计算 A/B valve innovation covariance、singular values、effective rank、condition number。
- 分别报告 per-side dose、`|Δv|`、活动时长，及 common/differential 模态能量；不得用 `R2 independent_channels_supported=False` 配置字段替代数据证据。
- 将日块增益回归到负荷/压力/燃烧 context、A/B valve activity、common/diff excitation；检查 fold 互换是否在控制活动度条件化后仍存在。
- full/common/diagonal MIMO 都只在其数据支持子空间内报告反事实。

### Z09 早期负增益与采样对齐

- 对 action timestamps 做 `-30/-20/-10/0/+10/+20/+30 s` 对齐敏感性；明确 lead 和 lag 定义。
- 报告 h=0、10、30、60、120、180、300、600 s 的 signed/absolute gain。
- 加入 `SP-T`、积分误差、动作创新条件后复算；上游 Tin 与错侧温度作为 placebo。
- 若负增益只在某个对齐消失，优先判 timing/alignment；若 conditioning 后消失，优先判 feedback confounding；均不自动证明瞬时因果边。

### Z10 selector、Pareto 与收敛复核

- 旧 A3/A4 的 `scope_selector_score` 只在各自目标内部使用，禁止跨目标排序。
- 用共同四任务权重重新评估所有 checkpoint 的 descriptive score，但不改变已选 checkpoint。
- 报告 learning-curve 最后 500 updates 斜率、best-update 距上限、最后值与 best 值差；据此判断是否值得解释 4000/8000 对照。
- Pareto 表只使用同一 checkpoint 选择规则下的 terminal/local/valve/Tin 与动作链指标。

### Z11 初始化可比性

- 给 encoder、valve policy、Tin、free/residual、response、downstream、bypass 分配固定 module seed。
- 可选模块存在与否不得改变共享 tensor 的初值；逐 tensor SHA/最大差必须为零。
- legacy RNG 仅作为审计复现候选，不再作为正式归因基础。

### Z12 KCI、非平稳与理论反馈边界

- KCI、environment invariance、noise independence 和 Jacobian sparsity 本批只作诊断，不作识别 PASS。
- environment/time index 与 measured context 分开；负荷/压力/燃烧量不自动称外生 environment。
- 检查 lead/placebo、first-stage strength、动作创新和闭环 policy residual；结果只决定 RM3-B 模块是否值得进入候选集。
- 非平稳性、KCI 或稀疏性无论通过与否，都不能突破未测喷水流量、阀位代理误差、双阀秩亏或 closed-loop action endogeneity。

同时建立逐论文 assumption ledger，而不是用论文名作架构背书：CD-NOD 的域索引 C 与 measured context 分开；NonstatSSM 的全观测线性瞬时 SEM 等假设、TDRL 的有限离散 Markov 域、CtrlNS 的机制/支持变化、IDOL 的稀疏潜过程与时滞父集、CaRiNG 的非可逆观测条件、LEAP 的独立机制噪声逐项标记 `met / unmet / not-testable`。只要有关键条件未满足，对应模块只能记为工程候选，不得引用定理升级识别结论。

### Z13 状态闭合与真正递推

- 逐项列出下一窗口 history 所需的全部特征：模型生成、控制策略生成、声明的外部 scenario、或缺失；只要存在未声明真实未来输入，就不能称 open-loop simulation。
- 分开报告：单窗口 teacher-forced H60、使用真实未来 context 的 oracle rolling、使用 hold/forecast scenario context 的 declared-context rolling、模型自身温度/阀位回灌的 recursive rolling。
- 报 10/30/60 min MAE 增长、漂移、边界命中、NaN/发散率和 action sensitivity；oracle future Tin/context 只作误差归因，不计部署能力。
- 若当前输出域无法构成下一状态，则判 `STATE_CLOSURE_BLOCKED`，而不是通过真实未来特征拼接伪造递推 PASS。

## 4. RM3-AV1：32 候选 × 2 folds × 1 seed

### 4.1 共同训练合同

- folds: `F0/F1`；seed: `0`；C00–C27/C31 固定完成 `4000 updates`，C28–C30 固定完成 `8000 updates`；不因 validation patience 提前终止，但仍按共同 selector 保存 best checkpoint。
- 历史长度 H96、共同预测/选择目标 H60、输入权限、batch、optimizer 和评估频率与 RM3 保持一致；所有候选从同一 H120 连续可用 anchor pool 抽取 train/selector/report anchors，使 C31 的两窗口递推与其余候选保持相同样本支持，非 C31 仍只训练/评分声明的 H60 输出。
- 所有新归因候选使用 Z11 module-scoped identical initialization；C00–C02 保留 legacy 架构/初始化路径，但在本批固定预算、共同 H120 anchor 支持和 UTC-day 隔离的新统一协议下重训，只作行为锚点，不宣称数值复现旧 run。
- 所有 full-multitask 候选用同一 selector 权重；不同 loss 权重只作为训练因素，不能拿各自 selector score 横排。
- validation 内 selector 使用较早 UTC 日块，reporting 使用后续且完全不重叠的 UTC 日块；只做到 anchor 行不重叠不合格。diagnostic anchors 只从 reporting 日块内确定性随机抽取。
- future logged valve 可进入声明的 training-only auxiliary，但 forecast 路径不得读取；产物必须同时记录 gradient reachability。
- 每个 unit 一次 attempt；失败原样回传，不自动重试、不缩模型、不改阈值；整批输出根目录一旦非空，runner 拒绝再次启动，网络中断或部分结果只能回传 Supervisor，不能自动续跑。

### 4.2 候选矩阵

| 组 | ID | 相对基线的唯一主要改变 | 主要回答 |
|---|---|---|---|
| Anchor | C00 | P3 current paired-free | 旧 local-free 锚点 |
| Anchor | C01 | P4 current A1 scheduled | 旧 additive-response 锚点 |
| Anchor | C02 | P5 current joint-latent+bypass | 旧 joint 锚点 |
| Head | C03 | P3 + action-invariant terminal bypass | P5 terminal 优势是否来自 bypass 位置 |
| Head | C04 | P5 no terminal bypass | joint latent 自身是否仍有 terminal 优势 |
| Head | C05 | P5 bypass-only terminal readout | terminal 是否可完全绕开物理链 |
| Head | C06 | P5 response-off during training and inference | P5 terminal 是否依赖 explicit response |
| Head | C07 | P4 free/residual small | free 容量减小时 response 是否增大 |
| Head | C08 | P4 free/residual large | free 容量增大时 response 是否缩小 |
| Head | C09 | P4 action-shielded residual/free information | 切断 action proxy 是否恢复分解 |
| Supervision | C10 | P4 + RM2-style logged-action auxiliary | structure penalty 是否恢复 response |
| Supervision | C11 | P4 + 真正接入训练的 OOF R-loss | 正交监督能否替代 logged aux |
| Supervision | C12 | P5 + 真正接入训练的 OOF R-loss | P5 动作链能否被激活 |
| Supervision | C13 | P5 + RM2-style logged-action auxiliary | P5 中 aux 与 bypass 的相互作用 |
| Valve | C14 | P5 + Δvalve loss + multiscale roughness match | 过平滑是否是损失造成 |
| Valve | C15 | P5 + structured PI policy | `SP-T`/积分反馈是否足够 |
| Valve | C16 | P5 + PI core + GRU residual/limit override | 结构控制器与未知逻辑的折中 |
| MIMO | C17 | P4 common-mode response only | 数据是否只支持公共喷水模态 |
| MIMO | C18 | P4 diagonal-only response | cross gain 是否主要吸收共模混杂 |
| Timing | C19 | P4 one-pole | 最小稳定动态基线 |
| Timing | C20 | P4 two-pole | 两时间尺度是否足够 |
| Timing | C21 | P4 power-basis response | 非指数形状对照 |
| Timing | C22 | P4 linear-ramp response | 低信息窗口负控 |
| Timing | C23 | P4 three-pole + learnable bounded dead-time | 物理响应是否只是未及时启动 |
| Timing | C24 | P4 signed/unconstrained diagnostic response | 数据自身是否支持稳定物理符号 |
| Init | C25 | P3 + module-scoped matched initialization | 清除 P3/P4 RNG 混淆 |
| Init | C26 | P4 + module-scoped matched initialization | 与 C25 作合法架构归因 |
| Init | C27 | P5 + module-scoped matched initialization, 4000 updates | P5 收敛对照的合法锚点 |
| Optim | C28 | P3 matched-init, 固定8000 vs 固定4000 | P3 是否 4000 未收敛 |
| Optim | C29 | P4 matched-init, 固定8000 vs 固定4000 | P4 是否 4000 未收敛 |
| Optim | C30 | P5 matched-init, 固定8000 vs 固定4000 | P5 是否 4000 未收敛 |
| Rollout | C31 | P5 + two-window state-continuation/declared-context rollout loss | 递推训练能否改善 10→20 min 漂移且不破坏动作链；30/60 min 本批强制判 `NOT_TESTABLE` |

算术闭合：`32 candidates × 2 folds × 1 seed = 64 training units`。C00–C02 即使已有旧 checkpoint，也在本批按统一 runner 重跑，因为本批要统一强制诊断产物；旧 checkpoint 仅在 AV0 作冻结参照，不能拿旧产物代替新合同。

C09 的 `action-shielded` 不是简单删除 future action（当前 residual 本来就不直接读 future valve），而是用 train-fold OOF action nuisance 从 history context 中投影掉可预测的未来 valve-innovation 子空间；projection 只在 train 内拟合并冻结到 validation。它与 C11/C12 的 outcome-side OOF R-loss 分开，分别检验“free 的 action proxy”与“response 的正交监督”。

### 4.3 为什么不做全因子组合

本批是机制宽筛，不是最终优化。logged-aux × OOF × bypass × valve decoder × shape 的全因子会把 32 个候选膨胀到不可审计规模。每个候选只改变一个主要因素；只有当单因素在两个 folds 出现同方向直接证据时，组合才有资格进入 RM3-B。未入选不等于路线被永久否决。

## 5. 共同输出与判读规则

### 5.1 预测指标

- valve/Tin/local/terminal MAE，A/B 分侧和 pooled 并列；
- persistence skill 与 H60/H180/H600 horizon curves；
- terminal/local 同目标 Pareto，不输出跨目标 composite champion。

### 5.2 动作链指标

- predicted/logged effect mean absolute、effect/local-error ratio；
- H60/H180/H600 finite-difference action sensitivity；
- correct vs shuffled/wrong-side/lead placebo advantage；
- common/differential 与 A-only/B-only response；
- constant-action identity、prefix causality、stable rollout。

### 5.3 阀位与反馈指标

- `|Δv|`、跨度、静止率、粗糙度/频谱与 persistence skill；
- `SP only` 相对 `SP-T + integral(error)` 的增量预测价值；
- policy residual 与温度/动作创新的 lead-lag 相关，仅作闭环诊断。

### 5.4 四态审计判决

每个 Q01–Q33 单独判定：

- `SUPPORTED`：两个 folds 的直接机制量方向一致，且没有同等强的替代解释；
- `REFUTED`：两个 folds 的直接机制量均与命题相反；
- `MIXED`：fold 不一致、只改善间接 MAE、或多个机制同时变化；
- `NOT_TESTABLE`：现有观测/支持域不足，不能用模型结果补成答案。

`2 folds × 1 seed` 只用于宽筛。不得使用 seed n=1 计算优化稳定性 CI，也不得把两个 folds 当独立同分布样本做显著性宣传。需要升级为 RM3-B 正式候选时，再冻结多 seed 方案。

## 6. RM3-B 消费规则

AV2 完成后只输出以下决策，不输出冠军：

1. **基线拓扑**：P3-style、P5-style、P3+bypass 或 P5-no-bypass 中谁能作为 RM3-B anchor；
2. **响应监督**：none、logged auxiliary、integrated OOF R-loss 哪些保留；
3. **阀位策略**：GRU、PI、PI+GRU 哪些保留；
4. **动作子空间**：full、diagonal、common-only 的支持边界；
5. **动态基底**：只保留得到 blocked OOS 支持的最小形状族；
6. **RM3-B 理论模块**：A1/A2/A3/A5/A6/A7/A8 逐项标记 `empirically_motivated / diagnostic_only / defer`。

以下情况直接阻止相应 RM3-B 主张，但不阻止继续做预测模型：

- 动作敏感性接近零且 response-off 无退化：不得称 explicit response 被使用；
- only common mode 有支持：不得做双侧独立 `do(valve_A/B)`；
- shape OOS 无区分：不得称真实三极点或真实时间常数已辨识；
- KCI/非平稳检查通过：仍不得单独称 causal identification；
- valve policy 改善但 local/terminal placebo 不改善：只称 controller prediction 改善。

## 7. Linux 执行边界

Linux/Hermes 只负责：

1. 拉取 Supervisor 明确授权的 commit；
2. 先运行 fail-closed preflight/dry-run；
3. 执行 AV0 批量推理与 AV1 的 64 units，可按 fold/候选并行；
4. 每 unit 最多一次 attempt，保存完整 stdout/stderr/resource/environment/git SHA；
5. 原样回传 checkpoint、episodes、metrics、manifest、ledger 和 root summary。

Linux/Hermes 不得：

- 自行删减“看起来差”的候选、补 seed、延长非 O 组预算、重试失败 run；
- 修改 loss、阈值、fold、输入权限、shape、初始化规则或报告字段；
- 根据中途结果组合新候选；
- 写 Supervisor 审计、TODO、项目状态或 RM3-B 决策；
- 访问 test、启动 MS4、把诊断结果写成因果结论。

## 8. ADR

**Decision:** 在 RM3-B 前加入独立 `RM3-AV`，以 0-training replay + 32-candidate `2 folds × 1 seed` 宽筛验证全部独立审计意见。

**Alternatives considered:**

- 直接按独立审计否决 P5/RM3-B：缺少成对实验，拒绝。
- 只做 free/bypass 两项：不能覆盖反馈、初始化、形状、MIMO、收敛和实现漂移，拒绝。
- 直接做 RM3-B 全量多 seed：基线与理论模块尚未定性，可能把算力花在错误计算图上，暂缓。

**Consequences:** 增加 64 个 validation training units 和一套严格诊断合同，但能把“架构可疑”转为可复算数据结论；RM3-B 的候选数和理论装置将由证据缩减，而不是由审计措辞提前决定。
