# Phase 3.5 MS3-R RM3-B 响应可识别性诊断门与机制噪声架构设计

> 状态：DRAFT FOR REVIEW，待本地/Codex 独立审计；Linux 未授权；validation-only；不访问 test、不启动 MS4；RM3/RM3-A 已审计结果不可变。

## Material Passport

- Material Type: active experiment-purpose specification
- Scope: Phase 3.5-MS3-R 后续批次（RM3-B）的评测门升级与架构增改设计；不改写 RM3/RM3-A 审计结论
- Verification Status: DRAFT（本文不包含对自身的审计判决；开放决策点见 §12）
- Paper Basis: `papers/causal_representation/` 七篇因果表征论文 + 三份精读笔记（`notes_*.md`）
- Data Boundary: A/B 现场 historian；喷水流量不作真值；实际阀位仅是有效喷水作用代理
- Claim Boundary: 可识别性（模型内部自由度唯一性）不等于观测因果效应；不建立 `do(valve)`

## 1. 要回答的问题

RM2/RM3/RM3-A 已给出三个硬事实：

1. **RM2**：四个响应算子族 logged response 幅值为 `0.319/1.088/0.435/0.493°C`（Koopman 相对 A1 放大 3.41×），local MAE 却无差异（`1.688–1.705°C`）→ `OPERATOR_GAIN_NOT_IDENTIFIED`。residual head 与 response operator 互相补偿，加性分解在观测数据下多解。
2. **RM3 R0**：滚动 fold 间端点方向不稳定（A→A `0.569→0.337`，B→B `0.309→0.456`）→ 数据支持时变、扰动条件的响应轨迹，不支持单一不变双侧增益。
3. **RM3**：P0 oracle 阀位 `0.661` vs P1 预测阀位 `0.948` → SP→阀位策略误差贡献约 0.29°C terminal，是当前最大单一误差源；A 侧动态响应 `0.0066–0.0085°C`（B 侧 `0.0429–0.0485°C`，5–7×），且 A 侧激励剂量比仅 1.05。

当前分解的正确性由结构先验保证（零初始化、加性、前缀因果），没有可证伪的统计裁判。RM3-B 要回答：**能否把"响应分解"从多解结构升级为可识别结构，并使识别性本身成为预注册门**。依据是论文包的统一原则：用条件独立、充分变异性、不变性约束替换不可检验的先验。

## 2. 论文基础（文件与结论）

| 论文 | 关键结论（章节/定理） | 本文使用处 |
|---|---|---|
| `LEAP_ICLR2022.pdf` | Thm1 非参数可识别需 2n+1 工况充分变异性；Thm2 广义 Laplacian(α<2)+满秩 B_τ；噪声 Total Correlation 惩罚 | E2/E4、A1/A2 |
| `Nonstationary_StateSpace_ICML2019.pdf` | Thm1 线性+高斯+非平稳即可识别时变因果系数（不需非高斯/faithfulness）；根因统计量 S(t,t+p) | E3、A6 |
| `TDRL_NeurIPS2023.pdf` | Thm1 ARHMM 由 ≥4 连续观测识别工况；Thm2 交叉导数向量线性无关→分量级识别 | A5、E4 |
| `CaRiNG_ICML2024.pdf` | Thm1 非可逆观测下 z_t=m(x_{t:t-μ}) 分量级识别；Corollary A1 非平稳助识别 | A8、§6 |
| `CD_NOD_JMLR2020.pdf` | Thm1 C 入条件集的骨架恢复；Thm2 Alg.2 不变性/Alg.3 独立变化定向；Alg.4 KNV 驱动力 | E2/E3、A6 |
| `CtrlNS_NeurIPS2024.pdf` | Thm1 机制稀疏性+变异性→域识别；Thm2 域识别后分量级识别；跨域图不变时假设失效 | A5（风险见 §5） |
| `IDOL_ICLR2025.pdf` | Thm1/Thm2 稀疏潜过程→图同构；Thm3 时滞父集互异→瞬时边定向；L_S=‖J_d‖+‖J_e‖ 稀疏 | A2/A3 |

精读笔记：`papers/causal_representation/notes_LEAP_NonstatSSM_TDRL.md`、`notes_CaRiNG_CDNOD.md`、`notes_CtrlNS_IDOL.md`。

## 3. 评测侧调整（E 系列）

以下 E1–E4 为**识别性门**，E5–E8 为**归因与报告层**。E1–E3、E6 可先在 RM3/RM3-A 既有冻结产物上回放（零新训练），用于校验门本身。

### E1 响应幅值分歧门（RM2 现象 → 正式门）

- 依据：RM2 `OPERATOR_GAIN_NOT_IDENTIFIED` 审计事实。
- 做法：同一数据、同一 fold/seed、同一 residual 初始化下，各候选的 logged response 幅值分歧超预注册阈值（建议 >2×，见 §12）→ 该单元判 `IDENTIFICATION_FAILED`，不再比较 MAE；分歧未超阈值才进入 MAE 对比。
- 失败含义：分解多解，任何"响应增益已识别"的主张撤回。
- 实现：`experiments/phase3_5/audit_ms3r_rm3b.py` 新门，汇总进 batch summary。

### E2 响应分量条件独立检验（KCI）

- 依据：CD-NOD Thm1 将 C 入条件集；KCI 检验（Zhang et al., 2012）。
- 做法：validation 集上非参检验 `response_effect ⟂ residual_local | context`（context=编码器输出的 d 维上下文或工况 bin）。逐 UTC 日算 p 值，报分布与整体判决（α 与带宽启发见 §12）。
- 失败含义：free/response 分解不唯一；分量 MAE 降级为"组件拟合质量"，不得称分量恢复。
- 实现：`src/phase35/multistep/rm3b_identifiability.py`；先以诊断级（报告不拦截）运行一个批次，判决级阈值待审计后冻结。

### E3 不变性检验（升级 Gate B 的 SP-IV）

- 依据：CD-NOD Alg.2；NonstatSSM 的时变系数可识别性。
- 做法：(a) 数据侧：`P(valve|SP)` 跨时间块/负荷 bin 的不变性检验——不变则 SP→a 定向成立（Gate B SP-IV partial R² 0.014 的框架升级为定向检验）；(b) 模型侧：fold 对的响应增益差异进 bootstrap CI，超 CI 即正式声明时变响应（R0 的 0.569→0.337 已预示）。
- 失败含义：不变性不成立→控制链定向存疑或响应必须时变参数化（A6）。

### E4 充分变异性资格门（训练前资格）

- 依据：LEAP Thm1（2n+1 工况）、TDRL Thm2（交叉导数线性无关）。
- 做法：复用 `RM3MomentAudit.condition_number`（动作 Gram 条件数）+ 新增每侧/每负荷 bin 的剂量多样性指标，预注册最小门槛。不达标的侧/工况**训练前**判 `INSUFFICIENT_EXCITATION`，不训完再报 FAIL。
- 失败含义：该侧响应先验不可识别；A 侧 0.0066°C 的结果属此类，不得通过调阈值补救。

### E5 oracle 阶梯归因（P0b/P0c）

- 做法：在 P0（oracle 阀位 0.661）基础上补两级——P0b=oracle 阀位+oracle Tin（复用 `boundary_mode="oracle_boundary"`，`gatec_data.py` 已支持）；P0c=P0b+oracle 局部温降（新增 contract：把 true `Tin−Tout` 直接注入 downstream，绕过 local response 与 residual head）。
- 输出：0.973（P5）→0.661（P0）的 0.31°C 逐层归属（阀位策略 / Tin 边界 / 响应动力学）。
- 实现：`rm3b_contracts.py` + `RM3FairPredictionAdapter` 扩展。

### E6 NNLS 经验轨迹形状基准

- 做法：R1 审计的 60/180/600s 基精确 NNLS 拟合（轨迹 RMSE 0.024–0.072）定为经验响应参考；各候选 learned IRF 与之比形状距离（幅值自由）。响应评估获得模型外锚点。

### E7 递归 rollout 评测（C2 最低门槛）

- 做法：validation 上 open-loop 递推（喂模型自身输出），报 10/30/60 min 误差增长、漂移、NaN 率；并列报告 teacher-forced / open-loop / closed-loop 三模式。
- 失败含义：仍称 predictor，不得称仿真器（W3 门）。

### E8 分层工况报告（Simpson 防线）

- 做法：terminal/local MAE 按负荷 bin、SP 升/降方向、阀位活跃度分层报告，检查子组反转。

## 4. 架构侧调整（A 系列）

### A1 机制噪声模块（LEAP/CaRiNG 转移先验的独立噪声条件）

- 做法：response effect 改为 `N(g(a,context), σ_r(context))`，σ_r 按工况学习；residual head 配独立 σ_f。smooth_l1 升级为似然（尺度自适应），两个噪声尺度的比值作为可识别性信号（比值漂移=RM2 的 3.4× 现象）。
- 改动：`rm3_joint_model.py`（JointLatentPhysicalInterfaces 加噪声参数）、`rm3_training.py`（loss 改似然）。
- 风险：闭环下 IN 可能被违反（见 §6），需与 A3/A4 联合才有理论意义。

### A2 下三角 Jacobian 转移先验（LEAP/IDOL/TDRL 同一技巧）

- 做法：g_response 逆映射按固定因果序（阀位→局部→末端）参数化，损失加 log|J| 项；结构从"加法习惯"变为体积保持变换先验。
- 改动：新模块 `rm3b_transition_prior.py`，挂在 JointLatent 转移上。
- 不变量：constant-action identity、prefix causality、稳定性约束全部保留。

### A3 瞬时耦合层 + Jacobian 稀疏（IDOL）

- 做法：readout 前加瞬时混合 `z_t = tanh(W z_t)`，对瞬时 Jacobian `J_e` 加 L1（IDOL L_S）；按 Thm3 给每个潜分量保留独特时滞父集（喷水只进汽温方程）保证方向可识别。
- 依据：10s 采样下喷水/蓄热/汽温瞬时互馈客观存在，当前被压进 residual/bypass。
- 风险：IDOL 假设稀疏潜过程（A3 条件）——热工系统大概率成立但需验证；高维下稀疏约束能力有限。

### A4 PI 结构化阀位策略（CD-NOD 不变性 + 伊敏 PID 解析）

- 做法：GRU 阀位策略替换为 `valve = baseline + Kp(e)(SP−T) + Ki(e)·∫e + 死区/限幅`，Kp/Ki 按功率调度（结构来自 `dcs_bmcs2_output` 伊敏主调解析：纯 PI、Td=0、增益随偏差×功率动态调度）。
- 动机：P1→P0 的 0.29°C 缺口 + `P(valve|SP)` 不变性成为可验证结构性质。
- 风险：实际控制器若含前馈/override 逻辑，PI 结构会 misfit；需保留 GRU 作对照。

### A5 离散工况开关（NCTRL/CtrlNS，作用在参数不在图）

- 做法：Gumbel-softmax 选 U 套增益/τ 参数（CtrlNS sparse transition），或 ARHMM 无监督切分工况段（NCTRL）。
- 风险：CtrlNS 可识别性要求跨域因果图不同——喷水物理跨工况图大概率不变，假设易失效；故域开关只作用参数，且 A5 仅作非平稳增益的显式化，不声称域已识别。

### A6 时变增益 AR（NonstatSSM）

- 做法：diagonal/cross gain 改为小递归状态 `b_t = α·b_{t-1} + β(load,pressure)`，非平稳性显式进生成模型。
- 依据：R0 已判时变响应；NonstatSSM Thm1 证明该参数化在近线性近高斯对象下可识别。

### A7 快/慢状态分层（IDOL 稀疏结构 + 物理）

- 做法：快层（局部温降，τ 20–1200s）与慢层（金属蓄热，τ 数百秒）分离，各配独特父集；对应 R1 的 60/180/600s 基。

### A8 递推训练（CaRiNG + C2）

- 做法：scheduled-sampling 微调 + 部分 rollout 损失；terminal readout 移入递推路径。非可逆观测（15 特征←高维热状态）按 CaRiNG 的 `z_t=m(x_{t:t-μ})` 处理，编码器继续用 96 步历史补全丢失状态。

## 5. 批次结构

| 批次 | 内容 | 训练量 |
|---|---|---|
| RM3-B0 | 结构合同：shape/finite/prefix/constant-action/noise-ratio/Jacobian/rollout 稳定性 + E1/E2/E3/E6 在 RM3/RM3-A 冻结产物上回放（校验门本身） | 零 |
| RM3-B1 | E4/E5/E7/E8 在冻结产物上补齐基线；E2 从诊断级升判决级的阈值数据 | 零 |
| RM3-B2 | A 系列 one-update smoke + 单 seed 单 fold 筛查（建议首批 A1+A4+A3，见 §12） | 微 |
| RM3-B3 | 正式 validation 比较：≤N 候选 × 3 seeds × 2 folds，4000-update 上限、fold/seed/输入权限与 RM3 相同；具体矩阵 B2 后冻结 | 全量 |

## 6. 理论警告：闭环反馈违反独立噪声假设

七篇的可识别性都以噪声独立性（IN/时空独立）为前提。PID 闭环下动作噪声是状态噪声的因果下游，IN 默认被违反——直接套 LEAP 式先验而不显式建模 SP→阀位→温度反馈环，识别理论静默失效。应对是结构性联合要求：

1. A4 PI 结构化阀位策略（反馈环显式）；
2. A3 瞬时耦合层（同采样步互馈显式）；
3. A1 动作噪声与状态噪声的相关项（σ 协方差而非对角）。

三者缺一，A1/A2 的识别主张不得升级。E2（KCI）即为 IN 是否成立的检测器：E2 不通过时，本批次结论降级为 `OBSERVATIONAL_CONSISTENCY`，不称可识别。

## 7. 放行门（RM3-B3 判决）

1. 识别性：E1 幅值分歧 ≤ 冻结阈值；E2 逐日 p 值分布过冻结 α；E3 不变性/时变性判决与架构选择一致；E4 每侧/每工况过资格线；
2. 架构不变量：constant-action identity、prefix causality、有限稳定 rollout、噪声比稳定、Jacobian 符号结构全部通过；
3. 预测非劣：terminal/local MAE 相对 P5 在预注册非劣界内；E5 oracle 阶梯完整报告；
4. E7：10/30/60 min open-loop 误差增长 ≤ 冻结阈值；
5. 统计单位 UTC 日/连续时间块；seed 只表示优化稳定性；
6. 失败时不缩阈值、不补 seed/fold、不打开 test、不事后改矩阵。

## 8. 不可提前声称

- 识别性门通过只证明"分解在模型族内唯一到置换+分量可逆变换"，不建立 `do(valve)`、唯一 plant gain、喷水流量标定或 closed-loop readiness；
- 可识别 ≠ 因果效应：外部锚点仍缺（W2 经验响应之外，W4 反事实继续 BLOCKED）；
- A5/A6 不声称域/增益"已恢复"，只声称非平稳性被显式参数化且可检验；
- 论文理论（IN、sufficiency、稀疏潜过程）在闭环数据下可能不成立，E2/E4 正是其检测器。

## 9. 被拒绝的备选

- 继续仅平滑调度：动态范围窄（exp(0.25·tanh(·))），且 R0 已显示时变响应——A6 显式化优于继续隐式；
- 纯容量扩充：RM3-A 已证 terminal 优势是架构性的（A0/A1 扩容仍差 0.079–0.080，A2 缩容仍优），容量杠杆已耗尽；
- 把幅值分歧当超参扫描：look-elsewhere 风险，与 fail-closed 纪律冲突；
- 在 synthetic known-truth 上验证识别理论：MS3 已证明 synthetic PASS 不迁移真实数据，识别性门必须在真实 A/B 数据上回放。

## 10. 与现行纪律的关系

- RM3/RM3-A 结果、fold/seed/输入权限、4000-update 上限不可变（RM3 final audit 第 5 点）；
- 本文不改 TODO、注册表、Supervisor 文档（本地职责）；
- Linux 执行仍须 `active_gate` 与 `linux_authorized_gate` 同时满足，本地冻结矩阵后才授权；
- 任一预检红项必须停止回传，不得自行豁免。

## 11. ADR：为什么以"识别性门 + 机制噪声"为 RM3-B 主轴

**状态：Proposed（待审计）。**

**背景：** RM2 已证响应幅值多解（3.4×），RM3-A 只解决容量/权重混淆，不触及分解唯一性；论文包（Supervisor 于 a3b003d 加入）全部指向同一升级路径：条件独立+充分变异性+不变性约束。

**决定：** 先落评测门（E 系列，可在冻结产物上零成本回放并校验），再动架构（A 系列，分 smoke 与正式两阶段）；架构改动优先 A1+A4+A3（识别入口+最大误差源+理论前提），其余按审计意见分批。

**后果：** 评测对象从"预测准"转为"分解可识别"，可能与部分既有指标（local MAE）解耦；代价是新增 KCI/方差/rollout 三套统计设施与噪声参数化，训练目标与审计复杂度上升。仍不改变 validation-only、fail-closed 的根本纪律。

## 12. 开放决策点（供独立审计，本文不作裁定）

1. E1 幅值分歧阈值具体值（建议 >2×；需从 RM2 四算子分布反推合理界）；
2. E2 KCI 的 α（建议 0.05）、RBF 带宽启发（中位数距离）、判决级 vs 诊断级的分批路径；
3. E4 最小条件数与剂量多样性门槛的数值（需 A/B 侧数据分布支撑）；
4. A 系列首批冻结矩阵包含项（建议 A1+A4+A3；A5/A6/A7 是否并入 RM3-B 或延后）；
5. 批次命名（RM3-B）与"回放既有产物"是否算新 Gate；
6. P0c 的 oracle-local 注入接口的 contract 边界（是否触碰 RM3 契约）；
7. A5 在 CtrlNS 假设失效时的降级声明措辞。
