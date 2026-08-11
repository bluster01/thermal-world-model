# Phase 3.5 MS3-R 真实响应辨识设计

> 状态：设计冻结，Gate 1 实现中；不授权 Linux，不访问 test，不启动 MS4。

## 1. 目标与声明边界

MS3-R 用真实点位扩充 MS3，但不预设历史闭环数据能够完成双侧开环 plant identification。当前目标依次是：

1. 判断 A/B 阀位在给定历史、SP 与运行工况后是否仍有独立创新；
2. 验证正确交叉链路是否优于 lead、错移、错侧和上游温度 placebo；
3. 判断只允许公共喷水模态、允许双侧 MIMO，还是只能保留 observed-policy closed-loop prediction；
4. 在点位门禁通过后，再比较分支归因、串级闭合和真实模型结构。

现场交叉关系作为已知拓扑冻结，不再作为待发现假设：

- A 标记 SP/阀位控制右侧（B 热列）；
- B 标记 SP/阀位控制左侧（A 热列）。

喷水流量传感器不作为真值。阀位仅是有效喷水作用的非线性代理。历史闭环结果最多支持 observed-policy prediction、扰动条件响应与支持域内小反事实；任意 `do(valve)` 仍需可信外生工具或现场预注册激励。

## 2. 分支语义合同

旧结构

\[
\hat y=f_{free}(H,c)+g_{phys}(H,c,u)
\]

只是函数分解。MS3-R 在证据闭合前统一改用以下术语：

| 旧称 | MS3-R 合同名称 | 允许含义 |
|---|---|---|
| free head | 历史状态/未测扰动残差分支 | 从过去状态外推未来并承接未测扰动 |
| physics head | 受约束阀位响应分支 | 满足稳定性、符号和 constant-action identity 的阀位条件响应 |

残差分支不是显式燃烧模型；响应分支没有喷水流量、喷水压力或焓平衡真值，因此不是已经验证的喷水物理模型。不得通过删除或极端压缩残差分支，强迫响应分支吸收燃烧扰动。

## 3. 三个批次级检查点

| 大门 | Linux 批次 | 本地唯一检查点 | 放行结果 |
|---|---|---|---|
| Gate A | MS3-R0S/R0P/R1/R2 | 点位、placebo、激励与输入秩独立重算 | common-only、dual-MIMO 或 closed-loop-only |
| Gate B | MS3-RA/R3/R4/R5 | 分支归因、串级闭合、不变性与 IV 可行性审计 | 冻结真实模型结构和允许声明 |
| Gate C | MS3-RM0/RM1 | 单 seed 结构筛查后，正式候选 3 seeds + rolling folds | 冻结模型；再决定是否存在 final lockbox 入口 |

Linux 只执行冻结命令、保存日志与机器产物，不作科学结论或修改状态文档。本地只在整批结果回传后审计一次；批内小实验不逐项审批。

Gate A 的远端执行边界固定为 `batch_id=ms3r_gate_a_v1`：单次 attempt、预计 30 分钟、2 小时硬超时、8 CPU threads、16 GiB RSS 上限、不使用 GPU。失败、超时、资源越界或产物缺失时停止并回传，不自动重试。即使结果“不符合预期”，Linux 也不得调参、换点位或追加模型；是否优化代码或签发新 batch 由本地决定。

## 4. Gate A：点位与可辨识性

### 4.1 MS3-R0S/R0P

R0S 固定分支名称、禁止声明和信息流合同。残差分支不得读取未来动作；历史/SP 能否预测未来阀位另作内生性诊断。

R0P 校验缓存来源、时间戳、交叉映射、缺失、冻结值、量程和 validation 边界。物理主检验只使用冻结点位；200 列全量扫描仅为探索性扰动发现，任何新候选必须重新冻结后才能进入主检验。

### 4.2 MS3-R1 动态影响图

先对每侧阀位差分做过去信息条件化：

\[
\tilde u_t=\Delta v_t-\widehat{E}[\Delta v_t\mid H_{t-1},\Delta SP_t,c_{t-1}]
\]

采用 validation 内过去训练、未来评估的 rolling cross-fit，禁止随机样本切分。再对未来点位变化做同样的控制变量残差化，并估计 residual-on-residual 局部投影。主输出为 60/180/300/600 s 的方向化系数、日块分布和增量解释度。

| 类型 | A 阀示例 |
|---|---|
| 正确局部路径 | A 阀创新 → 右侧 `Tin-Tout`、`Tout` |
| 正确末端路径 | A 阀创新 → 右侧末温 |
| 错侧 placebo | A 阀创新 → 左侧局部/末温 |
| 上游 placebo | A 阀创新 → 左右 `Tin` |
| 时间 placebo | action lead、跨日循环错移 |

正确链路必须在合理正滞后出现，且不能被 lead、错侧、上游或错移结果同样复现。R1 本身仍是闭环条件关联，不升级为因果效应。

### 4.3 MS3-R2 输入秩

构造

\[
u_c=(\tilde u_A+\tilde u_B)/2,\qquad
u_d=(\tilde u_A-\tilde u_B)/2.
\]

报告创新协方差、相关系数、条件数、common/differential 能量，以及 60/180/600 s block-Hankel 奇异谱。条件数不单独充当 PASS 阈值；本地 Gate-A 综合第二模态、稳定性、placebo 与有效日期支持判定：

- 两模态均有支持且正确链路闭合：允许双侧 MIMO；
- 只有 common 模态有支持：只允许公共喷水模态；
- 创新存在但正确链路不胜 placebo：只允许 closed-loop prediction；
- 激励与链路均不足：停止真实动作点辨识，等待外生激励。

## 5. Gate B：归因、串级与 IV

MS3-RA 固定 `small/base/large residual capacity × terminal-only/local-supervision × additive/context-scheduled × excitation stratum`。相同 folds、3 seeds。若容量增大而响应缩小、MAE基本不变，标记分解不唯一；若中间监督恢复局部响应，说明末温单目标缺少锚点；若仅工况调度恢复，说明简单加法结构错配。

MS3-R3 依次验证 `SP→valve→Tin-Tout/Tout→terminal`。物理时序要求控制器段先于局部喷水段，局部段先于末温传播段。局部 `Tin-Tout` 是 plant-response 主证据，末温是延迟下游验证。

MS3-R4 按日期、负荷、压力、Tin、阀位基准、opening/closing、燃烧工况和单双阀同步分层。方向应基本一致，时常处于相近量级；增益可随工况调度。

MS3-R5 只评估 SP innovation 作为工具变量的可行性。必须报告 first-stage strength、同步其他动作、lead/placebo 与三路线一致性；失败只表示不能升级成 `do(valve)`。

## 6. Gate C：真实模型候选

正式接口以未来 SP 场景为动作；模型内部预测阀位。valve 接口只作条件 plant 诊断。推荐 measured-boundary latent MIMO：

\[
x_{k+1}=A(c_k)x_k+B_T(c_k)Tin_k+B_u(c_k)[u_c,u_d]^T+G(c_k)\eta_k
\]

\[
[\hat T_{out,L},\hat T_{out,R},\hat T_{main,L},\hat T_{main,R}]=C(c_k)x_k.
\]

未来 Tin 必须由边界预测器产生或作为声明的外部场景输入；真实未来 Tin 只能用于 oracle 上限。训练采用最多 10% 更新量的短 warm-up，随后全量解冻 joint multitask。selector 同时包含 valve、局部温降、末温 rollout、placebo 与 collapse penalty，不使用 CFI，也不只看末温 MAE。

候选 B0–B5 先跑单 seed 结构筛查；只有通过结构门的候选进入 3 seeds + rolling folds。MS3-RA 属于归因证据实验，固定 3 seeds，不按单 seed 淘汰。

## 7. 失败模式与保护

- validation-only；Gate A/B 不访问 test；
- 不把 10 s 样本数当独立样本，统计单位为 UTC 日块或连续时间块；
- 不将 seed 当统计样本量；
- 缓存、配置、Git SHA、split bounds、日志和环境必须写入 manifest；
- 远端不得自行改阈值、补跑或更新 supervisor 结论；
- 任一自动标签只能是机器诊断，本地审计文档才是放行依据。
