# Phase 3.5 MS3-R RM3-B 成对组合筛查设计

> 状态：B0 DESIGN FROZEN / B1 LOCAL VERIFIED / Linux 一次执行已授权 / validation-only。唯一科学输入是 RM3-AV2 Supervisor 判决；本文不恢复旧 E 系列，也不把因果表征论文的定理直接外推到现场闭环数据。

## 1. 目标与边界

RM3-B 不再回答“再加哪一个听起来更物理的模块”，而回答三个可证伪问题：

1. 在 8000 optimizer updates、公平四任务 selector 和 module-scoped 初始化下，RM3-AV 保留模块能否在两个 rolling folds 给出同方向改善；
2. 改善来自 terminal bypass、valve policy、显式 response，还是三者之间的补偿；
3. 哪些模块可以进入后续 multi-seed 组合确认，哪些应停止。

允许声明：observed-policy prediction、disturbance-conditioned response diagnostic、validation 内的成对架构归因。

禁止声明：`do(valve)`、独立双侧 plant gain、真实喷水流量、已识别物理阶次、30–60 min state-closed simulator、模型冠军或论文结论。

## 2. 架构决策（ADR-RM3B-01）

### 背景

RM3-AV 已确认：P5 的 terminal 优势主要由 action-invariant bypass 贡献；action shield 和 OOF R-loss 能放大显式响应但尚未闭合 placebo；阀位预测过平滑；full MIMO 与三极点/纯迟延没有额外证据；4000 updates 混入了收敛偏差。

### 决定

RM3-B1 采用“角色锚点 + 单模块成对干预”，不堆叠全部模块：

- `C28`：高容量直接预测角色锚点；
- `C29`：显式响应角色锚点；
- `C30`：joint-latent + action-invariant bypass 角色锚点；
- 每个新候选只改变一个预声明模块，并与其角色锚点成对；
- 全部候选固定 H60、2 rolling folds、seed 0、8000 updates；
- 只有两个 folds 都同方向且合同门通过的模块，才可进入 B2 multi-seed 组合确认。

### 后果

这一设计牺牲一次性搜索组合最优的速度，换取归因可解释性。B1 不产生冠军；B2 组合数由 B1 结果决定，不能提前写入 runner。

## 3. B1 冻结候选表

| ID | 角色/干预 | RM3-AV 实现模板 | 成对基线 | 唯一改变 |
|---|---|---|---|---|
| B00 | P3 prediction anchor | C28 | — | 8000-update anchor |
| B01 | P4 response anchor | C29 | — | 8000-update anchor |
| B02 | P5 joint+bypass anchor | C30 | — | 8000-update anchor |
| B03 | action-shielded history residual | C09 | B01 | OOF action projection shield |
| B04 | integrated OOF R-loss | C11 | B01 | response calibration loss |
| B05 | valve delta + multiscale roughness | C14 | B02 | valve dynamics loss |
| B06 | structured PI + GRU residual | C16 | B02 | valve decoder |
| B07 | field-aligned diagonal response | C18 | B01 | response coordinates |
| B08 | one-pole response basis | C19 | B01 | response shape |
| B09 | linear-ramp sensitivity control | C22 | B01 | shape negative/sensitivity control |
| B10 | capacity-controlled action-invariant bypass | C03 | B00 | history-only terminal bypass |

矩阵为 `11 candidates × 2 folds × 1 seed = 22 training units`。B03–B10 继承模板的结构实现，但 optimizer cap 统一提升到 8000；模板编号只表示已审计代码路径，不继承 RM3-AV 的 4000-update 数字。

## 4. 公平合同

所有 22 个单元必须同时满足：

- 数据：真实 A/B paired cache，window=96、H60、10 s；F0=train 0–0.6/validation 0.6–0.7，F1=train 0–0.7/validation 0.7–0.8；test `[0.8,1.0)` 禁止；
- selector/reporting 按 UTC 日隔离，checkpoint 仅由共同四任务 loss 选择；
- 四任务权重相同，terminal/local/Tin/valve 分开报告；
- seed 只表示优化初始化，UTC 日/连续时间块才是统计单位；
- 每个成对比较必须校验共享模块初始化 hash，不能以不同初始化冒充模块效应；
- 每个 run 必须报告 normal、response-off、wrong-side、lead、shuffled、logged-valve 等既有诊断；
- 不计算综合排名，不填写自动科学 PASS，不访问 test。

## 5. 成对判决

B1 只做模块资格判决，不做模型选择。每一对逐 fold 给出 `SUPPORTED / MIXED / REJECTED / NOT_TESTABLE`：

| Pair | 主要问题 | 必报证据 |
|---|---|---|
| B03−B01 | shield 是否减少 free 对动作策略的代理，同时保住预测 | terminal/local 非劣、H60 response、wrong/lead/shuffled sensitivity |
| B04−B01 | OOF R-loss 是否让 response 真正进入 terminal | response-off terminal delta、OOF first-stage/残差诊断、placebo |
| B05−B02 | 动态 loss 是否修复阀位过平滑 | valve MAE、mean `|Δv|`、roughness、terminal/local |
| B06−B02 | PI+GRU 是否改善策略层而不伤害末温 | valve MAE/dynamics、terminal/local、PI residual contribution |
| B07−B01 | 数据是否只支持更简单的对角通道 | terminal/local 非劣、common/differential response、wrong-side |
| B08−B01 | one-pole 是否是足够的工程基底 | shape RMSE、terminal/local、跨 fold 稳定性 |
| B09−B01 | shape 结论是否对线性基敏感 | 与 B08 并列报告；不得据此声称阶次 |
| B10−B00 | bypass 是否只改善自然工况预测 | terminal 改善、response-off/bypass-off、动作不变性 |

“同方向”必须由各 pair 的角色指标定义，而不是只看一个 MAE：预测不劣但 response/placebo 不闭合时判 MIXED；两个 fold 方向相反时判 MIXED；任一合同失败直接 REJECTED。

## 6. B2 与 B3

- B2 只有在 B1 审计后才生成。最多从三个机制族各保留一个模块：response calibration、valve policy、terminal correction；只做预声明的最小交互，不进行全组合搜索。
- B3 才对 B2 surviving compositions 做多 seed 确认。seed 数、候选数和预算届时另行冻结。
- B1/B2/B3 均保持 validation-only；test 与 MS4 继续 HOLD。

## 7. 识别性诊断如何放置

KCI、nonstationarity、Jacobian sparsity、机制噪声独立性和 condition number 只作为诊断 ledger：

- 域标签和 measured context 分开；负荷/压力不能自动当外生 environment；
- 非平稳性不解决闭环动作内生性、阀位代理测量误差或双阀秩亏；
- lead/wrong-side/placebo 未闭合时，不用 KCI 或稀疏 Jacobian 升级因果声明；
- 本批不实现 A1 机制噪声、A2 Jacobian prior、A3 瞬时耦合或 A6 时变增益新模型，避免与 AV2 已保留模块混为一个不可归因大改动。

## 8. Linux 执行边界

本地负责：冻结矩阵与代码、micro smoke、全量回归、授权 SHA、结果回放与唯一 Supervisor 判决。

Linux/Hermes 只负责：在授权提交上执行冻结的 22 units 一次，回传原始六件套、root ledger、stdout/stderr、资源记录和 commit SHA。

Linux 禁止：改候选/阈值/seed/fold/loss、补跑失败单元、访问 test、代写判决、自动生成 B2、启动 RM3-B2/B3/MS4。预检失败时整批不启动；训练单元失败时停止该设备的后续分区，其他已经启动的设备分区允许完成，随后原样回传全部成功产物与 `failure.json`，不得发起第二次 attempt。

## 9. 放行检查点

1. B0：本文、AV2 parent hash 与候选角色表冻结；
2. B1-local：合同、22-unit 展开、模型映射、one-update smoke、产物 schema 和专项回归通过；
3. B1-authorize：registry 精确授权 `RM3-B1`，且授权 commit 工作区干净；
4. B1-remote：Linux 一次执行并回传，不作科学结论；
5. B1-audit：本地复算 hashes、checkpoint、轨迹与 paired verdict；之后才允许设计 B2。
