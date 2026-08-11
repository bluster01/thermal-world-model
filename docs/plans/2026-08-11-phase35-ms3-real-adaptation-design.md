# Phase 3.5-MS3 真实 A/B 观测适配设计

- 协议：`phase3.5-ms3-v1`
- 状态：本地实现与验证
- 证据范围：`real_ab_observational_validation_not_causal`
- 前置：MS5 已以 `JOINT_SELECTED` 关闭

## 1. 本 Gate 只回答什么

MS3 检验 MS5 选中的三极点、工况调度、单调阀门映射与 `free+response` 联合训练，在真实 A/B 数据上是否满足三个最低条件：

1. 总温度预测不因加入 response 分支而相对 free-only 明显退化；
2. response 分支在真实动态阀位 support 上没有坍缩为零；
3. 记录的未来阀位轨迹相对“保持当前阀位”和“置乱相对阀位轨迹”具有跨 UTC 日块稳定的条件预测增益。

第三点只说明动作路径包含时间对齐的预测信息。串级 PID 会让未来阀位对未来温度具有反馈内生性，因此即使该门通过，也不能解释成 `do(valve)`、开环 plant 响应或反事实已识别。物理闭环验证仍由 MS4 的 SP→阀位→温度 held-step 设计承担。

## 2. 三种方案及选择

| 方案 | 优点 | 硬问题 | 决定 |
|---|---|---|---|
| 继续使用旧 A/B 40-column cache | 现成、因果 LOCF age 完整 | 遗留同名侧接口不能显式保证 A阀→右温、B阀→左温 | 拒绝作为 MS3 主源 |
| 从 `all_merged_10s.csv` 生成交叉控制回路 cache | 同时含两阀与左右温度，可将现场交叉事实写死到 manifest | 源文件是已合并密集网格，缺少原始 tag age | **采用**，并把 age 边界写入 provenance |
| 立即重开 Graybox/Koopman/PI-ODE/DeepONet 大赛 | 可一次比较路线 | 把真实适配、表示选择和因果验证混在同一 Gate，算力与解释均失控 | 延后；MS3 只迁移 MS5 选中主干 |

## 3. 冻结数据映射

数据源 SHA-256 冻结为 `85a3f926...e4da6`，大小 `3,987,138,416` bytes。两个 side 标签表示控制回路，不表示未经配对的几何同侧：

| 回路 | 未来动作 | 物理温度链 |
|---|---|---|
| A | A 二级减温阀位反馈 | 右(B)二减入口 → 右二减出口 → 右末过出口 |
| B | B 二级减温阀位反馈 | 左(A)二减入口 → 左二减出口 → 左末过出口 |

共享 history 包含负荷、主汽压力、给水、煤量、主汽流量；本回路 history 还包含入口/出口/末过温度、SP 和实际阀位。没有把语义不一致的旧 `中间设定/阀门指令` 字段强行映射进来，也不使用不准的喷水流量。源文件含停机段，因此 anchor 只在处理前 16 min history 全程满足负荷≥250 MW、主汽压力≥10 MPa、目标/SP 位于 500–600°C 时进入训练或验证；未来段只检查阀位在 −2%–102% 的传感器容差内，不按未来温度筛选。

源文件时间戳必须严格递增；脚本不再插值或填充。实测存在 282 个非 10 s transition（其中 279 个为单点 20 s 间隔，另有 120 s、180 s 和 75,750 s 缺口），均原样保留。anchor 构造器用 transition-prefix 硬排除 history→future 跨越任一非 10 s transition 的窗口。cache `age=0` 只表示“该 dense merged row 有值”，不表示原始 DCS tag 在该时刻真实更新。该限制写入 cache manifest，禁止把它包装成原始测点时效证明。

本地只读 preflight 实际扫描 1,192,329 行并成功生成两个约 37 MB cache。应用连续性和运行工况门后，A/B train anchors 分别为 510,409/510,545，validation 均为 109,528；冻结 8192-anchor 抽样中，各 seed 动态窗口约 7,929–8,034，覆盖 27 个 UTC 日。它只证明矩阵具有足够支持并且代码可执行，不是模型效果结果。

## 4. 模型与信息流

模型保持

\[
\widehat T_{1:H}=f_{free}(h_{t-95:t})+
g_R(c,a_{t+1:t+H},a_t),\quad H=60.
\]

- `HistoryEncoder` 只读取过去 96×10 s；过去阀位属于处理前状态；
- `free` 输出未来 60×10 s 的绝对温度，不读取未来阀位；
- `response` 是 MS5 选中的 3-pole stable graybox，`K<0`、`τ>0`、learned monotone opening map、context-scheduled；
- 参考路径为“未来保持当前阀位”，因此 `g(c,r,r)=0` 是代码恒等式；
- joint 与 free-only 使用相同 history、anchor、seed 与 checkpoint selector；free-only 的 response 精确为零。

训练只使用 total-temperature Huber loss，不使用 component truth。checkpoint 只按 validation logged-action MAE 选择，所以 MS3 是开发/validation Gate，不是独立 test。

## 5. 冻结 12-run 矩阵

| 候选 | A/B | seeds | 数量 | 作用 |
|---|---:|---:|---:|---|
| `ms3_joint_total` | 2 | 3 | 6 | MS5 选中主策略 |
| `ms3_free_only` | 2 | 3 | 6 | prediction-only 负控 |

合计 12 runs。窗口 96、horizon 60、最多 40 epochs、每 epoch 100 updates；checkpoint selector 使用 validation 中冻结的 2048 anchors，最终诊断使用同 seed 的 8192 anchors。只运行 chronological 60/20 train/validation；不访问末 20% test。该预算约 4000 optimizer updates/run，和 MS5 同量级，避免以无上限真实数据扫描换取微小开发集收益。

## 6. 对照与指标

对每个 joint checkpoint 在同一 validation anchor 上计算：

1. `logged`：真实未来阀位；
2. `baseline`：未来保持 anchor 时刻阀位；
3. `shuffled`：在 10%-baseline bin 内循环置换相对阀位轨迹，再加回本样本 baseline；
4. `free-only`：相同 encoder/free 架构但 response 精确为零。

动态窗口定义为未来 10 min 内 `max|valve-baseline|≥1%`。推断顶层 cluster 是 UTC 日；每个窗口先得到 60-step MAE，再按日求均值，最后对日块 bootstrap 5000 次。窗口不是独立样本，不以重叠窗口数量虚增统计功效。

## 7. 冻结门禁

逐 side/seed 必须同时满足：

1. joint/free logged MAE ratio `≤1.05`；
2. 动态 support `≥512 windows` 且 `≥5 UTC days`；
3. 动态 mean absolute effect `≥0.02°C`，全体 max absolute effect `≤20°C`；
4. `baseline_error − logged_error` 的日块 bootstrap 95% CI 下界 `>0`；
5. `shuffled_error − logged_error` 的日块 bootstrap 95% CI 下界 `>0`；
6. reference identity、free action-blind、future-prefix causality、正开阀终端效应非正及 finite checks 全通过。

每侧至少 2/3 seeds 通过，双侧均通过才记 `OBSERVATIONAL_VALIDATION_PASS`。失败后原样回传并诊断，不在同批补阈值、seed 或超参数扫描。

## 8. 允许与禁止结论

允许：该架构在真实交叉回路 validation 上保持预测，并利用与温度时间对齐的 logged valve trajectory；或相反，未通过并定位具体门。

禁止：真实 free/response component 已唯一分解、阀位为外生干预、阀门非线性等于流量标定、三极点是现场唯一阶次、`do(valve)` 成立、模型已可用于 MPC/闭环。MS4 之前这些结论全部保持未建立。
