# Phase 3.5 MS3-R RM3-AV Supervisor Audit

> Decision: **AUDITED / ARCHITECTURE ATTRIBUTION CONFIRMED WITH MIXED SUPERVISION, TIMING AND CONTEXT / NO MODEL CHAMPION / RM3-B DESIGN ALLOWED, TRAINING NOT AUTHORIZED**

## 1. 审计范围与完整性

本审计消费 Linux 回传的 RM3-AV0 零训练回放和 RM3-AV1 `32 candidates × 2 folds × seed 0 = 64 units`，不访问 test，不把两个 rolling folds 当 IID 样本，不计算 seed 稳定性 CI。

- AV1：64/64 complete，0 `failure.json`，每 run 六件套齐全；root/run ledger、checkpoint、NPZ、manifest、metrics、diagnostics 全部由本地 AV2 复算哈希通过。
- AV0：RM3 36 + RM3-A 30 = 66 checkpoints，RM2 54/54 checkpoints，0 hash errors；`zero_training=true`。
- 所有 manifest/metrics/diagnostics 均为 `test_accessed=false`；selector/reporting UTC 日无重叠；Linux 未填写任何科学判决。
- 56 组预声明共享模块初始化比较全部 hash equal，`mismatch_count=0`。
- AV1 在 `75a0325` 执行，AV0 在 `0c90624` 执行。两个 executor 修复均保留原始失败日志并经 Supervisor 复核：本地 checkpoint 分支漏计数，以及 `int(0.8n)` 相对 `int(0.6n)+int(0.2n)` 的一行取整越界。二者不修改候选、fold、loss、seed、selector 或 AV1 权重。

权威机器证据为：

- `results/phase3_5/ms3r_rm3av2/supervisor_evidence_validation.json`
- `results/phase3_5/ms3r_rm3av2/supervisor_decision_validation.json`
- `results/phase3_5/ms3r_rm3av0/supervisor_replay_validation.json`

## 2. 最重要的架构结论

### 2.1 P5 的末温优势主要来自 bypass，不是已闭合的物理链

在共同 selector、module-scoped 初始化和相同 4000 updates 下，P5 anchor C27 的末温 MAE 为 0.9930°C，优于 P4 C26 的 1.0369°C，但局部 `Tin-Tout` MAE 为 2.0454°C，明显差于 C26 的 1.6451°C。

- C04 去掉 P5 bypass：末温平均退化 **+0.2027°C**。
- C05 只保留 bypass terminal path：末温仅 **+0.0048°C**，几乎保留全部末温表现。
- C06 从训练和推理关闭 response：末温仅 **+0.0060°C**。
- 冻结旧 P5 checkpoint 的 AV0 干预同样显示：bypass-off 平均 `+0.3362°C`，response-off 仅 `+0.0301°C`。

因此 P5 仍可作为高容量预测/末温 anchor，但不得称“显式喷水响应贯穿末温的物理冠军”。旁路保持 action-invariant 是必要安全合同，却不能替代 response path 的贡献证明。

### 2.2 free/action proxy 确实存在，但简单删 free 会制造伪物理

完整 history 对未来阀位的 blocked OOF 增量 R² 为 F0 `0.624`、F1 `0.698`，而 SP-only 和 PI-feature probes 为负，说明历史状态包含很强的 observed-policy proxy。

free small/large 会在预测 MAE几乎不变时改变显式响应幅值；C09 action shield 相对 C26：

- H60 response magnitude `+0.1074°C`；
- explicit/local-change ratio `+0.0889`；
- wrong-side 与 lead 均在两个 folds 明显恶化，说明正确时序/侧别开始被模型使用；
- 但 terminal/local MAE 分别退化 `+0.0152/+0.0621°C`。

这支持“分解不唯一、history 可代理动作”，也说明不能通过删除 free 强迫 response 承担全部扰动。RM3-B 应保留受控 residual，并把 action shield 作为独立候选而非默认真理。

### 2.3 响应监督能放大显式通道，但尚未把它变成有效末温链

C10 logged auxiliary 与 C11 integrated OOF R-loss 在 P4 上均把 H60 response 提高约 `0.165–0.179°C`，训练图有 7 个非零梯度 tensor、one-update 参数变化非零，且 forecast path 不读取 logged future valve。

但其 response-off 对末温的影响仍只有约 `0.0002–0.0004°C`，lead placebo 也没有同步改善。因此 Q01 判为 MIXED：旧结构锚点删除是响应塌缩因素之一，但不是唯一根因；监督增大幅值不等于物理贡献已经传到末温。

### 2.4 阀位预测仍过度平滑

真实未来阀位平均 `|Δv|≈0.31–0.32` point，当前 anchors 只有约 `0.05–0.08`；真实跨度约 `9.25` points，预测约 `3.1–3.8`。绝对阀位 MAE优于 persistence 并不代表动态已恢复。

- C14 `Δv + multiscale roughness`：valve MAE `-0.0906`，但动态幅值没有闭合。
- C15 structured PI-only：valve MAE `+0.4770`、预测 `|Δv|≈0.0096`，明确拒绝。
- C16 PI+GRU residual：valve MAE `-0.1041`，预测 `|Δv|≈0.093`，但 terminal `+0.0220°C`，保留为工程候选。

### 2.5 双阀数据有二维激励，但 full cross-gain 没有得到额外支持

残差化 innovation effective rank 均为 2，differential energy fraction 为 F0 `0.238`、F1 `0.368`，因此本批不能再简单说“只能辨识 common mode”。但 action innovation 相关仍为 `0.525/0.281`，且闭环 action-local residual covariance 非零，不能升级为 `do(valve)`。

C17 common-only、C18 diagonal 与 C26 full 的四任务 MAE几乎相同；diagonal 与 full 的 H60 response 也近乎一致。结论是：数据存在 differential excitation，但没有证据支付 full cross-gain 的额外自由度。RM3-B 优先使用现场交叉关系校正后的 diagonal explicit channel，full MIMO 只作敏感性对照。

### 2.6 三极点、纯迟延和早期负增益均未被识别

AV0 blocked OOS mean RMSE 在 linear/power/one/two/three-pole/dead-time 之间仅约 `0.1557–0.1567°C`，split wins 分散；AV1 C19–C23 的 terminal/local 差异约为千分之一度量级。三极点或 dead-time 不能称真实阶次。

早期 signed common response 在多个模型中先负后正，最佳 action alignment 在 folds/shape 间移动到 `-20/-30 s`，且 lead/placebo 结果不干净。Q15 判 MIXED：timing、闭环反馈和真实快速项仍无法分离。

### 2.7 4000 updates 不足以作稳定架构排名

三条 8000-update anchors 相对对应 4000-update anchors：

| Anchor | terminal ΔMAE | local ΔMAE | valve ΔMAE |
|---|---:|---:|---:|
| C28 vs C25 (P3) | -0.0414 | -0.0436 | -0.0631 |
| C29 vs C26 (P4) | -0.0382 | -0.0243 | -0.0251 |
| C30 vs C27 (P5) | -0.0346 | -0.1455 | -0.1081 |

8000 updates 下 C28/C29/C30 的 terminal MAE 分别为 `0.9976/0.9987/0.9584°C`；P5 仍是 terminal 较低、local 较差的不同 Pareto 点，而非综合冠军。

### 2.8 真正递推仍失败，当前不是状态闭合世界模型

C31 的 teacher-forced second window 仍有正 terminal skill，但 recursive second window 已跌至 persistence 以下：F0 terminal/local/valve/Tin skill 为 `-0.034/-0.303/-0.326/-0.150`。该候选只支持 1200 s declared-context rollout；1800/3600 s 未实现。当前结论仍是 `state_closed_simulator=false`。

## 3. Q01–Q33 四态判决

| Q | Verdict | Supervisor 结论 |
|---|---|---|
| Q01 | MIXED | auxiliary/OOF 可放大响应，但未证明旧 penalty 删除是主因，也未闭合末温贡献 |
| Q02 | SUPPORTED | 阀位 decoder 明显过平滑；PI-only 失败，PI+GRU/动态损失仅部分改善 |
| Q03 | SUPPORTED | AV0 恢复的干预能发现旧 RM3 报告遗漏的 bypass/response 失活 |
| Q04 | SUPPORTED | P5 terminal 优势由 bypass 主导，P3/P4 local residual 位置影响 local Pareto |
| Q05 | SUPPORTED | 关闭 response 基本保留 P5 terminal，动作链未成为主要末温通路 |
| Q06 | SUPPORTED | P5 是 terminal/local 权衡点，不是综合架构结论 |
| Q07 | SUPPORTED | 必须使用共同报告目标，禁止跨不同 selector 分数排名 |
| Q08 | SUPPORTED | 4000-update 比较混入未完成优化；8000 三条均改善 |
| Q09 | SUPPORTED | unconstrained response 未稳定恢复唯一符号，sign 只能作工程先验 |
| Q10 | SUPPORTED | history action proxy 强；容量与 shield 改变响应而 MAE近似/权衡 |
| Q11 | SUPPORTED | module-scoped 初始化闭合 56/56；旧 RNG 比较确有混淆风险 |
| Q12 | SUPPORTED | shape family 无稳定可分性，不能宣称三阶/迟延已识别 |
| Q13 | SUPPORTED | calibration 已从冻结 NPZ 重算并保留 provenance，未覆盖历史文件 |
| Q14 | SUPPORTED | diagonal/common/full 预测近似，full cross gain 无额外证据 |
| Q15 | MIXED | timing、feedback 与快速项不能由现有 lead/shift 唯一区分 |
| Q16 | SUPPORTED | A/B 误差在 fold 间反转，pooled 数字会掩盖侧别问题 |
| Q17 | SUPPORTED | baseline 显式响应/local-change 比仅约 2–7%，比值门必须配绝对幅值 |
| Q18 | SUPPORTED | action shield/placebo 与容量证据表明单一独立性诊断不能证明唯一分解 |
| Q19 | MIXED | context/activity 回归是描述性的，不能唯一解释 fold 增益互换 |
| Q20 | SUPPORTED | per-side innovation rank 必须实算；本批 rank=2，但不等于因果识别 |
| Q21 | SUPPORTED | NNLS/shape fit 只是 trajectory diagnostic，不是真实物理锚点 |
| Q22 | SUPPORTED | frozen ablation 能定位 bypass/response；retrained 结果显示补偿效应仍需报告 |
| Q23 | SUPPORTED | 旧 P5 不是 OOF-calibrated；新 OOF loss 已验证训练图可达但科学收益混合 |
| Q24 | SUPPORTED | 配置 flag 不能替代 rank；已用 innovation covariance/effective rank 重算 |
| Q25 | SUPPORTED | 闭环内生性、代理测量和 placebo 不因非平稳/KCI 自动消失 |
| Q26 | SUPPORTED | P2 future-SP 在 H6/H18/H60 对 first-target hold skill 均为负 |
| Q27 | SUPPORTED | teacher-forced loss 与 recursive rollout 明显不同，旧 rollout 命名过强 |
| Q28 | SUPPORTED | time/domain index 与 measured context 必须分开；context 不自动外生 |
| Q29 | SUPPORTED | 相关非平稳/latent 定理关键假设均 unmet 或 not-testable |
| Q30 | SUPPORTED | 同采样与早期负增益不能证明 IDOL 式瞬时因果边 |
| Q31 | SUPPORTED | 阀位代理/部分观测不自动满足 CaRiNG 识别条件 |
| Q32 | SUPPORTED | side mechanism residual correlation 约 0.26–0.29，独立噪声未成立 |
| Q33 | SUPPORTED | H60 latent 不等于状态闭合；递推第二窗已失败，30–60 min 不可检验 |

汇总：`SUPPORTED=30 / MIXED=3 / REFUTED=0 / NOT_TESTABLE=0`。这里的 SUPPORTED 表示“审计问题得到数据支持”，不是模型或因果识别 PASS。

## 4. RM3-B 输入清单

RM3-B 只允许进入设计，不授权 Linux 训练。三个 prediction anchors 为 C28/P3、C29/P4、C30/P5 的 module-scoped 8000-update 版本，分别代表高容量预测、显式响应无 bypass、joint latent + action-invariant bypass 三种不同 Pareto 角色。

允许进入成对组合设计的模块：

1. action-shielded residual；
2. integrated OOF R-loss（只称 response calibration candidate）；
3. `Δv + multiscale roughness`；
4. PI+GRU residual valve decoder；
5. 现场点位对齐后的 diagonal explicit response；
6. one-pole 主候选 + linear-ramp 敏感性；
7. 容量受控、对动作严格不变的 terminal bypass。

禁止直接进入主张：PI-only、三极点/迟延已识别、full-MIMO 优越、sign 作为数据证据、任意 `do(valve)`、30–60 min 状态闭合仿真。

RM3-B 不得把上述模块一次性全叠加。每个组合必须相对 C28/C29/C30 做 paired ablation；先要求两个 folds 同方向，再进入多 seed confirmation。selector 仍为共同四任务目标，terminal/local/Tin/valve 与 response-off/wrong/lead 分开报告，不设 composite champion。

## 5. 最终声明边界

当前证据支持：observed-policy prediction、扰动条件响应诊断、支持域内的小幅 action sensitivity，以及“高容量预测 backbone + 显式受约束局部响应接口”的工程方向。

当前证据不支持：完全 plant identification、任意策略 `do(valve)`、独立 A/B 喷水物理增益、三阶/纯迟延真实结构、状态闭合世界模型、30–60 min open-loop simulation 或论文最终结论。
