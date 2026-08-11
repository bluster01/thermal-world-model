# Phase 3.5 MS3-R Gate B Supervisor 审计

## 结论

Gate B 按冻结主门通过，但只通过到“短时局部条件 MIMO”这一层：

```text
GATE_B_FROZEN_PRIMARY_PASS /
SHORT_HORIZON_CONDITIONAL_LOCAL_MIMO_SUPPORTED /
SP_IV_NOT_SUPPORTED_AS_CAUSAL_ROUTE /
UPSTREAM_PLACEBO_NONZERO /
TERMINAL_SIDE_SPECIFICITY_FAIL
```

这允许本地继续设计 Gate C 的 measured-boundary latent MIMO 模型和消融，不允许把现有结果写成开环 plant identification、喷水流量物理增益、任意 `do(valve)`、末温侧别因果关系或有效 SP 工具变量。Linux 授权在本审计后关闭；Gate C 暂时只允许本地设计。

## 执行与产物审计

- Linux 结果提交：`0b6d948`；实际执行代码：`32ede073...`；随后注册表授权提交：`ef1489f...`。
- Linux 在未拉取授权状态提交时依据用户指示执行，构成授权竞态。`32ede073` 与 `ef1489f` 之间 Gate-B 科学代码和配置差异为 0，后者只改 registry、TODO、README 和状态测试，因此本批记为 `ACCEPT_WITH_PROCESS_ADVISORY`，不要求重跑。
- 配置列出的 11/11 产物齐全；ledger 对 10 个非 ledger 产物给出哈希。所有科学 JSON 和 43.66 MB NPZ 哈希精确。
- Windows checkout 将 `stdout.log` 与 `resource_usage.txt` 转为 CRLF，工作树字节哈希因而不同；两者的 Git LF blob 与 Linux ledger 精确一致，不是运行后篡改。
- exit=0、stderr 为空、`/usr/bin/time -v` wall time 4.51 s、峰值 RSS 1.315 GiB、无 swap；test 未访问、未训练模型、科学判决仍为 `null`。
- 辅助 `environment.txt` 来自 Python 3.13.9，而 timed runner manifest 是 Python 3.11.15/NumPy 2.3.5。实际 runner 可由 manifest 和 time command 定位，但该环境文件不能作为完整执行环境记录；后续批必须用运行同一解释器生成环境清单。

## 独立重放与统计单位

审计只读取 `replay_arrays_validation.npz`，没有读取 cache 或重跑实验。逐日 2×2 矩阵最大复算误差 `8.88e-16`，配对量最大误差 `4.44e-16`，冻结 bootstrap 区间逐位一致。

分析覆盖 103,408 个 cross-fit 行、4 个 rolling folds、24 个 UTC 日；日内行数 539–8,212，中位数 4,757。逐日输入 Gram 条件数 1.19–6.92，中位数 1.54，不存在以日为单位的明显双输入秩亏。`n=24` 指同一机组连续 validation 时段中的 UTC 日，不是 24 台机组或独立 plant replication。

## 冻结主门

主门直接构造逐日配对差，没有用“两条独立 CI 是否重叠”推断差异。两个 family 分别对 A/B 两侧使用 simultaneous 97.5% bootstrap interval 控制 familywise alpha=0.05。

| Family | Side | 正值日 | 日中位数 | simultaneous 97.5% interval | 判决 |
|---|---:|---:|---:|---:|---|
| correct local − \|wrong side\| | A | 22/24 | 0.5149 | [0.4266, 0.6453] | PASS |
| correct local − \|wrong side\| | B | 24/24 | 0.3950 | [0.3033, 0.4521] | PASS |
| positive lag − \|lead\| | A | 22/24 | 0.5950 | [0.5478, 0.7216] | PASS |
| positive lag − \|lead\| | B | 24/24 | 0.4590 | [0.4020, 0.5111] | PASS |

A 有两个日期方向相反，因此 PASS 是“UTC 日中位路径”的结论，不是每个日期不变。作为非阻断审计敏感性，2 日和 3 日 circular-block bootstrap 的四个 97.5% 下界仍分别至少为 0.3033 和 0.3078；leave-one-day-out 中位数也全部为正。它增强稳健性描述，但不是追加确认性主门。

## 2×2 局部条件响应

行是 residualized A/B 阀位创新，列是 A/B `Tin−Tout` 变化。日中位矩阵为：

```text
H60  [[0.5508, 0.1076],
      [0.0677, 0.4183]]

H180 [[0.7721, 0.1343],
      [0.0686, 0.4869]]
```

H60/H180 的 lead 对角均接近 0。负荷、基线阀位、煤量/负荷三个 tertile 和四个 rolling fold 内对角方向均保持为正；A 对角的 fold 中位数范围为 0.4068–0.6665，B 为 0.4395–0.4961。opening/closing 的点估计存在幅值差异，但没有冻结逐日配对区间，只能作为阀位非线性/滞环的模型设计线索。

这些矩阵是历史、SP 与工况残差化后的闭环条件系数。喷水流量没有可靠真值，不能把系数直接解释为喷水质量流量增益或开环传递函数。

## 未闭合的 placebo 与下游链路

上游 Tin placebo 仍非零：H60 对角为 A=0.1027、B=0.0599，H180 为 A=0.0509、B=0.0437。它们小于正确局部路径且 lead 很小，但说明闭环/未测扰动没有完全消除。

末温侧别归因继续失败。H600 冷却方向下的正确对角只有 A=0.0261、B=0.0252，而错侧路径为 A→B 0.3556、B→A 0.2446。H600 的 A 局部 future 对角也接近 0，而 lead 为 0.1998。因此 H300/H600、末温和侧别下游约束继续只作诊断；Gate C 不得给末温建立未经支持的硬侧别通道。

## SP-IV 路线

SP innovation 对阀位 innovation 的 partial R² 只有 A=0.0141、B=0.0040；A 与另一阀共动作相关为 −0.217，B 为 −0.090。A 的正确 2SLS 97.5% interval 跨 0；B 虽为正，但错侧 2SLS 日中位数 1.3051 反而大于正确侧 1.0553。非零 first stage 不等于排除限制或外生性成立。

因此本批明确关闭“把历史 SP innovation 当作有效工具变量来升级 `do(valve)`”的路线。SP 仍可作为闭环世界模型的观测输入或 controller action，不可作为已验证外生 instrument。

## Gate C 设计约束

下一步只放行本地模型设计，核心应是 measured-boundary latent MIMO：

1. 显式短时局部 2×2 阀位条件响应通道，同时保留 off-diagonal 和 common/differential 模态；
2. `Tin` 作为 measured disturbance boundary，未来 rollout 必须由上游模型预测或由场景显式给定；
3. 未测燃烧、流量、混合与金属蓄热合并为稳定 latent block，不把末温侧别写成硬约束；
4. 比较 residual/free 容量、局部中间监督、additive 与 context-scheduled response；
5. selector 同时约束局部响应、placebo、总预测和 rollout，不能只看末温 MAE；
6. 保持 observed-policy prediction 和支持域内小反事实声明，MS4 与任意策略闭环仍冻结。

机器可读审计见 `results/phase3_5/ms3r_gateb_point_closure/supervisor_audit_validation.json`，cache-free 重放见同目录 `supervisor_replay_validation.json`。
