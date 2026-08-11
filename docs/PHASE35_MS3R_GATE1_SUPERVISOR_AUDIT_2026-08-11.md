# Phase 3.5 MS3-R Gate A Supervisor Audit

> 审计日期：2026-08-11
>
> 远端结果：`57d3be8`，补交 replay arrays：`2132a3c`
>
> 执行代码：`13170981749b17c92b6a440716f4651f4587eee9`
>
> 最终标签：**GATE_A_CONDITIONAL_PASS**
>
> 下一状态：只放行本地 Gate-B 设计；无 Linux 授权，不访问 test，不启动 Gate C/MS4。

## 1. 结论

Gate A 排除了最简单的“双阀完全同步、输入矩阵秩亏”解释，但没有证明 A/B 两条独立 plant 通道已经因果辨识。

- A/B 标准化阀位创新相关系数 `0.1765`，条件数 `1.4288`；common/differential 能量为 `0.5883/0.4117`，比值 `0.6999`。60/180/600 s block-Hankel 谱没有出现简单数值塌缩。因此，**dual-input algebraic rank supported**。
- 正确侧局部 `Tin-Tout` 在 60/180 s 的逐日中位系数明显大于错侧：A 约 `3.6×/4.4×`，B 约 `3.5×/3.3×`。这支持把局部点位作为 Gate-B 的主监督。
- 但错侧局部路径和上游 Tin placebo 在短时程也不为零；A 侧 300/600 s action-lead 诊断还出现非零区间。现有产物没有保存逐日 correct-minus-placebo 配对差，不能用两组分开的 CI 声称“正确路径显著强于 placebo”。
- 末温侧归因失败。300/600 s 的错侧末温关联对 A、B 两个动作都大于正确侧关联，不能把 Gate-A 结果写成双侧独立末温 plant gain，也不能给末温施加硬 side-specific response 约束。

因此 Gate A 是**条件通过**：允许设计 measured-boundary latent MIMO，并比较 common-only 与 common+differential；所有动作响应必须经过 `Tin-Tout/Tout` 中间监督。禁止升级为独立 A/B plant identification、`do(valve)` 或完全物理世界模型。

## 2. 执行与产物审计

Linux 使用冻结命令，`exit=0`，报告 wall-clock `3 s`；`test_accessed=false`、`training_executed=false`、`automatic_scientific_pass=null`。补交后 8/8 必需产物齐全，ledger 中七个科学文件和 NPZ 的 SHA256 全部闭合。

NPZ 包含 `115,190` 个候选 anchor，其中四个 rolling folds 共评估 `103,408` 个样本，覆盖 24 个 UTC 日。输入秩复算最大标量误差 `2.22e-16`，Hankel 奇异值最大误差 `1.55e-15`。

执行合同仍有一个 P1 缺口：未记录 peak RSS。进程已经结束，按单 attempt 合同不重跑；后续 batch 必须用 `/usr/bin/time -v` 固定记录。

## 3. Statistics review scope

- 独立统计单位：UTC 日；10 s 样本只用于模型拟合和残差构造，不作为独立 `n`。
- 报告支持：24 个 UTC 日、4 个过去训练/未来评估 rolling folds。
- 路径面板：2 sides × 8 paths × 4 horizons × 3 timing variants = 192 个诊断量。
- 多重性：未作 correction。该面板只能作为预设诊断地图，不能把每个 bootstrap CI 当独立确认性检验。
- Gate-B 必须冻结少量 primary contrasts，并直接计算逐日配对 `correct − wrong-side` 与 `correct − upstream`，不能用“一组显著、另一组不显著”代替差异检验。

## 4. Major statistical issues

### P0：末温侧归因不成立

A 动作在 H300/H600 的正确侧逐日中位为 `0.0983/0.0371`，错侧为 `0.3315/0.3849`；B 动作为 `0.0460/0.0540` 对 `0.2209/0.2554`。错侧关联更强，说明末温仍被公共燃烧扰动、热惯性和闭环动作污染。

**处理：** Gate-B 的 primary plant evidence 固定在 `Tin-Tout/Tout`；末温只作 downstream rollout，不设侧别硬增益门。

### P0：尚无 correct-vs-placebo 直接推断

正确局部路径的点估计确实更大，但错侧和上游 placebo 也有非零区间。当前 JSON 只保存各路径日中位与 bootstrap CI，没有逐日路径差，因此不能检验 interaction/paired difference。

**处理：** Gate-B 预注册两项逐日配对主对比；其他路径和时程降为 secondary diagnostics，并冻结 multiplicity family。

### P1：秩健康不等于外生或唯一可辨识

阀位历史/SP 对未来阀位差分的 cross-fit R² 为 A `0.0930`、B `0.2114`；剩余创新标准差为 `0.7007/0.7994%`。这说明剩余变化存在，但残差化模型可能错配，且闭环扰动仍可同时影响创新和温度。

**处理：** 只称 algebraic rank；Gate-B 继续 residual-capacity × excitation 分层和 IV feasibility，不称 open-loop plant。

### P1：长时 action-lead 污染

A 侧局部路径在 300/600 s 的 lead 诊断不再接近零，表明慢时程受自相关、闭环反馈或工况漂移影响。

**处理：** Gate-B 局部响应主时程先冻结 60/180 s；300/600 s 作为传播与稳健性诊断，不作为局部因果主门。

## 5. Gate-B 设计约束

Gate-B 只允许本地设计以下内容：

1. measured-boundary latent MIMO，显式监督双侧 `Tin-Tout/Tout`；
2. common-only 对 common+differential 消融，不能默认 differential 就是独立 plant gain；
3. residual capacity `small/base/large` × additive/context-scheduled response；
4. terminal-only 对 local+terminal 监督；
5. 逐日配对 correct-minus-placebo 主对比；
6. 日期、负荷、方向和燃烧工况不变性；
7. SP-IV 只作 feasibility，要求 first-stage、lead 和同步动作排除。

Gate-B 设计、代码和本地测试全部完成前，不授权新的 Linux batch。
