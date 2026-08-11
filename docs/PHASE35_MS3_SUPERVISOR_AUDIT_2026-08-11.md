# Phase 3.5-MS3 Supervisor Audit

> 审计日期：2026-08-11
> 最终标签：`AUDITED / OBSERVATIONAL_VALIDATION_FAIL_ASYMMETRIC / NO_RETRY / MS4_HOLD`

## 1. Supervisor 判决

MS3 整体未通过。B 回路（B 阀→左温）3/3 seeds 通过冻结的 observational validation 门；A 回路（A 阀→右温）0/3 通过，核心失败是 response non-collapse：动态窗口平均绝对响应仅 `0.00663–0.00854°C`，低于预注册 `0.02°C` 下限。A seed 1 的 shuffled-action CI 和 seed 2 的两个 action-alignment CI 也未过零。

该结果不得通过降低 A 门槛、补 seed、重训或只保留 B 侧来“修正”。冻结决策要求双侧均至少 2/3 seeds 通过，因此：

- MS3 记为科学失败，不打开真实模型 test；
- B 侧只获得开发集上的条件预测与动作时间对齐证据，不是 `do(valve)`；
- A 侧不能声称动作响应已经被当前架构稳定识别；
- 正式 MS4 继续 HOLD，先做不训练的 A/B asymmetry diagnosis。

## 2. 产物与重放审计

| 项目 | 结果 |
|---|---:|
| Linux 执行代码 | `798fcde0b15c6cea89983f6e4d5f6080c6e57d87` |
| Linux 结果提交 | `597180f` |
| 完整 runs | 12/12 |
| checkpoint archive | 12/12，SHA-256 `93ff25c186208156e435d7ebc585000b112c6fb819549b989cd3d5a777943837` |
| manifest / matrix / cache timeline | 全部闭合；A阀→右温、B阀→左温未漂移 |
| anchor / trajectory hash | 12/12 逐字节一致 |
| checkpoint CPU 重放 | PASS；aggregate metric 最大绝对差 `1.1623e-5` |
| episode CPU 重放 | PASS；单窗口最大绝对差 `8.4734e-4°C`，冻结 CI 下界最大漂移 `<9.5e-8°C` |
| structural diagnostics | 12/12 PASS；free-only effect 精确为 0 |
| test artifacts / test access | 无 / `false` |

重放脚本为 `experiments/phase3_5/audit_ms3_real_adaptation.py`，机器结果为 `results/phase3_5/ms3_real_adaptation/supervisor_replay_validation.json`。远端为 aarch64 CUDA、审计为 x86 CPU，因此逐窗口浮点值不要求逐位一致；anchor、trajectory bytes、shuffle design 和所有冻结判决保持一致。

## 3. 冻结主结果

`baseline CI` 是保持 anchor 时刻阀位与 logged valve 的动态 UTC-day 配对差；`shuffled CI` 是 bin 内置换 delta-path 与 logged valve 的配对差。正值表示 logged future valve 的误差更低。

| side | seed | logged MAE (°C) | dynamic mean abs effect (°C) | baseline 95% CI (°C) | shuffled 95% CI (°C) | checkpoint +5% H60 median (°C) | median response grad |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 1.1550 | 0.00712 | [0.000096, 0.000848] | [0.000184, 0.000988] | −0.0286 | 0.000850 |
| A | 1 | 1.1588 | 0.00854 | [0.000312, 0.001231] | [−0.000008, 0.001049] | −0.0344 | 0.000982 |
| A | 2 | 1.1792 | 0.00663 | [−0.000361, 0.000493] | [−0.000118, 0.000813] | −0.0327 | 0.000814 |
| B | 0 | 1.1623 | 0.04565 | [0.001780, 0.007937] | [0.000815, 0.007876] | −0.1382 | 0.006157 |
| B | 1 | 1.1592 | 0.04289 | [0.002050, 0.007790] | [0.003867, 0.010883] | −0.1528 | 0.005371 |
| B | 2 | 1.1508 | 0.04851 | [0.001328, 0.009836] | [0.005400, 0.014895] | −0.1514 | 0.005672 |

预测非劣与动态 support 两侧全部通过：joint/free MAE ratio 为 A `0.9991–1.0128`、B `0.9915–1.0023`；每 run 有约 `7,929–8,034` 个动态窗口、27 个 UTC 日。它说明 failure 不是没有动作样本，也不是加入 response 后总预测明显恶化。

## 4. A/B 不对称诊断

### 4.1 不是简单的动作剂量不足

A/B 同 seed 使用完全相同的 validation anchor。B 的动态阀位剂量中位数只比 A 高 `5.2%–5.9%`，但 B 的动态平均绝对温度效应是 A 的 `5.03–7.32` 倍。因此不能把不对称简单归因于 B 的阀门动作更大。

### 4.2 不对称已经写入 checkpoint 的标准化响应

在相同 checkpoint、相同 history、相同 `+5%` 阀位阶跃下，A 的 H60 温降中位数仅 `−0.0286~−0.0344°C`，B 为 `−0.1382~−0.1528°C`。B 的 scheduled gain 中位数约 `−0.0523~−0.0618°C/effective-%`，A 为 `−0.0206~−0.0249`；B 的后两个极点也总体更快，尤其第三极点中位数约 `289–349 s`，A 为 `465–488 s`。

训练期 response gradient 的中位数同样分离：B 为 `0.00537–0.00616`，A 为 `0.00081–0.00098`。三个 seed 均复现这一方向，所以它不像单一 seed 的坏初始化；更像 A 数据中 total-loss 对动作分支提供的识别梯度明显较弱。

这仍不能区分三种机制：A 真实闭环增益更弱、A 阀位→有效喷水曲线/执行器状态不同、或串级反馈与 free head 使 A 动作信号更难识别。仅凭 observational total loss 不可判定哪一种成立。

### 4.3 B 的阳性稳定，但工程量级仍小

B 的 logged-action 平均改善约 `0.0044–0.0100°C`，相对约 `1.15°C` 的总体 MAE 只是小比例增益。以 1、2、3、5 个连续 UTC 日作 circular block bootstrap，B 的两种 action contrast、三个 seed 的 95% CI 下界仍全部大于 0；最低下界约 `0.000349°C`。这说明 B 的方向并非由单日驱动，但不等于工程上已经足够用于仿真或控制。

A seed 0 有很小的正向条件预测增益；seed 1 的 shuffled contrast 对 block length 敏感，seed 2 两个 contrast 均跨零。无论这些近零 CI 如何波动，A 的 response non-collapse 仍以约 2.3–3.0 倍距离低于 `0.02°C` 门槛，因此整体失败不依赖零附近的数值舍入。

## 5. 统计与协议边界

1. 主推断单位是 UTC 日，不是约 8,000 个重叠窗口；窗口仅在日内聚合。有效顶层 `n=27` days。
2. 三个 seed 不是独立实验单位。不同 seed 的 validation anchor 仅约 `7.37%–7.71%` 重叠，因此 seed 同时混合优化波动和窗口子抽样波动；不能把 `n=3 seeds` 当作统计重复。
3. checkpoint 在 validation selector 子集上选取，最终 8,192-anchor 评估仍属于同一 validation 时段；CI 未包含模型选择不确定性，也不是独立 test。
4. UTC-day bootstrap 仍假设日块可交换。连续 2–5 日 block 的后验稳健性分析只作诊断，不改写冻结 Gate。
5. 两个 action contrast、双侧和多 seed 均为预注册 conjunction Gate；本轮不做事后阈值选择。论文中仍应完整报告所有 contrasts，而不能只报告 B 阳性。
6. shuffled delta-path 在极少数 singleton baseline bins 中不可避免保留原轨迹：每 run `0–2/8192` 个 fixed points，比例不超过 `0.0244%`。这与“完全无 fixed point”的文字描述不一致，但数量不足以解释 A/B 结论；后续协议应明确排除 singleton 或合并稀疏 bins。

## 6. 下一步：MS3-D asymmetry diagnosis（本地、无训练）

不重跑 MS3 v1.1，也不直接启动正式 MS4。下一步只冻结一个数据/模型对齐诊断：

1. 在负荷、主汽压力、处理前主汽温稳定的 SP held-step 中，分别估计 A/B 的 `SP→实际阀位→末过温度` 经验响应；动态工况只作分层描述，不与稳态事件混成主估计。
2. 使用相同实际阀位剂量归一化和相同 horizon，把经验 A/B 响应与本次 checkpoint 的标准化 `±5%` IRF 对齐。
3. 若经验 A 响应同样明显弱于 B，则优先解释为 side-specific plant/actuator scale，后续 MS4 必须使用 side-specific empirical margin；不能事后把 MS3 改判 PASS。
4. 若经验 A 响应与 B 接近而模型仍弱，则判当前 total-loss/selector 存在 A-side response absorption，另立新协议比较 response-aware selector、分段 system identification 或监督闭环链；这将是新实验，不是 MS3 retry。

在该诊断冻结并审计前，Linux 无新训练任务，真实 test、MS4、模型选择和论文继续 HOLD。
