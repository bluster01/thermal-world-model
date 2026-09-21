# 物理状态三臂回传：响应更明显，精度目标尚未达到

Linux 回传提交 `09e340d`，训练代码提交 `db29660`。本轮使用原定 1/3 数据、seed 11；只有这一个优化种子，以下是探索性结果。P0/P1/P2 全部完成 60 轮、每臂 9600 次更新，产生 6 行评价，没有训练失败。当前需要补齐的是“是否达到预设验证停滞条件”的证据，而不是补跑崩溃丢失的模型。

来源：[运行状态](../../results/action_predictor_bench_20260919/physical33_seed11/state.json)、[配置及数据/代码指纹](../../results/action_predictor_bench_20260919/physical33_seed11/config.json)、[三臂训练状态](../../results/action_predictor_bench_20260919/physical33_seed11/CONVERGENCE.md)。各臂的更新次数来自下方 `fit.json`。

## 先说明三臂在比较什么

三臂共用“5 个流体温度＋3 个金属温度”的热状态递推，温度监督都覆盖 H128。P0 只训练温度预测；P1 增加 EMA 教师的未来状态匹配；P2 增加两个观察视角的状态交叉预测。P1/P2 的金属状态目标都是模型估计，并非实测金属温度。实现与目标定义见 [physical_models.py](physical_models.py) 和 [PHYSICAL.md](PHYSICAL.md)。

每臂都保留 short、balanced 两种选模结果。下文报告集精度统一用 short checkpoint，不按报告集结果另挑权重。H32/H128/H512 分别表示向前 32/128/512 个采样步。

## 60 轮之后还需不需要训练

下面的改善，是“前 40 轮内最佳 selector”与“前 60 轮内最佳 selector”的比较，不是拿某一轮的随机波动当趋势。selector 是训练期间选模用的数据，和后面的报告集精度表分开。

| 臂 | short selector：前40轮→前60轮 | balanced selector：前40轮→前60轮 | 第60轮后学习率 | 当前学习率下 stale |
|---|---:|---:|---:|---:|
| P0 | 0.759774→0.752722，改善0.93% | 1.128695→1.121935，改善0.60% | 0.000125 | 1 |
| P1 | 0.772100→0.768154，改善0.51% | 1.135980→1.128120，改善0.69% | 0.000100 | 5 |
| P2 | 0.791927→0.788070，改善0.49% | 1.150649→1.150649，无改善 | 0.000100 | 5 |

单位为 °C。来源：各臂逐轮 [P0 日志](../../results/action_predictor_bench_20260919/physical33_seed11/fits/seed11/P0/training.jsonl)、[P1 日志](../../results/action_predictor_bench_20260919/physical33_seed11/fits/seed11/P1/training.jsonl)、[P2 日志](../../results/action_predictor_bench_20260919/physical33_seed11/fits/seed11/P2/training.jsonl)。停止原因、更新次数与最优轮分别见 [P0 fit](../../results/action_predictor_bench_20260919/physical33_seed11/fits/seed11/P0/fit.json)、[P1 fit](../../results/action_predictor_bench_20260919/physical33_seed11/fits/seed11/P1/fit.json)、[P2 fit](../../results/action_predictor_bench_20260919/physical33_seed11/fits/seed11/P2/fit.json)。

P0 的 balanced 最优刚好在第 60 轮，仍有接续价值。P1 的 short/balanced 最优在第 55/59 轮；P2 则在第 49/40 轮。P1/P2 都已经降至最低学习率，并有 5 轮没有达到调度器要求的实质改善。这里的“实质改善”门槛为 0.002°C，因此刷新一个更小幅度的最佳值，并不一定会清零 stale。

三臂停止原因均为 `budget_limit`，不是 `validation_plateau`。建议将总上限从 60 提至 **90 轮**，三臂全部从各自第 60 轮接续，保留原 Adam、学习率、最佳权重和 stale。保持原来的最低学习率 0.0001、停滞耐心 6 轮；如果没有新的实质改善，P1/P2 可能在下一轮就正常停止。不要重置 stale，也不要求三臂强行多跑相同轮数。新目录保留这次回传，随后重新评价续训选出的权重。

这些训练规则来自 [配置](../../results/action_predictor_bench_20260919/physical33_seed11/config.json) 与 [advance_budget 实现](full_baselines.py)。90 轮是本次续训预算建议，不是已执行的实验。

## 精度目前处于什么位置

| 模型（short checkpoint） | 主温 H32 MAE | 主温 block H128 MAE | 主温 block H512 MAE | 主温 native H512 MAE |
|---|---:|---:|---:|---:|
| P0：热状态＋温度监督 | 0.7050 | 1.2539 | 2.6757 | 2.6757 |
| P1：P0＋EMA 状态目标 | 0.7152 | 1.2485 | 2.5742 | 2.5742 |
| P2：P0＋双视角状态目标 | 0.7281 | 1.2562 | 2.5928 | 2.5928 |
| B：SSM＋11路历史 | 0.5172 | 0.9789 | 1.4747 | 1.7959 |
| D：背景SSM＋冻结R4响应＋11路历史 | 0.5425 | 1.0509 | 1.5740 | 1.7827 |

单位为 °C。来源：[本轮 summary](../../results/action_predictor_bench_20260919/physical33_seed11/summary.json)、[上轮 focused summary](../../results/action_predictor_bench_20260919/focused33_seed11/summary.json)。

P1 相比同架构 P0，H512 MAE 降低约 3.8%，H32 增加约 1.4%；P2 分别降低约 3.1%、增加约 3.3%。目前看到的是有限的长短期权衡，还没有出现“状态预测辅助目标同时改善精度和响应”的证据。

与精度参照 B 相比，P0 的 H32 MAE 高约 36.3%，native H512 高约 49.0%。这些差距需要正视，但不能把尚未达到停滞条件的模型直接定性失败，也不能预先保证续训会消除差距。B/D 采用 H32 温度监督，P 系列采用 H128 监督；这是同一评价体系下的实用比较，不是只改变架构的严格消融。B/D 的训练设定见 [FOCUSED.md](FOCUSED.md)。

**本轮 P 系列的 block 与 native 实际使用完全相同的连续物理状态递推，没有分块重新编码。两列相同不是两份独立验证。** 原自动生成 `DIAGNOSIS.md` 末尾有关“两种不同协议”的通用描述不适用于 P 系列，具体应以本轮配置中的 `block_protocol` 为准。

## 动作响应的进展和仍未解决的部分

动作对齐评价包含 110 个方案，覆盖阶跃、斜坡、脉冲、双脉冲、正弦和平滑动作，每个方案最多使用 8 个窗口。每次动作之后都观察相同的 128 步；越界窗口会按原规则排除。

| short checkpoint | 方向不符比例的情景均值 | 阀1开大3pp：主温末端变化 | 阀2开大3pp：主温末端变化 |
|---|---:|---:|---:|
| P0 | 0.5200% | −0.3119°C | −0.5027°C |
| P1 | 0.5139% | −0.3058°C | −0.4971°C |
| P2 | 0.5072% | −0.3058°C | −0.5066°C |
| D | 0% | −0.0552°C | −0.0730°C |

这里的方向指标取 native 下 80 个具有该指标的单阀方案，先按各方案的可达温度、窗口和时间计算，再对方案求平均；不包含正弦等不能直接套用恒定方向的动作。它不是“真实干预预测错误率”。末端变化使用 `step_at+0_dose+0.03_v1_0` / `step_at+0_dose+0.03_v0_1`，两种方案均有 8 个有效窗口，单位 pp 表示阀门开度百分点。

来源：动作对齐指标 [P0](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P0_short/aligned_response_metrics.json)、[P1](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P1_short/aligned_response_metrics.json)、[P2](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P2_short/aligned_response_metrics.json)、[D](../../results/action_predictor_bench_20260919/focused33_seed11/seed11/D_short/aligned_response_metrics.json)；指标定义见 [response_summary](evaluation.py)。

三臂在这些方案中的提前响应、不可达上游响应均为 0，近零响应比例为 0。P0 两个阀的上述末端响应幅度分别约为 D 的 5.65 倍、6.89 倍，说明这套结构保留了明显的动作敏感度，而非让响应趋零来满足方向。但没有真实干预轨迹作标签，幅度更大仍不能解释为幅度更准确。

同时已经看到局部反向：P0 在“阀1开大3pp”的方案中，第 4 个温度最多升高约 **0.01875°C**，该方案主温没有反向。数值来自 [P0 aligned_responses.npz](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P0_short/aligned_responses.npz) 中对应 case 的 native 数组。因此，目前可以说路径约束已生效、上述主温响应幅度比 D 更大，不能说热量平衡结构已经保证所有温度的全部瞬态单调。

## 一个值得接着验证的假设

三臂第一喷水节点的 `capacity / nominal_flow` 都学到约 **1400 秒**，而初始化为 **20 秒**；主温节点约为 **729–902 秒**。这些只是常比热代理模型中的有效惯性，不能当成辨识出的真实设备热容。它提示一种可能：模型正在通过很强的平滑惯性，补偿尚未表达出来的热源扰动或动态过程。

依据是 short checkpoint 的物理参数 [P0](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P0_short/physical_parameters.json)、[P1](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P1_short/physical_parameters.json)、[P2](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P2_short/physical_parameters.json)，按 [physical_models.py](physical_models.py) 中训练均值 `nominal_flow` 计算；初始化也在同一文件中。与这一假设相容的现象是，本轮主温 60 秒变化相关约 **0.16**，B 约 **0.72**，来源为上述两个 `summary.json` 的 `delta60_correlation`。这是待检验解释，不是已经确认的原因。

P2 的估计状态一致性 MSE 为 **0.407**，P0 为 **0.751**，来源分别为 [P2 状态诊断](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P2_short/state_diagnostics.json)、[P0 状态诊断](../../results/action_predictor_bench_20260919/physical33_seed11/seed11/P0_short/state_diagnostics.json)。但不同臂的目标金属状态也随各自观察器变化，而且诊断只使用前 8 个报告窗口。这个数值降低不能证明金属状态更真实，更不能证明共因扰动已经被识别或消除。

当前先完成三臂原条件续训。若停滞后仍保留明显精度差距，下一步优先拆分验证热源/扰动动态表达与有效惯性的作用，再判断状态预测辅助目标能否提供额外收益；保留 B 的精度参照与 D 的方向参照，暂不提前裁掉 P0/P1/P2。
