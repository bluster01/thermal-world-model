# 当前Linux入口：四臂历史信息 × 连续响应

Linux数据回传711abf5已接收。该提交提供旁路数据草稿，不是新训练结果。本轮继续执行[部分收敛设计](../../docs/plans/2026-09-20-response-focused-design.md)，不扩展模型名单。

## 数据接入后的两个决定

1. 原13维A侧任务的一级动作取A阀、二级动作取B阀，这是已有交叉接线口径（`analysis/world_model_plant/prepare_tenth.py`）。v0旁路的二级A设定不匹配第二动作，v1改用二级B设定。原计划中“A侧二级设定”的表述在此修正，不改变温度和动作原通道。v0留存。
2. `fuel_ctl`与原煤量相关系数约0.999874，v1包仍保存12路便于追溯，但四臂入口显式选11路，排除此项。分段恒定的设定值/负荷变化率不因相邻相等而自动判为坏点；燃料、负荷等高相关但语义不同的信号保留。

v1由相同SHA256的原始合并表重建，与v0原起点及其他11路数值逐元素一致；全部20,371/128/256个窗口可用，无删行。归一化只使用训练历史唯一时刻。额外输入仅在起点64步历史中存在，未来槽位填零且不读取；预测过程中原始历史摘要保持不变，不伪造未来观测。

## 四次训练，八行评价

| Arm | 背景 | 新增历史 | 动作路径 |
|---|---|---|---|
| A | full33 SSM-short继续训练 | 无 | 原自由路径 |
| B | 同一SSM初始权重 | 11路 | 原自由路径 |
| C | 同一SSM作为保持阀位的参考背景 | 无 | 冻结full33 R4-short，响应连续 |
| D | 同一SSM参考背景 | 11路 | 同一冻结R4，响应连续 |

历史适配器：11路历史各池化至8点，88→48→48小MLP，末层零初始化，输出以0.1×tanh注入SSM初始记忆。B/D初始化一致。块式推演只重编码原13维的模拟历史，保持初始旁路摘要；C/D背景只接参考动作及参考预测，候选响应不回流背景。R4响应状态从整个计划起点连续推演，不在块边界重置。

同一seed11/样本顺序，H32事实损失，batch128，Adam lr=0.0003（继续训练，区别于full33从头的0.001），降至0.0001，最低12/最高60轮；沿用验证停滞判据，预算到顶保留budget_limit。所有臂相同规则；short/balanced选模均保留。R4响应参数冻结但仍保留对候选动作的可微依赖；不增加蒸馏或响应惩罚。

## Linux执行

在`codex/action-predictor-bench`最新提交执行：

```bash
python -m pytest experiments/action_predictor_bench_20260919/test_bench.py experiments/action_predictor_bench_20260919/test_round2.py experiments/action_predictor_bench_20260919/test_round3.py experiments/action_predictor_bench_20260919/test_full_baselines.py experiments/action_predictor_bench_20260919/test_focused.py -q
python -m experiments.action_predictor_bench_20260919.focused --output results/action_predictor_bench_20260919/focused33_seed11
```

默认CPU单线程；可显式指定`--device cuda`。原始数据无需重建，v1包与两组父checkpoint均随分支提供。`--resume`只跳过完成的拟合/评价，要求配置、代码、数据、父权重一致；未完成拟合从共同父权重重来，不提供逐batch恢复。完整回传输出目录，包括权重、预测数组、两套动作数组和日志。

输出：`CONVERGENCE.md`、`summary.csv/json`、`DIAGNOSIS.md`、`state.json`和每臂short/balanced结果。失败写入`failure_<arm>.txt`并在state保留，不静默删臂。旧全基线结果保留在full33_seed11。

## 评价

- 复用全部原精度评价：主汽温/五温度、多horizon、block/native、尾段、温差及held-boundary等。
- 原110动作方案、固定评分窗口仍保存为`response_metrics.json` / `responses.npz`。
- 新增同样110方案，每次从动作起效后观察128步，保存`aligned_response_metrics.json` / `aligned_responses.npz`。各方案最大推演272步，旁路历史固定。动作测试沿用起点保持边界的条件情景，没有新喂未来测温。该数组时间轴以动作起效为零，原数组以计划起点为零，不可混用。
- 重点看B−A、C−A、D−B；不能只拿D与旧的六轮SSM比较。冻结响应预期让C/D保持结构性质，本轮不声称已学准幅度，主要问题是额外信息能否降低分解的精度代价。

## 本地验证

47个测试通过，包括原40个回归测试和7个新测试：零适配器逐值保留原预测；候选差异仅经连续响应、跨块及动作前隔离；未来旁路值不进入边界/标签；背景与适配器可训练而响应冻结；110长动作保留原计划前缀。

四臂真实数据微型通路测试完成：每臂16个训练窗口、1轮2更新，八行评价全部成功。优化后C/D响应权重与父权重逐值相同。此结果只验证通路，不是精度结论；没有本地正式训练。

另外用D初始模型、两个真实起点运行完整原110方案及全部110个动作对齐方案，两种协议分别共220条记录；全部成功，对齐数组每协议形状110×2×128×5，动作前与不可达通道响应最大均为0。这验证新评价通路覆盖全部方案，不是训练后效果结论。
