# 预测器 → 规划器/模拟器：快速基线实验

目的：在同一套数据与指标下看清预测精度、动态外推和动作响应各自的长短处，再组合有效思路。本轮是基线探索，不做论文收口、不上现场闭环。

## 当前执行：全部原始基线的较充分训练

按用户纠正，主线恢复到全部模型比较。**9个原始训练家族全部保留，另有persistence与anchored_hold。** 每个训练家族至少12轮、上限60轮，统一的学习率下降与验证停滞规则，明确列出到顶仍未确认停滞的模型。设计与完整名单见 [FULL_BASELINES.md](FULL_BASELINES.md)。

```bash
python -m pytest experiments/action_predictor_bench_20260919/test_bench.py experiments/action_predictor_bench_20260919/test_round2.py experiments/action_predictor_bench_20260919/test_round3.py experiments/action_predictor_bench_20260919/test_full_baselines.py -q
python -m experiments.action_predictor_bench_20260919.full_baselines --output results/action_predictor_bench_20260919/full33_seed11
```

同一1/3训练数据，所有家族从头训练H32；每家族同时保留short/balanced两个checkpoint。共9次训练、21行评价（每种选模规则均有完整11个概念基线，persistence共用）。优先回传 `ALL_BASELINES.md`、`CONVERGENCE.md` 及整个结果目录。不要自动追加SSM支线。

最新收到的`b1d933a`是此前已安排的SSM支线，6次拟合/12行评价，不是全基线充分训练。发现见 [ROUND3_RETURN.md](ROUND3_RETURN.md)。

## 已回传的SSM支线：H32/H128 × 保持/预测名义动作

以下保留历史round3命令与设计，不是当前默认任务。1/3基线和六个响应消融回传于`8aa9204`，后续round3已回传于`b1d933a`。原设计见 [RETURN33_AND_ROUND3.md](RETURN33_AND_ROUND3.md)。

```bash
python -m pytest experiments/action_predictor_bench_20260919/test_bench.py experiments/action_predictor_bench_20260919/test_round2.py experiments/action_predictor_bench_20260919/test_round3.py -q
python -m experiments.action_predictor_bench_20260919.round3 --output results/action_predictor_bench_20260919/round3_33_seed11
```

新入口默认读取已提交的H128延长包与1/3父权重。6个温度拟合各3epoch，加一次共享名义动作预测器预训练；每个温度拟合保留short/balanced两个checkpoint，共12行评价。训练起点仍是20,371个，报告集不变。回传整个 `round3_33_seed11`，优先看 `COMPARISON.md`、`DIAGNOSIS.md`、`policy/result.json`。

## 已完成：1/3 数据基线 + 响应消融

用户在首轮回传后要求将训练数据扩大到1/3。**当前默认包是 `screen_A_33pct.npz`**，已提交，无需Linux重新准备原始数据。完整的结果解读和下一轮实验说明见 [ROUND2.md](ROUND2.md)。

```bash
python -m pip install -r experiments/action_predictor_bench_20260919/requirements.txt
python -m pytest experiments/action_predictor_bench_20260919/test_bench.py experiments/action_predictor_bench_20260919/test_round2.py -q
# 9个基线全部在1/3包上重新训练，得到11行结果。
python -m experiments.action_predictor_bench_20260919.run --output results/action_predictor_bench_20260919/screen33_seed11
# 上一步成功后：2个免训练组合 + 4个有等预算对照的继续训练消融。
python -m experiments.action_predictor_bench_20260919.round2 --output results/action_predictor_bench_20260919/round2_33_seed11
```

20,371训练窗，包含原6,111窗；selector128、reporting256及归一化逐值一致。每epoch160次更新，6epoch共960次；不将相对1/10的变化解读成纯数据量效应。旧结果和数据包保留。

回传 `screen33_seed11` 与 `round2_33_seed11` 两个完整目录，优先读后者的 `COMPARISON.md` 和 `DIAGNOSIS.md`。如需重启完成部分，分别加 `--resume`，仍要求配置、代码和父checkpoint完全一致。

## 首轮1/10历史协议（已执行）

从仓库根目录执行。小数据包已随本分支提交，Linux 不需要 Windows 原始 CSV 或其他未提交的侧会话目录。

```bash
python -m pip install -r experiments/action_predictor_bench_20260919/requirements.txt
python -m pytest experiments/action_predictor_bench_20260919/test_bench.py -q
python -m experiments.action_predictor_bench_20260919.run \
  --output results/action_predictor_bench_20260919/smoke --smoke
python -m experiments.action_predictor_bench_20260919.run \
  --data experiments/action_predictor_bench_20260919/data/screen_A_10pct.npz \
  --output results/action_predictor_bench_20260919/screen_seed11
```

CPU 默认单线程、顺序执行，避免小模型并发争抢；可用 `--device cuda`，但不假定这种小循环模型在 GPU 更快。相同 seed 的所有模型使用相同 epoch 样本顺序。首轮默认单 seed11，6轮，每轮完整遍历6,111窗，batch128、48次更新/轮、共288更新/模型，无早停。每轮在固定 selector 上选 H32 主汽 MAE 最优 checkpoint。首轮9个拟合＋persistence＋无新增拟合的组合臂，共11行模型结果。

`--resume` 仅接受数据/代码/配置一致的目录，跳过完成模型；未完成模型从原 seed 重启，不宣称优化器断点续训。不要并发写同一个结果目录。

后续多seed复核示例（不是当前执行入口）：

```bash
python -m experiments.action_predictor_bench_20260919.run \
  --output results/action_predictor_bench_20260919/confirm \
  --models direct attention_concat ait r4 r4_mlp r4_directref --seeds 11 23 37
```

上述多seed示例尚未启用；当前只执行文档顶部的full_baselines入口。保留所有成功/失败模型，回传结果后讨论。

## 模型

| 名称 | 结构 | 本轮要了解什么 |
|---|---|---|
| persistence | 保持五路当前温度 | 最低复杂度基准、零响应参照 |
| direct_no_action | RevIN＋按变量patch编码＋直接H32；不读取未来阀位 | 未来动作是否值得使用；历史阀位仍保留 |
| direct | 同结构，额外读取整段未来阀位 | 强预测型黑箱，可能存在窗口内动作前缀泄漏 |
| gru | GRU历史编码＋逐步温度/动作/边界递推 | 通用非线性自回归黑箱 |
| ssm | 可学习多时常潜态松弛＋动作条件驱动 | 没有R4拓扑约束的潜态SSM |
| attention_concat | 状态、动作/边界特征融合为一个token | AIT的等参数融合对照 |
| ait | 状态/动作交错token＋一步更新＋H32自由rollout训练 | 借鉴CEDAR动作模态的紧凑基线 |
| r4 | 原GRU参考分支＋独立carrier传播＋正值路径读出 | 结构化响应基准 |
| r4_mlp | R4＋零初始化历史MLP适配器 | 用参考信息增强争取精度与响应兼得 |
| r4_directref | 直接多步参考预测＋R4传播/读出 | 放开参考预测能力能否改善曲线 |
| anchored_hold | 已训练direct的保持阀位预测＋R4候选/保持轨迹差 | 强预测轨迹与动作调整量的组合 |

这是新统一协议下的模型家族比较，不把旧实验数字直接搬来排名。Direct是Phase1思路的13维五输出轻量实现，并非40维M0原模型复现。AIT是一个32维、4头、单attention块、8个转移的有限token记忆模型，使用历史GRU上下文；不是完整CEDAR复现，没有电商文本事件模块。`attention_concat`与`ait`参数相同、可见8个转移，token数不同；其他模型记录参数/耗时，不声称等FLOP。R4+MLP也为本协议的轻量实现，不等同旧DTF实验的29k参数MLP。

全部拟合五路温度，统一训练损失：训练集尺度归一化的多步MSE，通道权重 `[.125,.125,.125,.125,.5]`；Adam lr=.001、梯度范数上限5。首轮不叠加JEPA、动态损失和响应正则，先看结构本身。所有递推模型也用H32自由rollout训练，不让AIT额外得到未来真温度teacher forcing。

## 数据与时间

- Side A，统一13维历史：五路温度、双阀位、六个边界。历史64步，每步10秒。
- 首轮从原61,114个合法stride80（800秒步幅）训练窗中，按时间分箱抽取6,111窗（1/10），覆盖整个训练时段。比例指窗口数量，不是互不重叠的原始时长比例；压缩包约19.5MB。
- 标准化沿用唯一训练行统计。selector128窗和reporting256窗来自原validation的不同时间段，中间留出完整长窗间隔；所有模型共用。
- 训练H32=320秒；评价H128=21分20秒、H512=85分20秒。当前轮的外推主要指**超训练时长外推**，不是新电厂、新工况或锁定extension测试。
- 保留原train/validation划分；未读取test/extension作拟合或评价。报告早/晚validation子组，便于看时间变化，但不称新独立测试。
- 左端动作/边界驱动下一时刻温度：历史止于t，输入u[t]/d[t]对应标签T[t+1]。
- 主表使用记录的未来阀位和边界条件；另列边界保持压力测试。不同条件不混排。

当前1/3包已提交。以下命令仅用于从原准备数据重新生成同一嵌套样本：

```bash
python -m experiments.action_predictor_bench_20260919.data \
  --source results/side_full_windows_20260913/v1 \
  --fraction 0.3333333333333333 --side A \
  --include-pack experiments/action_predictor_bench_20260919/data/screen_A_10pct.npz \
  --output /path/to/screen_A_33pct.npz
# run命令加 --data /path/to/screen_A_33pct.npz
```

## 精度评价：三把尺子

1. **多步MAE**：H1/H6/H18/H32累计、端点、逐步、五路分别计算；H32在第一次重编码之前，与原生预测一致。
2. **共同block rollout**：预测完整32步，将预测五温度＋对应右端阀位/边界写回64步历史，然后再预测；到H128/H512。绝不用未来真实温度刷新历史。参考与响应都重初始化，这是明确的对照协议。
3. **不中断native rollout/extrapolation**：GRU/SSM/attention/R4连续维持状态到H128/H512，不重启、不喂真温度；分列33–128和129–512尾段MAE。固定长度direct和r4_directref没有native H512，标为不适用，仍有共同block指标。

`anchored_hold`的native列是**混合方式**：direct名义轨迹仍按32步生成，R4动作差连续不中断；单独注明，不能写成direct也拥有原生长时状态。block列则重启两个组件，以观察响应保护在哪种调用方式下成立。

另报60/180/600秒温差MAE、温差相关性/幅度比、峰谷时刻误差、预测有限性。比较两类指标，寻找水平误差与变化形态同时改善的候选。

## 动作响应：110个方案 × 默认8个固定起点

全部固定同一初始历史、同一保持边界与保持阀位基准；保存真实输入剂量和完整两条预测之差。不是通过重训制造不同响应。

- 单阀正负1/3/6个百分点阶跃。
- 起始、中段、末段：评分窗口0/640/1120秒改变动作；另在评分窗口前240秒发生动作。
- 窗口前动作：从共同真实历史先推演320秒，两条轨迹经历各自动作；评分从第320秒开始。它是在模型中产生的前史，不是只篡改历史阀位而保持温度不变。
- 120秒脉冲及释放、双脉冲、160秒斜坡、正弦、两频平滑连续动作。
- 双阀同向、反向；block边界前10秒、边界时刻、边界后10秒起动。
- native与block分别做同一套方案；越0–1阀位边界的方案/起点不裁剪凑剂量，排除数明确记录。

输出：逐通道响应曲线、峰值/末值/面积、10%峰值到达时间、近零比例、预动作变化、不可达通道变化、方向、释放后尾部、block/native差异。配对统计剂量排序、正负对称、剂量线性、双阀叠加误差，始终用共同合法起点。它们揭示非线性和交互，不将线性/对称越强当作越好。响应探针临时使用float64、同批成对推演，避免550°C附近float32舍入被误读成微弱动作效果；训练与预测MAE仍用float32。

规划接口加一个float64自动微分/有限差分探针，报告模型能否对动作给出可用梯度。没有在本轮增加优化器搜动作或闭环控制任务。

特别值得看的现象：`direct_no_action`第一块不读未来阀位，但后续块历史含此前动作，所以长程差值可以非零；R4的native路径约束不保证在block重编码后仍成立。把现象保存出来，正是本轮了解不同预测器如何变成模拟器的目的。

## 组合臂的名义计划

`T(a;a0)=F(H,a0,d)+[R(H,a,d)-R(H,a0,d)]`，a0固定为最初历史末阀位保持计划，整次候选比较中不跟随候选重新选a0。a=a0时精确还原F的名义预测。

实际预测评价仍输入记录动作a并与真实温度比较；没有把未来实测动作偷设成a0来获得平凡的精度一致。组合臂是否保住记录轨迹的精度，仍由本轮MAE回答。它不新增拟合，参数量/训练成本计入F和R两部分。

## 看结果与回传

优先看 `REPORT.md`、`summary.csv`、`seed_summary.json`、`figures/comparison_mae.png`，再看各模型的固定六窗曲线和响应图。每个模型保存训练日志、best/last、原始预测/响应数组、响应指标与梯度探针。模型失败保留日志，不静默跳过。

Linux执行完成后，将整个结果目录打包回传（不用提交生成的大数组到Git）：

```bash
tar -czf action_predictor_screen_seed11.tar.gz \
  -C results/action_predictor_bench_20260919 screen_seed11
```

下一轮再从“预测更准、曲线更对、响应通路更可用”的组合里选方向，加动态损失、局部响应学习或更大样本。单种子首筛不做显著性结论。

## R4来源

`r4_base.py`与`r4_transport.py`保存2026-09-11侧会话的v3基础结构及R4 transport实现，来源分别为原`adhoc_latent_jepa_20260910/v3/dual_model.py`、`r4_transport/transport_model.py`。仅调整相对import；行为检查核对原实现与包装模型逐值一致，零初始化MLP也保持一致。其他历史实验代码和结果无需随本实验复制。
