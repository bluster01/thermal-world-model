# 当前主线：全部基线的较充分训练比较

本轮已回传并分析：9/9训练家族达到预设验证停滞条件，21行评价完成。见[FULL33_RETURN.md](FULL33_RETURN.md)的完整解读、动作响应问题和SSM通俗架构说明；原始两套选模结果均保留。

用户要求先看清全部基线、训练是否充分和模型结构，再决定路线。上一轮SSM支线不替代这项任务。本入口固定运行下面全部9个训练家族，不提供筛掉模型的`--models`参数。

| 家族 | 本仓库实现 | 参数量 |
|---|---|---:|
| direct_no_action | 历史patch编码＋固定32步输出，不读取未来阀位 | 98,832 |
| direct | 同结构，读取未来阀位 | 102,928 |
| gru | GRU历史记忆＋逐步温度增量 | 18,389 |
| ssm | GRU历史编码＋48维多时常状态更新＋起点相对读出 | 16,453 |
| attention_concat | 状态和动作特征合并的逐步attention | 13,765 |
| ait | 状态与动作交错的逐步attention | 13,765 |
| r4 | 参考预测＋有方向和路径约束的响应传播 | 9,642 |
| r4_mlp | R4历史编码器增加MLP适配器 | 18,117 |
| r4_directref | Direct参考分支与R4响应一起训练 | 108,474 |

另保留persistence和已拟合direct＋r4的anchored_hold，共11个概念上的基线。这里的Direct/AIT仍是此前快筛的轻量实现；不是原Phase1完整40维模型或完整CEDAR复现。本次不改变这些实现，不以换模型的方式改善排行榜。

## 训练与停止规则

- 同一1/3数据：20,371个训练起点；同一128个selector和256个报告窗口；默认seed11，全部从头初始化，保持原epoch样本顺序。
- 所有家族训练H32，batch128，原五温度加权归一化MSE，Adam初始lr=.001，梯度裁剪5。训练只读取前32步目标，未扩展某个家族的监督长度。
- 至少12epoch，上限60epoch。所有家族遵循同一规则，但允许在不同轮数达到停滞；这不是等FLOP实验。
- 每轮计算两种共同selector：`short=H32主汽MAE`；`balanced=.5×H32主汽MAE+.5×block H128主汽MAE`。Direct也能使用同一block指标，因此不把只有部分模型支持的native放进选模公式。
- 两种selector**都连续4轮没有累计超过0.002°C的改善**时，学习率减半，最低=.0001。任一种有改善就重置等待。
- 到最低学习率后，两种selector仍连续6轮无上述改善，且已经至少训练12轮，停止并标记`validation_plateau`。每次降低学习率都重新计算等待轮数。
- 上限60轮先到，则标记`budget_limit`，**不宣称已收敛**。回传后逐模型判断是否需要延长；保存last.pt中的模型、优化器和停止状态，方便后续延长预算。

raw MAE只要更低就保存checkpoint；0.002°C阈值只决定调整学习率/停止，不丢弃小幅改善的最佳权重。初始化epoch0也可保留。训练完成不等于理论收敛，`validation_plateau`是预先规定的可操作判据。

## 输出和评价

每次训练保留short/balanced两套checkpoint。9次训练产生18行评价，再加persistence一次、两套parent选择的anchored_hold，共21行。这些是11个基线在两种选模规则下的结果，不是21次训练。两种规则的anchored_hold均使用同规则选出的direct/r4权重。

- `ALL_BASELINES.md`：short/balanced分表，各自包含全部11个基线。
- `CONVERGENCE.md`：每个训练家族的实际轮数、停止原因、最终学习率、两个最佳轮次。
- `state.json`：`not_confirmed_plateau`明确列出尚未满足停滞判据的模型；失败也保留。
- 原有完整精度指标：H1/H6/H18/H32、block H128/H512、可用模型的native H128/H512、尾段MAE、温差相关性与幅度、峰谷误差。
- 原有110种动作方案：正负阶跃、连续动作、脉冲及释放、窗口前/早中晚、双阀、跨块时刻，另含剂量关系和梯度探针。

所有selector都只读取selector窗口。报告窗口不决定学习率、停止或checkpoint。H128扩展包仅用于提供相同起点的长selector标签；温度训练仍为H32。训练成本按fit_id去重，不把short/balanced两行成本重复相加。

## Linux执行

```bash
python -m pytest experiments/action_predictor_bench_20260919/test_bench.py experiments/action_predictor_bench_20260919/test_round2.py experiments/action_predictor_bench_20260919/test_round3.py experiments/action_predictor_bench_20260919/test_full_baselines.py -q
python -m experiments.action_predictor_bench_20260919.full_baselines --output results/action_predictor_bench_20260919/full33_seed11
```

默认CPU单线程，可指定`--device cuda`。这一轮会明显长于6epoch快筛，目的是给全部家族训练机会。先完整回传seed11的训练曲线与状态；若到顶仍改善，继续补训练，再做多seed稳定性复核。没有自动启动下一条SSM支线。

回传完整 `full33_seed11`。`--resume`要求代码、配置和数据一致，跳过已经完成的fit/评价；未完成fit从固定seed重训。last优化器状态为后续预算延长保留，本入口不假装提供未实现的逐batch断点恢复。

## 本次已收到的是哪批结果

后续设计输入见[侧窗口研究纪要](../../docs/SSM_PHYSICS_RESPONSE_DECONFOUNDING_IDEAS_20260920.md)：物理响应结构、参考工况调度与共因扰动分离。均为未实现、未验证的候选消融，不改变本轮全部家族的训练和评价协议，也未据此启动新的 SSM 支线。

`b1d933a`回传的是此前发布的round3支线：6次拟合，各追加3epoch，两个selector，共12行评价。它不是上述全部家族的充分训练结果。支线的发现已单列在[ROUND3_RETURN.md](ROUND3_RETURN.md)，保留有用结论，但不据此删减全基线名单。
