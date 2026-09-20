# 全基线训练状态

validation_plateau=满足预设验证停滞规则；budget_limit=预算耗尽，未确认停滞。

| Seed | 模型 | 实际轮数 | 停止原因 | 最终学习率 | 短期最优轮 | 综合最优轮 |
|---|---|---:|---|---:|---:|---:|
| 11 | ait | 43 | validation_plateau | 0.0001 | 36 | 29 |
| 11 | attention_concat | 58 | validation_plateau | 0.0001 | 52 | 25 |
| 11 | direct | 35 | validation_plateau | 0.0001 | 13 | 12 |
| 11 | direct_no_action | 38 | validation_plateau | 0.0001 | 16 | 12 |
| 11 | gru | 49 | validation_plateau | 0.0001 | 48 | 25 |
| 11 | r4 | 38 | validation_plateau | 0.0001 | 20 | 20 |
| 11 | r4_directref | 33 | validation_plateau | 0.0001 | 25 | 19 |
| 11 | r4_mlp | 38 | validation_plateau | 0.0001 | 35 | 20 |
| 11 | ssm | 47 | validation_plateau | 0.0001 | 46 | 25 |
