# 全部基线：两种选模规则分别展示

9个训练家族＋persistence＋anchored_hold。两种选择复用同一次训练，不是重复拟合。
所有家族的balanced均为0.5×selector H32＋0.5×selector block H128；不使用native指标选模。
native列的—表示固定输出头不支持该模式；anchored_hold的native是Direct分块＋R4响应连续的混合模式。
是否观察到训练停滞请看CONVERGENCE.md，不把完整运行自动称为充分收敛。

## short

| Seed | 模型 | H32 | block H128 | block H512 | native H512 |
|---|---|---:|---:|---:|---:|
| 11 | ait_short | 0.5906 | 1.2349 | 2.2756 | 39.6461 |
| 11 | anchored_hold_short | 0.6676 | 1.3036 | 1.8565 | 1.8210 |
| 11 | attention_concat_short | 0.5880 | 1.1720 | 1.8377 | 4.8998 |
| 11 | direct_no_action_short | 0.6309 | 1.2750 | 1.8523 | — |
| 11 | direct_short | 0.6148 | 1.2293 | 1.8211 | — |
| 11 | gru_short | 0.5770 | 1.2067 | 1.7715 | 3.1595 |
| 11 | persistence | 1.0750 | 1.5879 | 2.0520 | 2.0520 |
| 11 | r4_directref_short | 0.6363 | 1.2712 | 1.8616 | — |
| 11 | r4_mlp_short | 0.5984 | 1.1305 | 1.6662 | 1.8156 |
| 11 | r4_short | 0.5999 | 1.1493 | 1.6374 | 1.7879 |
| 11 | ssm_short | 0.5725 | 1.1675 | 1.6945 | 1.8778 |

## balanced

| Seed | 模型 | H32 | block H128 | block H512 | native H512 |
|---|---|---:|---:|---:|---:|
| 11 | ait_balanced | 0.5911 | 1.3115 | 2.4602 | 36.7485 |
| 11 | anchored_hold_balanced | 0.6921 | 1.3415 | 1.9140 | 1.8710 |
| 11 | attention_concat_balanced | 0.6611 | 1.3412 | 2.0432 | 4.2787 |
| 11 | direct_balanced | 0.6324 | 1.2525 | 1.8315 | — |
| 11 | direct_no_action_balanced | 0.6500 | 1.2797 | 1.8416 | — |
| 11 | gru_balanced | 0.6265 | 1.3335 | 1.9456 | 3.2423 |
| 11 | persistence | 1.0750 | 1.5879 | 2.0520 | 2.0520 |
| 11 | r4_balanced | 0.5999 | 1.1493 | 1.6374 | 1.7879 |
| 11 | r4_directref_balanced | 0.6292 | 1.2619 | 1.8450 | — |
| 11 | r4_mlp_balanced | 0.6121 | 1.1447 | 1.6624 | 1.7870 |
| 11 | ssm_balanced | 0.6144 | 1.2418 | 1.8692 | 1.8277 |
