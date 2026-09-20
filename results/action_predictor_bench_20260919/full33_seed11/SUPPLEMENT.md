# Full33 补充读数

从回传数组重算；仅描述性分析，不重新选模。精度单位 °C。

## 全五温度平均误差（short checkpoint）

| 模型 | 协议 | H32 | H512 |
|---|---|---:|---:|
| ait_short | block | 1.2217 | 3.8590 |
| ait_short | native | 1.2217 | 43.2364 |
| anchored_hold_short | block | 1.2880 | 3.8445 |
| anchored_hold_short | native | 1.2880 | 3.9803 |
| attention_concat_short | block | 1.2083 | 3.4301 |
| attention_concat_short | native | 1.2083 | 15.2879 |
| direct_no_action_short | block | 1.2260 | 3.6111 |
| direct_short | block | 1.1497 | 3.5281 |
| gru_short | block | 1.2426 | 3.3556 |
| gru_short | native | 1.2426 | 7.2608 |
| persistence | block | 1.7460 | 3.9753 |
| persistence | native | 1.7460 | 3.9753 |
| r4_directref_short | block | 1.1952 | 3.6389 |
| r4_mlp_short | block | 1.2657 | 3.3165 |
| r4_mlp_short | native | 1.2657 | 3.9835 |
| r4_short | block | 1.2832 | 3.4453 |
| r4_short | native | 1.2832 | 3.9367 |
| ssm_short | block | 1.1694 | 3.0511 |
| ssm_short | native | 1.1694 | 3.5459 |

## 响应按形状和动作发生位置分组（short checkpoint）

每行是同组方案等权平均，不把不同有效窗口数当成相同的统计样本量。
方向不符比例仅针对定义了固定符号指标的阶跃/脉冲/斜坡/双脉冲；连续正负摆动不套用该指标。
响应绝对幅度是敏感度，不是动作响应真值误差。窗口前=-24，早=0，中=64，晚=112，跨块=31/32/33。

| 模型 | 协议 | 分组 | 方案数 | 平均响应绝对幅度 | 方向不符均值 | 动作发生前最大变化（全通道） |
|---|---|---|---:|---:|---:|---:|
| ait_short | block | double_pulse | 2 | 0.14014 | 45.53% | 0.00000 |
| ait_short | block | pulse | 12 | 0.09971 | 26.84% | 0.00000 |
| ait_short | block | ramp | 12 | 0.08540 | 46.66% | 0.00000 |
| ait_short | block | sine | 12 | 0.13509 | — | 0.00000 |
| ait_short | block | smooth | 12 | 0.07470 | — | 0.00000 |
| ait_short | block | step | 60 | 0.17944 | 31.84% | 0.00000 |
| ait_short | block | onset=-24 | 12 | 0.25092 | 32.61% | 0.00000 |
| ait_short | block | onset=0 | 32 | 0.24244 | 21.82% | 0.00000 |
| ait_short | block | onset=31 | 2 | 0.20492 | 25.00% | 0.00000 |
| ait_short | block | onset=32 | 2 | 0.20300 | 19.69% | 0.00000 |
| ait_short | block | onset=33 | 2 | 0.15016 | 30.74% | 0.00000 |
| ait_short | block | onset=64 | 30 | 0.11299 | 17.67% | 0.00000 |
| ait_short | block | onset=112 | 30 | 0.01702 | 65.83% | 0.00000 |
| ait_short | native | double_pulse | 2 | 0.28670 | 16.54% | 0.00000 |
| ait_short | native | pulse | 12 | 0.13416 | 44.21% | 0.00000 |
| ait_short | native | ramp | 12 | 0.28719 | 51.92% | 0.00000 |
| ait_short | native | sine | 12 | 0.10932 | — | 0.00000 |
| ait_short | native | smooth | 12 | 0.08072 | — | 0.00000 |
| ait_short | native | step | 60 | 0.63815 | 42.28% | 0.00000 |
| ait_short | native | onset=-24 | 12 | 1.60961 | 22.56% | 0.00000 |
| ait_short | native | onset=0 | 32 | 0.64432 | 23.66% | 0.00000 |
| ait_short | native | onset=31 | 2 | 0.34296 | 26.55% | 0.00000 |
| ait_short | native | onset=32 | 2 | 0.32899 | 27.15% | 0.00000 |
| ait_short | native | onset=33 | 2 | 0.31563 | 27.63% | 0.00000 |
| ait_short | native | onset=64 | 30 | 0.12122 | 46.26% | 0.00000 |
| ait_short | native | onset=112 | 30 | 0.02179 | 79.52% | 0.00000 |
| anchored_hold_short | block | double_pulse | 2 | 0.11821 | 33.24% | 0.00000 |
| anchored_hold_short | block | pulse | 12 | 0.07463 | 10.07% | 0.00000 |
| anchored_hold_short | block | ramp | 12 | 0.02140 | 4.72% | 0.00000 |
| anchored_hold_short | block | sine | 12 | 0.06991 | — | 0.00000 |
| anchored_hold_short | block | smooth | 12 | 0.04211 | — | 0.00000 |
| anchored_hold_short | block | step | 60 | 0.09567 | 8.28% | 0.00000 |
| anchored_hold_short | block | onset=-24 | 12 | 0.09176 | 11.80% | 0.00000 |
| anchored_hold_short | block | onset=0 | 32 | 0.15032 | 13.01% | 0.00000 |
| anchored_hold_short | block | onset=31 | 2 | 0.12183 | 10.10% | 0.00000 |
| anchored_hold_short | block | onset=32 | 2 | 0.12317 | 10.77% | 0.00000 |
| anchored_hold_short | block | onset=33 | 2 | 0.03550 | 5.77% | 0.00000 |
| anchored_hold_short | block | onset=64 | 30 | 0.06655 | 10.49% | 0.00000 |
| anchored_hold_short | block | onset=112 | 30 | 0.00014 | 0.00% | 0.00000 |
| anchored_hold_short | native | double_pulse | 2 | 0.00623 | 0.00% | 0.00000 |
| anchored_hold_short | native | pulse | 12 | 0.00283 | 0.00% | 0.00000 |
| anchored_hold_short | native | ramp | 12 | 0.01118 | 0.00% | 0.00000 |
| anchored_hold_short | native | sine | 12 | 0.00148 | — | 0.00000 |
| anchored_hold_short | native | smooth | 12 | 0.00167 | — | 0.00000 |
| anchored_hold_short | native | step | 60 | 0.02249 | 0.00% | 0.00000 |
| anchored_hold_short | native | onset=-24 | 12 | 0.04934 | 0.00% | 0.00000 |
| anchored_hold_short | native | onset=0 | 32 | 0.02161 | 0.00% | 0.00000 |
| anchored_hold_short | native | onset=31 | 2 | 0.01942 | 0.00% | 0.00000 |
| anchored_hold_short | native | onset=32 | 2 | 0.01900 | 0.00% | 0.00000 |
| anchored_hold_short | native | onset=33 | 2 | 0.01858 | 0.00% | 0.00000 |
| anchored_hold_short | native | onset=64 | 30 | 0.00554 | 0.00% | 0.00000 |
| anchored_hold_short | native | onset=112 | 30 | 0.00014 | 0.00% | 0.00000 |
| attention_concat_short | block | double_pulse | 2 | 0.14292 | 50.90% | 0.00000 |
| attention_concat_short | block | pulse | 12 | 0.09772 | 31.86% | 0.00000 |
| attention_concat_short | block | ramp | 12 | 0.10606 | 60.20% | 0.00000 |
| attention_concat_short | block | sine | 12 | 0.15355 | — | 0.00000 |
| attention_concat_short | block | smooth | 12 | 0.07003 | — | 0.00000 |
| attention_concat_short | block | step | 60 | 0.20295 | 41.12% | 0.00000 |
| attention_concat_short | block | onset=-24 | 12 | 0.33770 | 49.55% | 0.00000 |
| attention_concat_short | block | onset=0 | 32 | 0.26739 | 28.78% | 0.00000 |
| attention_concat_short | block | onset=31 | 2 | 0.18605 | 31.28% | 0.00000 |
| attention_concat_short | block | onset=32 | 2 | 0.20074 | 27.28% | 0.00000 |
| attention_concat_short | block | onset=33 | 2 | 0.16421 | 33.57% | 0.00000 |
| attention_concat_short | block | onset=64 | 30 | 0.10896 | 25.69% | 0.00000 |
| attention_concat_short | block | onset=112 | 30 | 0.02037 | 75.06% | 0.00000 |
| attention_concat_short | native | double_pulse | 2 | 0.09529 | 35.72% | 0.00000 |
| attention_concat_short | native | pulse | 12 | 0.06479 | 60.97% | 0.00000 |
| attention_concat_short | native | ramp | 12 | 0.09714 | 62.59% | 0.00000 |
| attention_concat_short | native | sine | 12 | 0.06694 | — | 0.00000 |
| attention_concat_short | native | smooth | 12 | 0.04961 | — | 0.00000 |
| attention_concat_short | native | step | 60 | 0.20504 | 52.91% | 0.00000 |
| attention_concat_short | native | onset=-24 | 12 | 0.47081 | 36.59% | 0.00000 |
| attention_concat_short | native | onset=0 | 32 | 0.18850 | 38.58% | 0.00000 |
| attention_concat_short | native | onset=31 | 2 | 0.11706 | 45.54% | 0.00000 |
| attention_concat_short | native | onset=32 | 2 | 0.12466 | 46.89% | 0.00000 |
| attention_concat_short | native | onset=33 | 2 | 0.12262 | 46.18% | 0.00000 |
| attention_concat_short | native | onset=64 | 30 | 0.09743 | 60.35% | 0.00000 |
| attention_concat_short | native | onset=112 | 30 | 0.01672 | 81.97% | 0.00000 |
| direct_no_action_short | block | double_pulse | 2 | 0.20594 | 15.45% | 0.00000 |
| direct_no_action_short | block | pulse | 12 | 0.10115 | 13.80% | 0.00000 |
| direct_no_action_short | block | ramp | 12 | 0.03890 | 11.45% | 0.00000 |
| direct_no_action_short | block | sine | 12 | 0.04878 | — | 0.00000 |
| direct_no_action_short | block | smooth | 12 | 0.07552 | — | 0.00000 |
| direct_no_action_short | block | step | 60 | 0.13894 | 17.91% | 0.00000 |
| direct_no_action_short | block | onset=-24 | 12 | 0.09687 | 36.45% | 0.00000 |
| direct_no_action_short | block | onset=0 | 32 | 0.20914 | 20.16% | 0.00000 |
| direct_no_action_short | block | onset=31 | 2 | 0.54498 | 14.64% | 0.00000 |
| direct_no_action_short | block | onset=32 | 2 | 0.30820 | 14.21% | 0.00000 |
| direct_no_action_short | block | onset=33 | 2 | 0.04201 | 16.20% | 0.00000 |
| direct_no_action_short | block | onset=64 | 30 | 0.07583 | 16.51% | 0.00000 |
| direct_no_action_short | block | onset=112 | 30 | 0.00000 | 0.00% | 0.00000 |
| direct_short | block | double_pulse | 2 | 0.15604 | 43.02% | 0.00000 |
| direct_short | block | pulse | 12 | 0.13531 | 28.02% | 0.58978 |
| direct_short | block | ramp | 12 | 0.05712 | 63.01% | 0.26007 |
| direct_short | block | sine | 12 | 0.16958 | — | 0.35920 |
| direct_short | block | smooth | 12 | 0.08331 | — | 0.33802 |
| direct_short | block | step | 60 | 0.12065 | 37.27% | 3.54009 |
| direct_short | block | onset=-24 | 12 | 0.19967 | 38.12% | 3.54009 |
| direct_short | block | onset=0 | 32 | 0.23074 | 23.32% | 0.00000 |
| direct_short | block | onset=31 | 2 | 0.26214 | 15.80% | 0.66264 |
| direct_short | block | onset=32 | 2 | 0.19972 | 9.21% | 0.00000 |
| direct_short | block | onset=33 | 2 | 0.10108 | 26.79% | 0.15537 |
| direct_short | block | onset=64 | 30 | 0.05756 | 23.21% | 0.00000 |
| direct_short | block | onset=112 | 30 | 0.00874 | 82.63% | 0.73381 |
| gru_short | block | double_pulse | 2 | 0.11711 | 52.48% | 0.00000 |
| gru_short | block | pulse | 12 | 0.12596 | 27.18% | 0.00000 |
| gru_short | block | ramp | 12 | 0.11891 | 52.49% | 0.00000 |
| gru_short | block | sine | 12 | 0.13409 | — | 0.00000 |
| gru_short | block | smooth | 12 | 0.06255 | — | 0.00000 |
| gru_short | block | step | 60 | 0.23658 | 34.07% | 0.00000 |
| gru_short | block | onset=-24 | 12 | 0.41492 | 35.13% | 0.00000 |
| gru_short | block | onset=0 | 32 | 0.29062 | 22.18% | 0.00000 |
| gru_short | block | onset=31 | 2 | 0.24273 | 21.28% | 0.00000 |
| gru_short | block | onset=32 | 2 | 0.23188 | 19.66% | 0.00000 |
| gru_short | block | onset=33 | 2 | 0.18036 | 31.63% | 0.00000 |
| gru_short | block | onset=64 | 30 | 0.12059 | 19.66% | 0.00000 |
| gru_short | block | onset=112 | 30 | 0.01736 | 72.65% | 0.00000 |
| gru_short | native | double_pulse | 2 | 0.11692 | 32.02% | 0.00000 |
| gru_short | native | pulse | 12 | 0.06441 | 49.69% | 0.00000 |
| gru_short | native | ramp | 12 | 0.16177 | 54.67% | 0.00000 |
| gru_short | native | sine | 12 | 0.09079 | — | 0.00000 |
| gru_short | native | smooth | 12 | 0.06839 | — | 0.00000 |
| gru_short | native | step | 60 | 0.29410 | 44.06% | 0.00000 |
| gru_short | native | onset=-24 | 12 | 0.57699 | 28.53% | 0.00000 |
| gru_short | native | onset=0 | 32 | 0.31147 | 33.66% | 0.00000 |
| gru_short | native | onset=31 | 2 | 0.25674 | 36.89% | 0.00000 |
| gru_short | native | onset=32 | 2 | 0.25273 | 37.04% | 0.00000 |
| gru_short | native | onset=33 | 2 | 0.24873 | 37.27% | 0.00000 |
| gru_short | native | onset=64 | 30 | 0.11820 | 42.84% | 0.00000 |
| gru_short | native | onset=112 | 30 | 0.01837 | 76.67% | 0.00000 |
| persistence | block | double_pulse | 2 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | pulse | 12 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | ramp | 12 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | sine | 12 | 0.00000 | — | 0.00000 |
| persistence | block | smooth | 12 | 0.00000 | — | 0.00000 |
| persistence | block | step | 60 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | onset=-24 | 12 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | onset=0 | 32 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | onset=31 | 2 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | onset=32 | 2 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | onset=33 | 2 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | onset=64 | 30 | 0.00000 | 0.00% | 0.00000 |
| persistence | block | onset=112 | 30 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | double_pulse | 2 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | pulse | 12 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | ramp | 12 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | sine | 12 | 0.00000 | — | 0.00000 |
| persistence | native | smooth | 12 | 0.00000 | — | 0.00000 |
| persistence | native | step | 60 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | onset=-24 | 12 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | onset=0 | 32 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | onset=31 | 2 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | onset=32 | 2 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | onset=33 | 2 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | onset=64 | 30 | 0.00000 | 0.00% | 0.00000 |
| persistence | native | onset=112 | 30 | 0.00000 | 0.00% | 0.00000 |
| r4_directref_short | block | double_pulse | 2 | 0.26414 | 15.12% | 0.00000 |
| r4_directref_short | block | pulse | 12 | 0.07506 | 22.16% | 0.00000 |
| r4_directref_short | block | ramp | 12 | 0.03455 | 5.05% | 0.00000 |
| r4_directref_short | block | sine | 12 | 0.04104 | — | 0.00000 |
| r4_directref_short | block | smooth | 12 | 0.04994 | — | 0.00000 |
| r4_directref_short | block | step | 60 | 0.10196 | 11.38% | 0.00000 |
| r4_directref_short | block | onset=-24 | 12 | 0.10146 | 13.35% | 0.00000 |
| r4_directref_short | block | onset=0 | 32 | 0.16734 | 21.20% | 0.00000 |
| r4_directref_short | block | onset=31 | 2 | 0.25658 | 7.47% | 0.00000 |
| r4_directref_short | block | onset=32 | 2 | 0.24178 | 8.51% | 0.00000 |
| r4_directref_short | block | onset=33 | 2 | 0.05083 | 2.65% | 0.00000 |
| r4_directref_short | block | onset=64 | 30 | 0.04592 | 15.38% | 0.00000 |
| r4_directref_short | block | onset=112 | 30 | 0.00016 | 0.00% | 0.00000 |
| r4_mlp_short | block | double_pulse | 2 | 0.14075 | 27.75% | 0.00000 |
| r4_mlp_short | block | pulse | 12 | 0.07395 | 10.06% | 0.00000 |
| r4_mlp_short | block | ramp | 12 | 0.02601 | 6.02% | 0.00000 |
| r4_mlp_short | block | sine | 12 | 0.06430 | — | 0.00000 |
| r4_mlp_short | block | smooth | 12 | 0.03697 | — | 0.00000 |
| r4_mlp_short | block | step | 60 | 0.10331 | 8.48% | 0.00000 |
| r4_mlp_short | block | onset=-24 | 12 | 0.11044 | 15.04% | 0.00000 |
| r4_mlp_short | block | onset=0 | 32 | 0.15361 | 12.83% | 0.00000 |
| r4_mlp_short | block | onset=31 | 2 | 0.12781 | 9.79% | 0.00000 |
| r4_mlp_short | block | onset=32 | 2 | 0.12647 | 8.25% | 0.00000 |
| r4_mlp_short | block | onset=33 | 2 | 0.03700 | 6.76% | 0.00000 |
| r4_mlp_short | block | onset=64 | 30 | 0.06888 | 9.72% | 0.00000 |
| r4_mlp_short | block | onset=112 | 30 | 0.00016 | 0.00% | 0.00000 |
| r4_mlp_short | native | double_pulse | 2 | 0.00615 | 0.00% | 0.00000 |
| r4_mlp_short | native | pulse | 12 | 0.00270 | 0.00% | 0.00000 |
| r4_mlp_short | native | ramp | 12 | 0.01088 | 0.00% | 0.00000 |
| r4_mlp_short | native | sine | 12 | 0.00143 | — | 0.00000 |
| r4_mlp_short | native | smooth | 12 | 0.00159 | — | 0.00000 |
| r4_mlp_short | native | step | 60 | 0.02126 | 0.00% | 0.00000 |
| r4_mlp_short | native | onset=-24 | 12 | 0.04538 | 0.00% | 0.00000 |
| r4_mlp_short | native | onset=0 | 32 | 0.02055 | 0.00% | 0.00000 |
| r4_mlp_short | native | onset=31 | 2 | 0.01859 | 0.00% | 0.00000 |
| r4_mlp_short | native | onset=32 | 2 | 0.01821 | 0.00% | 0.00000 |
| r4_mlp_short | native | onset=33 | 2 | 0.01784 | 0.00% | 0.00000 |
| r4_mlp_short | native | onset=64 | 30 | 0.00568 | 0.00% | 0.00000 |
| r4_mlp_short | native | onset=112 | 30 | 0.00017 | 0.00% | 0.00000 |
| r4_short | block | double_pulse | 2 | 0.11821 | 33.24% | 0.00000 |
| r4_short | block | pulse | 12 | 0.07463 | 10.07% | 0.00000 |
| r4_short | block | ramp | 12 | 0.02140 | 4.72% | 0.00000 |
| r4_short | block | sine | 12 | 0.06991 | — | 0.00000 |
| r4_short | block | smooth | 12 | 0.04211 | — | 0.00000 |
| r4_short | block | step | 60 | 0.09567 | 8.28% | 0.00000 |
| r4_short | block | onset=-24 | 12 | 0.09176 | 11.80% | 0.00000 |
| r4_short | block | onset=0 | 32 | 0.15032 | 13.01% | 0.00000 |
| r4_short | block | onset=31 | 2 | 0.12183 | 10.10% | 0.00000 |
| r4_short | block | onset=32 | 2 | 0.12317 | 10.77% | 0.00000 |
| r4_short | block | onset=33 | 2 | 0.03550 | 5.77% | 0.00000 |
| r4_short | block | onset=64 | 30 | 0.06655 | 10.49% | 0.00000 |
| r4_short | block | onset=112 | 30 | 0.00014 | 0.00% | 0.00000 |
| r4_short | native | double_pulse | 2 | 0.00623 | 0.00% | 0.00000 |
| r4_short | native | pulse | 12 | 0.00283 | 0.00% | 0.00000 |
| r4_short | native | ramp | 12 | 0.01118 | 0.00% | 0.00000 |
| r4_short | native | sine | 12 | 0.00148 | — | 0.00000 |
| r4_short | native | smooth | 12 | 0.00167 | — | 0.00000 |
| r4_short | native | step | 60 | 0.02249 | 0.00% | 0.00000 |
| r4_short | native | onset=-24 | 12 | 0.04934 | 0.00% | 0.00000 |
| r4_short | native | onset=0 | 32 | 0.02161 | 0.00% | 0.00000 |
| r4_short | native | onset=31 | 2 | 0.01942 | 0.00% | 0.00000 |
| r4_short | native | onset=32 | 2 | 0.01900 | 0.00% | 0.00000 |
| r4_short | native | onset=33 | 2 | 0.01858 | 0.00% | 0.00000 |
| r4_short | native | onset=64 | 30 | 0.00554 | 0.00% | 0.00000 |
| r4_short | native | onset=112 | 30 | 0.00014 | 0.00% | 0.00000 |
| ssm_short | block | double_pulse | 2 | 0.13567 | 52.05% | 0.00000 |
| ssm_short | block | pulse | 12 | 0.09897 | 25.08% | 0.00000 |
| ssm_short | block | ramp | 12 | 0.10596 | 55.23% | 0.00000 |
| ssm_short | block | sine | 12 | 0.09803 | — | 0.00000 |
| ssm_short | block | smooth | 12 | 0.07301 | — | 0.00000 |
| ssm_short | block | step | 60 | 0.19078 | 34.47% | 0.00000 |
| ssm_short | block | onset=-24 | 12 | 0.27903 | 35.79% | 0.00000 |
| ssm_short | block | onset=0 | 32 | 0.24009 | 22.45% | 0.00000 |
| ssm_short | block | onset=31 | 2 | 0.17557 | 39.21% | 0.00000 |
| ssm_short | block | onset=32 | 2 | 0.20639 | 18.13% | 0.00000 |
| ssm_short | block | onset=33 | 2 | 0.16102 | 37.50% | 0.00000 |
| ssm_short | block | onset=64 | 30 | 0.11687 | 20.16% | 0.00000 |
| ssm_short | block | onset=112 | 30 | 0.02021 | 70.65% | 0.00000 |
| ssm_short | native | double_pulse | 2 | 0.08520 | 36.50% | 0.00000 |
| ssm_short | native | pulse | 12 | 0.06470 | 50.43% | 0.00000 |
| ssm_short | native | ramp | 12 | 0.10135 | 53.55% | 0.00000 |
| ssm_short | native | sine | 12 | 0.10962 | — | 0.00000 |
| ssm_short | native | smooth | 12 | 0.09876 | — | 0.00000 |
| ssm_short | native | step | 60 | 0.15731 | 40.42% | 0.00000 |
| ssm_short | native | onset=-24 | 12 | 0.23659 | 27.49% | 0.00000 |
| ssm_short | native | onset=0 | 32 | 0.19480 | 31.80% | 0.00000 |
| ssm_short | native | onset=31 | 2 | 0.16031 | 33.12% | 0.00000 |
| ssm_short | native | onset=32 | 2 | 0.15871 | 33.22% | 0.00000 |
| ssm_short | native | onset=33 | 2 | 0.15711 | 33.37% | 0.00000 |
| ssm_short | native | onset=64 | 30 | 0.11282 | 39.85% | 0.00000 |
| ssm_short | native | onset=112 | 30 | 0.02309 | 73.88% | 0.00000 |
