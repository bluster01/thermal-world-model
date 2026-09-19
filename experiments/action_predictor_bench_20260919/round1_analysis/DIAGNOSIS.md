# Returned baseline diagnosis

One optimization seed; exploratory validation results. Main-temperature MAE in C.

| Model | Mode | H32 | H128 | H512 | Active H32 | Quiet H32 | Upstream max | Wrong-sign scenario mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ait | block | 0.9439 | 1.6658 | 1.9917 | 1.1306 | 0.8817 | 3.2454 | 0.3670 |
| ait | native | 0.9439 | 2.5641 | 14.3070 | 1.1306 | 0.8817 | 10.0036 | 0.3671 |
| anchored_hold | block | 0.9330 | 1.5445 | 2.0916 | 1.2139 | 0.8393 | 3.2235 | 0.1464 |
| anchored_hold | native | 0.9330 | 1.5126 | 2.0198 | 1.2139 | 0.8393 | 0.0000 | 0.0000 |
| attention_concat | block | 0.8606 | 1.4781 | 1.8497 | 1.0454 | 0.7991 | 3.8154 | 0.3626 |
| attention_concat | native | 0.8606 | 1.8317 | 3.6744 | 1.0454 | 0.7991 | 3.5648 | 0.3753 |
| direct | block | 0.8398 | 1.4401 | 1.9864 | 1.0260 | 0.7778 | 7.7415 | 0.4532 |
| direct_no_action | block | 0.8577 | 1.4456 | 1.9475 | 1.0830 | 0.7826 | 1.1357 | 0.2009 |
| gru | block | 0.9697 | 1.5860 | 2.0702 | 1.1699 | 0.9030 | 2.9296 | 0.3281 |
| gru | native | 0.9697 | 1.5609 | 2.0519 | 1.1699 | 0.9030 | 2.9973 | 0.3082 |
| persistence | block | 1.0750 | 1.5879 | 2.0520 | 1.5564 | 0.9146 | 0.0000 | 0.0000 |
| persistence | native | 1.0750 | 1.5879 | 2.0520 | 1.5564 | 0.9146 | 0.0000 | 0.0000 |
| r4 | block | 0.9464 | 1.5358 | 1.7625 | 1.1525 | 0.8777 | 3.2235 | 0.1464 |
| r4 | native | 0.9464 | 1.5199 | 1.9454 | 1.1525 | 0.8777 | 0.0000 | 0.0000 |
| r4_directref | block | 0.8773 | 1.4726 | 1.9909 | 1.1027 | 0.8021 | 1.1111 | 0.1318 |
| r4_mlp | block | 0.8750 | 1.4281 | 1.7120 | 1.0416 | 0.8195 | 3.1093 | 0.1263 |
| r4_mlp | native | 0.8750 | 1.4532 | 1.9097 | 1.0416 | 0.8195 | 0.0000 | 0.0000 |
| ssm | block | 0.8232 | 1.4468 | 1.8091 | 0.9501 | 0.7809 | 3.2323 | 0.4423 |
| ssm | native | 0.8232 | 1.3929 | 1.8746 | 0.9501 | 0.7809 | 1.9421 | 0.5480 |

Active = top quartile of mean absolute recorded valve displacement over H32 (64 windows).
Wrong-sign mean includes only single-valve step/pulse/ramp/double-pulse cases with the metric defined.
Response amplitude is a model sensitivity, not intervention accuracy. Zero response alone cannot qualify.
Native and block are different simulation protocols; protected models explicitly retain carrier state in block mode.
