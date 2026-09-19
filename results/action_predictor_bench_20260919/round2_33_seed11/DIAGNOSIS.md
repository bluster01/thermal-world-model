# Returned baseline diagnosis

One optimization seed; exploratory validation results. Main-temperature MAE in C.

| Model | Mode | H32 | H128 | H512 | Active H32 | Quiet H32 | Upstream max | Wrong-sign scenario mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r4_protected | block | 0.7234 | 1.2614 | 1.7470 | 0.8911 | 0.6675 | 0.0000 | 0.0000 |
| r4_protected | native | 0.7234 | 1.2634 | 1.7775 | 0.8911 | 0.6675 | 0.0000 | 0.0000 |
| ssm_continue | block | 0.6225 | 1.2000 | 1.8205 | 0.7247 | 0.5884 | 2.9434 | 0.3601 |
| ssm_continue | native | 0.6225 | 1.2388 | 1.8490 | 0.7247 | 0.5884 | 3.5077 | 0.3904 |
| ssm_protected | block | 0.7878 | 1.3353 | 1.7724 | 1.0957 | 0.6852 | 0.0000 | 0.0000 |
| ssm_protected | native | 0.7878 | 1.3530 | 1.8600 | 1.0957 | 0.6852 | 0.0000 | 0.0000 |
| ssm_r4_joint | block | 0.6710 | 1.2416 | 1.8588 | 0.8194 | 0.6215 | 0.0000 | 0.0000 |
| ssm_r4_joint | native | 0.6710 | 1.2763 | 1.8661 | 0.8194 | 0.6215 | 0.0000 | 0.0000 |
| ssm_r4_teacher | block | 0.6594 | 1.2256 | 1.8486 | 0.8206 | 0.6057 | 0.0000 | 0.0000 |
| ssm_r4_teacher | native | 0.6594 | 1.2674 | 1.8674 | 0.8206 | 0.6057 | 0.0000 | 0.0000 |
| ssm_soft_response | block | 0.6544 | 1.2227 | 1.7040 | 0.8154 | 0.6008 | 2.6289 | 0.2003 |
| ssm_soft_response | native | 0.6544 | 1.2710 | 1.8454 | 0.8154 | 0.6008 | 0.4541 | 0.2621 |

Active = top quartile of mean absolute recorded valve displacement over H32 (64 windows).
Wrong-sign mean includes only single-valve step/pulse/ramp/double-pulse cases with the metric defined.
Response amplitude is a model sensitivity, not intervention accuracy. Zero response alone cannot qualify.
Native and block are different simulation protocols; protected models explicitly retain carrier state in block mode.
