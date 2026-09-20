# Returned baseline diagnosis

One optimization seed; exploratory validation results. Main-temperature MAE in C.

| Model | Mode | H32 | H128 | H512 | Active H32 | Quiet H32 | Upstream max | Wrong-sign scenario mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ait | block | 0.7621 | 1.4350 | 2.0457 | 0.9279 | 0.7069 | 3.0279 | 0.3129 |
| ait | native | 0.7621 | 2.8944 | 26.9983 | 0.9279 | 0.7069 | 10.5835 | 0.3941 |
| anchored_hold | block | 0.7040 | 1.3619 | 1.9401 | 0.8916 | 0.6415 | 2.5263 | 0.1026 |
| anchored_hold | native | 0.7040 | 1.3290 | 1.8500 | 0.8916 | 0.6415 | 0.0000 | 0.0000 |
| attention_concat | block | 0.7680 | 1.5202 | 2.2674 | 0.8946 | 0.7258 | 2.7113 | 0.4115 |
| attention_concat | native | 0.7680 | 2.7441 | 8.7962 | 0.8946 | 0.7258 | 5.7325 | 0.3958 |
| direct | block | 0.6539 | 1.2627 | 1.8002 | 0.7597 | 0.6186 | 7.8288 | 0.3933 |
| direct_no_action | block | 0.6722 | 1.2894 | 1.8212 | 0.8115 | 0.6258 | 1.6947 | 0.1594 |
| gru | block | 0.6899 | 1.2961 | 1.7959 | 0.8199 | 0.6465 | 3.5917 | 0.3443 |
| gru | native | 0.6899 | 1.4618 | 2.4223 | 0.8199 | 0.6465 | 3.2489 | 0.3963 |
| persistence | block | 1.0750 | 1.5879 | 2.0520 | 1.5564 | 0.9146 | 0.0000 | 0.0000 |
| persistence | native | 1.0750 | 1.5879 | 2.0520 | 1.5564 | 0.9146 | 0.0000 | 0.0000 |
| r4 | block | 0.7265 | 1.3586 | 1.9866 | 0.9086 | 0.6657 | 2.5263 | 0.1026 |
| r4 | native | 0.7265 | 1.3010 | 1.8367 | 0.9086 | 0.6657 | 0.0000 | 0.0000 |
| r4_directref | block | 0.6734 | 1.2848 | 1.8013 | 0.8479 | 0.6152 | 1.0944 | 0.1064 |
| r4_mlp | block | 0.7234 | 1.2575 | 1.7001 | 0.8911 | 0.6675 | 2.5883 | 0.1041 |
| r4_mlp | native | 0.7234 | 1.2634 | 1.7775 | 0.8911 | 0.6675 | 0.0000 | 0.0000 |
| ssm | block | 0.6438 | 1.1884 | 1.6029 | 0.7765 | 0.5996 | 2.0687 | 0.3957 |
| ssm | native | 0.6438 | 1.1979 | 1.7467 | 0.7765 | 0.5996 | 2.2157 | 0.4082 |

Active = top quartile of mean absolute recorded valve displacement over H32 (64 windows).
Wrong-sign mean includes only single-valve step/pulse/ramp/double-pulse cases with the metric defined.
Response amplitude is a model sensitivity, not intervention accuracy. Zero response alone cannot qualify.
Native and block are different simulation protocols; protected models explicitly retain carrier state in block mode.
