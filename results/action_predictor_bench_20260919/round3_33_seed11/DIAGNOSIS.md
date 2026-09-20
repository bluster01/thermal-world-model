# Returned baseline diagnosis

One optimization seed; exploratory validation results. Main-temperature MAE in C.

| Model | Mode | H32 | H128 | H512 | Active H32 | Quiet H32 | Upstream max | Wrong-sign scenario mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hold_h128_balanced | block | 0.7051 | 1.2465 | 1.6947 | 0.8889 | 0.6439 | 0.0000 | 0.0000 |
| hold_h128_balanced | native | 0.7051 | 1.1977 | 1.5912 | 0.8889 | 0.6439 | 0.0000 | 0.0000 |
| hold_h128_short | block | 0.7051 | 1.2465 | 1.6947 | 0.8889 | 0.6439 | 0.0000 | 0.0000 |
| hold_h128_short | native | 0.7051 | 1.1977 | 1.5912 | 0.8889 | 0.6439 | 0.0000 | 0.0000 |
| hold_h32_balanced | block | 0.6724 | 1.2253 | 1.7503 | 0.8364 | 0.6177 | 0.0000 | 0.0000 |
| hold_h32_balanced | native | 0.6724 | 1.2705 | 1.8461 | 0.8364 | 0.6177 | 0.0000 | 0.0000 |
| hold_h32_short | block | 0.6724 | 1.2253 | 1.7503 | 0.8364 | 0.6177 | 0.0000 | 0.0000 |
| hold_h32_short | native | 0.6724 | 1.2705 | 1.8461 | 0.8364 | 0.6177 | 0.0000 | 0.0000 |
| planned_h128_balanced | block | 0.7314 | 1.2577 | 1.7039 | 0.9339 | 0.6639 | 0.0000 | 0.0000 |
| planned_h128_balanced | native | 0.7314 | 1.2042 | 1.6066 | 0.9339 | 0.6639 | 0.0000 | 0.0000 |
| planned_h128_short | block | 0.7176 | 1.2353 | 1.6508 | 0.9215 | 0.6496 | 0.0000 | 0.0000 |
| planned_h128_short | native | 0.7176 | 1.2042 | 1.5884 | 0.9215 | 0.6496 | 0.0000 | 0.0000 |
| planned_h32_balanced | block | 0.6984 | 1.2098 | 1.6464 | 0.8648 | 0.6429 | 0.0000 | 0.0000 |
| planned_h32_balanced | native | 0.6984 | 1.3079 | 1.8900 | 0.8648 | 0.6429 | 0.0000 | 0.0000 |
| planned_h32_short | block | 0.6984 | 1.2098 | 1.6464 | 0.8648 | 0.6429 | 0.0000 | 0.0000 |
| planned_h32_short | native | 0.6984 | 1.3079 | 1.8900 | 0.8648 | 0.6429 | 0.0000 | 0.0000 |
| ssm_h128_balanced | block | 0.6435 | 1.2256 | 1.6917 | 0.7835 | 0.5969 | 2.6742 | 0.4189 |
| ssm_h128_balanced | native | 0.6435 | 1.1224 | 1.5528 | 0.7835 | 0.5969 | 2.6948 | 0.4199 |
| ssm_h128_short | block | 0.6435 | 1.2256 | 1.6917 | 0.7835 | 0.5969 | 2.6742 | 0.4189 |
| ssm_h128_short | native | 0.6435 | 1.1224 | 1.5528 | 0.7835 | 0.5969 | 2.6948 | 0.4199 |
| ssm_h32_balanced | block | 0.6438 | 1.1884 | 1.6029 | 0.7765 | 0.5996 | 2.0687 | 0.3957 |
| ssm_h32_balanced | native | 0.6438 | 1.1979 | 1.7467 | 0.7765 | 0.5996 | 2.2157 | 0.4082 |
| ssm_h32_short | block | 0.6125 | 1.2007 | 1.7937 | 0.7219 | 0.5760 | 3.0251 | 0.3831 |
| ssm_h32_short | native | 0.6125 | 1.2399 | 1.8392 | 0.7219 | 0.5760 | 3.1370 | 0.3993 |

Active = top quartile of mean absolute recorded valve displacement over H32 (64 windows).
Wrong-sign mean includes only single-valve step/pulse/ramp/double-pulse cases with the metric defined.
Response amplitude is a model sensitivity, not intervention accuracy. Zero response alone cannot qualify.
Native and block are different simulation protocols; protected models explicitly retain carrier state in block mode.
