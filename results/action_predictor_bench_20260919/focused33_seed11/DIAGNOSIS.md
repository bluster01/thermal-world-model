# Returned baseline diagnosis

One optimization seed; exploratory validation results. Main-temperature MAE in C.

| Model | Mode | H32 | H128 | H512 | Active H32 | Quiet H32 | Upstream max | Wrong-sign scenario mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_balanced | block | 0.5761 | 1.2246 | 1.8071 | 0.6357 | 0.5562 | 4.7535 | 0.3687 |
| A_balanced | native | 0.5761 | 1.2805 | 1.8836 | 0.6357 | 0.5562 | 4.2636 | 0.4494 |
| A_short | block | 0.5764 | 1.1901 | 1.6987 | 0.6426 | 0.5543 | 4.5965 | 0.3597 |
| A_short | native | 0.5764 | 1.2925 | 1.8934 | 0.6426 | 0.5543 | 4.2596 | 0.4451 |
| B_balanced | block | 0.5276 | 0.9839 | 1.4684 | 0.6352 | 0.4917 | 4.9568 | 0.3656 |
| B_balanced | native | 0.5276 | 1.2073 | 1.7985 | 0.6352 | 0.4917 | 4.6861 | 0.4300 |
| B_short | block | 0.5172 | 0.9789 | 1.4747 | 0.6206 | 0.4827 | 5.1246 | 0.3723 |
| B_short | native | 0.5172 | 1.2083 | 1.7959 | 0.6206 | 0.4827 | 4.7108 | 0.4319 |
| C_balanced | block | 0.5912 | 1.1821 | 1.6959 | 0.6987 | 0.5554 | 0.0000 | 0.0000 |
| C_balanced | native | 0.5912 | 1.2228 | 1.7580 | 0.6987 | 0.5554 | 0.0000 | 0.0000 |
| C_short | block | 0.5966 | 1.1799 | 1.6717 | 0.6938 | 0.5642 | 0.0000 | 0.0000 |
| C_short | native | 0.5966 | 1.2305 | 1.7564 | 0.6938 | 0.5642 | 0.0000 | 0.0000 |
| D_balanced | block | 0.5464 | 1.0524 | 1.5772 | 0.6520 | 0.5112 | 0.0000 | 0.0000 |
| D_balanced | native | 0.5464 | 1.1768 | 1.7935 | 0.6520 | 0.5112 | 0.0000 | 0.0000 |
| D_short | block | 0.5425 | 1.0509 | 1.5740 | 0.6453 | 0.5082 | 0.0000 | 0.0000 |
| D_short | native | 0.5425 | 1.1772 | 1.7827 | 0.6453 | 0.5082 | 0.0000 | 0.0000 |

Active = top quartile of mean absolute recorded valve displacement over H32 (64 windows).
Wrong-sign mean includes only single-valve step/pulse/ramp/double-pulse cases with the metric defined.
Response amplitude is a model sensitivity, not intervention accuracy. Zero response alone cannot qualify.
Native and block are different simulation protocols; protected models explicitly retain carrier state in block mode.
