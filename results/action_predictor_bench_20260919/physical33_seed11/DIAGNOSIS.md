# Returned baseline diagnosis

One optimization seed; exploratory validation results. Main-temperature MAE in C.

| Model | Mode | H32 | H128 | H512 | Active H32 | Quiet H32 | Upstream max | Wrong-sign scenario mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0_balanced | block | 0.7080 | 1.2577 | 2.7444 | 0.9716 | 0.6202 | 0.0000 | 0.0132 |
| P0_balanced | native | 0.7080 | 1.2577 | 2.7444 | 0.9716 | 0.6202 | 0.0000 | 0.0132 |
| P0_short | block | 0.7050 | 1.2539 | 2.6757 | 0.9708 | 0.6164 | 0.0000 | 0.0134 |
| P0_short | native | 0.7050 | 1.2539 | 2.6757 | 0.9708 | 0.6164 | 0.0000 | 0.0134 |
| P1_balanced | block | 0.7280 | 1.2742 | 2.6516 | 0.9922 | 0.6400 | 0.0000 | 0.0133 |
| P1_balanced | native | 0.7280 | 1.2742 | 2.6516 | 0.9922 | 0.6400 | 0.0000 | 0.0133 |
| P1_short | block | 0.7152 | 1.2485 | 2.5742 | 0.9831 | 0.6258 | 0.0000 | 0.0133 |
| P1_short | native | 0.7152 | 1.2485 | 2.5742 | 0.9831 | 0.6258 | 0.0000 | 0.0133 |
| P2_balanced | block | 0.7347 | 1.2630 | 2.6835 | 1.0119 | 0.6424 | 0.0000 | 0.0130 |
| P2_balanced | native | 0.7347 | 1.2630 | 2.6835 | 1.0119 | 0.6424 | 0.0000 | 0.0130 |
| P2_short | block | 0.7281 | 1.2562 | 2.5928 | 1.0061 | 0.6355 | 0.0000 | 0.0129 |
| P2_short | native | 0.7281 | 1.2562 | 2.5928 | 1.0061 | 0.6355 | 0.0000 | 0.0129 |

Active = top quartile of mean absolute recorded valve displacement over H32 (64 windows).
Wrong-sign mean includes only single-valve step/pulse/ramp/double-pulse cases with the metric defined.
Response amplitude is a model sensitivity, not intervention accuracy. Zero response alone cannot qualify.
Native and block are different simulation protocols; protected models explicitly retain carrier state in block mode.
