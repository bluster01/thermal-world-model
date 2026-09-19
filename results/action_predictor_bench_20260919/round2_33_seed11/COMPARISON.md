# Round2 comparison

Main-temperature MAE (C). Same data, horizons and selector as round1.
Protected block = nominal reference block32 + continuous R4 carrier. Native = both uninterrupted.
Original SSM/R4 parents: 6 epochs. Continued arms: up to 6 additional epochs; epoch0 can be retained.
Single seed discovery; no blanket accuracy-preservation claim.

| Arm | H32 | Block H128 | Block H512 | Native H512 | H32 change vs original SSM |
|---|---:|---:|---:|---:|---:|
| r4_mlp | 0.7234 | 1.2575 | 1.7001 | 1.7775 | +12.36% |
| ssm | 0.6438 | 1.1884 | 1.6029 | 1.7467 | +0.00% |
| r4_protected | 0.7234 | 1.2614 | 1.7470 | 1.7775 | +12.36% |
| ssm_continue | 0.6225 | 1.2000 | 1.8205 | 1.8490 | -3.32% |
| ssm_protected | 0.7878 | 1.3353 | 1.7724 | 1.8600 | +22.36% |
| ssm_r4_joint | 0.6710 | 1.2416 | 1.8588 | 1.8661 | +4.22% |
| ssm_r4_teacher | 0.6594 | 1.2256 | 1.8486 | 1.8674 | +2.42% |
| ssm_soft_response | 0.6544 | 1.2227 | 1.7040 | 1.8454 | +1.65% |

Read DIAGNOSIS.md for per-mode response direction/topology and action-active strata.
Use ssm_continue to separate extra optimization from response constraints; report failures too.
