# Phase 3.5 Result Summary

> Split: `validation`. Seed variation and event-level uncertainty are reported separately.

## Forecast aggregation

| Side | Config | Seeds | Integrated MAE mean±SD |
|---|---|---:|---:|
| A | absolute_equal_percentage_r50 | 3 | 0.7132 ± 0.0096 |
| A | absolute_identity | 3 | 0.7167 ± 0.0096 |
| A | absolute_nonlinear | 3 | 0.7179 ± 0.0084 |
| A | absolute_nonlinear_rate | 3 | 0.7170 ± 0.0086 |
| A | delta_no_baseline | 3 | 0.7139 ± 0.0118 |
| A | delta_with_baseline | 3 | 0.7167 ± 0.0096 |
| A | free_only | 3 | 0.7136 ± 0.0103 |
| B | absolute_equal_percentage_r50 | 3 | 0.9651 ± 0.0026 |
| B | absolute_identity | 3 | 0.9743 ± 0.0043 |
| B | absolute_nonlinear | 3 | 0.9723 ± 0.0052 |
| B | absolute_nonlinear_rate | 3 | 0.9725 ± 0.0048 |
| B | delta_no_baseline | 3 | 0.9670 ± 0.0030 |
| B | delta_with_baseline | 3 | 0.9743 ± 0.0043 |
| B | free_only | 3 | 0.9625 ± 0.0012 |

## Preregistered gates

| Side | Gate | Status | Evidence |
|---|---|---|---|
| A | E1_action_representation | PASS | reconstructed/naive-delta ratio=1.0039; absolute/reconstructed ratio=1.0000 |
| A | E2_nonlinear_opening | INCONCLUSIVE | fixed: forecast=0.9951, IRF=1.0084, dose gain=-0.023; learned: forecast=1.0017, IRF=1.0071, dose gain=-0.009 |
| A | E3_empirical_response | FAIL | events=1000 (open=296, close=704); day blocks=17; direction=0.394; oriented H60 CI upper=0.824 C; max|SMD|=2.028; pretrend diff=-0.119 C |
| A | E4_model_response | PASS | model direction=0.874; IRF-WMAE=0.499 C |
| A | E5_sp_negative_control | FAIL | n(no-execution/executed)=4/134; no-execution=0.0000 C; executed/no-execution=1088274662.23 |
| B | E1_action_representation | PASS | reconstructed/naive-delta ratio=1.0076; absolute/reconstructed ratio=1.0000 |
| B | E2_nonlinear_opening | INCONCLUSIVE | fixed: forecast=0.9906, IRF=1.0059, dose gain=-0.043; learned: forecast=0.9979, IRF=1.0017, dose gain=-0.003 |
| B | E3_empirical_response | FAIL | events=1000 (open=629, close=371); day blocks=18; direction=0.477; oriented H60 CI upper=0.856 C; max|SMD|=1.959; pretrend diff=0.041 C |
| B | E4_model_response | FAIL | model direction=0.828; IRF-WMAE=1.682 C |
| B | E5_sp_negative_control | FAIL | n(no-execution/executed)=2/136; no-execution=0.0000 C; executed/no-execution=25145.63 |

PASS denotes a preregistered operational gate, not randomized causal proof. INCONCLUSIVE is retained when required runs/events are absent.
