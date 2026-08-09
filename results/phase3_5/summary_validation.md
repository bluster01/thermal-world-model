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
| A | E2_nonlinear_opening | INCONCLUSIVE | fixed: forecast=0.9951, IRF=1.0010, dose gain=-0.069; learned: forecast=1.0017, IRF=0.9995, dose gain=-0.026 |
| A | E3_empirical_response | INCONCLUSIVE | events=93 (open=93, close=0); day blocks=11; direction=0.323; oriented H60 CI upper=1.032 C; max|SMD|=0.302; pretrend diff=0.045 C; insufficient common support |
| A | E4_model_response | BLOCKED | E3 empirical reference did not pass; model-response comparison is not identifiable |
| A | E5_sp_negative_control | INCONCLUSIVE | n(no-execution/executed)=4/134; no-execution=0.0000 C; executed/no-execution=undefined (no-execution≈0) |
| B | E1_action_representation | PASS | reconstructed/naive-delta ratio=1.0076; absolute/reconstructed ratio=1.0000 |
| B | E2_nonlinear_opening | INCONCLUSIVE | fixed: forecast=0.9906, IRF=1.0017, dose gain=-0.019; learned: forecast=0.9979, IRF=1.0009, dose gain=-0.011 |
| B | E3_empirical_response | INCONCLUSIVE | events=122 (open=121, close=1); day blocks=12; direction=0.050; oriented H60 CI upper=4.144 C; max|SMD|=0.709; pretrend diff=-0.034 C; insufficient common support |
| B | E4_model_response | BLOCKED | E3 empirical reference did not pass; model-response comparison is not identifiable |
| B | E5_sp_negative_control | INCONCLUSIVE | n(no-execution/executed)=2/136; no-execution=0.0000 C; executed/no-execution=undefined (no-execution≈0) |

PASS denotes a preregistered operational gate, not randomized causal proof. INCONCLUSIVE is retained when required runs/events are absent.
