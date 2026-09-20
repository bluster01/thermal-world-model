# Horizon and nominal-plan comparison

Six temperature fits; short/balanced rows reuse the same training trajectory.
Protected block refreshes the reference only; the response stays continuous.
H128 cells supervise steps33–128; H512 remains beyond training horizon.
Compare within each selector rule. Neither rule selects on reporting windows.

| Model | H32 MAE | Block H128 | Block H512 | Native H512 |
|---|---:|---:|---:|---:|
| r4_mlp | 0.7234 | 1.2575 | 1.7001 | 1.7775 |
| ssm | 0.6438 | 1.1884 | 1.6029 | 1.7467 |
| hold_h128_balanced | 0.7051 | 1.2465 | 1.6947 | 1.5912 |
| hold_h128_short | 0.7051 | 1.2465 | 1.6947 | 1.5912 |
| hold_h32_balanced | 0.6724 | 1.2253 | 1.7503 | 1.8461 |
| hold_h32_short | 0.6724 | 1.2253 | 1.7503 | 1.8461 |
| planned_h128_balanced | 0.7314 | 1.2577 | 1.7039 | 1.6066 |
| planned_h128_short | 0.7176 | 1.2353 | 1.6508 | 1.5884 |
| planned_h32_balanced | 0.6984 | 1.2098 | 1.6464 | 1.8900 |
| planned_h32_short | 0.6984 | 1.2098 | 1.6464 | 1.8900 |
| ssm_h128_balanced | 0.6435 | 1.2256 | 1.6917 | 1.5528 |
| ssm_h128_short | 0.6435 | 1.2256 | 1.6917 | 1.5528 |
| ssm_h32_balanced | 0.6438 | 1.1884 | 1.6029 | 1.7467 |
| ssm_h32_short | 0.6125 | 1.2007 | 1.7937 | 1.8392 |

Policy quality: policy/result.json (valve MAE in percentage points versus hold).
Response quality: DIAGNOSIS.md and per-row response metrics; all110 scenarios retained.
Temperature fit cost is shared by two selectors; do not sum duplicated row costs.
