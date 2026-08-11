# Phase 3.5-MS3-D Supervisor Audit

> Audit date: 2026-08-11
> Final label: `AUDITED / MODEL_A_RESPONSE_ATTENUATION_EXCEEDS_FIELD_EVIDENCE / CONTROLLER_PERSISTENCE_ASYMMETRY / SIDE_ATTRIBUTION_INCONCLUSIVE / MS4_HOLD`

## 1. Supervisor decision

MS3-D does not support the explanation that the A thermal response is physically four to five times weaker than B. The validation-only SP held-step diagnosis instead shows a robust difference in the persistence of the observed valve response: B retains more expected-direction valve motion at 300 and 600 s. The paired-day local spray-temperature-drop and terminal-temperature contrasts do not show the same consistent B>A separation.

The MS3 checkpoint nevertheless produces a median B/A standardized `+5%` H600 terminal-effect ratio of `4.632`. That attenuation is distributed across the learned opening map, scheduled gain and dynamics. It is therefore a real property of the fitted response operator, but it is not corroborated as a plant-level side ratio by the available field diagnosis.

The correct conclusion is not that A and B are physically equivalent. Most events contain coordinated motion in the other loop, the clean 600 s subgroup is extremely small, and a confidence interval crossing zero is not an equivalence test. MS3 remains FAIL, formal MS4 stays on hold, and the next work is a new local response-identification design rather than an MS3 retry.

## 2. Provenance and replay

| Item | Audit result |
|---|---:|
| Execution commit | `1ad334704157faec38a14794e1fe7b92fce17432` |
| Source SHA-256 | `85a3f92648d5f88a4543f500859b200207fb55a32555900ca88f7c339c4e4da6` |
| Parent MS3 matrix SHA-256 | `b2e69d78b949334bfd5d92d100bcb730f444758c57d90920a451138a5b831f8a` |
| Evaluated split | validation only, 2026-03-16 21:42:50 to 2026-04-13 12:16:00 |
| Training / test access | `false / false` |
| Event and paired-day independent replay | PASS; maximum numeric error `0` |
| Event IDs, SP bounds, steady flags, 600 s separation | all exact |

The independent replay is `results/phase3_5/ms3d_asymmetry_diagnosis/supervisor_replay_validation.json`. It reconstructs all UTC-day medians, paired contrasts and 1/2/3-day diagnostic bootstrap intervals from the event JSONL rather than trusting the summary.

## 3. Support and identifiability

| Support layer | A | B | Interpretation |
|---|---:|---:|---|
| held, operating events | 78 | 95 | includes dynamic operation |
| primary 60 s strict and 180 s moderate intersection | 41 events / 19 days | 42 events / 19 days | 17 common UTC days for paired contrasts |
| SP up / SP down in primary layer | 25 / 16 | 27 / 15 | both directions present |
| pre-valve clean-chain subgroup | 9 / 7 days | 9 / 8 days | too small for a replacement primary analysis |
| primary events with other loop quiet | 2 | 1 | single-side attribution is weak |
| strict 600 s state plus valve-clean subgroup | 1 | 3 | insufficient for a strict steady-state claim |

The primary layer is suitable for a closed-loop A/B diagnostic because it has events from both directions and many days. It is not sufficient for an isolated plant-gain claim: coordinated cascade-PID actions dominate, and conditioning on post-SP valve execution would itself be post-treatment selection.

## 4. Primary paired-day results

All responses below are normalized by `|delta SP|`; intervals are 95% UTC-day bootstrap intervals for the paired `B-A` contrast.

| Endpoint | Paired days | Median B-A | 95% interval |
|---|---:|---:|---:|
| expected valve motion, H60 (%/degC) | 17 | +1.300 | [-0.714, +2.426] |
| expected valve motion, H180 | 17 | +2.617 | [-0.802, +3.425] |
| expected valve motion, H300 | 17 | +2.947 | [+0.544, +5.486] |
| expected valve motion, H600 | 17 | +2.627 | [+1.655, +4.107] |
| local spray drop, H180 (degC/degC-SP) | 17 | -0.308 | [-1.593, +1.437] |
| local spray drop, H300 | 17 | -0.060 | [-1.887, +1.585] |
| valve-normalized local drop, H180 (degC/%) | 14 | +0.105 | [-0.178, +0.649] |
| valve-normalized local drop, H300 | 15 | -0.018 | [-0.218, +0.423] |
| terminal response, H600 (degC/degC-SP) | 17 | +0.803 | [-0.284, +1.126] |

The H300/H600 valve-persistence difference remains positive under 1, 2 and 3 consecutive-day circular block diagnostics. None of the prespecified local-drop or terminal-response comparisons has a lower interval bound above zero. This is evidence for a controller/actuator trajectory difference, not for a confirmed four-to-five-fold thermal-gain difference.

## 5. Opening versus closing

SP up corresponds to expected valve closing and SP down to expected valve opening. Event medians suggest stronger direction nonlinearity, especially weak sustained A opening, but the day-paired direction strata are smaller:

| Direction | Paired days | H600 valve B-A | 95% interval |
|---|---:|---:|---:|
| expected closing | 13 | +2.931 %/degC | [+1.560, +3.419] |
| expected opening | 8 | +3.086 %/degC | [+0.965, +4.642] |

Direction-stratified local-drop, valve-normalized local-drop and terminal-temperature intervals all cross zero. Thus the sustained valve difference is present in both directions, while the apparent opening-specific thermal difference is not day-level robust. A future model must still allow directional and operating-point nonlinearity because valve opening is only a nonlinear proxy for spray flow, but the current data do not identify a unique valve characteristic.

## 6. Checkpoint attenuation decomposition

For a constant raw-valve `+5%` trajectory, the checkpoint B/A absolute H600 effect ratios are `4.839`, `4.440` and `4.632` across matched seeds. A marginal-median decomposition gives:

| Seed | opening-map ratio | scheduled-gain ratio | H600 dynamics ratio | observed effect ratio |
|---:|---:|---:|---:|---:|
| 0 | 1.583 | 2.426 | 1.428 | 4.839 |
| 1 | 1.369 | 2.100 | 1.756 | 4.440 |
| 2 | 1.359 | 2.998 | 1.146 | 4.632 |

The product is only an approximation because context-level covariance is not retained. Still, it shows that no single scalar explains the learned asymmetry: A is compressed by the opening map, fitted gain and slower dynamics together. This is why merely rescaling the final gain or lowering the MS3 non-collapse threshold would be an inadequate repair.

## 7. Statistical and causal boundary

1. The independent unit is UTC day; 83 primary event rows and their 10 s samples are nested observations.
2. A/B contrasts use only common days. Opening/closing strata are diagnostics with only 13 and 8 paired days.
3. Intervals do not include checkpoint-selection or event-rule-development uncertainty. Validation is not an independent lockbox.
4. Other-loop motion is nearly ubiquitous. The analysis estimates a coordinated closed-loop response, not an isolated `do(valve)` effect.
5. Failure to reject B=A is not evidence of equivalence. No equivalence margin or adequate minimum detectable effect was prespecified for the sparse clean subgroup.
6. The valve is a nonlinear proxy for untrusted spray-flow instrumentation. Ratios in degC/% cannot be interpreted as degC/(kg/s).

## 8. Next architecture task: MS3-R local design

The next protocol should make measured intermediate states supervise the response chain instead of asking terminal total loss to identify everything:

1. a directional, side-aware controller/actuator block for `SP_A/SP_B -> valve_A/valve_B` trajectories;
2. a two-input/two-output plant mediator for both actual valves to both local `Tin-Tout` drops, with load, pressure and operating point as context;
3. a downstream thermal block from local outlet/drop states to both terminal temperatures;
4. a joint residual/free branch that cannot read future actions and is selected by both terminal forecast loss and intermediate response loss;
5. ablations for shared physics plus side scale, fully independent sides, directional opening maps, MIMO versus separate SISO mediators, and response-aware versus terminal-only checkpoint selection.

Training is not yet authorized. First freeze estimands, information flow, selector, compute matching and no-test rules in a new design. Linux has no task until that code and matrix pass local tests.
