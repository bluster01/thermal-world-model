# Phase 3.5 MS3-R RM3 provisional Supervisor audit

## Status

`PROVISIONAL / ARTIFACT INCOMPLETE / CHECKPOINT SUPPLEMENT REQUIRED`

Hermes returned 48/48 complete validation units with exit code 0: 36 prediction runs and 12 orthogonal calibration units. Test was not accessed and no automatic scientific decision was emitted. The Linux authorization is now closed and the registry is `results_returned`, not `audited`.

The returned Git commit omits all 36 `checkpoint_best_validation.pt` files even though every per-run ledger requires and hashes one. The remaining 132 per-unit ledger entries verify byte-for-byte. The 36 NPZ files reproduce terminal MAE with maximum absolute error `2.15e-8`. Therefore the numerical report is replayable, but the model-state and checkpoint-to-manifest audit is not complete.

Hermes must supplement the exact original checkpoint bytes from the completed output directories. It must not retrain, resume, alter manifests or ledgers, regenerate checkpoints, change the matrix, access test, or start MS4.

## Prediction result boundary

The common descriptive terminal MAE means are:

| Candidate | Output scope | Terminal MAE (°C) |
|---|---|---:|
| P0 M7 oracle valve | terminal only, oracle action | 0.6615 |
| P1 M7 predicted valve | valve + terminal | 0.9482 |
| P2 M9 future SP | terminal only | 1.4453 |
| P3 Gate C paired free | full multitask | 1.0584 |
| P4 Gate C A1 scheduled | full multitask | 1.0578 |
| P5 hybrid joint latent | full multitask | 0.9727 |

These are not one cross-scope leaderboard. P0 is an oracle-action upper bound and P2 is comparable to P0 only within the terminal-only scope. P3/P4/P5 form the main full-multitask comparison:

- P4 minus P3 terminal MAE is `-0.00056°C` on average and improves 4/6 fold-seed pairs: no meaningful terminal advantage from simply adding the scheduled A1 slot.
- P5 minus P3 is `-0.08567°C` and P5 minus P4 is `-0.08511°C`, improving 6/6 pairs in both contrasts.
- P5 nevertheless has worse mean local-drop MAE (`1.9510°C`) than P3/P4 (`1.6387/1.6383°C`). The hybrid improves terminal prediction through the shared latent/terminal path while exposing a multitask trade-off; it is not evidence of a more accurate local physical response.

No prediction champion is declared before checkpoint supplementation and capacity/compute-aware review.

## Orthogonal response audit

R0 input-rank support reports independent A/B channels in 12/12 calibration units. Endpoint matrices are predominantly positive, but their magnitudes drift materially across rolling folds. At H180, for example:

- A→A mean changes from `0.569` in F0 to `0.337` in F1;
- B→B mean changes from `0.309` in F0 to `0.456` in F1.

This supports an estimable, time-varying disturbance-conditioned response trajectory, not one invariant plant gain.

The returned R1 projection used an invalid operation: unconstrained least squares followed by coefficient clipping. With correlated three-pole bases, negative and positive coefficients cancel; clipping afterward produced artificial coefficients up to about 220 and RMSE up to 22.32. The local fix uses exact active-set non-negative least squares over the frozen three basis columns. Replaying the returned R0 trajectories reduces all projection RMSE values to `0.024–0.072`, without retraining.

Thus the aggregate three-pole A1 shape remains a viable approximation. This replay does not identify context scheduling, measured spray-flow physics, a unique bilateral plant gain, or arbitrary `do(valve)` response.

## Current conclusion

RM3 provisionally supports:

1. a useful joint-latent/high-capacity direction for terminal prediction;
2. an identifiable local innovation subspace under the current rank diagnostic;
3. a compact non-negative three-pole approximation to the returned aggregate response trajectories.

RM3 does not yet establish completely physical response, invariant plant identification, causal valve intervention, closed-loop deployment, test performance, or a final architecture champion. Final Supervisor audit waits only for the original checkpoint supplement; no training rerun is authorized.
