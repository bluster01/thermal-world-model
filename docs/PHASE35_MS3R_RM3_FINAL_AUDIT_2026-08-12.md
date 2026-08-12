# Phase 3.5 MS3-R RM3 final Supervisor audit

## Decision

`AUDITED / JOINT-LATENT DIRECTION RETAINED / A1 AGGREGATE SHAPE VIABLE / UNIQUE PHYSICAL GAIN NOT IDENTIFIED / TEST AND MS4 HOLD`

All 48 validation units completed: 36 prediction runs and 12 orthogonal calibration units. The final supplement closes all 168 per-unit ledger entries. All 36 checkpoints match their ledger hashes, manifests, run specifications, best updates and selector scores; all six model families load with `strict=True`, and stored normalization buffers match exactly. Terminal MAE from the returned NPZ files replays with maximum absolute error `2.15e-8`. Test was not accessed.

## Prediction architecture

The terminal MAE comparison is descriptive across scopes, not a single leaderboard. Within the full-multitask scope:

- P4 scheduled A1 versus P3 paired-free: mean difference `-0.00056°C`, P4 better in 4/6 pairs. This is practically equivalent at terminal level.
- P5 hybrid joint-latent versus P3: `-0.08567°C`, P5 better in 6/6 pairs.
- P5 versus P4: `-0.08511°C`, P5 better in 6/6 pairs.

P3, P4 and P5 all used the full 4000 optimizer updates, so P5 did not receive more updates. However, P5 stores about 122k state elements versus about 87k for P3/P4. Architecture and capacity are therefore not separated. P5 also has worse local-drop MAE (`1.951°C`) than P3/P4 (`1.639/1.638°C`). The supported decision is to retain the joint-latent/high-capacity direction for a capacity-matched, local-versus-terminal trade-off ablation—not to declare P5 a final champion or superior local physics.

P0 M7 oracle-valve MAE `0.6615°C` remains an oracle-action upper-bound diagnostic. P1 predicted-valve MAE is `0.9482°C`; the oracle gap quantifies a material controller/actuator forecasting bottleneck. P2 M9 is not selected by these results.

## Response identification

R0 reports adequate independent-channel rank in 12/12 units. The endpoint direction is predominantly stable, but magnitudes vary across rolling folds. At H180, A→A changes from mean `0.569` to `0.337`, while B→B changes from `0.309` to `0.456`. The data therefore support a time-varying disturbance-conditioned response trajectory, not one invariant bilateral plant gain.

The returned R1 post-processing was not valid NNLS: it fitted unconstrained coefficients and clipped negative values afterward. This produced spurious large coefficients and RMSE as high as 22.32. Exact active-set NNLS over the frozen 60/180/600 s basis reduces all replayed trajectory RMSE values to `0.024–0.072`. This keeps a compact aggregate A1 three-pole response shape viable. It does not identify context scheduling, measured spray-water flow physics, a unique plant gain, or arbitrary `do(valve)` effects.

## Next gate

Do not open test or MS4. The next allowed work is an RM3-A validation-only ablation with no broad architecture search:

1. capacity-match P3/P4/P5;
2. vary local-versus-terminal loss weight on P5;
3. preserve the orthogonal response calibration and action-invariant terminal bypass contract;
4. report terminal gain, local degradation and response trajectory stability jointly;
5. keep folds, seeds, input permissions and 4000-update cap unchanged.

Only if P5's terminal advantage survives capacity matching without unacceptable local-response degradation should it become the retained Phase 3.5 world-model backbone. This audit establishes neither complete physical response nor closed-loop deployability.
