# FMTS-M7R1: M7-family encoder and independent state head

Registered 2026-09-13 after observing the parent v0.2 results and token diagnostic,
before any M7R1 formal training. This is a **post-result exploratory supplement**,
not a retrospective preregistration of v0.2. All parent results remain available.

## Question and single new candidate

Does replacing the new state-query observer with an M7-family encoder and an
independent vector state readout improve the no-rewet hybrid's main-steam
prediction, while retaining its explicit physical action channel?

`fusion_m7var_norew`, seeds 0/1/2. A bundled observer repair, not an experiment
isolating encoder versus readout effects. No old weights, old temperature head,
direct temperature residual, extra closure input, or new action path.

The exact executable contract is `experiments/fmts_m7fusion_20260913/spec.py`
and its equal `frozen_spec.json`. The training template is the full parent
`fusion_gru_norew` TrainSpec, changing only the arm name. Its `observer_encoder=gru`
describes the inherited vector-posterior dispatch interface, **not the actual
encoder**. Every checkpoint must additionally identify `M7Observer_v1` and the
effective model contract; loading through the old generic GRU factory is invalid.

## Encoder and head

23 historical variables = obs5 + actions2 + base boundary7 + extension9, W96.
Parent base physical normalization and train-only extension statistics first;
then per-window RevIN, patch16/stride8, d64, two shared per-variable TCN blocks,
four-head VariableAttention, dropout0.1. Reuse the historical M7 building blocks,
but bind patch dimensions locally rather than through mutable global config.

Retain all 23 variable embeddings by flattening (1472 dimensions). Concatenate
the pre-RevIN history mean/std (46 dimensions), project 1518→256→128 through
GELU/dropout, and apply non-affine LayerNorm. Preserve absolute-history information
for physical initial-state inference; no temperature output de-normalization.

Reuse the GRU observer's pressure features, independent 11-row mean/logvariance
heads, zero-initialized mean, and `0.1 * state_scale * tanh(raw)` bounds. Keep only
the same four slow-state corrections (metal temperatures and fuel offset).
Physical transition, conservative closure, no-rewet constants, observation model
and all other interfaces are unchanged. No added sign loss or saturation penalty.

## Fixed training and data

- Full 24000 update cap (120×200), batch32, Adam 0.001, gradient clip10,
  five-temperature NLL, validation every200, patience20; same parent minibatch RNG.
- Best checkpoint solely by equal-day cumulative main-steam H18 MAE.
- Exact audited v0.2 record/mapping/IAPWS and source hashes. Copy `indices.npz`
  byte-for-byte: train20000, prediction256, response64; no replacement windows.
- Extension access past-only; future base boundaries/actions are oracle
  prediction inputs. Held-last boundary dual-valve +0.05 probes, same support
  flags and doses; H18/10 seconds. Test remains locked. No H60 expansion.
- No retraining of parent blackbox, GRU, old token or rewet-on greybox.
- Run failures remain saved. No overwrite/resume, hyperparameter sweep, new seed,
  extra optimizer steps after inspection or sign-selected checkpoint.

## Diagnostics and analysis, frozen before new results

Before training, diagnose the retained three GRU and three token observers on the
same real validation histories. Save active-state raw corrections, bounded
fractions, local tanh derivatives and reversed-past sensitivity (keep final
observation/actions/boundaries fixed). Targets are not used by this calculation.
These finite synthetic reversals are diagnostic, not plant counterfactual truth.

Log M7 observer gradient L2 norm before clipping at each validation checkpoint,
plus health on the first32 fixed validation windows. Save full 256-window health
for the MAE-selected best checkpoint. Saturation `abs(tanh(raw))>0.99` is a
descriptive diagnostic, **not** a checkpoint/seed/window selector or new scientific
PASS threshold. LayerNorm and independent heads do not guarantee healthy training.

Recompute predictions and two responses for all three checkpoints. Pair against
all four parent arms, report cumulative H1…H18 MAE, persistence, paired H18
differences with 1000 day-block resamples, full valve curves and unsupported counts.
M7 is MAE-eligible only with ≥2/3 paired wins and mean no worse than retained GRU,
following the same numerical logic as v0.2. Eligibility alone is not scientific
adoption: source/numerical audit and explicit health review must also finish.
Response sign/magnitude/appearance must not select checkpoints or seeds.

## Physical-response recovery is a separate registered change

Run the existing **FMTS-GNR1** unchanged: pure steady/no-closure greybox, only
disable rewetting, seeds0/1/2, full original budget and fixed windows. Do not mix a
new observer into the pure greybox or alter its fit to enforce a negative curve.
If the expected cooling direction is not recovered, report that rather than
searching again. Neither negative signs nor no-rewet establish calibrated plant
gain, physical truth over the full horizon, deployable forecasting or control.

Final side-by-side M7/GNR1/parent paper plots are pending both result audits.
New results do not overwrite the retained manuscript numbers until then.
