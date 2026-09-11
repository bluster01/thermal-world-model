# FMTS three-model main-steam comparison: prospective protocol freeze

Date: 2026-09-11  
Status: **FROZEN FOR VALIDATION DEVELOPMENT — LOCKED TEST DISABLED**

Current revision: **0.2**, author-requested historical boundary extension.
This supersedes the revision 0.1 input contract before any formal training.
The revised input and execution paths are implemented in
`experiments/fmts_mainsteam_20260911/run.py`; revision 0.2 synthetic verification
is recorded separately from the historical revision 0.1 smoke results.

## Revision 0.2: common rich historical context

The black-box iTransformer and both hybrid observers receive exactly 23 past
variables: five temperatures, two valve positions, seven base boundary channels,
and all nine canonical v2.2 extension channels. The extension registry is:
fuel_corrected, mill_count_on, mill_gas_temp_wavg, flue_o2, secondary_air_total,
rh_gas_in_temp_a, rh_gas_in_temp_b, water_coal_ratio, unit_load.

All nine extension channels are history-only. No recorded future extension is
supplied to any model: these include process responses that cannot be presumed
known under a changed action. The existing conditional/oracle base-boundary
comparison remains explicitly labelled. The 96-step history and all window
indices are shared. Extension normalization uses training rows only, with
statistics saved in every checkpoint; missing or invalid extension windows are
excluded by one common input-quality manifest before training any arm. The
numeric quality rules must be bound to the canonical v2.2 mapping, not tuned
after observing model outputs.

The pure greybox consumes the physical base channels and valve actions; it has
no registered equations for the extension channels. It shares the sample
population but does not consume the same effective feature set. This distinction
must be disclosed: the three-family comparison evaluates complete methods,
whereas GRU versus token isolates observer architecture under identical inputs.
No arbitrary learned adapter is added to the pure greybox.

Water-coal ratio is a history feature here. The rejected A5 metal-power correction
is not reinstated, and its result does not establish whether historical water-coal
information is useful to a learned observer. Both hybrid candidates must be
trained under revision 0.2; historical seven-channel checkpoints remain references.

The author approved the minimal architecture direction on 2026-09-11: add a
tokenized cross-attention initial-state observer candidate without replacing the
physical transition, closure, action path or boundary model. This approval does
not authorize access to the locked chronological test split.

This document narrows the FMTS evidence to one observed output and two actions.
It does not alter the v0.7 matrix, its registered gates, or any existing verdict.
Existing validation results informed this design, so a repeated validation result
will remain post-hoc. A confirmatory claim requires a one-time evaluation on the
still-locked chronological test segment after the author explicitly authorizes
test access.

## 1. Question and falsifiable claims

Question: under the same data, information, training and evaluation contract,
how do a pure black-box forecaster, a pure grey-box thermal model, and the current
physics-structured world model trade main-steam-temperature prediction against
their internal response to first- and second-stage spray-valve changes?

Primary prediction claim, if supported: the three models have different
main-steam MAE trajectories over 10–180 s. No model is declared better from a
single horizon or seed.

Primary response claim, descriptive only: the three models produce different
conditional main-steam response trajectories when either spray valve is changed.
Because no matched plant intervention truth is available, direction and magnitude
are model-response diagnostics, not action-fidelity scores.

Combined claim: logged-trajectory accuracy and conditional model response are
separate properties. A “sweet spot” may be claimed only if the same fixed model
has competitive prediction and a better independently justified response target.
The present real-data comparison alone cannot establish the latter.

## 2. Frozen model families

The final headline compares exactly three model families. During validation-only
development, the hybrid family has two prospectively registered observer arms;
the selection rule below is applied before any locked-test access.

1. **Black box — action-conditioned iTransformer.** Use the canonical-protocol
   adaptation in `experiments/final_wm/v05_blackbox_baselines.py`, descended from
   the Phase 1 pure-black-box forecasting line. It has no physical state or
   balance equations. The revised runner must save its checkpoints and raw
   prediction/response arrays; historical summary JSON is not sufficient.
2. **Pure grey box — steady + none.** Use the final physical transition with
   trainable bounded physical parameters, deterministic physical anchoring
   (`initial_state_mode=steady`), no neural state correction, no latent state,
   and no neural closure (`closure_mode=none`). This is a new arm. The existing
   T1 `physics_only` arm is not a pure grey box because T1 fixes hybrid learned
   initialization for every transition arm.
3. **Physics-structured fusion model — selected hybrid.** Both candidates use
   `initial_state_mode=hybrid`, `closure_mode=conservative_norew`, latent dimension
   zero, and the identical existing physical transition. The control retains the
   current GRU observer. The candidate replaces only that observer with 16-step
   per-variable patches (stride 8), one 128-dimensional attention layer with four
   heads, and learned physical-state queries cross-attending to past tokens. The
   v0.7 R1 verdict remains `INCOMPLETE` and is not changed by this comparison.

Seeds are 0, 1 and 2 for every trainable arm. Parameter counts and actual update
counts are reported; they are not forced to match.

### 2.1 Frozen observer selection rule

The token candidate advances only if every implementation/stability gate passes,
its equal-day cumulative main-steam H18 MAE is lower than the GRU control in at
least two of three paired seeds, and its equal-seed mean H18 MAE is no greater
than the GRU mean. Otherwise the GRU is retained. Response sign, magnitude and
visual appearance are explicitly forbidden as selection criteria because no
matched plant intervention truth exists. The selected hybrid identity is frozen
before any optional one-time test evaluation; the non-selected observer remains
a reported ablation rather than disappearing from the record.

## 3. Shared data and information contract

- Record: canonical Side A, 10 s sampling, with its frozen channel mapping and
  chronological 75/15/10 train/validation/locked-test split.
- History: 96 steps (960 s). Blackbox and hybrid observers use 23 variables
  under revision 0.2; the pure greybox uses its physical subset as disclosed above.
- Forecast horizon: 18 steps (180 s).
- Reported prediction target: `final_outlet_temp` only. Other temperature channels
  may be required internally by a model but are not pooled into the headline MAE.
- Conditional/oracle comparison: recorded future actions and the same recorded
  future boundaries are supplied to all arms. This isolates transition quality
  and is not described as autonomous deployment forecasting.
- Training and validation windows must be materialized once into immutable index
  manifests. All arms use identical train-bank indices, validation indices and
  extension-normalization statistics fitted on training data only. Hybrid base
  inputs retain the physical fixed normalization; blackbox base inputs use
  train-only statistics. These model-native preprocessing differences are disclosed.
- The reserved test split remains unread until a separate author authorization.
  If authorized, it is read once with the frozen checkpoints and index-generation
  rule; no model or plotting choice may be changed afterward.

Model-native training losses must be disclosed. If the black box is trained only
on main steam while the structured models use auxiliary temperature supervision,
the comparison is labelled task-performance rather than loss-identical. The
frozen implementation uses main-steam MSE for the black box and the existing
five-temperature observation NLL for structured models; it is therefore a
task-performance comparison, not a loss-identical one. Early stopping and the
GRU-versus-token decision use the same main-steam validation MAE. No auxiliary
channel is used to select the final hybrid.

## 4. Prediction estimand and statistics

For window i, seed s and horizon h,

`e[i,s,h] = abs(predicted_final_outlet_temp - observed_final_outlet_temp)`.

The primary curve is cumulative MAE from steps 1 through h for h=1,…,18. The
single-step MAE curve is saved as source data and may be shown only if explicitly
labelled. Each model line averages the same windows within UTC day, days equally,
then the three optimization seeds equally. Bands in the figure are the range of
the three seed-level equal-day curves, not confidence intervals.

Pairwise numerical comparisons at H18 use within-seed, equal-day paired
differences with 1,000 day-block bootstrap resamples and 95% intervals. All three
pairs and all seeds are reported; there is no multiplicity-adjusted significance
claim. Persistence is retained as a non-neural diagnostic in the supplementary
table but not as a fourth headline model.

## 5. Action-response estimand

Two panels are fixed: first-stage valve and second-stage valve. For each panel,
the estimand is the model's paired difference

`delta_T[h] = prediction(u_plus)[h] - prediction(u_zero)[h]`, h=1,…,18,

for main steam only. Both rollouts share history and model weights; all seven
boundary channels are held at their cut-point values for all 18 steps.
`u_zero` holds the cut-point two-valve vector for all 18 steps;
`u_plus` adds +0.05 to exactly one selected valve, capped at 1.0, and holds the
other valve unchanged. This keeps the dose aligned with the frozen v0.7 response
probe; it does not revise R1.

The response-window manifest is selected without reading model outputs. It uses
the same 64 validation histories for every model and optimization seed within a
valve panel. Every history is retained in the primary matched curve. The existing
history-derived action-support check is recomputed and its violation count is
printed on the panel. A supported-only curve may appear in the appendix as a
predeclared sensitivity analysis, with its retained n, but cannot replace the
primary curve.

For each model, the plotted line gives the equal-day, then equal-seed mean
response. The band gives the range across three seed-level equal-day curves.
Per-seed day-block intervals and negative-window fractions are saved in a table.
Negative temperature change may be described as the model's cooling response
under held boundaries, not as measured plant truth. No “correct response”,
“response recovery”, causal gain or controller benefit is inferred.

## 6. Figure and table contract

The main comparison figure has exactly three panels and consistent model colors:

- **a, prediction:** cumulative main-steam MAE versus forecast time, 10–180 s;
- **b, valve 1:** paired model-response curve `delta_T` versus time;
- **c, valve 2:** paired model-response curve `delta_T` versus time.

All panels show the same three models. Panels b–c share a y scale, include a zero
line, and report the +5 percentage-point dose and support count. Color is paired
with line style. Curves use all registered points; no smoothing or selected
example window is allowed.

A compact table reports H1/H6/H18 main-steam MAE, H18 paired differences, response
at 60/180 s, negative-window fraction, parameter count and actual optimization
updates for all three seeds. The figure communicates shape; the table carries
exact values.

## 7. Execution order and fail-closed checks

1. Save code revision, environment, canonical-record hash, mapping hash, property
   table hash, split counts, train/validation index manifests and normalization
   statistics.
2. Verify blackbox checkpoint round-trip and saved-array replay before interpreting
   any comparison. Revision 0.2 changes the feature set and window population, so
   the old v0.5 validation score is not a numerical acceptance target. Retain its
   architecture and budget as the historical baseline definition; disclose the
   feature/preprocessing/window changes. Synthetic replay is a software check,
   not reproduction of the old real-data performance.
3. Run a smoke test for all arms. Confirm identical batch indices, targets,
   future actions and future boundaries, finite outputs, and exact zero response
   when `u_plus == u_zero`.
4. Train all three seeds for every development arm under the frozen budget. Apply
   the registered GRU-versus-token selection rule on validation only.
   Failed or unstable seeds remain in the report. No automatic hyperparameter
   search, seed replacement or post-result architecture change.
5. Run prediction and response inference once on the frozen validation manifests;
   save raw per-window/per-step arrays, day identifiers and support masks.
6. Audit hashes, shapes, pairing, units, channel index and both valve mappings.
7. Only after author approval, optionally unlock the test segment once. Validation
   results remain labelled exploratory; test results become the primary comparison.

## 8. Existing evidence and the actual gap

`results/final_wm/v05_blackbox_comparison_20260823.json` already provides
canonical H18 main-steam summaries for several black boxes. The three-seed
iTransformer mean is approximately 0.365 °C, but checkpoints and raw response
curves were not preserved there. `results/final_wm/v04_comparison_report_20260823.json`
contains main-steam metrics for existing structured checkpoints, but evaluation
windows differ by seed and its response summaries are endpoints on a different
60/240-step protocol. It also lacks a true steady/no-closure grey-box arm.

Consequently, historical values may motivate the comparison but may not be joined
into the new figure. The minimum new work is: train/save the black-box reference,
train the true grey-box arm, train both registered hybrid observers on validation,
freeze the selected hybrid, and export matched prediction and two-valve response
trajectories for every arm and seed. Only the three final families enter the main
figure; the observer comparison is retained in the table or appendix.

## 9. Paper boundary

This comparison can support the FMTS claim that forecast accuracy and conditional
model response must be evaluated separately. It cannot establish field
counterfactual fidelity because the response reference is not a matched plant
intervention. The synthetic known-truth experiment remains the direct evidence
that low total error can conceal a missing response; the real-data curves show how
the three model classes behave under one shared diagnostic interface.
