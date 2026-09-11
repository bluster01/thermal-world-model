# Protocol 0.2 implementation verification — 2026-09-11

Implemented: 23-channel history for iTransformer and GRU/token hybrid observers;
true steady/no-closure greybox; four-arm training with fixed shared windows;
main-steam MAE selection; two held-boundary valve probes; checkpoint/source/input
hashes; raw array export; three-seed summaries and paired day-block intervals;
read-only checkpoint reload and numerical replay.

## Evidence

- `python -m pytest tests/final_wm -q`: **213 passed**, 167.97 seconds.
- New rich-history tests cover historical extension gradients, future-extension
  exclusion, train-only normalization, invalid histories, time gaps, greybox
  feature consumption, equal-day weighting, full smoke and overwrite rejection.
- Synthetic pipeline: four arms, seed0, two optimizer updates each; **SMOKE**.
- Saved artifacts reloaded: four checkpoints, eight prediction/response array
  sets; all numerical replay checks passed. No failed runs.
- Bundle: `results/fmts_mainsteam_20260911/synthetic_v02_final_verification/`.
- `audit.json` gives the machine-readable replay result. The first synthetic
  verification bundle is preserved as an intermediate implementation checkpoint.

These checks establish software behavior, not scientific performance. Analytic
thermodynamic properties and synthetic inputs were used locally. No real-data
training, held-out test evaluation, paper verdict change or Linux dispatch occurred.

## Protocol details made explicit during implementation

- Exact 10-second continuity, base valid/finite values across the whole window,
  and nine extension range/finite checks on history determine shared eligibility.
- All nine extensions are standardized with common training-row statistics.
  Physical base normalization remains fixed; blackbox base normalization is fitted
  on training rows. The pure greybox has no extension equations.
- Future recorded base boundaries are an oracle comparison. The blackbox can use
  measured total spray among these; the physical action-driven transition ignores
  that channel. This difference must be disclosed and does not demonstrate equal
  effective causal information or deployment forecasting.
- Full input bytes must match the audited Side A v0.7 reference manifest. Future
  extension measurements are never passed into models. A5 is not reintroduced.
- Full budgets: structured 120 epochs × 200 batches, batch32, patience20;
  blackbox 3000 steps, batch128, validation every500, patience3 evaluations.
  Differences in budgets, training losses and active parameter counts are reported.
- All model selectors use equal-day cumulative main-steam H18 MAE. Response
  magnitude/sign is not a selector. All seeds and failures are preserved.
- Historical blackbox v0.5 scores have different inputs/windows and are not
  numerical acceptance thresholds for this experiment. Saved-array replay is
  explicitly a software validation, not reproduction of those scientific scores.

## Handoff

Code is ready for Linux execution using `RUN_LINUX.md`; full training and return
audit remain pending. Source changes are uncommitted local work at this checkpoint.
