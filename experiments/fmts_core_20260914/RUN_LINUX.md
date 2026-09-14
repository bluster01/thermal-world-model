# FMTS-CORE1 Linux handoff — inference only

Read `docs/fmts2026/PREREG_CORE_ABLATION_20260914.md`. Execute only after this
package and registration are committed and published; preparation is not a run
receipt. No new training, parameter tuning, locked-test scoring or paper verdict.

Scientific purpose and paper-entry boundaries: `EVIDENCE_PLAN_ZH.md` in this
directory. The experiment uses REAL canonical plant records and existing trained
checkpoints. Synthetic fixtures below are software checks, not experimental data.

From the repository root, fetch and fast-forward the published `origin/main`
containing this package. Do not reset local changes or run a stale worktree.
Use the existing environment that produced the audited parent/GNR1 results.
No package/model update is needed for this inference-only task.

## Preflight

Use the unchanged audited v0.2 / GNR1 result bundles and actual canonical record /
IAPWS grid. Source, input, index and checkpoint hashes must pass. Do not rebuild
indices, change tolerances or change source fingerprints after a failure.

```bash
python -m pytest tests/final_wm/test_fmts_core.py -q
```

Tests include a tiny synthetic parent fixture; no plant training is part of CORE1.

## Execute and audit

```bash
python -m experiments.fmts_core_20260914.run \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --out results/fmts_core_20260914/linux_core1 \
  --device cuda

python -m experiments.fmts_core_20260914.audit \
  --out results/fmts_core_20260914/linux_core1 \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --device cuda \
  --save results/fmts_core_20260914/linux_core1/audit.json
```

Expect **15 inference cells**: 4 same-core variants × 3 seeds + 3 unchanged GNR
references. Audit requires 30 prediction/response array sets and 15 state-trace
sets replayed, `complete=true`, `training_updates=0`. This is not 15 new fits.
Artifact-only audit without the record must return `complete=false`.

C11 must reproduce parent GRU and GNR must reproduce original GNR. Both absolute
temperature and step-minus-base difference gates apply; zero-action identity is
exact. A mismatch stops execution and remains saved; no retry with modified code,
seeds, windows or tolerance. Return failure evidence for author review.

Return the entire NEW output directory: immutable identity and indices,
per-cell report/prediction/response/trace, progress, summary, contrasts, explicit
audit and every failure. Do not overwrite originals or return raw plant records.
State traces include initial state, base/two stepped states and target spray rates;
they are model diagnostics, not measured plant states.

After execution, update ONLY this package's `experiment_state.json` and
`RESULTS.md`: record source/return identity, actual completed cells, failure or
array/trace replay counts, `results_returned=true` after successful publication,
and keep `audited=false` until the author's return review. Do not edit the paper,
parent results or old experiment states. Commit/push this new return directory
and its two status files via the existing Git handoff. On failure, return the
available directory and error evidence with failed status; do not report complete.

## Separate temporal preflight — do not score

The optional metadata module accepts only JSON; it never opens CSV/NPZ or labels:

```bash
python -m experiments.fmts_core_20260914.preflight \
  --metadata analysis/fmts_greybox_core_20260914/temporal_candidate_metadata.json \
  --out /path/to/new_fmts_temporal_preflight.json
```

This returns blockers, not an authorization to evaluate. The existing 13-channel
dataset is NOT a compatible FMTS 23-variable dataset. Historical Phase3.5 source,
manifests and outputs CONFIRM candidate exposure through May 11; it cannot become
a clean independent test by fixing mapping or freezing models/windows. A later
temporal-robustness exploration needs its own registration and explicit prior-use
disclosure. Do not unlock either old test or extension here.
