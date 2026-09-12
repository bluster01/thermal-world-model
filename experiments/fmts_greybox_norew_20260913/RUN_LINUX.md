# FMTS-GNR1 — pure greybox no-rewet, three seeds only

Status: released for Linux on the user's 2026-09-13 push request. Execute from
the origin/main commit containing this package. Do not rerun the old four-arm matrix.
Read [registration](../../docs/fmts2026/PREREG_GREYBOX_NOREW_20260913.md) first.

## Verify

From the reviewed checkout root, confirm source and frozen protocol are committed
and unchanged. Input paths below are the same as the completed v0.2 run; only
filesystem paths/device may be adapted. The runner rejects changed input bytes,
parent source fingerprints, normalization or index sets. Do not update hashes
to make a mismatch pass.

```bash
git status --short
python -m pytest tests/final_wm/test_fmts_greybox_norew.py -q
```

The test includes a synthetic one-seed/two-update smoke and both array replays;
it is not a formal plant experiment. The following is the ONLY new full batch:

```bash
python -m experiments.fmts_greybox_norew_20260913.run \
  --parent results/fmts_mainsteam_20260911/linux_full_v02 \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --out results/fmts_greybox_norew_20260913/linux_full_gnr1 \
  --device cuda
```

It runs `greybox_steady_none_norew` seeds 0/1/2, 24,000-update cap each,
and preserves failures. It refuses an existing output directory. Do not silently
resume or rerun a failed seed; return the failure for review.

## Replay and save an explicit audit record

```bash
python -m experiments.fmts_greybox_norew_20260913.audit \
  --parent results/fmts_mainsteam_20260911/linux_full_v02 \
  --out results/fmts_greybox_norew_20260913/linux_full_gnr1 \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --device cuda \
  --save results/fmts_greybox_norew_20260913/linux_full_gnr1/audit.json
```

Require `runs_checked=3`, `array_sets_replayed=6`, `failed_runs=[]`, `complete=true`.
Return the actual audit JSON, not merely a commit-message assertion.

## Return whitelist

Return this new result directory only: `.gitattributes`, `identity.json`, exact
`indices.npz`, `progress.json`, `summary.json`, `comparison.json`, `audit.json`,
each seed's `ledger.jsonl`, `report.json`, `prediction.npz`, `response.npz`, `best.pt`,
and every failure JSON. Do not alter the existing `linux_full_v02` bundle.
Preserve bytes (`* -text` in the output directory prevents JSON newline rewriting).

Checkpoint weights use the unchanged parent trainer. Its protocol metadata is
finalized to GNR1 on the NEW checkpoint, with the old protocol retained under
`parent_protocol`; the new report stores its final hash. No old checkpoint is edited.

Report negative/null outcomes unchanged. Do not tune response signs, swap
checkpoints by response, unlock test, or claim field-gain correctness.
