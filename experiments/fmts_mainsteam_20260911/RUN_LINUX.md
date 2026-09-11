# FMTS protocol 0.2 — Linux execution

Run from the repository root after receiving this implementation. This is a new
FMTS validation experiment, separate from matrix v0.7 and other ongoing routes.
The full command runs four arms × seeds 0,1,2 (12 runs), sequentially. The test
split has no runner option. Do not edit settings on Linux after seeing results.

## Inputs

- Side A canonical v2.2, including all nine `boundary_ext` channels.
- IAPWS property grid; analytic properties are permitted only for smoke tests.
- `configs/final_wm/channel_mapping_v2.json` for historical extension ranges.
- Audited reference manifest:
  `artifacts/final_wm/v07_full_reissue_v1/sideA/manifest.json`.

The full runner checks the record and properties against that manifest before
training. Different bytes fail closed; investigate the input identity rather
than replacing the expected hashes. The record SHA is
`24da77960e05e3636cc7b97a60a75e9b4ba470a3abb4f8ba920ddf11c6dad1d0` and
property SHA is `9fd7a1dba96a5b968661f644fa185ac755266c85853d689d065432a2d41f6e92`.

## Software verification

```bash
python -m pytest tests/final_wm/test_fmts_rich_pipeline.py -q
python -m experiments.fmts_mainsteam_20260911.smoke_pipeline \
  --out results/fmts_mainsteam_20260911/linux_synthetic_v02
```

Use a new output directory. Smoke produces no model selection or paper verdict.

## Full validation run

Adjust only the input filesystem paths and device to the Linux host:

```bash
python -m experiments.fmts_mainsteam_20260911.run \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --out results/fmts_mainsteam_20260911/linux_full_v02 \
  --device cuda
```

The runner refuses an existing output path, preserves failures in progress and
summary files, and returns nonzero if any arm fails. It does not silently resume
or replace a failed seed. Do not start another full batch automatically.

## Read-only audit

```bash
python -m experiments.fmts_mainsteam_20260911.audit \
  --out results/fmts_mainsteam_20260911/linux_full_v02
```

Add `--record`, `--properties`, and `--device cuda` with the same paths to reload
all checkpoints and replay every saved prediction/response array. Replay uses
`rtol=1e-5, atol=1e-5`; discrepancies are errors requiring inspection.

## Return artifacts

Return `identity.json`, `indices.npz`, `progress.json`, `summary.json`, every run's
`ledger.jsonl`, `report.json`, `prediction.npz`, `response.npz`, and `best.pt`, plus
any failure JSON. Preserve original filenames and byte hashes. Checkpoints carry
normalization, model specification, full protocol and the run identity hash.
Raw arrays include first-future indices, UTC days, per-step predictions/targets,
persistence, both valve counterfactual predictions, actual capped doses, and
support masks. These files support the three-panel figure without further model
training. Base versus stepped response uses the same past history and held base
boundaries; no future extension channels are consumed.

Do not promote the exploratory validation selection into a v0.7 verdict or a
claim of measured plant response fidelity. All four arms remain in the summary.
