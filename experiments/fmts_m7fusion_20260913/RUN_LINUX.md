# FMTS-M7R1 + GNR1: six-run follow-up

Released for Linux with this package. Execute only from the committed published
package; read [M7R1 registration](../../docs/fmts2026/PREREG_M7_FUSION_20260913.md)
and [GNR1 registration](../../docs/fmts2026/PREREG_GREYBOX_NOREW_20260913.md).
No extra seed, sweep, old-arm rerun or locked-test evaluation.

## Preflight

Fetch/pull the reviewed published main with a clean relevant source tree. Parent
source hashes must match `results/fmts_mainsteam_20260911/linux_full_v02`.
Do not change hashes, regenerate inputs or resample windows to bypass a failure.
Exact filesystem paths/device may be adapted; input bytes may not.

```bash
git status --short
python -m pytest tests/final_wm/test_fmts_m7fusion.py tests/final_wm/test_fmts_greybox_norew.py -q
```

Tests use only synthetic two-update training, not paper experiments.

## 1. M7R1: three fresh fusion fits

The runner first diagnoses the six old GRU/token observers on the 256 actual
validation histories, read-only; then trains only the new M7 arm. Abnormal old
health is recorded, not used to select inputs or retune training. Source/data/
nonfinite errors stop execution and must be returned, not worked around.

```bash
python -m experiments.fmts_m7fusion_20260913.run \
  --parent results/fmts_mainsteam_20260911/linux_full_v02 \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --out results/fmts_m7fusion_20260913/linux_full_m7r1 \
  --device cuda

python -m experiments.fmts_m7fusion_20260913.audit \
  --parent results/fmts_mainsteam_20260911/linux_full_v02 \
  --out results/fmts_m7fusion_20260913/linux_full_m7r1 \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --device cuda \
  --save results/fmts_m7fusion_20260913/linux_full_m7r1/audit.json
```

Require 3 complete runs, 6 prediction/response array replays, **9 health replays**
(6 old + 3 new), zero failed runs and `complete=true`. An audit without the record
cannot count as a full replay audit.

## 2. GNR1: three fresh pure-greybox no-rewet fits

This is the unchanged registered task. No observer, loss, budget or selector change.

```bash
python -m experiments.fmts_greybox_norew_20260913.run \
  --parent results/fmts_mainsteam_20260911/linux_full_v02 \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --out results/fmts_greybox_norew_20260913/linux_full_gnr1 \
  --device cuda

python -m experiments.fmts_greybox_norew_20260913.audit \
  --parent results/fmts_mainsteam_20260911/linux_full_v02 \
  --out results/fmts_greybox_norew_20260913/linux_full_gnr1 \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --device cuda \
  --save results/fmts_greybox_norew_20260913/linux_full_gnr1/audit.json
```

Require 3 complete runs, 6 array replays, zero failures. If either output already
exists, inspect/report its status: no overwrite, implicit resume, or extra run.
These two jobs are separate registered questions; do not use one to retune the other.

## Return and stop

Return both complete output directories with `.gitattributes`, immutable source/
input identity, byte-identical indices, all checkpoints, per-run ledgers/reports,
raw prediction/response arrays, comparisons, summaries and explicit audit JSONs.
M7 additionally returns all old/new observer-health arrays. Keep failures. Do not
commit raw plant records or input IAPWS grids again.

Record exact execution commit/device and actual started/completed status. Do not
change registrations or paper/scientific verdicts. Negative model curves alone
do not establish calibrated physical response. Author-side health/response review
of all seeds precedes any revised paper plots; parent v0.2 remains visible.
