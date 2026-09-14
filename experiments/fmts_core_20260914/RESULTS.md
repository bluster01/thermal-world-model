# FMTS-CORE1 status

Registration: [PREREG_CORE_ABLATION_20260914.md](../../docs/fmts2026/PREREG_CORE_ABLATION_20260914.md) · plan [EVIDENCE_PLAN_ZH.md](EVIDENCE_PLAN_ZH.md) · handoff [RUN_LINUX.md](RUN_LINUX.md).

## Linux execution return (2026-09-14) — author audit pending

Executed from published `1ae343f` on the audited existing environment (`device=cuda`, torch 2.11.0+cu130). Return bundle: `results/fmts_core_20260914/linux_core1/` (immutable identity + indices, 15 per-cell report/prediction/response/trace, progress, summary, contrasts, explicit audit). **15/15 inference cells complete** (C00/C10/C01/C11 × seeds 0/1/2 + GNR × seeds 0/1/2); **training_updates=0, locked test not evaluated, no new fitting, no window / tolerance / source changes**. Linux preflight: 11/11 package contract tests passed; parent/GNR fingerprints and index bytes enforced at start. Explicit audit: `complete=true`, runs_checked=15, **30 prediction/response array sets and 15 state-trace sets replayed**, failures=0. C11 reproduces the parent GRU and GNR reproduces the original GNR exactly (max absolute array error 0.0; max step-minus-base difference error 0.0).

### Descriptive fixed-weight outcomes (saved-array replay; author audit pending; NOT plant truth)

H18 cumulative MAE °C, 3-seed mean ± sample SD; two-valve H18 model ΔT (single-valve +0.05 normalized opening against hold-last; 64 registered windows, 0 unsupported windows excluded):

| configuration | H18 MAE °C | valve1 ΔT °C | valve2 ΔT °C |
|---|---:|---:|---:|
| C00 (observer off / closure off) | 3.222132 ± 0.313362 | −0.0058158 | −0.0408534 |
| C10 (observer on / closure off) | 3.242752 ± 0.282960 | −0.0058014 | −0.0409009 |
| C01 (observer off / closure on) | 0.956230 ± 0.076917 | −0.1466473 | −0.1209489 |
| C11 (observer on / closure on) | 0.443139 ± 0.022406 | −0.1472726 | −0.1215690 |
| GNR (independent identification) | 0.973432 ± 0.008449 | −0.0002803 | −0.0028085 |

C00–GNR differences include both the identified physical parameters and the initial-state anchors they determine; this table does not state which configuration is closer to the field. No re-identification, parameter optimization, threshold change or extension scoring was run. Paper numbers, figures and verdicts remain unchanged pending the author's return audit.

## Local verification (pre-release; software only)

CORE1 contract tests passed locally (10, then 11 with the exposure-guard test); the related five-file regression passed 34 before and **35 after** the release-sync permission update (43.08 s), and 35 again after the final progress/array-preservation edit (44.00 s). These are engineering checks; the scientific evidence is the Linux REAL-record inference above. See [VERIFICATION.md](VERIFICATION.md).

## Temporal exposure (unchanged)

Temporal metadata preflight has no scoring route. Source-matched historical Phase3.5 code, run manifests and blocked-model outputs confirm prior development, fitting/selection and evaluation on the March-to-May candidate. It is NOT a clean independent test; fixing its FMTS 23-variable mapping cannot reverse that exposure. See `analysis/fmts_greybox_core_20260914/EXPOSURE_REVIEW_ZH.md`.

## Status

`results_returned=true`; `audited=false` until the author's return review.
